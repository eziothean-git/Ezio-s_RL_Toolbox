#!/usr/bin/env python3
"""fleet_manager.py — 远程 GPU 服务器舰队管理

stdlib only，无外部依赖。
由 train_manager.py --fleet 模式导入使用。

职责：
- 服务器注册表 CRUD（~/.myrl/servers.json）
- SSH 隧道生命周期管理
- 远程操作（sync/setup/start-manager/deploy）
- HTTP 代理到远程 train_manager
- 健康检查轮询
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── 常量 ─────────────────────────────────────────────────────────────────────

_SERVER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,31}$")
_TUNNEL_PORT_BASE = 7010
_TUNNEL_PORT_RANGE = 990  # 7010..7999
_HEALTH_INTERVAL = 10.0
_OP_TIMEOUT = 1800  # 30 min（setup 可能很慢）

# rsync 排除列表（与 push.sh 一致）
_RSYNC_EXCLUDES = [
    ".git", "myrl/work/", "__pycache__", "*.pyc",
    ".env", "*.tar.gz", "image.png",
]


# ── FleetManager ─────────────────────────────────────────────────────────────

class FleetManager:
    """管理远程 GPU 服务器舰队。"""

    def __init__(self, registry_path: str, broadcaster, repo_root: Path):
        self._registry_path = registry_path
        self._bc = broadcaster  # SSEBroadcaster (duck-typed)
        self._repo_root = repo_root
        self._tunnels: dict[str, dict] = {}  # server_id -> {proc, local_port}
        self._op_locks: dict[str, threading.Lock] = {}
        self._health_cache: dict[str, dict] = {}
        self._port_counter = 0
        self._lock = threading.Lock()  # 保护 _tunnels / _port_counter

    # ── Registry CRUD ────────────────────────────────────────────────────────

    def _load_registry(self) -> dict:
        if not os.path.exists(self._registry_path):
            return {"version": 1, "servers": {}}
        with open(self._registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
        tmp = self._registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._registry_path)

    def add_server(self, cfg: dict) -> tuple[bool, str]:
        """添加服务器。cfg 需要 name, ssh_host, 可选其他字段。"""
        name = cfg.get("name", "").strip()
        ssh_host = cfg.get("ssh_host", "").strip()
        if not name:
            return False, "name is required"
        if not ssh_host:
            return False, "ssh_host is required"

        # 生成 server_id：name 转小写，非字母数字替换为 -
        sid = re.sub(r"[^a-z0-9]", "-", name.lower()).strip("-")[:32]
        if not _SERVER_ID_RE.match(sid):
            return False, f"invalid server id derived from name: {sid}"

        reg = self._load_registry()
        if sid in reg["servers"]:
            return False, f"server '{sid}' already exists"

        reg["servers"][sid] = {
            "name": name,
            "ssh_host": ssh_host,
            "ssh_port": int(cfg.get("ssh_port", 22)),
            "ssh_key": cfg.get("ssh_key", ""),
            "remote_dir": cfg.get("remote_dir", "~/Ezios_RL_Toolbox"),
            "manager_port": int(cfg.get("manager_port", 7001)),
            "connect_mode": cfg.get("connect_mode", "tunnel"),
            "labels": cfg.get("labels", []),
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_registry(reg)
        return True, sid

    def remove_server(self, server_id: str) -> bool:
        reg = self._load_registry()
        if server_id not in reg["servers"]:
            return False
        del reg["servers"][server_id]
        self._save_registry(reg)
        self._kill_tunnel(server_id)
        self._health_cache.pop(server_id, None)
        return True

    def update_server(self, server_id: str, updates: dict) -> tuple[bool, str]:
        """更新服务器配置字段。"""
        reg = self._load_registry()
        if server_id not in reg["servers"]:
            return False, "server not found"
        for k, v in updates.items():
            if k in ("name", "ssh_host", "ssh_port", "ssh_key", "remote_dir",
                      "manager_port", "connect_mode", "labels"):
                reg["servers"][server_id][k] = v
        self._save_registry(reg)
        return True, "updated"

    def _get_server(self, server_id: str) -> dict | None:
        reg = self._load_registry()
        return reg["servers"].get(server_id)

    def list_servers(self) -> list[str]:
        return list(self._load_registry()["servers"].keys())

    def list_servers_with_health(self) -> dict:
        """返回完整服务器列表 + 健康状态（供 /fleet 端点使用）。"""
        reg = self._load_registry()
        result = {}
        for sid, cfg in reg["servers"].items():
            result[sid] = {
                **cfg,
                "health": self._health_cache.get(sid, {"status": "unknown"}),
            }
        return result

    # ── SSH Tunnel Manager ───────────────────────────────────────────────────

    def _alloc_port(self) -> int:
        with self._lock:
            port = _TUNNEL_PORT_BASE + self._port_counter
            self._port_counter = (self._port_counter + 1) % _TUNNEL_PORT_RANGE
            return port

    def _ssh_base_args(self, cfg: dict) -> list[str]:
        """构建 SSH 基础参数列表。"""
        args = ["ssh"]
        if cfg.get("ssh_key"):
            key_path = os.path.expanduser(cfg["ssh_key"])
            args += ["-i", key_path]
        if cfg.get("ssh_port", 22) != 22:
            args += ["-p", str(cfg["ssh_port"])]
        args += [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
        ]
        return args

    def _ensure_tunnel(self, server_id: str) -> tuple[int, str]:
        """确保 SSH 隧道存活，返回 (local_port, host)。"""
        cfg = self._get_server(server_id)
        if not cfg:
            raise ValueError(f"server not found: {server_id}")

        # direct 模式（Tailscale）：直接连远程端口
        if cfg.get("connect_mode") == "direct":
            host = cfg["ssh_host"].split("@")[-1] if "@" in cfg["ssh_host"] else cfg["ssh_host"]
            return cfg.get("manager_port", 7001), host

        # tunnel 模式：检查已有隧道是否可用（通过快速 HTTP probe）
        with self._lock:
            if server_id in self._tunnels:
                info = self._tunnels[server_id]
                port = info["local_port"]
                try:
                    urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                    return port, "127.0.0.1"
                except Exception:
                    # 隧道可能已死，清理后重建
                    del self._tunnels[server_id]

        # 创建新隧道（ssh -f 会 fork 到后台后退出）
        local_port = self._alloc_port()
        remote_port = cfg.get("manager_port", 7001)

        ssh_args = self._ssh_base_args(cfg)
        ssh_args += [
            "-fNL", f"{local_port}:127.0.0.1:{remote_port}",
            "-o", "ExitOnForwardFailure=yes",
            cfg["ssh_host"],
        ]

        proc = subprocess.Popen(
            ssh_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise ConnectionError(f"SSH tunnel timeout for {server_id}")

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise ConnectionError(f"SSH tunnel failed for {server_id}: {stderr}")

        # ssh -f 成功后前台进程已退出，隧道在后台 SSH 进程中运行
        with self._lock:
            self._tunnels[server_id] = {
                "proc": proc,  # 已退出但保留端口信息
                "local_port": local_port,
                "created_at": time.time(),
            }

        return local_port, "127.0.0.1"

    def _kill_tunnel(self, server_id: str) -> None:
        with self._lock:
            info = self._tunnels.pop(server_id, None)
        if not info:
            return
        port = info["local_port"]
        # ssh -f 已退出，后台 SSH 进程需要通过端口查找
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def _get_tunnel_url(self, server_id: str) -> str:
        """获取可用的代理 URL base（含端口）。"""
        port, host = self._ensure_tunnel(server_id)
        return f"http://{host}:{port}"

    # ── Health Check ─────────────────────────────────────────────────────────

    def _health_check(self, server_id: str) -> dict:
        """检查远程 train_manager 健康状态。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return {"status": "unknown"}

        # 先检查 SSH 连通性
        ssh_args = self._ssh_base_args(cfg)
        ssh_args += [cfg["ssh_host"], "echo", "ok"]
        try:
            r = subprocess.run(
                ssh_args, capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return {"status": "ssh_unreachable"}
        except (subprocess.TimeoutExpired, Exception):
            return {"status": "ssh_unreachable"}

        # 再检查 train_manager 是否响应
        try:
            base = self._get_tunnel_url(server_id)
            with urlopen(f"{base}/health", timeout=5) as resp:
                data = json.loads(resp.read())
                return {"status": "online", **data}
        except Exception:
            return {"status": "manager_down"}

    def start_health_loop(self) -> None:
        threading.Thread(target=self._health_loop, daemon=True,
                         name="FleetHealth").start()

    def _health_loop(self) -> None:
        time.sleep(2.0)  # 启动延迟
        while True:
            sids = self.list_servers()
            for sid in sids:
                try:
                    h = self._health_check(sid)
                    h["checked_at"] = int(time.time())
                    self._health_cache[sid] = h
                except Exception:
                    self._health_cache[sid] = {
                        "status": "error", "checked_at": int(time.time()),
                    }
            if sids:
                self._bc.publish("fleet_health", {
                    "servers": {
                        sid: self._health_cache.get(sid, {"status": "unknown"})
                        for sid in sids
                    },
                })
            time.sleep(_HEALTH_INTERVAL)

    # ── Remote Operations ────────────────────────────────────────────────────

    def _run_op(self, server_id: str, op_name: str, cmd: list,
                timeout: int = _OP_TIMEOUT) -> None:
        """在后台线程中运行命令，逐行流式发布到 SSE。"""
        lock = self._op_locks.setdefault(server_id, threading.Lock())
        if not lock.acquire(blocking=False):
            self._bc.publish("fleet_op", {
                "server": server_id, "op": op_name,
                "status": "error", "msg": "另一操作进行中",
            })
            return
        try:
            self._bc.publish("fleet_op", {
                "server": server_id, "op": op_name, "status": "running",
            })
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self._bc.publish("fleet_console", {
                    "server": server_id, "op": op_name,
                    "line": line.rstrip("\n"),
                })
            rc = proc.wait(timeout=timeout)
            status = "done" if rc == 0 else "error"
            self._bc.publish("fleet_op", {
                "server": server_id, "op": op_name,
                "status": status, "returncode": rc,
            })
        except subprocess.TimeoutExpired:
            proc.kill()
            self._bc.publish("fleet_op", {
                "server": server_id, "op": op_name,
                "status": "error", "msg": f"timeout ({timeout}s)",
            })
        except Exception as e:
            self._bc.publish("fleet_op", {
                "server": server_id, "op": op_name,
                "status": "error", "msg": str(e),
            })
        finally:
            lock.release()

    def _start_op(self, server_id: str, op_name: str, cmd: list,
                  timeout: int = _OP_TIMEOUT) -> tuple[bool, str]:
        """启动后台操作线程。返回 (ok, msg)。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, f"server not found: {server_id}"
        threading.Thread(
            target=self._run_op, args=(server_id, op_name, cmd, timeout),
            daemon=True, name=f"fleet-{op_name}-{server_id}",
        ).start()
        return True, f"{op_name} started"

    def sync_code(self, server_id: str) -> tuple[bool, str]:
        """rsync 代码到远程服务器。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, "server not found"

        cmd = ["rsync", "-avz", "--progress"]
        for excl in _RSYNC_EXCLUDES:
            cmd += ["--exclude", excl]
        # SSH 参数
        ssh_cmd_parts = ["ssh"]
        if cfg.get("ssh_key"):
            ssh_cmd_parts += ["-i", os.path.expanduser(cfg["ssh_key"])]
        if cfg.get("ssh_port", 22) != 22:
            ssh_cmd_parts += ["-p", str(cfg["ssh_port"])]
        cmd += ["-e", " ".join(ssh_cmd_parts)]

        cmd += [
            str(self._repo_root) + "/",
            f"{cfg['ssh_host']}:{cfg['remote_dir']}/",
        ]
        return self._start_op(server_id, "sync", cmd)

    def setup_remote(self, server_id: str) -> tuple[bool, str]:
        """在远程服务器上执行 setup.sh（先同步代码）。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, "server not found"

        def _do_setup():
            lock = self._op_locks.setdefault(server_id, threading.Lock())
            if not lock.acquire(blocking=False):
                self._bc.publish("fleet_op", {
                    "server": server_id, "op": "setup",
                    "status": "error", "msg": "另一操作进行中",
                })
                return
            try:
                # Step 1: sync code
                self._bc.publish("fleet_op", {
                    "server": server_id, "op": "setup", "status": "running",
                    "msg": "Step 1/2: syncing code...",
                })
                sync_cmd = ["rsync", "-avz", "--progress"]
                for excl in _RSYNC_EXCLUDES:
                    sync_cmd += ["--exclude", excl]
                ssh_cmd_parts = ["ssh"]
                if cfg.get("ssh_key"):
                    ssh_cmd_parts += ["-i", os.path.expanduser(cfg["ssh_key"])]
                if cfg.get("ssh_port", 22) != 22:
                    ssh_cmd_parts += ["-p", str(cfg["ssh_port"])]
                sync_cmd += ["-e", " ".join(ssh_cmd_parts)]
                sync_cmd += [
                    str(self._repo_root) + "/",
                    f"{cfg['ssh_host']}:{cfg['remote_dir']}/",
                ]
                proc = subprocess.Popen(
                    sync_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    self._bc.publish("fleet_console", {
                        "server": server_id, "op": "setup",
                        "line": line.rstrip("\n"),
                    })
                rc = proc.wait(timeout=300)
                if rc != 0:
                    self._bc.publish("fleet_op", {
                        "server": server_id, "op": "setup",
                        "status": "error", "msg": f"sync failed (rc={rc})",
                    })
                    return

                # Step 2: run setup.sh
                self._bc.publish("fleet_console", {
                    "server": server_id, "op": "setup",
                    "line": "--- Step 2/2: running setup.sh ---",
                })
                ssh_args = self._ssh_base_args(cfg)
                remote_dir = cfg["remote_dir"]
                ssh_args += [
                    cfg["ssh_host"],
                    f"cd {remote_dir} && bash myrl/scripts/deploy/setup.sh",
                ]
                proc2 = subprocess.Popen(
                    ssh_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc2.stdout:
                    self._bc.publish("fleet_console", {
                        "server": server_id, "op": "setup",
                        "line": line.rstrip("\n"),
                    })
                rc2 = proc2.wait(timeout=_OP_TIMEOUT)
                status = "done" if rc2 == 0 else "error"
                self._bc.publish("fleet_op", {
                    "server": server_id, "op": "setup",
                    "status": status, "returncode": rc2,
                })
            except Exception as e:
                self._bc.publish("fleet_op", {
                    "server": server_id, "op": "setup",
                    "status": "error", "msg": str(e),
                })
            finally:
                lock.release()

        threading.Thread(
            target=_do_setup, daemon=True,
            name=f"fleet-setup-{server_id}",
        ).start()
        return True, "setup started (sync + setup.sh)"

    def start_remote_manager(self, server_id: str) -> tuple[bool, str]:
        """SSH 启动远程 train_manager.py（nohup 后台）。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, "server not found"

        remote_dir = cfg["remote_dir"]
        manager_port = cfg.get("manager_port", 7001)
        ssh_args = self._ssh_base_args(cfg)
        # 先 kill 已有 train_manager，再启动新的
        remote_cmd = (
            f"cd {remote_dir} && "
            f"pkill -f 'train_manager.py.*--port {manager_port}' 2>/dev/null; "
            f"sleep 1; "
            f"nohup python3 myrl/scripts/train_manager.py "
            f"--port {manager_port} --bind 0.0.0.0 "
            f"--compose-file myrl/docker/compose.yaml "
            f"> /tmp/train_manager.log 2>&1 & "
            f"sleep 2 && "
            f"curl -sf http://localhost:{manager_port}/health && "
            f"echo 'train_manager started' || "
            f"echo 'train_manager may have failed to start'"
        )
        ssh_args += [cfg["ssh_host"], remote_cmd]
        return self._start_op(server_id, "start-manager", ssh_args, timeout=30)

    def stop_remote_manager(self, server_id: str) -> tuple[bool, str]:
        """SSH 停止远程 train_manager.py。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, "server not found"

        manager_port = cfg.get("manager_port", 7001)
        ssh_args = self._ssh_base_args(cfg)
        ssh_args += [
            cfg["ssh_host"],
            f"pkill -f 'train_manager.py.*--port {manager_port}' && "
            f"echo 'stopped' || echo 'not running'",
        ]
        return self._start_op(server_id, "stop-manager", ssh_args, timeout=15)

    def deploy_package(self, server_id: str, pkg_path: str) -> tuple[bool, str]:
        """rsync .myrlpkg 到远程服务器。"""
        cfg = self._get_server(server_id)
        if not cfg:
            return False, "server not found"
        if not pkg_path or not os.path.exists(pkg_path):
            return False, f"package not found: {pkg_path}"

        remote_dir = cfg["remote_dir"]
        cmd = ["rsync", "-avz", "--progress"]
        ssh_cmd_parts = ["ssh"]
        if cfg.get("ssh_key"):
            ssh_cmd_parts += ["-i", os.path.expanduser(cfg["ssh_key"])]
        if cfg.get("ssh_port", 22) != 22:
            ssh_cmd_parts += ["-p", str(cfg["ssh_port"])]
        cmd += ["-e", " ".join(ssh_cmd_parts)]
        cmd += [
            pkg_path,
            f"{cfg['ssh_host']}:{remote_dir}/myrl/work/_packages/",
        ]
        return self._start_op(server_id, "deploy", cmd, timeout=300)

    # ── HTTP Proxy ───────────────────────────────────────────────────────────

    def proxy_get(self, server_id: str, path: str) -> tuple[int, dict | str]:
        """GET 请求代理到远程 train_manager。"""
        base = self._get_tunnel_url(server_id)
        try:
            with urlopen(f"{base}{path}", timeout=10) as resp:
                body = resp.read()
                try:
                    return resp.status, json.loads(body)
                except json.JSONDecodeError:
                    return resp.status, {"_raw": body.decode(errors="replace")}
        except URLError as e:
            return 502, {"error": f"proxy failed: {e}"}
        except Exception as e:
            return 502, {"error": str(e)}

    def proxy_post(self, server_id: str, path: str, body: dict) -> tuple[int, dict]:
        """POST 请求代理到远程 train_manager。"""
        base = self._get_tunnel_url(server_id)
        data = json.dumps(body, ensure_ascii=False).encode()
        req = Request(
            f"{base}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except URLError as e:
            return 502, {"error": f"proxy failed: {e}"}
        except Exception as e:
            return 502, {"error": str(e)}

    def proxy_sse(self, server_id: str, handler, filt: str = "") -> None:
        """SSE 流代理：从远程 train_manager 的 /stream 读取并转发给客户端。

        handler 是 BaseHTTPRequestHandler 实例，已发送 SSE headers。
        """
        base = self._get_tunnel_url(server_id)
        url = f"{base}/stream"
        if filt:
            url += f"?filter={filt}"
        try:
            with urlopen(url, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace")
                    if filt and line.startswith("data: "):
                        if f'"type": "{filt}"' not in line:
                            continue
                    handler.wfile.write(line.encode())
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """清理所有 SSH 隧道。"""
        with self._lock:
            for sid in list(self._tunnels.keys()):
                info = self._tunnels.pop(sid, None)
                if info and info["proc"].poll() is None:
                    info["proc"].terminate()
