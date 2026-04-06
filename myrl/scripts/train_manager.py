#!/usr/bin/env python3
"""train_manager.py — myrl 训练管控服务端

stdlib only，无外部依赖。
默认监听 :7001；代理训练进程内嵌 SSELogServer（默认 :7000）。

用法:
    python myrl/scripts/train_manager.py [--port 7001] [--bind 0.0.0.0]
"""

import argparse
import collections
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from urllib.error import URLError

_START_TIME = time.time()

# Editor 静态文件目录（editor/ 子目录）
_EDITOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")

# 仓库根目录：scripts/ 的上两级
_REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


def _parse_yaml_scalars(text: str) -> dict:
    """从 YAML 文本提取顶层标量字段（不依赖 PyYAML）。"""
    result = {}
    for line in text.splitlines():
        if line.startswith((" ", "\t", "#")) or not line.strip():
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key in ("name", "version", "description"):
            result[key] = val
    return result


# ── SSE 广播器 ─────────────────────────────────────────────────────────────────

class SSEBroadcaster:
    """线程安全 SSE 多播器。每个 SSE 连接通过 subscribe() 获取独立队列。"""

    def __init__(self, maxlen: int = 2000):
        self._clients: set = set()
        self._lock = threading.Lock()
        self._history: collections.deque = collections.deque(maxlen=maxlen)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    def publish(self, event_type: str, payload: dict) -> None:
        data = dict(payload)
        data["type"] = event_type
        data["ts"] = int(time.time())
        line = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
        with self._lock:
            self._history.append(line)
            for q in list(self._clients):
                try:
                    q.put_nowait(line)
                except queue.Full:
                    pass  # 慢客户端丢帧，不阻塞

    def get_history(self, n: int = 0) -> list:
        with self._lock:
            items = list(self._history)
        return items[-n:] if n > 0 else items


# ── 进程控制 ───────────────────────────────────────────────────────────────────

class ProcessCtrl:
    """训练子进程的生命周期管理（启动 / 停止 / 暂停 / 恢复）。

    宿主机模式下，训练进程通过 docker exec 启动。信号通过 docker exec pkill 发送，
    因为 proc.send_signal() 对 docker CLI 进程无效（信号不会传递到容器内的 train.py）。
    """

    def __init__(self, broadcaster: SSEBroadcaster, container: str = "",
                 console_maxlen: int = 2000):
        self._proc: subprocess.Popen = None
        self._task: str = ""
        self._config: dict = {}
        self._state: str = "stopped"  # stopped | starting | running | halted | stopping
        self._start_ts: float = 0.0
        self._console: collections.deque = collections.deque(maxlen=console_maxlen)
        self._lock = threading.Lock()
        self._bc = broadcaster
        self._container = container  # 空 = 直接模式（容器内运行）

    # ── 生命周期 ───────────────────────────────────────────────────────────────

    def start(self, cmd: list, task: str, config: dict) -> tuple:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, "训练进程已在运行"
            self._task = task
            self._config = config
            self._state = "starting"
            self._console.clear()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
        except Exception as e:
            with self._lock:
                self._state = "stopped"
            return False, str(e)
        with self._lock:
            self._proc = proc
            self._start_ts = time.time()
            self._state = "running"
        threading.Thread(target=self._read_output, daemon=True, name="ConsoleReader").start()
        self._bc.publish("status", {"state": "running", "pid": proc.pid, "task": task})
        return True, "OK"

    def _send_signal(self, sig_name: str) -> None:
        """发送信号：容器模式用 docker exec pkill，直接模式用 proc.send_signal。"""
        if self._container:
            # 先发给 train.py，KILL 时额外清理 Isaac Sim 子进程
            subprocess.run(
                ["docker", "exec", self._container, "pkill", f"-{sig_name}", "-f", "train.py"],
                capture_output=True, timeout=5,
            )
            if sig_name == "KILL":
                # 清理可能残留的 Isaac Sim/Kit 子进程
                subprocess.run(
                    ["docker", "exec", self._container,
                     "bash", "-c", "pkill -KILL -f 'isaac-sim|kit/kernel|omni.kit' 2>/dev/null; true"],
                    capture_output=True, timeout=5,
                )
        else:
            proc = self._proc
            if proc is None:
                return
            sig = getattr(signal, f"SIG{sig_name}", None)
            if sig:
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    pass

    def stop(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            self._state = "stopping"
        self._send_signal("TERM")
        self._bc.publish("status", {"state": "stopping"})
        # Isaac Sim 非 headless 模式下 simulation_app.close() 可能挂起，
        # 超时后自动 SIGKILL 确保进程退出
        def _force_kill_after_timeout():
            time.sleep(15)
            with self._lock:
                if self._proc is not None and self._proc.poll() is None:
                    pass  # 还活着
                else:
                    return
            self._send_signal("KILL")
        threading.Thread(target=_force_kill_after_timeout, daemon=True).start()

    def kill(self) -> None:
        with self._lock:
            if self._proc is None:
                return
        self._send_signal("KILL")

    def halt(self) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            self._state = "halted"
        self._send_signal("USR1")
        self._bc.publish("status", {"state": "halted"})

    def resume(self) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            self._state = "running"
        self._send_signal("USR2")
        self._bc.publish("status", {"state": "running"})

    def checkpoint(self, wait_s: float = 10.0) -> None:
        """SIGUSR1 → 等待 wait_s 秒 → SIGUSR2（保存 checkpoint 但不停止）。"""
        self.halt()
        def _resume():
            time.sleep(wait_s)
            self.resume()
        threading.Thread(target=_resume, daemon=True).start()

    # ── 状态查询 ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return "stopped"
            return self._state

    @property
    def pid(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            return self._proc.pid

    @property
    def task(self) -> str:
        return self._task

    @property
    def config(self) -> dict:
        return self._config

    @property
    def uptime(self) -> float:
        return time.time() - self._start_ts if self._start_ts else 0.0

    def get_console(self, n: int = 200) -> list:
        with self._lock:
            items = list(self._console)
        return items[-n:] if n > 0 else items

    # ── 输出捕获 ───────────────────────────────────────────────────────────────

    def _read_output(self) -> None:
        proc = self._proc
        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                with self._lock:
                    self._console.append(line)
                self._bc.publish("console", {"line": line})
        finally:
            proc.stdout.close()
            ret = proc.wait()
            with self._lock:
                self._state = "stopped"
                self._proc = None
            self._bc.publish("status", {"state": "stopped", "returncode": ret})


# ── GPU 指标 ───────────────────────────────────────────────────────────────────

class GPUMetrics:
    """nvidia-smi 轮询 + /proc 系统指标（stdlib only，无需 psutil）。"""

    def __init__(self, poll_interval: float = 2.0):
        self._interval = poll_interval
        self._data: dict = {"gpus": [], "cpu": 0.0, "ram_used": 0, "ram_total": 0}
        self._lock = threading.Lock()
        self._prev_stat: tuple = (0, 1)  # (idle, total)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="GPUMetrics").start()

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def _loop(self) -> None:
        while True:
            try:
                gpus = self._query_gpus()
                cpu = self._query_cpu()
                ram_used, ram_total = self._query_ram()
                with self._lock:
                    self._data = {"gpus": gpus, "cpu": cpu,
                                  "ram_used": ram_used, "ram_total": ram_total}
            except Exception:
                pass
            time.sleep(self._interval)

    def _query_gpus(self) -> list:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            result = []
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) < 6:
                    continue
                try:
                    result.append({
                        "idx": int(p[0]),
                        "util": int(p[1]),
                        "mem_used": int(p[2]),
                        "mem_total": int(p[3]),
                        "temp": int(p[4]),
                        "power": float(p[5]) if p[5] not in ("[N/A]", "N/A") else 0.0,
                    })
                except (ValueError, IndexError):
                    pass
            return result
        except Exception:
            return []

    def _query_cpu(self) -> float:
        try:
            with open("/proc/stat") as f:
                vals = list(map(int, f.readline().split()[1:]))
            idle, total = vals[3], sum(vals)
            p_idle, p_total = self._prev_stat
            d_idle, d_total = idle - p_idle, total - p_total
            self._prev_stat = (idle, total)
            return 100.0 * (1.0 - d_idle / d_total) if d_total else 0.0
        except Exception:
            return 0.0

    def _query_ram(self) -> tuple:
        try:
            info: dict = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k.strip()] = int(v.strip().split()[0])
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            return (total - avail) // 1024, total // 1024  # MB
        except Exception:
            return 0, 0


# ── SSE 代理（from :inner_port）───────────────────────────────────────────────

class SSEProxy:
    """代理训练进程内嵌 SSELogServer（:inner_port/stream）的 LogEvent 到主广播器。"""

    def __init__(self, broadcaster: SSEBroadcaster, inner_port: int = 7000):
        self._bc = broadcaster
        self._port = inner_port
        self._running = False
        # 最新训练快照（供 /status 端点使用）
        self.latest_iteration: int = 0
        self.tot_iter: int = 0
        self.latest_metrics: dict = {}
        self.latest_extras: dict = {}

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="SSEProxy").start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                url = f"http://localhost:{self._port}/stream"
                with urlopen(url, timeout=60) as resp:
                    backoff = 1.0
                    for raw in resp:
                        line = raw.decode("utf-8").rstrip()
                        if line.startswith("data: "):
                            try:
                                ev = json.loads(line[6:])
                                self._on_event(ev)
                            except json.JSONDecodeError:
                                pass
            except URLError:
                time.sleep(min(backoff, 30.0))
                backoff = min(backoff * 1.5, 30.0)
            except Exception:
                time.sleep(5.0)

    def _on_event(self, ev: dict) -> None:
        it = ev.get("iteration", self.latest_iteration)
        metrics = ev.get("metrics", {})
        extras = ev.get("extras", {})
        self.latest_iteration = it
        self.latest_metrics = metrics
        self.latest_extras = extras
        self.tot_iter = extras.get("tot_iter", self.tot_iter)
        self._bc.publish("train", {"iteration": it, "metrics": metrics, "extras": extras})


# ── 管理器主体 ─────────────────────────────────────────────────────────────────

class TrainManager:
    def __init__(self, train_script: str, inner_port: int, log_root: str,
                 container: str = "", compose_file: str = ""):
        self.train_script = train_script
        self.inner_port = inner_port
        self.log_root = log_root
        self.container = container  # 空 = 直接模式（容器内运行）
        self.compose_file = compose_file
        self.bc = SSEBroadcaster()
        self.proc = ProcessCtrl(self.bc, container=container)
        self.gpu = GPUMetrics()
        self.proxy = SSEProxy(self.bc, inner_port)

    def start_background(self) -> None:
        self.gpu.start()
        self.proxy.start()
        threading.Thread(target=self._sys_loop, daemon=True, name="SysLoop").start()

    def _sys_loop(self) -> None:
        while True:
            snap = self.gpu.get()
            self.bc.publish("system", snap)
            time.sleep(2.0)

    def _wrap_cmd(self, inner_cmd: list) -> list:
        """容器模式：用 docker exec + entrypoint 包装命令。直接模式：原样返回。"""
        if self.container:
            return ["docker", "exec", self.container,
                    "/opt/myrl/entrypoint.sh"] + inner_cmd
        return inner_cmd

    def start_training(self, task: str, num_envs: int, extra_args: list,
                       experiment: str = None, headless: bool = True) -> tuple:
        # 容器模式下检查容器是否运行
        if self.container:
            cs = self.container_status()
            if not cs["running"]:
                return False, f"容器 {self.container} 未运行，请先启动"

        # 构造容器内路径的 train.py 命令
        train_script = "/workspace/myrl/scripts/train.py" if self.container else self.train_script

        if experiment:
            yaml_path = _REPO_ROOT / "myrl" / "assets" / "experiments" / f"{experiment}.yaml"
            if not yaml_path.exists():
                return False, f"实验不存在: {yaml_path}"
            pkg_dir = os.path.join(self.log_root, "_packages")
            os.makedirs(pkg_dir, exist_ok=True)
            try:
                from myrl.assets.packager import PackageBuilder
                builder = PackageBuilder.from_yaml_file(str(yaml_path))
                pkg_path = builder.build(pkg_dir)
            except Exception as e:
                return False, f"打包失败: {e}"
            inner = [
                "python3", train_script,
                "--package", pkg_path,
                "--num_envs", str(num_envs),
                "--log_server_port", str(self.inner_port),
            ]
            if headless:
                inner.append("--headless")
            inner += extra_args
            config = {"experiment": experiment, "num_envs": num_envs, "extra_args": extra_args}
            os.makedirs(self.log_root, exist_ok=True)
            return self.proc.start(self._wrap_cmd(inner), f"exp:{experiment}", config)
        else:
            inner = [
                "python3", train_script,
                "--task", task,
                "--num_envs", str(num_envs),
                "--log_server_port", str(self.inner_port),
            ]
            if headless:
                inner.append("--headless")
            inner += extra_args
            config = {"task": task, "num_envs": num_envs, "extra_args": extra_args}
            os.makedirs(self.log_root, exist_ok=True)
            return self.proc.start(self._wrap_cmd(inner), task, config)

    # ── Discovery（无需 Isaac Sim）────────────────────────────────────────────

    def list_experiments(self) -> list:
        """扫描 myrl/assets/experiments/*.yaml，含 tasks 列表。"""
        exp_dir = _REPO_ROOT / "myrl" / "assets" / "experiments"
        results = []
        if not exp_dir.exists():
            return results
        for f in sorted(exp_dir.glob("*.yaml")):
            text = f.read_text(encoding="utf-8")
            if _yaml:
                cfg = _yaml.safe_load(text) or {}
            else:
                cfg = _parse_yaml_scalars(text)
            entry = {
                "name": cfg.get("name", f.stem),
                "version": cfg.get("version", ""),
                "description": cfg.get("description", ""),
                "path": str(f),
            }
            # 新格式：experiment YAML 包含 tasks 列表
            if "tasks" in cfg and isinstance(cfg["tasks"], list):
                entry["tasks"] = cfg["tasks"]
            results.append(entry)
        return results

    def get_experiment(self, name: str) -> dict | None:
        """返回 experiment YAML 全文（JSON 格式）。"""
        yaml_path = _REPO_ROOT / "myrl" / "assets" / "experiments" / f"{name}.yaml"
        if not yaml_path.exists():
            return None
        text = yaml_path.read_text(encoding="utf-8")
        if _yaml:
            return _yaml.safe_load(text)
        return {"_raw": text, "name": name}

    def list_tasks(self) -> list:
        """从 config/__init__.py 提取 gym.register(id=...)，无需 import Isaac Sim。"""
        tasks_dir = _REPO_ROOT / "myrl" / "src" / "myrl" / "tasks"
        pattern = re.compile(r'gym\.register\s*\(\s*id\s*=\s*["\']([^"\']+)')
        results = []
        if not tasks_dir.exists():
            return results
        for init_file in sorted(tasks_dir.rglob("config/*/__init__.py")):
            for m in pattern.finditer(init_file.read_text(encoding="utf-8")):
                task_id = m.group(1)
                results.append({
                    "id": task_id,
                    "type": "play" if "Play" in task_id else "train",
                })
        return results

    # ── 容器管理 ──────────────────────────────────────────────────────────────

    def container_status(self) -> dict:
        if not self.container:
            return {"running": False, "name": "(no container)", "direct": True}
        try:
            out = subprocess.check_output(
                ["docker", "inspect", self.container, "--format", "{{.State.Running}}"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
            return {"running": out == "true", "name": self.container}
        except Exception:
            return {"running": False, "name": self.container}

    def container_start(self) -> tuple:
        if not self.compose_file:
            return False, "no compose file configured"
        r = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "up", "-d"],
            capture_output=True, text=True, timeout=120,
        )
        ok = r.returncode == 0
        return ok, (r.stderr.strip() or "OK") if ok else r.stderr.strip()

    def container_stop(self) -> tuple:
        if not self.compose_file:
            return False, "no compose file configured"
        r = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "down"],
            capture_output=True, text=True, timeout=30,
        )
        ok = r.returncode == 0
        return ok, (r.stderr.strip() or "OK") if ok else r.stderr.strip()

    def get_status(self) -> dict:
        gpu = self.gpu.get()
        extras = self.proxy.latest_extras
        tot_time = extras.get("tot_time", 0)
        start_iter = extras.get("start_iter", 0)
        cur = self.proxy.latest_iteration
        tot = self.proxy.tot_iter
        elapsed = cur - start_iter
        eta_s = (tot_time / elapsed * (tot - cur)) if (elapsed > 0 and tot > cur) else 0.0
        return {
            "state": self.proc.state,
            "pid": self.proc.pid,
            "task": self.proc.task,
            "config": self.proc.config,
            "uptime": self.proc.uptime,
            "iteration": cur,
            "tot_iter": tot,
            "eta_s": eta_s,
            "metrics": self.proxy.latest_metrics,
            **gpu,
        }


# ── Pipeline CRUD 辅助 ────────────────────────────────────────────────────────

_ASSETS_DIR = _REPO_ROOT / "myrl" / "assets"


_reward_schema_cache = None

def _read_reward_schema() -> dict:
    """从 reward_lib 注册表实时读取 schema（自动发现所有 @reward_fn）。
    首次调用时扫描 myrl/tasks/ 下的 reward 模块触发注册，结果缓存。
    """
    global _reward_schema_cache
    if _reward_schema_cache is not None:
        return _reward_schema_cache
    try:
        # 扫描 reward 模块触发 @reward_fn/@transform_fn 注册
        _discover_reward_modules()
        from myrl.core.task.reward_lib import get_reward_library, get_transform_library
        _reward_schema_cache = {
            "terms": get_reward_library().to_dict(),
            "transforms": get_transform_library().to_dict(),
        }
    except Exception as e:
        print(f"[TrainManager] reward schema auto-discover failed: {e}")
        # fallback：读静态 YAML
        _reward_schema_cache = _read_reward_schema_static()
    return _reward_schema_cache


def _discover_reward_modules():
    """扫描 myrl/tasks/**/rewards/*.py，import 触发 @reward_fn 注册。"""
    import importlib.util
    rewards_dir = _REPO_ROOT / "myrl" / "src" / "myrl" / "tasks"
    if not rewards_dir.exists():
        return
    for py in sorted(rewards_dir.rglob("rewards/*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"myrl_reward_scan.{py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[TrainManager] skip {py.name}: {e}")


def _read_reward_schema_static() -> dict:
    """Fallback：读取 reward_schemas/ 下的静态 YAML 文件。"""
    result = {"terms": {}, "transforms": {}}
    for name, key in [("rewards_latest.yaml", "terms"), ("transforms_latest.yaml", "transforms")]:
        path = _ASSETS_DIR / "reward_schemas" / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if _yaml:
            data = _yaml.safe_load(text) or {}
        else:
            data = json.loads(text) if text.strip().startswith("{") else {}
        result[key] = data.get(key, data.get("transforms", {})) if key == "transforms" else data.get("terms", {})
    return result


def _read_pipeline(subdir: str, name: str) -> dict:
    """读取 assets/{subdir}/{name}.yaml 并返回 JSON。"""
    path = _ASSETS_DIR / subdir / f"{name}.yaml"
    if not path.exists():
        return {"error": f"not found: {subdir}/{name}.yaml"}
    text = path.read_text(encoding="utf-8")
    if _yaml:
        return _yaml.safe_load(text) or {}
    return {"_raw": text, "name": name}


def _write_pipeline(subdir: str, name: str, data: dict) -> dict:
    """将修改后的配置写回 YAML 文件（先备份 .bak）。"""
    if not _yaml:
        return {"ok": False, "error": "PyYAML not installed on server"}
    path = _ASSETS_DIR / subdir / f"{name}.yaml"
    if not path.parent.exists():
        return {"ok": False, "error": f"directory not found: {subdir}"}
    # 备份
    if path.exists():
        bak = path.with_suffix(".yaml.bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    # 写入
    try:
        text = _yaml.safe_dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── HTTP 处理器 ────────────────────────────────────────────────────────────────

class HttpHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默 access log

    @property
    def manager(self) -> TrainManager:
        return self.server.manager  # type: ignore[attr-defined]

    @property
    def fleet(self):
        return getattr(self.server, "fleet", None)

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)

        if path == "/" or path == "/ui":
            self._serve_static("index.html")
        elif path.startswith("/editor/"):
            self._serve_static(path[len("/editor/"):])
        elif path == "/health":
            self._json({"status": "ok", "uptime": round(time.time() - _START_TIME, 1)})
        elif path == "/status":
            self._json(self.manager.get_status())
        elif path == "/experiments":
            self._json(self.manager.list_experiments())
        elif path.startswith("/experiment/"):
            name = path[len("/experiment/"):]
            data = self.manager.get_experiment(name)
            if data is None:
                self._json({"error": "not found"}, 404)
            else:
                self._json(data)
        elif path == "/tasks":
            self._json(self.manager.list_tasks())
        elif path == "/container":
            self._json(self.manager.container_status())
        elif path == "/stream":
            filt = qs.get("filter", [""])[0]
            self._sse_stream(filt)
        elif path == "/history":
            n = int(qs.get("n", ["200"])[0])
            lines = self.manager.bc.get_history(n)
            body = "".join(lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/console":
            n = int(qs.get("n", ["200"])[0])
            self._json({"lines": self.manager.proc.get_console(n)})
        # ── Pipeline CRUD endpoints ─────────────────────────────────────────
        elif path == "/reward-schema":
            self._json(_read_reward_schema())
        elif path.startswith("/pipeline/reward/"):
            name = path[len("/pipeline/reward/"):]
            self._json(_read_pipeline("reward_pipelines", name))
        elif path.startswith("/pipeline/obs/"):
            name = path[len("/pipeline/obs/"):]
            self._json(_read_pipeline("obs_pipelines", name))
        elif path.startswith("/pipeline/algo/"):
            name = path[len("/pipeline/algo/"):]
            self._json(_read_pipeline("algo_cfgs", name))
        # ── Fleet GET endpoints ──────────────────────────────────────────────
        elif path == "/fleet" and self.fleet:  # always true
            self._json(self.fleet.list_servers_with_health())
        elif path.startswith("/fleet/") and self.fleet:  # always true
            parts = path.split("/")  # ["", "fleet", server_id, ...]
            if len(parts) >= 4:
                sid = parts[2]
                subpath = "/" + "/".join(parts[3:])
                if subpath == "/stream":
                    filt = qs.get("filter", [""])[0]
                    self._fleet_sse_proxy(sid, filt)
                else:
                    try:
                        code, data = self.fleet.proxy_get(sid, subpath)
                        self._json(data, code)
                    except Exception as e:
                        self._json({"error": str(e)}, 502)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        path = urlparse(self.path).path
        body: dict = {}
        cl = self.headers.get("Content-Length")
        if cl:
            try:
                body = json.loads(self.rfile.read(int(cl)))
            except (json.JSONDecodeError, ValueError):
                pass

        if path == "/start":
            task = body.get("task", "")
            experiment = body.get("experiment", "")
            if not task and not experiment:
                self._json({"ok": False, "error": "task or experiment is required"}, 400)
                return
            extra = list(body.get("extra_args", []))
            max_iter = body.get("max_iterations")
            if max_iter is not None:
                extra.extend(["--max_iterations", str(int(max_iter))])
            signal_port = body.get("signal_server_port")
            if signal_port is not None:
                extra.extend(["--signal_server_port", str(int(signal_port))])
            ok, msg = self.manager.start_training(
                task=task,
                num_envs=int(body.get("num_envs", 16)),
                extra_args=extra,
                experiment=experiment or None,
                headless=body.get("headless", True),
            )
            self._json({"ok": ok, "msg": msg})
        elif path == "/stop":
            self.manager.proc.stop()
            self._json({"ok": True})
        elif path == "/kill":
            self.manager.proc.kill()
            self._json({"ok": True})
        elif path == "/halt":
            self.manager.proc.halt()
            self._json({"ok": True})
        elif path == "/resume":
            self.manager.proc.resume()
            self._json({"ok": True})
        elif path == "/checkpoint":
            self.manager.proc.checkpoint(float(body.get("wait_s", 10.0)))
            self._json({"ok": True})
        # ── Pipeline save endpoints ──────────────────────────────────────────
        elif path.startswith("/pipeline/reward/"):
            name = path[len("/pipeline/reward/"):]
            self._json(_write_pipeline("reward_pipelines", name, body))
        elif path.startswith("/pipeline/obs/"):
            name = path[len("/pipeline/obs/"):]
            self._json(_write_pipeline("obs_pipelines", name, body))
        elif path.startswith("/pipeline/algo/"):
            name = path[len("/pipeline/algo/"):]
            self._json(_write_pipeline("algo_cfgs", name, body))
        elif path == "/container/start":
            ok, msg = self.manager.container_start()
            self._json({"ok": ok, "msg": msg})
        elif path == "/container/stop":
            ok, msg = self.manager.container_stop()
            self._json({"ok": ok, "msg": msg})
        # ── Fleet POST endpoints ─────────────────────────────────────────────
        elif path == "/fleet/add" and self.fleet:  # always true
            ok, msg = self.fleet.add_server(body)
            self._json({"ok": ok, "msg": msg})
        elif path == "/fleet/remove" and self.fleet:  # always true
            ok = self.fleet.remove_server(body.get("id", ""))
            self._json({"ok": ok})
        elif path.startswith("/fleet/") and self.fleet:  # always true
            parts = path.split("/")
            if len(parts) >= 4:
                sid, action = parts[2], parts[3]
                if action == "setup":
                    ok, msg = self.fleet.setup_remote(sid)
                    self._json({"ok": ok, "msg": msg})
                elif action == "sync":
                    ok, msg = self.fleet.sync_code(sid)
                    self._json({"ok": ok, "msg": msg})
                elif action == "deploy":
                    ok, msg = self.fleet.deploy_package(sid, body.get("package", ""))
                    self._json({"ok": ok, "msg": msg})
                elif action == "start-manager":
                    ok, msg = self.fleet.start_remote_manager(sid)
                    self._json({"ok": ok, "msg": msg})
                elif action == "stop-manager":
                    ok, msg = self.fleet.stop_remote_manager(sid)
                    self._json({"ok": ok, "msg": msg})
                elif action == "update":
                    ok, msg = self.fleet.update_server(sid, body)
                    self._json({"ok": ok, "msg": msg})
                else:
                    # 其余路径代理到远程 train_manager
                    subpath = "/" + "/".join(parts[3:])
                    try:
                        code, data = self.fleet.proxy_post(sid, subpath, body)
                        self._json(data, code)
                    except Exception as e:
                        self._json({"error": str(e)}, 502)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    _MIME_MAP = {".js": "application/javascript", ".css": "text/css",
                 ".html": "text/html; charset=utf-8", ".json": "application/json",
                 ".svg": "image/svg+xml", ".png": "image/png"}

    def _serve_static(self, rel_path: str) -> None:
        """从 editor/ 目录提供静态文件。"""
        import mimetypes
        safe = os.path.normpath(rel_path)
        if safe.startswith("..") or os.path.isabs(safe):
            self.send_response(403); self.end_headers(); return
        full = os.path.join(_EDITOR_DIR, safe)
        if not os.path.isfile(full):
            self.send_response(404); self.end_headers()
            self.wfile.write(b"not found"); return
        ext = os.path.splitext(safe)[1].lower()
        mime = self._MIME_MAP.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fleet_sse_proxy(self, server_id: str, filt: str) -> None:
        """SSE 代理：从远程 train_manager 的 /stream 转发到当前客户端。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.flush()
        except Exception:
            return
        try:
            self.fleet.proxy_sse(server_id, self, filt)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_stream(self, filt: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.flush()
        except Exception:
            return
        q = self.manager.bc.subscribe()
        try:
            while True:
                try:
                    line: str = q.get(timeout=20)
                    if filt and f'"type": "{filt}"' not in line:
                        continue
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.manager.bc.unsubscribe(q)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    manager: TrainManager  # injected after construction


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="myrl TrainManager — 训练管控服务端")
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--bind", default="0.0.0.0", help="监听地址（Tailscale 侧传入 100.x.x.x）")
    parser.add_argument("--train-script", default=os.path.join(here, "train.py"),
                        help="train.py 绝对路径")
    parser.add_argument("--inner-port", type=int, default=7000,
                        help="train.py 内嵌 SSELogServer 端口")
    parser.add_argument("--log-root", default=os.path.join(here, "..", "work", "logs"),
                        help="训练日志根目录")
    parser.add_argument("--container", default="myrl-dev",
                        help="Docker 容器名（空=直接模式，即容器内运行）")
    parser.add_argument("--compose-file",
                        default=os.path.join(here, "..", "docker", "compose.yaml"),
                        help="docker compose 文件路径")
    parser.add_argument("--fleet-registry",
                        default=os.path.expanduser("~/.myrl/servers.json"),
                        help="服务器注册表路径")
    args = parser.parse_args()

    # 宿主机模式：确保 myrl 包可导入（experiment packing 需要）
    myrl_src = str(_REPO_ROOT / "myrl" / "src")
    if myrl_src not in sys.path:
        sys.path.insert(0, myrl_src)

    manager = TrainManager(
        train_script=os.path.abspath(args.train_script),
        inner_port=args.inner_port,
        log_root=os.path.abspath(args.log_root),
        container=args.container,
        compose_file=os.path.abspath(args.compose_file),
    )
    manager.start_background()

    from fleet_manager import FleetManager
    fleet = FleetManager(args.fleet_registry, manager.bc, _REPO_ROOT)
    fleet.start_health_loop()

    server = ThreadedHTTPServer((args.bind, args.port), HttpHandler)
    server.manager = manager
    server.fleet = fleet
    print(f"[TrainManager] 监听 {args.bind}:{args.port}  "
          f"(SSELogServer 代理 → :{args.inner_port})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[TrainManager] 收到 Ctrl+C，退出。")


if __name__ == "__main__":
    main()
