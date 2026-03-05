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
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from urllib.error import URLError

_START_TIME = time.time()


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
    """训练子进程的生命周期管理（启动 / 停止 / 暂停 / 恢复）。"""

    def __init__(self, broadcaster: SSEBroadcaster, console_maxlen: int = 2000):
        self._proc: subprocess.Popen = None
        self._task: str = ""
        self._config: dict = {}
        self._state: str = "stopped"  # stopped | starting | running | halted | stopping
        self._start_ts: float = 0.0
        self._console: collections.deque = collections.deque(maxlen=console_maxlen)
        self._lock = threading.Lock()
        self._bc = broadcaster

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

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            self._state = "stopping"
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._bc.publish("status", {"state": "stopping"})

    def kill(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def halt(self) -> None:
        if not hasattr(signal, "SIGUSR1"):
            return
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            self._state = "halted"
        try:
            proc.send_signal(signal.SIGUSR1)
        except ProcessLookupError:
            pass
        self._bc.publish("status", {"state": "halted"})

    def resume(self) -> None:
        if not hasattr(signal, "SIGUSR2"):
            return
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            self._state = "running"
        try:
            proc.send_signal(signal.SIGUSR2)
        except ProcessLookupError:
            pass
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
    def __init__(self, train_script: str, inner_port: int, log_root: str):
        self.train_script = train_script
        self.inner_port = inner_port
        self.log_root = log_root
        self.bc = SSEBroadcaster()
        self.proc = ProcessCtrl(self.bc)
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

    def start_training(self, task: str, num_envs: int, extra_args: list) -> tuple:
        cmd = [
            sys.executable, self.train_script,
            "--task", task,
            "--num_envs", str(num_envs),
            "--log_server_port", str(self.inner_port),
            "--headless",
        ] + extra_args
        config = {"task": task, "num_envs": num_envs, "extra_args": extra_args}
        os.makedirs(self.log_root, exist_ok=True)
        return self.proc.start(cmd, task, config)

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


# ── HTTP 处理器 ────────────────────────────────────────────────────────────────

class HttpHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默 access log

    @property
    def manager(self) -> TrainManager:
        return self.server.manager  # type: ignore[attr-defined]

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)

        if path == "/health":
            self._json({"status": "ok", "uptime": round(time.time() - _START_TIME, 1)})
        elif path == "/status":
            self._json(self.manager.get_status())
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
            if not task:
                self._json({"ok": False, "error": "task is required"}, 400)
                return
            ok, msg = self.manager.start_training(
                task,
                int(body.get("num_envs", 16)),
                list(body.get("extra_args", [])),
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
        else:
            self.send_response(404)
            self.end_headers()

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                    if filt and f'"type":"{filt}"' not in line:
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
    args = parser.parse_args()

    manager = TrainManager(
        train_script=os.path.abspath(args.train_script),
        inner_port=args.inner_port,
        log_root=os.path.abspath(args.log_root),
    )
    manager.start_background()

    server = ThreadedHTTPServer((args.bind, args.port), HttpHandler)
    server.manager = manager
    print(f"[TrainManager] 监听 {args.bind}:{args.port}  "
          f"(SSELogServer 代理 → :{args.inner_port})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[TrainManager] 收到 Ctrl+C，退出。")


if __name__ == "__main__":
    main()
