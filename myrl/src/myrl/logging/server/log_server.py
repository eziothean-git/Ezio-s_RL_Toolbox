"""SSELogServer — 嵌入训练进程的 HTTP SSE 日志服务端（纯 stdlib，无额外依赖）。

以 daemon thread 运行，训练结束后自动退出。

HTTP 端点：
    GET /stream         Server-Sent Events 实时流（先发历史再推新事件）
    GET /history?n=100  最近 N 条 LogEvent（JSON 数组）
    GET /metrics        最新指标 flat JSON
    GET /health         健康检查

线程安全模型：
    - _history (deque):       主线程 append，handler 线程 list() 快照读取（GIL 保护）
    - _client_queues (set):   _queues_lock 保护增删
    - _latest_metrics (dict): _metrics_lock 保护写，handler 线程 copy 读
    - 每个 client Queue:      Queue 内置线程安全，主线程 put_nowait，handler 线程 get
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from myrl.logging.sinks.base import LogSink, LogEvent


def _event_to_dict(event: LogEvent) -> dict:
    return {
        "iter": event.iteration,
        "t": event.timestamp,
        "metrics": event.metrics,
        "extras": {
            k: v
            for k, v in event.extras.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """每个连接独立线程，互不阻塞。"""
    daemon_threads = True  # handler 线程随进程退出


class SSELogServer(LogSink):
    """SSE 日志服务端，同时也是一个 LogSink。

    训练主线程调用 write(event)，广播给所有已连接的 SSE 客户端。

    Args:
        host:           绑定地址，默认 "0.0.0.0"
        port:           监听端口，默认 7000
        history_maxlen: 内存中保留的最大历史条数，默认 1000
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7000,
        history_maxlen: int = 1000,
    ) -> None:
        self._host = host
        self._port = port
        self._history: deque[LogEvent] = deque(maxlen=history_maxlen)
        self._client_queues: set[queue.Queue] = set()
        self._queues_lock = threading.Lock()
        self._latest_metrics: dict[str, float] = {}
        self._metrics_lock = threading.Lock()

        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                p = parsed.path

                if p == "/health":
                    self._json({"status": "ok", "uptime": time.time()})
                elif p == "/metrics":
                    with server_ref._metrics_lock:
                        data = dict(server_ref._latest_metrics)
                    self._json(data)
                elif p == "/history":
                    n = int(params.get("n", ["100"])[0])
                    snap = list(server_ref._history)[-n:]
                    self._json([_event_to_dict(e) for e in snap])
                elif p == "/stream":
                    self._sse()
                else:
                    self.send_error(404)

            def _json(self, data):
                body = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.flush()  # 必须显式 flush，否则 headers 可能卡在 socket 缓冲区

                client_q: queue.Queue = queue.Queue(maxsize=200)
                with server_ref._queues_lock:
                    server_ref._client_queues.add(client_q)

                try:
                    # catchup：先发完整历史
                    for event in list(server_ref._history):
                        self._send(event)
                    # 实时推送
                    while True:
                        try:
                            event = client_q.get(timeout=30)
                        except queue.Empty:
                            # heartbeat comment，防代理超时断连
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                            continue
                        self._send(event)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with server_ref._queues_lock:
                        server_ref._client_queues.discard(client_q)

            def _send(self, event: LogEvent):
                data = json.dumps(_event_to_dict(event), ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()

            def log_message(self, fmt, *args):
                pass  # 抑制 HTTP 请求日志，避免污染训练输出

        self._httpd = _ThreadedHTTPServer((host, port), _Handler)
        self._httpd.allow_reuse_address = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="SSELogServer",
            daemon=True,
        )
        self._thread.start()

    # ── LogSink 接口 ───────────────────────────────────────────────────────

    def write(self, event: LogEvent) -> None:
        """主线程调用：追加历史 + 广播给所有 SSE 客户端。"""
        self._history.append(event)

        with self._metrics_lock:
            self._latest_metrics.update(event.metrics)

        with self._queues_lock:
            queues_snap = list(self._client_queues)
        for q in queues_snap:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # 客户端处理太慢，丢弃（不阻塞训练）

    def close(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)
