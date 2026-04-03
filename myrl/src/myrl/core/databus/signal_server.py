"""SignalServer — HTTP server 暴露 DataBus channel 数据（纯 stdlib）。

daemon thread 运行，训练进程退出后自动退出。

HTTP 端点：
    GET /channels                    → JSON list of channel paths
    GET /snapshot?channels=a,b,c     → 各 channel 最新统计（mean/std/shape）
    GET /stream?channels=a,b,c&hz=10 → SSE 实时推送
    GET /health                      → {"ok": true}
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from myrl.core.databus.bus import DataBus
from myrl.core.databus.tap import Tap


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SignalServer:
    """将 DataBus channel 数据通过 HTTP/SSE 暴露给外部消费者。"""

    def __init__(self, bus: DataBus, host: str = "0.0.0.0", port: int = 7002):
        self._bus = bus
        self._host = host
        self._port = port
        self._server: _ThreadedHTTPServer | None = None

    def start(self) -> None:
        """启动 HTTP server（daemon thread）。"""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass  # 静默 HTTP 日志

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                if path == "/health":
                    self._json_response({"ok": True})
                elif path == "/channels":
                    self._json_response(outer._bus.list_channels())
                elif path == "/snapshot":
                    chs = self._parse_channels(params)
                    self._json_response(outer._snapshot(chs))
                elif path == "/stream":
                    chs = self._parse_channels(params)
                    hz = float(params.get("hz", [10])[0])
                    self._sse_stream(chs, hz)
                else:
                    self.send_error(404)

            def _parse_channels(self, params) -> list[str]:
                raw = params.get("channels", [""])[0]
                if not raw:
                    return outer._bus.list_channels()
                return [c.strip() for c in raw.split(",") if c.strip()]

            def _json_response(self, data):
                body = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse_stream(self, channels: list[str], hz: float):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.flush()

                # 为每个 channel 创建 Tap（env_id=0 降低数据量）
                taps: dict[str, Tap] = {}
                for ch in channels:
                    taps[ch] = outer._bus.tap(ch, buffer_len=1, env_id=0)

                interval = 1.0 / max(hz, 0.1)
                try:
                    while True:
                        snapshot = {}
                        for ch, tap in taps.items():
                            latest = tap.latest
                            if latest is not None:
                                flat = latest.float().flatten()
                                snapshot[ch] = {
                                    "mean": float(flat.mean()),
                                    "std": float(flat.std()) if flat.numel() > 1 else 0.0,
                                    "min": float(flat.min()),
                                    "max": float(flat.max()),
                                    "shape": list(latest.shape),
                                    "count": tap.count,
                                }
                        if snapshot:
                            line = f"data: {json.dumps(snapshot)}\n\n"
                            self.wfile.write(line.encode())
                            self.wfile.flush()
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    for tap in taps.values():
                        tap.close()

        self._server = _ThreadedHTTPServer((self._host, self._port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

    def _snapshot(self, channels: list[str]) -> dict:
        """获取指定 channel 的最新统计。"""
        result = {}
        for ch in channels:
            info = self._bus.channel_info(ch)
            if info is not None:
                result[ch] = {
                    "shape": list(info.shape) if info.shape else None,
                    "dtype": info.dtype,
                    "num_taps": info.num_taps,
                    "publish_count": info.publish_count,
                }
            else:
                result[ch] = None
        return result

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
