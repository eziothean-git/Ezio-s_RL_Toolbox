"""SignalServer — HTTP server 暴露 DataBus channel 数据（纯 stdlib）。

daemon thread 运行，训练进程退出后自动退出。

HTTP 端点：
    GET /                            → Oscilloscope WebUI（单页 HTML）
    GET /channels                    → JSON list of channel paths
    GET /snapshot?channels=a,b,c     → 各 channel 最新统计（mean/std/shape）
    GET /data?channels=a,b&frames=64 → Tap buffer 原始 float 数组（波形用）
    GET /stream?channels=a,b,c&hz=10 → SSE 实时推送
    GET /health                      → {"ok": true}
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from myrl.core.databus.bus import DataBus
from myrl.core.databus.tap import Tap

# Oscilloscope HTML 文件路径（与本文件同目录）
_HTML_PATH = os.path.join(os.path.dirname(__file__), "oscilloscope.html")


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SignalServer:
    """将 DataBus channel 数据通过 HTTP/SSE 暴露给外部消费者。"""

    def __init__(self, bus: DataBus, host: str = "0.0.0.0", port: int = 7002):
        self._bus = bus
        self._host = host
        self._port = port
        self._server: _ThreadedHTTPServer | None = None
        # 持久化 Tap（不随请求关闭，供 /data 波形拉取）
        self._persistent_taps: dict[str, Tap] = {}
        self._taps_lock = threading.Lock()

    def _get_or_create_tap(self, channel: str) -> Tap:
        """获取或创建持久化 Tap（env_id=0, buffer_len=256）。"""
        tap = self._persistent_taps.get(channel)
        if tap is None:
            with self._taps_lock:
                tap = self._persistent_taps.get(channel)
                if tap is None:
                    tap = self._bus.tap(channel, buffer_len=256, env_id=0)
                    self._persistent_taps[channel] = tap
        return tap

    def start(self) -> None:
        """启动 HTTP server（daemon thread）。"""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                if path == "/" or path == "/ui":
                    self._serve_html()
                elif path == "/health":
                    self._json_response({"ok": True})
                elif path == "/channels":
                    self._json_response(outer._bus.list_channels())
                elif path == "/snapshot":
                    chs = self._parse_channels(params)
                    self._json_response(outer._snapshot(chs))
                elif path == "/data":
                    chs = self._parse_channels(params)
                    frames = int(params.get("frames", [64])[0])
                    self._json_response(outer._get_data(chs, frames))
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

            def _serve_html(self):
                try:
                    with open(_HTML_PATH, "rb") as f:
                        body = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except FileNotFoundError:
                    self.send_error(404, "oscilloscope.html not found")

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

    def _get_data(self, channels: list[str], frames: int) -> dict:
        """返回各 channel 的 Tap buffer 原始值（float 数组）。"""
        result = {}
        for ch in channels:
            tap = self._get_or_create_tap(ch)
            buf = tap.buffer
            if buf.numel() == 0:
                result[ch] = {"values": [], "shape": [], "count": tap.count}
                continue
            # buf shape: (buffered_frames, *per_frame_shape)
            # 取最近 frames 帧
            n = min(frames, buf.shape[0])
            recent = buf[-n:]
            # 对于标量 channel（shape=()），values 就是 flat list
            # 对于向量 channel（shape=(D,)），取 mean 或第一个元素
            if recent.dim() == 1:
                values = recent.float().tolist()
            else:
                # 多维：取每帧的 mean 作为标量波形
                values = recent.float().flatten(1).mean(dim=1).tolist()
            result[ch] = {
                "values": values,
                "shape": list(recent.shape[1:]),
                "count": tap.count,
            }
        return result

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
        for tap in self._persistent_taps.values():
            tap.close()
