"""SignalServer — HTTP server 暴露 DataBus channel 数据（纯 stdlib）。

daemon thread 运行，训练进程退出后自动退出。

HTTP 端点：
    GET /                                → Oscilloscope WebUI（单页 HTML）
    GET /info                            → {"num_envs": N}
    GET /channels                        → JSON list of channel paths
    GET /meta?channel=name               → channel 元数据（shape/dim_labels/dtype）
    GET /snapshot?channels=a,b&env_id=0  → 各 channel 最新统计
    GET /data?channels=a,b&env_id=0&dim=3&frames=64 → Tap buffer 标量/向量波形
    GET /stream?env_id=0&hz=10           → SSE 实时推送
    GET /health                          → {"ok": true}
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
        # 持久化 Tap，key = (channel_path, env_id)
        self._persistent_taps: dict[tuple[str, int], Tap] = {}
        self._taps_lock = threading.Lock()

    def _get_or_create_tap(self, channel: str, env_id: int = 0) -> Tap:
        """获取或创建持久化 Tap（buffer_len=256）。"""
        key = (channel, env_id)
        tap = self._persistent_taps.get(key)
        if tap is None:
            with self._taps_lock:
                tap = self._persistent_taps.get(key)
                if tap is None:
                    tap = self._bus.tap(channel, buffer_len=256, env_id=env_id)
                    self._persistent_taps[key] = tap
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
                elif path == "/info":
                    self._json_response(outer._get_info())
                elif path == "/channels":
                    self._json_response(outer._bus.list_channels())
                elif path == "/meta":
                    ch = params.get("channel", [""])[0]
                    self._json_response(outer._get_meta(ch))
                elif path == "/snapshot":
                    chs = self._parse_channels(params)
                    env_id = int(params.get("env_id", [0])[0])
                    self._json_response(outer._snapshot(chs, env_id))
                elif path == "/data":
                    chs = self._parse_channels(params)
                    frames = int(params.get("frames", [64])[0])
                    env_id = int(params.get("env_id", [0])[0])
                    dim = params.get("dim", [None])[0]
                    dim = int(dim) if dim is not None else None
                    self._json_response(outer._get_data(chs, frames, env_id, dim))
                elif path == "/stream":
                    raw_ch = params.get("channels", [""])[0]
                    chs = self._parse_channels(params)
                    hz = float(params.get("hz", [10])[0])
                    env_id = int(params.get("env_id", [0])[0])
                    self._sse_stream(chs, hz, env_id, discover=not raw_ch)
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

            def _sse_stream(self, channels: list[str], hz: float,
                            env_id: int = 0, discover: bool = False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.flush()

                taps: dict[str, Tap] = {}
                for ch in channels:
                    taps[ch] = outer._bus.tap(ch, buffer_len=1, env_id=env_id)

                interval = 1.0 / max(hz, 0.1)
                _DISCOVER_INTERVAL = 3.0
                last_discover = 0.0
                try:
                    while True:
                        # 动态发现新 channel
                        if discover:
                            now = time.time()
                            if now - last_discover > _DISCOVER_INTERVAL:
                                for ch in outer._bus.list_channels():
                                    if ch not in taps:
                                        taps[ch] = outer._bus.tap(ch, buffer_len=1, env_id=env_id)
                                last_discover = now

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
                            snapshot["_meta"] = outer._get_info()
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

    def _get_info(self) -> dict:
        """返回全局元信息（num_envs 从首个有 shape 的 channel 推断）。"""
        num_envs = 0
        for ch_path in self._bus.list_channels():
            info = self._bus.channel_info(ch_path)
            if info and info.shape and len(info.shape) >= 1:
                num_envs = info.shape[0]
                break
        return {"num_envs": num_envs}

    def _get_meta(self, channel: str) -> dict:
        """返回单个 channel 的元数据（shape, dim_labels, dtype）。"""
        info = self._bus.channel_info(channel)
        if info is None:
            return {"error": "channel not found"}
        return {
            "shape": list(info.shape) if info.shape else None,
            "dim_labels": info.dim_labels,
            "dtype": info.dtype,
            "publish_count": info.publish_count,
        }

    def _snapshot(self, channels: list[str], env_id: int = 0) -> dict:
        result = {}
        for ch in channels:
            info = self._bus.channel_info(ch)
            if info is not None:
                result[ch] = {
                    "shape": list(info.shape) if info.shape else None,
                    "dtype": info.dtype,
                    "num_taps": info.num_taps,
                    "publish_count": info.publish_count,
                    "dim_labels": info.dim_labels,
                }
            else:
                result[ch] = None
        return result

    def _get_data(self, channels: list[str], frames: int,
                  env_id: int = 0, dim: int | None = None) -> dict:
        """返回各 channel 的 Tap buffer 值（支持 env_id 和 dim 切片）。"""
        result = {}
        for ch in channels:
            tap = self._get_or_create_tap(ch, env_id)
            buf = tap.buffer
            if buf.numel() == 0:
                result[ch] = {"values": [], "shape": [], "count": tap.count}
                continue
            # buf shape: (buffered_frames, *per_env_shape)
            n = min(frames, buf.shape[0])
            recent = buf[-n:]
            # dim 切片：提取特定维度的标量时序
            if dim is not None and recent.dim() > 1:
                recent = recent[:, dim]
            # 转标量列表
            if recent.dim() == 1:
                values = recent.float().tolist()
            else:
                # 多维未指定 dim：取 mean 兼容旧行为
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
