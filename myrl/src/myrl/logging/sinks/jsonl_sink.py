"""JSONLSink — 结构化 JSONL 文件日志。

每次迭代写一行：
    {"iter": 100, "t": 1234567890.123, "metrics": {...}, "extras": {...}}

每行立即 flush，`tail -f metrics.jsonl` 可实时查看。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from myrl.logging.sinks.base import LogSink, LogEvent


class JSONLSink(LogSink):
    """写结构化 JSONL 文件。

    Args:
        log_dir:        日志目录（与 TensorBoard log_dir 相同）
        filename:       文件名，默认 "metrics.jsonl"
        include_extras: 是否写入 extras 字段，默认 True
    """

    def __init__(
        self,
        log_dir: str,
        filename: str = "metrics.jsonl",
        include_extras: bool = True,
    ) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = os.path.join(log_dir, filename)
        self._include_extras = include_extras
        self._file = open(self._path, "a", encoding="utf-8")

    def write(self, event: LogEvent) -> None:
        record: dict = {
            "iter": event.iteration,
            "t": event.timestamp,
            "metrics": event.metrics,
        }
        if self._include_extras and event.extras:
            # 只序列化可 JSON 化的字段
            safe: dict = {}
            for k, v in event.extras.items():
                try:
                    json.dumps(v)
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)
            record["extras"] = safe

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
