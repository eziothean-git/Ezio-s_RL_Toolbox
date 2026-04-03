"""Channel — DataBus 内部 channel 状态。

每个命名 channel（如 "robot/joints/pos"）对应一个 Channel 实例。
拥有该 channel 的 Tap 列表，负责将 publish 数据分发给所有 Tap。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from torch import Tensor

from myrl.core.databus.tap import Tap


@dataclass
class ChannelInfo:
    """Channel 只读元数据，由 DataBus.channel_info() 返回。"""
    path: str
    shape: tuple[int, ...] | None
    dtype: str | None
    num_taps: int
    publish_count: int


class Channel:
    """单个 channel 的内部状态。非公开 API。"""

    __slots__ = ("path", "_taps", "_lock", "_shape", "_dtype", "_publish_count")

    def __init__(self, path: str) -> None:
        self.path = path
        self._taps: list[Tap] = []
        self._lock = threading.Lock()
        self._shape: tuple[int, ...] | None = None
        self._dtype: str | None = None
        self._publish_count: int = 0

    @property
    def has_taps(self) -> bool:
        """快速检查——bool(list) 在 CPython GIL 下是原子的，无需锁。"""
        return bool(self._taps)

    def add_tap(self, tap: Tap) -> None:
        with self._lock:
            self._taps.append(tap)

    def remove_tap(self, tap: Tap) -> None:
        with self._lock:
            try:
                self._taps.remove(tap)
            except ValueError:
                pass

    def deliver(self, data: Tensor) -> None:
        """将数据推送给所有 Tap。由 sim 线程调用。

        使用 _taps 引用快照迭代（无锁），最坏情况下
        并发增删 Tap 会多发/少发一帧——诊断工具可接受。
        """
        self._publish_count += 1
        if self._shape is None:
            self._shape = tuple(data.shape)
            self._dtype = str(data.dtype)
        taps = self._taps  # 引用快照
        for tap in taps:
            tap._deliver(data)

    def info(self) -> ChannelInfo:
        return ChannelInfo(
            path=self.path,
            shape=self._shape,
            dtype=self._dtype,
            num_taps=len(self._taps),
            publish_count=self._publish_count,
        )
