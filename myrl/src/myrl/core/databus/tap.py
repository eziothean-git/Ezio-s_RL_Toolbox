"""Tap — DataBus 消费者句柄。

由 DataBus.tap() 创建，消费者通过它读取 channel 数据。
支持环形缓冲、降采样、幅度增益/偏移、单环境切片。
线程安全：sim 线程写（_deliver），UI 线程读（latest/buffer）。
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from myrl.core.databus.channel import Channel


class Tap:
    """订阅句柄。通过 DataBus.tap() 获取，Tap.close() 取消订阅。"""

    __slots__ = (
        "_channel", "_buffer", "_write_idx", "_count", "_buffered",
        "_downsample", "_ds_counter", "_gain", "_offset",
        "_env_id", "_lock", "_closed", "_buffer_len",
    )

    def __init__(
        self,
        channel: Channel,
        buffer_len: int = 256,
        downsample: int = 1,
        gain: float = 1.0,
        offset: float = 0.0,
        env_id: int | None = None,
    ) -> None:
        self._channel = channel
        self._buffer_len = buffer_len
        self._buffer: Tensor | None = None  # 首次 _deliver 时分配
        self._write_idx = 0
        self._count = 0       # 总 _deliver 调用次数
        self._buffered = 0    # 实际写入缓冲区的帧数
        self._downsample = max(1, downsample)
        self._ds_counter = 0
        self._gain = gain
        self._offset = offset
        self._env_id = env_id
        self._lock = threading.Lock()
        self._closed = False

    # ── 消费者 API ─────────────────────────────────────────────

    @property
    def latest(self) -> Tensor | None:
        """最新一帧数据（经 gain/offset 处理后的 clone）。无数据返回 None。"""
        with self._lock:
            if self._buffer is None or self._count == 0:
                return None
            idx = (self._write_idx - 1) % self._buffer_len
            frame = self._buffer[idx].clone()
        return self._apply_transform(frame)

    @property
    def buffer(self) -> Tensor:
        """环形缓冲内容，oldest-first 排列。返回 contiguous clone。
        形状为 (min(buffered_count, buffer_len), *shape)。"""
        with self._lock:
            if self._buffer is None:
                return torch.empty(0)
            n = min(self._buffered, self._buffer_len)
            if n < self._buffer_len:
                out = self._buffer[:n].clone()
            else:
                out = torch.roll(self._buffer, -self._write_idx, dims=0).clone()
        return self._apply_transform(out)

    def _apply_transform(self, data: Tensor) -> Tensor:
        """仅在 gain/offset 非默认值时应用变换，保留原始 dtype。"""
        if self._gain != 1.0 or self._offset != 0.0:
            return data * self._gain + self._offset
        return data

    @property
    def count(self) -> int:
        """_deliver 被调用的总次数（含被降采样跳过的帧）。"""
        return self._count

    @property
    def channel_path(self) -> str:
        return self._channel.path

    # ── 配置 ──────────────────────────────────────────────────

    def set_gain(self, gain: float) -> None:
        self._gain = gain

    def set_offset(self, offset: float) -> None:
        self._offset = offset

    def set_downsample(self, rate: int) -> None:
        self._downsample = max(1, rate)

    def close(self) -> None:
        """取消订阅。幂等。"""
        if self._closed:
            return
        self._closed = True
        self._channel.remove_tap(self)

    # ── 内部（由 Channel.deliver 调用） ───────────────────────

    def _deliver(self, data: Tensor) -> None:
        """接收一帧数据。由 sim 线程调用。"""
        if self._closed:
            return
        self._count += 1

        # 降采样
        self._ds_counter += 1
        if self._ds_counter < self._downsample:
            return
        self._ds_counter = 0

        # env_id 切片
        frame = data[self._env_id] if self._env_id is not None else data

        with self._lock:
            # 首次分配缓冲区
            if self._buffer is None:
                shape = frame.shape
                self._buffer = torch.zeros(
                    self._buffer_len, *shape,
                    dtype=frame.dtype, device=frame.device,
                )
            self._buffer[self._write_idx] = frame.detach()
            self._write_idx = (self._write_idx + 1) % self._buffer_len
            self._buffered += 1
