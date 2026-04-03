"""DataBus — 全局数据总线单例。

Publisher（View / RewardBuilder / ObsBuilder / Backend）调用 publish()
将 tensor 推送到命名 channel。Consumer（Oscilloscope / logger）通过
tap() 订阅。

零开销原则：
  - 未启用 DataBus 时 get_databus() 返回 None，所有 publish 站点短路。
  - DataBus 启用但 channel 无 Tap 时 publish() = dict.get + bool 检查。
"""
from __future__ import annotations

import os
import threading

from torch import Tensor

from myrl.core.databus.channel import Channel, ChannelInfo
from myrl.core.databus.tap import Tap


class DataBus:
    """全局数据总线。通过 get_databus() 获取单例。"""

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._lock = threading.Lock()

    def publish(self, channel_path: str, data: Tensor) -> None:
        """发布 tensor 到 channel。

        热路径：dict.get + bool(list)。
        无 Tap 时不拷贝 tensor、不分配内存。
        """
        ch = self._channels.get(channel_path)
        if ch is not None and ch.has_taps:
            ch.deliver(data)

    def tap(
        self,
        channel_path: str,
        *,
        buffer_len: int = 256,
        downsample: int = 1,
        gain: float = 1.0,
        offset: float = 0.0,
        env_id: int | None = None,
    ) -> Tap:
        """订阅 channel，返回 Tap 句柄。

        channel 不存在时自动创建（publisher 可能还没运行）。

        Args:
            channel_path: 如 "robot/joints/pos"。
            buffer_len: 环形缓冲容量（帧数）。
            downsample: 每 N 帧存 1 帧。
            gain: 读取时的乘法缩放。
            offset: 读取时的加法偏移（在 gain 之后）。
            env_id: 仅缓冲指定环境的数据。
        """
        ch = self._ensure_channel(channel_path)
        t = Tap(
            channel=ch,
            buffer_len=buffer_len,
            downsample=downsample,
            gain=gain,
            offset=offset,
            env_id=env_id,
        )
        ch.add_tap(t)
        return t

    def list_channels(self) -> list[str]:
        """返回所有已知 channel 路径（排序）。"""
        return sorted(self._channels.keys())

    def channel_info(self, channel_path: str) -> ChannelInfo | None:
        """返回 channel 元数据，不存在则返回 None。"""
        ch = self._channels.get(channel_path)
        return ch.info() if ch is not None else None

    def register_channel(self, channel_path: str) -> None:
        """预注册 channel（可选，供 list_channels 发现）。"""
        self._ensure_channel(channel_path)

    def _ensure_channel(self, path: str) -> Channel:
        """获取或创建 Channel（double-checked locking）。"""
        ch = self._channels.get(path)
        if ch is None:
            with self._lock:
                ch = self._channels.get(path)
                if ch is None:
                    ch = Channel(path)
                    self._channels[path] = ch
        return ch


# ── 全局单例 ────────────────────────────────────────────────────

_databus: DataBus | None = None
_databus_lock = threading.Lock()


def get_databus() -> DataBus | None:
    """获取全局 DataBus。未启用时返回 None（Train-time 零开销）。"""
    return _databus


def enable_databus() -> DataBus:
    """启用全局 DataBus 单例。Code-time 调用一次。"""
    global _databus
    with _databus_lock:
        if _databus is None:
            _databus = DataBus()
        return _databus


def disable_databus() -> None:
    """禁用全局 DataBus（测试用）。"""
    global _databus
    with _databus_lock:
        _databus = None


def auto_enable_databus() -> DataBus | None:
    """检查 MYRL_OSCILLOSCOPE 环境变量，设为 1/true 时自动启用。"""
    if os.environ.get("MYRL_OSCILLOSCOPE", "0").lower() in ("1", "true"):
        return enable_databus()
    return None
