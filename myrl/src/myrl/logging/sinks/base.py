"""LogSink ABC — 所有日志后端的统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LogEvent:
    """单次训练迭代的结构化日志事件。

    Attributes:
        iteration:  当前训练迭代号（current_learning_iteration）
        timestamp:  Unix 时间戳（time.time()）
        metrics:    所有 scalar 指标，键为 TensorBoard 路径格式
                    例如 {"Loss/value_loss": 0.12, "Train/mean_reward_0": 1.5}
        extras:     非 scalar 信息，供 TUI 等扩展使用
                    例如 {"tot_timesteps": 12345, "eta_sec": 1200.0}
    """

    iteration: int
    timestamp: float
    metrics: dict[str, float]
    extras: dict = field(default_factory=dict)


class LogSink(ABC):
    """日志后端抽象基类。

    write() 从训练主线程调用，实现者需保证线程安全
    （如 SSELogServer 会将事件跨线程传递给 HTTP handler）。
    """

    @abstractmethod
    def write(self, event: LogEvent) -> None:
        """写入一条日志事件。"""
        ...

    def close(self) -> None:
        """关闭资源（训练结束时调用）。默认空实现。"""
