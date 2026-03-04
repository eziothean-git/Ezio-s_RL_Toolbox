"""WandbSink — Weights & Biases 集成。

wandb 的初始化（wandb.init）由 train.py 负责，使用 sync_tensorboard=True
让 wandb 自动同步 TensorBoard 的所有数据。

此 sink 仅额外记录 LogEvent.extras 中的非 scalar 信息，
避免与 TensorBoard 同步的指标重复。
"""
from __future__ import annotations

from myrl.logging.sinks.base import LogSink, LogEvent


class WandbSink(LogSink):
    """将 LogEvent.extras 中的补充信息写入已初始化的 wandb run。

    使用前提：train.py 已调用 wandb.init(sync_tensorboard=True)，
    即 metrics 由 TensorBoard 同步处理，此 sink 只处理 extras。

    Args:
        log_extras: 要写入 wandb summary 的 extras 字段列表，
                    None 表示不额外写（此时 sink 仅作为占位）
    """

    def __init__(self, log_extras: list[str] | None = None) -> None:
        try:
            import wandb as _wandb
        except ImportError:
            raise ImportError("WandbSink 需要 wandb，请安装：pip install wandb") from None
        self._wandb = _wandb
        self._log_extras = log_extras or []

    def write(self, event: LogEvent) -> None:
        if not self._log_extras:
            return
        # 将指定 extras 字段作为 wandb summary 更新（不重复记录 metrics）
        summary = {
            k: v
            for k in self._log_extras
            if (v := event.extras.get(k)) is not None
        }
        if summary:
            self._wandb.run.summary.update(summary)

    def close(self) -> None:
        pass  # wandb.finish() 由 train.py 负责
