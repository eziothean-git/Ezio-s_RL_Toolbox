"""TimeScale 插件 — 时间流速控制。

功能：
    - 慢放（time_scale < 1.0）：每步后 sleep
    - 暂停（paused=True）：自旋等待
    - 单步（single_step_requested）：释放一次暂停

时间控制逻辑在 env_patch.py 的 step() 钩子中执行，
本插件仅提供状态读写接口。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from myrl.debug_tools.plugin_base import DebugPlugin

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class TimeScale(DebugPlugin):
    name = "time_scale"

    def attach(self, ctx: DebugContext) -> None:
        ctx.time_scale = 1.0
        ctx.paused = False
        ctx.single_step_requested = False

    def set_scale(self, ctx: DebugContext, scale: float) -> None:
        """设置时间缩放。1.0=正常，0.5=半速，0.1=10倍慢放。"""
        ctx.time_scale = max(0.01, min(scale, 1.0))

    def toggle_pause(self, ctx: DebugContext) -> None:
        ctx.paused = not ctx.paused

    def pause(self, ctx: DebugContext) -> None:
        ctx.paused = True

    def resume(self, ctx: DebugContext) -> None:
        ctx.paused = False

    def single_step(self, ctx: DebugContext) -> None:
        """从暂停状态释放一步。"""
        ctx.single_step_requested = True
