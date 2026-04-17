"""DebugPlugin ABC — 调试工具插件基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class DebugPlugin(ABC):
    """所有调试工具插件的抽象基类。

    每个插件实现一个独立功能（力施加、MUX、锚点等）。
    插件通过 DebugContext 读写共享状态，不直接互相通信。
    """

    name: str  # 子类必须设置，用作注册 key

    @abstractmethod
    def attach(self, ctx: DebugContext) -> None:
        """插件注册时调用。初始化资源、填充 ctx 状态。"""

    def detach(self) -> None:
        """关闭时调用。释放资源。默认无操作。"""

    def pre_step(self, ctx: DebugContext, actions: Tensor) -> Tensor:
        """env.step() 之前调用。可修改 actions 张量。

        默认直接返回原始 actions（pass-through）。
        ActionMux 插件重写此方法实现动作覆盖。
        """
        return actions

    def post_step(
        self,
        ctx: DebugContext,
        obs: Tensor,
        rew: Tensor,
        dones: Tensor,
        extras: dict,
    ) -> None:
        """env.step() 之后调用。用于锚点复位、数据记录等。"""

    def on_render(self, ctx: DebugContext) -> None:
        """sim.render() 期间调用。用于 debug draw 可视化更新。"""
