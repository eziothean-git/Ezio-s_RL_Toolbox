"""ForceApplicator 插件 — 外力施加。

在 Isaac Sim 视口中选中刚体后施加力。
力通过 Articulation.set_external_force_and_torque() 缓冲，
在 scene.write_data_to_sim() 时自动应用（每个 decimation 子步）。

力的生命周期：
    - 默认持续施加，直到 clear_forces() 清除
    - impulse 模式：施加一步后自动清除（由 post_step 处理）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from myrl.debug_tools.plugin_base import DebugPlugin

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class ForceApplicator(DebugPlugin):
    name = "force_applicator"

    def __init__(self) -> None:
        self._impulse_pending: bool = False  # 脉冲模式标记

    def attach(self, ctx: DebugContext) -> None:
        # 获取 Articulation 引用
        try:
            art = ctx.env.scene["robot"]
        except (KeyError, AttributeError):
            print("[force_applicator] No 'robot' in scene, force application disabled")
            return

        ctx._articulation = art
        ctx._body_names = list(art.body_names)
        ctx._num_bodies = art.num_bodies

        # 初始化力缓冲张量
        ctx.external_forces = torch.zeros(
            ctx.num_envs, art.num_bodies, 3, device=ctx.device
        )
        ctx.external_torques = torch.zeros(
            ctx.num_envs, art.num_bodies, 3, device=ctx.device
        )
        ctx.force_active = False

    def set_force(
        self,
        ctx: DebugContext,
        env_id: int,
        body_id: int,
        force_w: list[float],
        torque_w: list[float] | None = None,
        impulse: bool = False,
    ) -> None:
        """设置外力。force_w 在世界坐标系下。"""
        if ctx.external_forces is None:
            return
        ctx.external_forces[env_id, body_id] = torch.tensor(
            force_w, device=ctx.device, dtype=torch.float32
        )
        if torque_w:
            ctx.external_torques[env_id, body_id] = torch.tensor(
                torque_w, device=ctx.device, dtype=torch.float32
            )
        ctx.force_active = True
        self._impulse_pending = impulse

    def clear_forces(self, ctx: DebugContext) -> None:
        """清除所有外力。"""
        if ctx.external_forces is not None:
            ctx.external_forces.zero_()
        if ctx.external_torques is not None:
            ctx.external_torques.zero_()
        ctx.force_active = False
        self._impulse_pending = False

    def clear_body(self, ctx: DebugContext, env_id: int, body_id: int) -> None:
        """清除特定体的外力。"""
        if ctx.external_forces is not None:
            ctx.external_forces[env_id, body_id].zero_()
            ctx.external_torques[env_id, body_id].zero_()
        # 检查是否还有任何力
        if ctx.external_forces is not None and not ctx.external_forces.any():
            ctx.force_active = False

    def post_step(self, ctx: DebugContext, obs, rew, dones, extras) -> None:
        """脉冲模式：施加一步后自动清除。"""
        if self._impulse_pending:
            self.clear_forces(ctx)
