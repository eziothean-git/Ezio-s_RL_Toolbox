"""ActionMux 插件 — 动作覆盖 MUX。

SONIC 风格：per-env per-joint 地覆盖策略输出。
未被 MUX 的 env/joint 正常接收策略动作，不受影响。

用法（HTTP）：
    POST /debug/mux/set   {"env_id": 0, "joint_idx": 5, "value": 0.3}
    POST /debug/mux/clear  {"env_id": 0}
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from myrl.debug_tools.plugin_base import DebugPlugin

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class ActionMux(DebugPlugin):
    name = "action_mux"

    def attach(self, ctx: DebugContext) -> None:
        # 从 env 获取 action 维度
        if hasattr(ctx.env, "action_manager"):
            num_actions = ctx.env.action_manager.total_action_dim
        else:
            num_actions = ctx.env.num_actions if hasattr(ctx.env, "num_actions") else 0

        ctx._num_actions = num_actions
        ctx.action_override = torch.zeros(
            ctx.num_envs, num_actions, device=ctx.device
        )
        ctx.action_mask = torch.zeros(
            ctx.num_envs, num_actions, dtype=torch.bool, device=ctx.device
        )
        ctx.mux_active_envs = set()

        # 缓存关节名称（供 UI 使用）
        try:
            robot = ctx.env.scene["robot"]
            ctx._joint_names = list(robot.joint_names)
        except Exception:
            ctx._joint_names = [f"joint_{i}" for i in range(num_actions)]

    def pre_step(self, ctx: DebugContext, actions: Tensor) -> Tensor:
        """在 env.step() 前拦截并覆盖 MUX'd envs 的动作。"""
        if not ctx.mux_active_envs:
            return actions

        actions = actions.clone()
        for env_id in ctx.mux_active_envs:
            if env_id >= actions.shape[0]:
                continue
            mask = ctx.action_mask[env_id]
            actions[env_id, mask] = ctx.action_override[env_id, mask]
        return actions

    # ── 外部调用接口 ──────────────────────────────────────────────────

    def set_override(
        self, ctx: DebugContext, env_id: int, joint_idx: int, value: float
    ) -> None:
        """覆盖单个关节的动作值。"""
        ctx.action_override[env_id, joint_idx] = value
        ctx.action_mask[env_id, joint_idx] = True
        ctx.mux_active_envs.add(env_id)

    def set_all_overrides(
        self, ctx: DebugContext, env_id: int, values: list[float]
    ) -> None:
        """覆盖一个 env 的全部动作。"""
        t = torch.tensor(values, device=ctx.device)
        ctx.action_override[env_id, : len(values)] = t
        ctx.action_mask[env_id, : len(values)] = True
        ctx.mux_active_envs.add(env_id)

    def clear_override(
        self, ctx: DebugContext, env_id: int, joint_idx: int | None = None
    ) -> None:
        """清除覆盖。joint_idx=None 清除该 env 全部。"""
        if joint_idx is None:
            ctx.action_mask[env_id].zero_()
            ctx.mux_active_envs.discard(env_id)
        else:
            ctx.action_mask[env_id, joint_idx] = False
            if not ctx.action_mask[env_id].any():
                ctx.mux_active_envs.discard(env_id)

    def clear_all(self, ctx: DebugContext) -> None:
        """清除所有 env 的覆盖。"""
        ctx.action_mask.zero_()
        ctx.mux_active_envs.clear()
