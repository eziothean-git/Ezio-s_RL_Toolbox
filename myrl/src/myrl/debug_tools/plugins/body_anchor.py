"""BodyAnchor 插件 — 刚体锚定。

选中刚体后 toggle 锚定：捕获当前位姿，每步后复位到该位姿。
效果类似 MuJoCo 的 Ctrl+click 固定刚体。

锚定策略：
    - root body (body_id=0)：write_root_pose_to_sim + 零速度
    - 非 root body：暂不支持（需要关节级锁定，留作 P2）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from myrl.debug_tools.plugin_base import DebugPlugin

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class BodyAnchor(DebugPlugin):
    name = "body_anchor"

    def attach(self, ctx: DebugContext) -> None:
        ctx.anchor_poses = {}
        ctx.anchor_active = False

    def toggle_anchor(
        self, ctx: DebugContext, env_id: int, body_id: int
    ) -> bool:
        """切换锚定状态。返回 True=已锚定，False=已解除。"""
        if env_id in ctx.anchor_poses and body_id in ctx.anchor_poses[env_id]:
            # 解除锚定
            del ctx.anchor_poses[env_id][body_id]
            if not ctx.anchor_poses[env_id]:
                del ctx.anchor_poses[env_id]
            ctx.anchor_active = bool(ctx.anchor_poses)
            return False
        else:
            # 捕获当前位姿并锚定
            art = ctx._articulation
            if art is None:
                return False
            if body_id == 0:
                pos = art.data.root_pos_w[env_id]      # (3,)
                quat = art.data.root_quat_w[env_id]    # (4,)
                pose = torch.cat([pos, quat])           # (7,)
            else:
                pos = art.data.body_pos_w[env_id, body_id]   # (3,)
                quat = art.data.body_quat_w[env_id, body_id] # (4,)
                pose = torch.cat([pos, quat])
            ctx.anchor_poses.setdefault(env_id, {})[body_id] = pose.clone()
            ctx.anchor_active = True
            return True

    def set_anchor(
        self, ctx: DebugContext, env_id: int, body_id: int, pose: Tensor
    ) -> None:
        """直接设置锚定位姿（7 维：xyz + quat）。"""
        ctx.anchor_poses.setdefault(env_id, {})[body_id] = pose.clone()
        ctx.anchor_active = True

    def clear_anchor(self, ctx: DebugContext, env_id: int, body_id: int) -> None:
        """清除特定锚点。"""
        if env_id in ctx.anchor_poses:
            ctx.anchor_poses[env_id].pop(body_id, None)
            if not ctx.anchor_poses[env_id]:
                del ctx.anchor_poses[env_id]
        ctx.anchor_active = bool(ctx.anchor_poses)

    def clear_all(self, ctx: DebugContext) -> None:
        """清除所有锚点。"""
        ctx.anchor_poses.clear()
        ctx.anchor_active = False

    def post_step(self, ctx: DebugContext, obs, rew, dones, extras) -> None:
        """每步后复位锚定体的位姿。"""
        if not ctx.anchor_active or ctx._articulation is None:
            return

        art = ctx._articulation
        for env_id, body_map in ctx.anchor_poses.items():
            for body_id, target_pose in body_map.items():
                if body_id == 0:
                    # root body：直接写入位姿 + 零速度
                    art.write_root_pose_to_sim(
                        target_pose.unsqueeze(0),  # (1, 7)
                        env_ids=torch.tensor([env_id], device=ctx.device),
                    )
                    art.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=ctx.device),
                        env_ids=torch.tensor([env_id], device=ctx.device),
                    )
                # 非 root body 暂不支持，需要关节级约束
