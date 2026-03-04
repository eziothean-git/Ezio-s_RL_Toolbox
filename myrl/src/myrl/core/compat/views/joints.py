from __future__ import annotations
import torch
from torch import Tensor
from isaaclab.assets import Articulation


class JointView:
    """封装 Isaac Lab Articulation 的关节状态读取。"""

    def __init__(self, asset: Articulation, joint_ids: list[int] | None = None):
        self._asset = asset
        self._ids = joint_ids  # None = 全部关节

    # ── 只读属性 ──────────────────────────────────────────
    @property
    def pos(self) -> Tensor:
        """关节位置 (num_envs, J)。"""
        d = self._asset.data.joint_pos
        return d[:, self._ids] if self._ids is not None else d

    @property
    def pos_rel(self) -> Tensor:
        """相对默认姿态的偏差 = pos - default_pos。"""
        return self.pos - self.default_pos

    @property
    def vel(self) -> Tensor:
        d = self._asset.data.joint_vel
        return d[:, self._ids] if self._ids is not None else d

    @property
    def acc(self) -> Tensor:
        d = self._asset.data.joint_acc
        return d[:, self._ids] if self._ids is not None else d

    @property
    def torque(self) -> Tensor:
        d = self._asset.data.applied_torque
        return d[:, self._ids] if self._ids is not None else d

    @property
    def default_pos(self) -> Tensor:
        d = self._asset.data.default_joint_pos
        return d[:, self._ids] if self._ids is not None else d

    # ── 子集选取 ──────────────────────────────────────────
    def select(self, joint_ids: list[int]) -> JointView:
        """返回关节子集视图（不复制数据）。"""
        return JointView(self._asset, joint_ids)
