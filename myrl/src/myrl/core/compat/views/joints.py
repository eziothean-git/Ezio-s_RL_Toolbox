from __future__ import annotations
import torch
from torch import Tensor
from isaaclab.assets import Articulation

from myrl.core.databus.bus import get_databus as _get_databus
_bus = None


class JointView:
    """封装 Isaac Lab Articulation 的关节状态读取。"""

    def __init__(self, asset: Articulation, joint_ids: list[int] | None = None):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._asset = asset
        self._ids = joint_ids  # None = 全部关节

    # ── 只读属性 ──────────────────────────────────────────
    @property
    def pos(self) -> Tensor:
        """关节位置 (num_envs, J)。"""
        d = self._asset.data.joint_pos
        result = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/joints/pos", result)
        return result

    @property
    def pos_rel(self) -> Tensor:
        """相对默认姿态的偏差 = pos - default_pos。"""
        result = self.pos - self.default_pos
        if _bus: _bus.publish("robot/joints/pos_rel", result)
        return result

    @property
    def vel(self) -> Tensor:
        d = self._asset.data.joint_vel
        result = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/joints/vel", result)
        return result

    @property
    def acc(self) -> Tensor:
        d = self._asset.data.joint_acc
        result = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/joints/acc", result)
        return result

    @property
    def torque(self) -> Tensor:
        d = self._asset.data.applied_torque
        result = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/joints/torque", result)
        return result

    @property
    def applied_torque(self) -> Tensor:
        """applied_torque 别名（与 reward term 命名保持一致）。"""
        return self.torque

    @property
    def default_pos(self) -> Tensor:
        d = self._asset.data.default_joint_pos
        result = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/joints/default_pos", result)
        return result

    # ── 子集选取 ──────────────────────────────────────────
    def select(self, joint_ids: list[int]) -> JointView:
        """返回关节子集视图（不复制数据）。"""
        return JointView(self._asset, joint_ids)
