from __future__ import annotations
from torch import Tensor
from isaaclab.assets import Articulation

from myrl.core.databus.bus import get_databus as _get_databus
_bus = None


class BodyView:
    """封装 Isaac Lab Articulation 的刚体状态读取。"""

    def __init__(self, asset: Articulation, body_ids: list[int] | None = None):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._asset = asset
        self._ids = body_ids

    # ── 世界系 ────────────────────────────────────────────
    @property
    def root_pos_w(self) -> Tensor:        # (num_envs, 3)
        r = self._asset.data.root_pos_w
        if _bus: _bus.publish("robot/bodies/root_pos_w", r)
        return r

    @property
    def root_quat_w(self) -> Tensor:       # (num_envs, 4) wxyz
        r = self._asset.data.root_quat_w
        if _bus: _bus.publish("robot/bodies/root_quat_w", r)
        return r

    @property
    def root_lin_vel_w(self) -> Tensor:    # (num_envs, 3)
        r = self._asset.data.root_lin_vel_w
        if _bus: _bus.publish("robot/bodies/root_lin_vel_w", r)
        return r

    @property
    def root_ang_vel_w(self) -> Tensor:    # (num_envs, 3)
        r = self._asset.data.root_ang_vel_w
        if _bus: _bus.publish("robot/bodies/root_ang_vel_w", r)
        return r

    # ── 机体系 ────────────────────────────────────────────
    @property
    def root_lin_vel_b(self) -> Tensor:    # (num_envs, 3)
        r = self._asset.data.root_lin_vel_b
        if _bus: _bus.publish("robot/bodies/root_lin_vel_b", r)
        return r

    @property
    def root_ang_vel_b(self) -> Tensor:    # (num_envs, 3)
        r = self._asset.data.root_ang_vel_b
        if _bus: _bus.publish("robot/bodies/root_ang_vel_b", r)
        return r

    @property
    def projected_gravity_b(self) -> Tensor:  # (num_envs, 3)
        r = self._asset.data.projected_gravity_b
        if _bus: _bus.publish("robot/bodies/projected_gravity_b", r)
        return r

    # ── 多体支持（用于 feet 等） ────────────────────────────
    def body_pos_w(self, body_ids: list[int] | None = None) -> Tensor:
        ids = body_ids if body_ids is not None else self._ids
        d = self._asset.data.body_pos_w
        return d[:, ids] if ids is not None else d

    def body_quat_w(self, body_ids: list[int] | None = None) -> Tensor:
        ids = body_ids if body_ids is not None else self._ids
        d = self._asset.data.body_quat_w
        return d[:, ids] if ids is not None else d

    def body_lin_vel_w(self, body_ids: list[int] | None = None) -> Tensor:
        ids = body_ids if body_ids is not None else self._ids
        d = self._asset.data.body_lin_vel_w
        return d[:, ids] if ids is not None else d
