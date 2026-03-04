from __future__ import annotations
from torch import Tensor
from isaaclab.assets import Articulation


class BodyView:
    """封装 Isaac Lab Articulation 的刚体状态读取。"""

    def __init__(self, asset: Articulation, body_ids: list[int] | None = None):
        self._asset = asset
        self._ids = body_ids

    # ── 世界系 ────────────────────────────────────────────
    @property
    def root_pos_w(self) -> Tensor:        # (num_envs, 3)
        return self._asset.data.root_pos_w

    @property
    def root_quat_w(self) -> Tensor:       # (num_envs, 4) wxyz
        return self._asset.data.root_quat_w

    @property
    def root_lin_vel_w(self) -> Tensor:    # (num_envs, 3)
        return self._asset.data.root_lin_vel_w

    @property
    def root_ang_vel_w(self) -> Tensor:    # (num_envs, 3)
        return self._asset.data.root_ang_vel_w

    # ── 机体系 ────────────────────────────────────────────
    @property
    def root_lin_vel_b(self) -> Tensor:    # (num_envs, 3)
        return self._asset.data.root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> Tensor:    # (num_envs, 3)
        return self._asset.data.root_ang_vel_b

    @property
    def projected_gravity_b(self) -> Tensor:  # (num_envs, 3)
        return self._asset.data.projected_gravity_b

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
