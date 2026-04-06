"""传感器数据协议——myrl 的设备驱动接口。

各后端（IsaacLab / MuJoCo / 真机）实现这些 Protocol，
View 层只消费 Protocol，不 import 任何后端类型。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DepthCameraProto(Protocol):
    """深度相机数据协议。

    后端驱动需提供深度图和分辨率元数据。
    深度图应已完成 inf→0 替换。
    """

    @property
    def depth(self) -> Tensor:
        """深度图 (num_envs, H, W)，inf 已替换为 0。"""
        ...

    @property
    def resolution(self) -> tuple[int, int]:
        """(H, W) 分辨率。"""
        ...

    @property
    def depth_history(self) -> Tensor | None:
        """深度图历史 (num_envs, T, H, W)，无历史时返回 None。"""
        ...


@runtime_checkable
class HeightScanProto(Protocol):
    """高度扫描数据协议。

    后端驱动需提供射线击中点坐标和传感器挂载点位置。
    """

    @property
    def ray_hits_w(self) -> Tensor:
        """射线击中点世界坐标 (num_envs, num_rays, 3)。"""
        ...

    @property
    def pos_w(self) -> Tensor:
        """传感器挂载点世界坐标 (num_envs, 3)。"""
        ...


@runtime_checkable
class ForceSensorProto(Protocol):
    """力传感器数据协议。

    后端驱动需提供接触力，力矩为可选（3 轴传感器返回 None）。
    """

    @property
    def net_forces_w(self) -> Tensor:
        """接触净力 (num_envs, num_sensors, 3)。"""
        ...

    @property
    def net_torques_w(self) -> Tensor | None:
        """力矩 (num_envs, num_sensors, 3)，3 轴传感器返回 None。"""
        ...
