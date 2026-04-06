"""SensorView 抽象及具体传感器 View 实现。

SensorView 是所有传感器视图的基类。具体 View（DepthCameraView 等）
消费 myrl 传感器协议（DepthCameraProto 等），不直接依赖任何后端类型。
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from myrl.core.compat.sensors.protocols import (
    DepthCameraProto,
    HeightScanProto,
    ForceSensorProto,
)
from myrl.core.databus.bus import get_databus as _get_databus
_bus = None


class SensorView(ABC):
    """传感器视图基类。"""

    @property
    @abstractmethod
    def data(self) -> Tensor:
        """传感器默认输出张量。"""
        ...


class ImuView(SensorView):
    """封装 Isaac Lab IMUSensor。"""

    def __init__(self, sensor):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._sensor = sensor
        if _bus:
            _bus.set_labels("robot/sensors/imu/lin_acc_b", [["x", "y", "z"]])
            _bus.set_labels("robot/sensors/imu/ang_vel_b", [["x", "y", "z"]])

    @property
    def data(self) -> Tensor:
        return self.lin_acc_b

    @property
    def lin_acc_b(self) -> Tensor:
        """机体系线加速度 (num_envs, 3)。"""
        r = self._sensor.data.lin_acc_b
        if _bus: _bus.publish("robot/sensors/imu/lin_acc_b", r)
        return r

    @property
    def ang_vel_b(self) -> Tensor:
        """机体系角速度 (num_envs, 3)。"""
        r = self._sensor.data.ang_vel_b
        if _bus: _bus.publish("robot/sensors/imu/ang_vel_b", r)
        return r


class DepthCameraView(SensorView):
    """深度相机 View——消费 DepthCameraProto。

    提供深度图的多种访问方式：原始 2D、展平 1D（供 obs pipeline）、历史帧。
    """

    def __init__(self, source: DepthCameraProto, *,
                 channel_prefix: str = "robot/sensors/depth_camera"):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._src = source
        self._prefix = channel_prefix

    @property
    def data(self) -> Tensor:
        """默认输出——展平深度图。"""
        return self.depth_flat

    @property
    def depth(self) -> Tensor:
        """深度图 (num_envs, H, W)。"""
        r = self._src.depth
        if _bus: _bus.publish(f"{self._prefix}/depth", r)
        return r

    @property
    def depth_flat(self) -> Tensor:
        """展平深度图 (num_envs, H*W)，供 obs pipeline 直接消费。"""
        return self.depth.flatten(start_dim=1)

    @property
    def history(self) -> Tensor | None:
        """深度图历史 (num_envs, T, H, W)，无历史时返回 None。"""
        return self._src.depth_history

    @property
    def resolution(self) -> tuple[int, int]:
        """(H, W) 分辨率。"""
        return self._src.resolution

    @property
    def shape(self) -> tuple[int, int]:
        """单帧深度图 shape (H, W)。"""
        return self._src.resolution


class HeightScanView(SensorView):
    """高度扫描 View——消费 HeightScanProto。

    提供世界系高度、相对高度（去除挂载点）、原始射线击中点。
    """

    def __init__(self, source: HeightScanProto, *,
                 channel_prefix: str = "robot/sensors/height_scan"):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._src = source
        self._prefix = channel_prefix

    @property
    def data(self) -> Tensor:
        """默认输出——世界系高度。"""
        return self.heights_w

    @property
    def heights_w(self) -> Tensor:
        """世界系 Z 高度 (num_envs, num_rays)，inf → 0。"""
        z = self._src.ray_hits_w[..., 2]
        r = torch.where(torch.isinf(z), torch.zeros_like(z), z)
        if _bus: _bus.publish(f"{self._prefix}/heights_w", r)
        return r

    @property
    def heights_rel(self) -> Tensor:
        """相对传感器挂载点高度 (num_envs, num_rays)。"""
        mount_z = self._src.pos_w[:, 2:3]  # (N, 1)
        r = self.heights_w - mount_z
        if _bus: _bus.publish(f"{self._prefix}/heights_rel", r)
        return r

    @property
    def ray_hits_w(self) -> Tensor:
        """原始击中点世界坐标 (num_envs, num_rays, 3)。"""
        return self._src.ray_hits_w

    @property
    def num_rays(self) -> int:
        """射线数量。"""
        return self._src.ray_hits_w.shape[1]


class ForceSensorView(SensorView):
    """力传感器 View——消费 ForceSensorProto。

    提供力向量、力矩（可选）、力的模。
    """

    def __init__(self, source: ForceSensorProto, *,
                 channel_prefix: str = "robot/sensors/force"):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._src = source
        self._prefix = channel_prefix

    @property
    def data(self) -> Tensor:
        """默认输出——力向量。"""
        return self.forces

    @property
    def forces(self) -> Tensor:
        """力向量 (num_envs, num_sensors, 3)。"""
        r = self._src.net_forces_w
        if _bus: _bus.publish(f"{self._prefix}/forces", r)
        return r

    @property
    def torques(self) -> Tensor | None:
        """力矩向量 (num_envs, num_sensors, 3)，3 轴传感器返回 None。"""
        return self._src.net_torques_w

    @property
    def magnitude(self) -> Tensor:
        """力的模 (num_envs, num_sensors)。"""
        r = self.forces.norm(dim=-1)
        if _bus: _bus.publish(f"{self._prefix}/magnitude", r)
        return r
