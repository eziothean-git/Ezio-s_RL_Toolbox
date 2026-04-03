"""SensorView 抽象及 IMU 实现。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor

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
