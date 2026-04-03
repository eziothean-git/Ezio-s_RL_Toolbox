from __future__ import annotations
from torch import Tensor
from isaaclab.sensors import ContactSensor

from myrl.core.databus.bus import get_databus as _get_databus
_bus = None


class ContactView:
    """封装 Isaac Lab ContactSensor 的读取。"""

    def __init__(self, sensor: ContactSensor, body_ids: list[int] | None = None,
                 step_dt: float = 0.0):
        global _bus
        if _bus is None: _bus = _get_databus()
        self._sensor = sensor
        self._ids = body_ids
        self._step_dt = step_dt

    @property
    def net_forces_w(self) -> Tensor:
        """当前接触净力，世界系 (num_envs, num_bodies, 3)。"""
        d = self._sensor.data.net_forces_w
        r = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/contacts/net_forces_w", r)
        return r

    @property
    def net_forces_w_history(self) -> Tensor:
        """接触净力历史 (num_envs, history, num_bodies, 3)。"""
        d = self._sensor.data.net_forces_w_history
        return d[:, :, self._ids] if self._ids is not None else d

    @property
    def force_magnitude(self) -> Tensor:
        """接触力幅值 (num_envs, num_bodies)。"""
        r = self.net_forces_w.norm(dim=-1)
        if _bus: _bus.publish("robot/contacts/force_magnitude", r)
        return r

    @property
    def in_contact(self) -> Tensor:
        """是否接触 bool (num_envs, num_bodies)。"""
        r = self.force_magnitude > 1.0
        if _bus: _bus.publish("robot/contacts/in_contact", r)
        return r

    @property
    def air_time(self) -> Tensor:
        """当前空中时长 (num_envs, num_bodies)。"""
        d = self._sensor.data.current_air_time
        r = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/contacts/air_time", r)
        return r

    @property
    def contact_time(self) -> Tensor:
        """当前接触时长 (num_envs, num_bodies)。"""
        d = self._sensor.data.current_contact_time
        r = d[:, self._ids] if self._ids is not None else d
        if _bus: _bus.publish("robot/contacts/contact_time", r)
        return r

    @property
    def just_landed(self) -> Tensor:
        """本步骤内首次接触的 mask（需构造时传入 step_dt）。"""
        return self.first_contact(self._step_dt)

    def first_contact(self, step_dt: float) -> Tensor:
        """本步骤内首次接触的 mask (num_envs, num_bodies)。"""
        return (self.contact_time > 0.0) & (self.contact_time < step_dt)

    def first_air(self, step_dt: float) -> Tensor:
        """本步骤内首次离地的 mask (num_envs, num_bodies)。"""
        return (self.air_time > 0.0) & (self.air_time < step_dt)
