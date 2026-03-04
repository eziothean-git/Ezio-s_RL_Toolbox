from __future__ import annotations
from functools import wraps
from typing import Callable
from torch import Tensor

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

from .joints import JointView
from .bodies import BodyView
from .contacts import ContactView


class RobotHandle:
    """机器人状态访问句柄，聚合所有 View。每次 make_term() 调用时轻量创建。"""

    def __init__(self, asset: Articulation, env: ManagerBasedEnv):
        self._asset = asset
        self._env = env

    @classmethod
    def from_env(cls, env: ManagerBasedEnv, robot_name: str = "robot") -> RobotHandle:
        asset = env.scene[robot_name]
        return cls(asset, env)

    # ── 根体快捷访问（最常用，避免每次 .bodies.xxx） ──────────────
    @property
    def root_pos_w(self) -> Tensor:
        return self._asset.data.root_pos_w

    @property
    def root_quat_w(self) -> Tensor:
        return self._asset.data.root_quat_w

    @property
    def root_lin_vel_w(self) -> Tensor:
        return self._asset.data.root_lin_vel_w

    @property
    def root_ang_vel_w(self) -> Tensor:
        return self._asset.data.root_ang_vel_w

    @property
    def root_lin_vel_b(self) -> Tensor:
        return self._asset.data.root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> Tensor:
        return self._asset.data.root_ang_vel_b

    @property
    def projected_gravity_b(self) -> Tensor:
        return self._asset.data.projected_gravity_b

    # ── View 访问器 ────────────────────────────────────────────
    @property
    def joints(self) -> JointView:
        return JointView(self._asset)

    @property
    def bodies(self) -> BodyView:
        return BodyView(self._asset)

    def contacts(self, sensor_name: str = "contact_forces",
                 body_ids: list[int] | None = None) -> ContactView:
        sensor = self._env.scene[sensor_name]
        return ContactView(sensor, body_ids)

    # ── 环境上下文 ─────────────────────────────────────────────
    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def step_dt(self) -> float:
        return self._env.step_dt

    @property
    def episode_length_buf(self) -> Tensor:
        return self._env.episode_length_buf

    def get_command(self, command_name: str) -> Tensor:
        """读取 command（ManagerBasedRLEnv 才有 command_manager）。"""
        if not isinstance(self._env, ManagerBasedRLEnv):
            raise RuntimeError("get_command() requires ManagerBasedRLEnv")
        return self._env.command_manager.get_command(command_name)


# ── Bridge 函数 ────────────────────────────────────────────────────────

def make_term(
    fn: Callable[[RobotHandle], Tensor],
    robot_name: str = "robot",
) -> Callable:
    """将 (RobotHandle)->Tensor 包装为 ObservationTermCfg.func 兼容签名
    (env: ManagerBasedEnv, **kwargs)->Tensor。"""
    @wraps(fn)
    def wrapped(env: ManagerBasedEnv, **kwargs) -> Tensor:
        robot = RobotHandle.from_env(env, robot_name)
        return fn(robot)
    wrapped.__myrl_fn__ = fn
    return wrapped


def make_rew(
    fn: Callable[[RobotHandle], Tensor],
    robot_name: str = "robot",
) -> Callable:
    """与 make_term 相同，语义别名用于 reward 函数。"""
    return make_term(fn, robot_name)
