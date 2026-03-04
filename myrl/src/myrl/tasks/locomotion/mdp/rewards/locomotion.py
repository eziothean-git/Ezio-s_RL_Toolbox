"""行走类奖励 term（速度跟踪、空中时间等）。

每个 term = Params 类 + 被 @reward_fn 装饰的函数。
函数签名：(robot: RobotHandle, params: Params) -> Tensor[num_envs]
"""
from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from typing import TYPE_CHECKING

from myrl.core.task.reward_lib import reward_fn

if TYPE_CHECKING:
    from myrl.core.compat.views.robot import RobotHandle


# ── 1. 水平线速度跟踪（指数核） ─────────────────────────────────────

class TrackLinVelXYExpParams(BaseModel):
    std: float = Field(
        0.25,
        ge=0.01,
        le=5.0,
        description="高斯宽度（m/s）；越大奖励越宽容",
        json_schema_extra={"unit": "m/s"},
    )
    command_name: str = Field("base_velocity", description="速度指令名称（CommandManager 中的键）")


@reward_fn(
    description="指数核跟踪水平线速度命令",
    long_description=(
        "exp(-||v_cmd_xy - v_robot_xy||² / std²)\n"
        "适用平地行走，std 控制宽容度。"
    ),
    tags=["locomotion", "command_tracking", "dense", "biped"],
    params=TrackLinVelXYExpParams,
    version="1.0.0",
    author="ezio",
    added_in="2026-03-04",
)
def track_lin_vel_xy_exp(robot: RobotHandle, params: TrackLinVelXYExpParams) -> Tensor:
    cmd = robot.get_command(params.command_name)[:, :2]    # (N, 2) xy 分量
    vel = robot.root_lin_vel_b[:, :2]                      # (N, 2) base frame
    error_sq = ((cmd - vel) ** 2).sum(dim=-1)              # (N,)
    return torch.exp(-error_sq / params.std ** 2)


# ── 2. 偏航角速度跟踪（指数核） ─────────────────────────────────────

class TrackAngVelZExpParams(BaseModel):
    std: float = Field(
        0.25,
        ge=0.01,
        le=5.0,
        description="高斯宽度（rad/s）",
        json_schema_extra={"unit": "rad/s"},
    )
    command_name: str = Field("base_velocity", description="速度指令名称")


@reward_fn(
    description="指数核跟踪偏航角速度命令",
    long_description=(
        "exp(-(ω_cmd_z - ω_robot_z)² / std²)\n"
        "鼓励机器人按指令转向。"
    ),
    tags=["locomotion", "command_tracking", "dense", "biped"],
    params=TrackAngVelZExpParams,
    version="1.0.0",
    author="ezio",
    added_in="2026-03-04",
)
def track_ang_vel_z_exp(robot: RobotHandle, params: TrackAngVelZExpParams) -> Tensor:
    cmd_yaw = robot.get_command(params.command_name)[:, 2]  # (N,) yaw rate
    ang_vel_z = robot.root_ang_vel_b[:, 2]                  # (N,)
    error_sq = (cmd_yaw - ang_vel_z) ** 2
    return torch.exp(-error_sq / params.std ** 2)


# ── 3. 双足空中时间奖励 ──────────────────────────────────────────────

class FeetAirTimeBipedParams(BaseModel):
    foot_body_ids: list[int] = Field(
        description="左右脚 body index（在 ContactSensor 中的索引）"
    )
    threshold: float = Field(
        0.35,
        ge=0.0,
        le=2.0,
        description="最小空中时间阈值（秒）；超过才给奖励",
        json_schema_extra={"unit": "s"},
    )
    sensor_name: str = Field("contact_forces", description="ContactSensor 名称")


@reward_fn(
    description="双足最小空中时间奖励——促进抬脚行走",
    long_description=(
        "当脚离地时间超过 threshold 后奖励 (air_time - threshold)。\n"
        "落地时触发一次奖励（不是持续奖励），防止机器人滑步。"
    ),
    tags=["locomotion", "gait", "biped", "contact"],
    params=FeetAirTimeBipedParams,
    version="1.0.0",
    output_description="scalar reward per environment (sum over two feet)",
    author="ezio",
    added_in="2026-03-04",
)
def feet_air_time_biped(robot: RobotHandle, params: FeetAirTimeBipedParams) -> Tensor:
    contact_view = robot.contacts(
        sensor_name=params.sensor_name,
        body_ids=params.foot_body_ids,
    )
    air_time = contact_view.air_time          # (N, num_feet)
    # 落地时刻触发奖励
    just_landed = contact_view.just_landed    # (N, num_feet) bool
    reward_per_foot = torch.clamp(air_time - params.threshold, min=0.0) * just_landed.float()
    return reward_per_foot.sum(dim=-1)        # (N,)
