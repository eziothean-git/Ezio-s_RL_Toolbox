"""正则化惩罚项（关节力矩、线加速度、姿态偏差等）。

注意：这里的"正则化"指对机器人行为的惩罚性约束，
不是 TransformLibrary 中的奖励后处理归一化算子。
"""
from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from typing import TYPE_CHECKING

from myrl.core.task.reward_lib import reward_fn

if TYPE_CHECKING:
    from myrl.core.compat.views.robot import RobotHandle


# ── 1. 关节力矩 L2 惩罚 ──────────────────────────────────────────────

class PenalizeJointTorqueL2Params(BaseModel):
    joint_ids: list[int] | None = Field(
        None,
        description="限定关节索引，None=所有关节",
    )


@reward_fn(
    description="关节力矩 L2 惩罚——降低能耗",
    long_description=(
        "-sum(tau²) over selected joints\n"
        "用负权重（如 -0.01）添加，鼓励节能步态。"
    ),
    tags=["regularization", "energy", "torque"],
    params=PenalizeJointTorqueL2Params,
    version="1.0.0",
    output_description="non-positive scalar per environment (use negative weight)",
    author="ezio",
    added_in="2026-03-04",
)
def penalize_joint_torque_l2(
    robot: RobotHandle, params: PenalizeJointTorqueL2Params
) -> Tensor:
    torques = robot.joints.applied_torque    # (N, num_joints)
    if params.joint_ids is not None:
        torques = torques[:, params.joint_ids]
    return (torques ** 2).sum(dim=-1)        # (N,)  正值，需负权重


# ── 2. 根体线加速度惩罚 ──────────────────────────────────────────────

class PenalizeLinAccelParams(BaseModel):
    pass  # 无超参数，保留 Params 以统一接口


@reward_fn(
    description="根体线加速度 L2 惩罚——减少冲击",
    long_description=(
        "-||a_root_lin||²（世界系）\n"
        "抑制机器人躯干剧烈震动，提升运动平稳性。"
    ),
    tags=["regularization", "smoothness"],
    params=PenalizeLinAccelParams,
    version="1.0.0",
    output_description="non-positive scalar per environment (use negative weight)",
    author="ezio",
    added_in="2026-03-04",
)
def penalize_lin_accel(robot: RobotHandle, params: PenalizeLinAccelParams) -> Tensor:
    # Isaac Lab Articulation 提供 root_lin_vel_w，用有限差分近似加速度
    # 注意：首帧无法准确计算，通常误差可接受
    accel = robot._asset.data.body_lin_vel_w[:, 0, :]  # 根体 (N, 3)
    return (accel ** 2).sum(dim=-1)                     # (N,)  正值，需负权重


# ── 3. 姿态偏差惩罚（Roll/Pitch） ───────────────────────────────────

class PenalizeOrientationParams(BaseModel):
    pass  # 无超参数


@reward_fn(
    description="Roll/Pitch 偏差惩罚——保持躯干直立",
    long_description=(
        "-||gravity_projected_xy||²（base 系投影重力的 xy 分量）\n"
        "重力向量在 base 坐标系的 xy 分量反映倾斜程度：直立时接近 0。"
    ),
    tags=["regularization", "orientation", "biped"],
    params=PenalizeOrientationParams,
    version="1.0.0",
    output_description="non-positive scalar per environment (use negative weight)",
    author="ezio",
    added_in="2026-03-04",
)
def penalize_orientation(robot: RobotHandle, params: PenalizeOrientationParams) -> Tensor:
    gravity_b = robot.projected_gravity_b    # (N, 3)  base 坐标系重力方向
    return (gravity_b[:, :2] ** 2).sum(dim=-1)   # (N,) roll/pitch 偏差
