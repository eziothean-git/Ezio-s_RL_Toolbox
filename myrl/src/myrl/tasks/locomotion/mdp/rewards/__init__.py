"""myrl 行走类奖励包。

导入此包会自动将所有 @reward_fn 装饰的函数注册到 RewardLibrary。
调用方只需 `import myrl.tasks.locomotion.mdp.rewards` 即可触发注册。

内置 term：
    locomotion.py:
        track_lin_vel_xy_exp      — 水平线速度跟踪
        track_ang_vel_z_exp       — 偏航角速度跟踪
        feet_air_time_biped       — 双足空中时间

    regularization.py:
        penalize_joint_torque_l2  — 关节力矩惩罚
        penalize_lin_accel        — 根体线加速度惩罚
        penalize_orientation      — Roll/Pitch 姿态偏差惩罚
"""
from __future__ import annotations

# 导入子模块触发 @reward_fn 注册
from . import locomotion, regularization

__all__ = ["locomotion", "regularization"]
