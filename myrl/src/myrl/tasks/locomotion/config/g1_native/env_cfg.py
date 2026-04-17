"""G1 原生任务配置：使用 myrl RewardBuilder 驱动奖励。

设计：保留 G1FlatEnvCfg 的场景/物理/终止/事件/命令配置，
用单一 passthrough term（_compute_native_rewards）替换所有奖励，
内部通过 RewardBuilder + myrl reward_lib 计算 6 个 term 的加权和。
"""
from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from myrl.assets import resolve_asset
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)
from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── 资产优先级：myrl/assets/ > instinctlab 内置 ──────────────────────────
_MYRL_G1_URDF = resolve_asset("robots/g1/urdf/g1_29dof_torsobase_popsicle.urdf")

if _MYRL_G1_URDF is not None:
    _orig_spawn = G1_29DOF_TORSOBASE_POPSICLE_CFG.spawn
    _ROBOT_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(
        spawn=_orig_spawn.replace(asset_path=_MYRL_G1_URDF)
    )
else:
    _ROBOT_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG


# ── RewardBuilder 单例（懒初始化，首次 step 时创建） ───────────────────────
_builder = None


def _compute_native_rewards(env: "ManagerBasedRLEnv") -> Tensor:
    """Passthrough 奖励函数：内部调用 RewardBuilder，写 per-term 日志。"""
    global _builder
    if _builder is None:
        from myrl.core.task.reward_builder import RewardBuilder

        _builder = RewardBuilder()

        # 查找脚部 body ID（ankle roll link 对应 ContactSensor 中的索引）
        contact_sensor = env.scene.sensors["contact_forces"]
        foot_body_ids = list(contact_sensor.find_bodies([".*ankle_roll_link"])[0])

        # 注意：add_from_lib 需要 import myrl.tasks 已触发 @reward_fn 装饰器注册
        import myrl.tasks.locomotion.mdp.rewards  # noqa: F401 确保注册

        _builder.add_from_lib("track_lin_vel_xy_exp", weight=1.5, std=0.25)
        _builder.add_from_lib("track_ang_vel_z_exp",  weight=0.75, std=0.25)
        _builder.add_from_lib(
            "feet_air_time_biped",
            weight=0.5,
            foot_body_ids=foot_body_ids,
            threshold=0.35,
        )
        _builder.add_from_lib("penalize_joint_torque_l2", weight=-0.001)
        _builder.add_from_lib("penalize_orientation",     weight=-1.0)
        _builder.add_from_lib("penalize_lin_accel",       weight=-0.01)

    step = int(getattr(env, "common_step_counter", 0))
    total, per_term = _builder.compute(env, step=step)

    # 将 per-term 均值写入 extras["log"]，供日志体系收集
    log = env.extras.setdefault("log", {})
    for name, val in per_term.items():
        log[f"rew/{name}"] = val.mean().item()

    return total


# ── 奖励配置：单一 passthrough term ───────────────────────────────────────

@configclass
class G1NativeRewardsCfg:
    """通过 RewardBuilder 驱动的奖励配置。"""
    native_reward = RewTerm(func=_compute_native_rewards, weight=1.0)


# ── 环境配置 ──────────────────────────────────────────────────────────────

@configclass
class G1NativeEnvCfg(G1FlatEnvCfg):
    """myrl G1 原生任务：RewardBuilder + myrl reward_lib 奖励体系。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 替换奖励为 RewardBuilder passthrough
        self.rewards = G1NativeRewardsCfg()


@configclass
class G1NativeEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play 变体（可视化用）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards = G1NativeRewardsCfg()
