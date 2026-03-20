"""env_script 资产模板 — g1_flat_packaged。

ExperimentComposer 在 import 后注入以下模块级变量：
    _COMPOSER_REWARD_BUILDER  : RewardBuilder 实例（或 None）
    _COMPOSER_ACTUATOR_CFG    : actuator_cfg dict（或 None）
    _COMPOSER_SENSOR_CFG      : sensor_cfg dict（或 None）
    _COMPOSER_TERRAIN_META    : terrain_meta dict（或 None）

约定：模块需暴露 __myrl_env_cfg__ 指向 EnvCfg 类，ExperimentComposer 通过此属性获取配置。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1FlatEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from myrl.core.task.reward_builder import RewardBuilder

# ── ExperimentComposer 注入点 ─────────────────────────────────────────────────
_COMPOSER_REWARD_BUILDER: RewardBuilder | None = None
_COMPOSER_ACTUATOR_CFG: dict | None = None
_COMPOSER_SENSOR_CFG: dict | None = None
_COMPOSER_TERRAIN_META: dict | None = None


# ── 奖励函数 ──────────────────────────────────────────────────────────────────

def _compute_packaged_rewards(env: "ManagerBasedRLEnv"):
    """调用 ExperimentComposer 注入的 RewardBuilder 计算奖励。"""
    if _COMPOSER_REWARD_BUILDER is None:
        raise RuntimeError("RewardBuilder not injected by ExperimentComposer")
    total, per_term = _COMPOSER_REWARD_BUILDER.compute(env)
    log = env.extras.setdefault("log", {})
    for k, v in per_term.items():
        log[f"rew/{k}"] = v.mean().item()
    return total


@configclass
class PackagedRewardsCfg:
    """使用 ExperimentComposer 注入的 RewardBuilder 计算奖励。"""
    packaged_reward = RewTerm(func=_compute_packaged_rewards, weight=1.0)


# ── 环境配置 ──────────────────────────────────────────────────────────────────

def _apply_actuator_cfg(env_cfg, actuator_cfg: dict) -> None:
    """将 actuator_cfg dict 应用到 env_cfg（简化版）。"""
    # 此处仅记录，完整实现需配合 Isaac Lab actuator system
    pass


def _apply_sensor_cfg(env_cfg, sensor_cfg: dict) -> None:
    """将 sensor_cfg dict 应用到 env_cfg（简化版）。"""
    pass


@configclass
class G1FlatPackagedEnvCfg(G1FlatEnvCfg):
    """打包环境配置：保留 G1FlatEnvCfg 场景，替换奖励为 PackagedRewardsCfg。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = PackagedRewardsCfg()
        if _COMPOSER_ACTUATOR_CFG:
            _apply_actuator_cfg(self, _COMPOSER_ACTUATOR_CFG)
        if _COMPOSER_SENSOR_CFG:
            _apply_sensor_cfg(self, _COMPOSER_SENSOR_CFG)


# ExperimentComposer 通过此属性读取 env_cfg 类
__myrl_env_cfg__ = G1FlatPackagedEnvCfg
