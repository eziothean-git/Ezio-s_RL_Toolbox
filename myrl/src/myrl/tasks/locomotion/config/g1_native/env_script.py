"""env_script 资产模板 — g1_flat_packaged。

ExperimentComposer 在 import 后注入以下模块级变量：
    _COMPOSER_REWARD_BUILDER  : RewardBuilder 实例（或 None）
    _COMPOSER_OBS_CFG         : obs_pipeline 解析后的 dict（或 None）
    _COMPOSER_ACTUATOR_CFG    : actuator_cfg dict（或 None）
    _COMPOSER_SENSOR_CFG      : sensor_cfg dict（或 None）
    _COMPOSER_TERRAIN_META    : terrain_meta dict（或 None）

约定：模块需暴露 __myrl_env_cfg__ 指向 EnvCfg 类，ExperimentComposer 通过此属性获取配置。
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1FlatEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from myrl.core.task.reward_builder import RewardBuilder

# ── ExperimentComposer 注入点 ─────────────────────────────────────────────────
_COMPOSER_REWARD_BUILDER: RewardBuilder | None = None
_COMPOSER_OBS_CFG: dict | None = None
_COMPOSER_ACTUATOR_CFG: dict | None = None
_COMPOSER_SENSOR_CFG: dict | None = None
_COMPOSER_TERRAIN_META: dict | None = None


# ── 奖励函数 ──────────────────────────────────────────────────────────────────

def _compute_packaged_rewards(env: "ManagerBasedRLEnv"):
    """调用 ExperimentComposer 注入的 RewardBuilder 计算奖励。"""
    if _COMPOSER_REWARD_BUILDER is None:
        raise RuntimeError("RewardBuilder not injected by ExperimentComposer")
    step = int(getattr(env, "common_step_counter", 0))
    total, per_term = _COMPOSER_REWARD_BUILDER.compute(env, step=step)
    log = env.extras.setdefault("log", {})
    for k, v in per_term.items():
        log[f"rew/{k}"] = v.mean().item()
    return total


@configclass
class PackagedRewardsCfg:
    """使用 ExperimentComposer 注入的 RewardBuilder 计算奖励。"""
    packaged_reward = RewTerm(func=_compute_packaged_rewards, weight=1.0)


# ── obs 管线：从 YAML 生成 ObservationsCfg ────────────────────────────────────

def _resolve_obs_func(func_str: str):
    """将 obs_pipeline YAML 中的 func 字符串解析为 Python callable。

    支持格式：
        "mdp.base_ang_vel"           → isaaclab.envs.mdp.base_ang_vel
        "module.path:func"           → 动态 import
        "sensor:{name}.{output}"     → RobotHandle.{sensor_type}(name).{output}
    """
    if func_str.startswith("sensor:"):
        # 传感器观测：sensor:front_depth_camera.depth_flat
        sensor_ref = func_str[7:]
        sensor_name, output_prop = sensor_ref.rsplit(".", 1)

        def _sensor_obs(env, _sn=sensor_name, _op=output_prop):
            from myrl.core.compat.views.robot import RobotHandle
            robot = RobotHandle.from_env(env)
            view = _resolve_sensor_view(robot, env, _sn)
            return getattr(view, _op)

        return _sensor_obs

    if func_str.startswith("mdp."):
        # Isaac Lab 内置 mdp 函数
        from isaaclab.envs import mdp
        attr = func_str[4:]  # 去掉 "mdp." 前缀
        if hasattr(mdp, attr):
            return getattr(mdp, attr)
        # 尝试 instinctlab 的 mdp
        try:
            from instinctlab.tasks.locomotion import mdp as ilab_mdp
            if hasattr(ilab_mdp, attr):
                return getattr(ilab_mdp, attr)
        except ImportError:
            pass
        raise ValueError(f"Cannot resolve obs func '{func_str}': not found in mdp modules")

    if ":" in func_str:
        # module.path:func_name 格式
        module_path, func_name = func_str.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)

    raise ValueError(f"Cannot resolve obs func '{func_str}': unknown format")


def _resolve_sensor_view(robot, env, sensor_name: str):
    """根据 sensor_name 从 env.scene.sensors 查找传感器并返回对应 View。"""
    sensor = env.scene.sensors.get(sensor_name)
    if sensor is None:
        raise ValueError(f"Sensor '{sensor_name}' not found in env.scene.sensors. "
                         f"Available: {list(env.scene.sensors.keys())}")
    # 按 sensor 对象的类型名判断
    cls_name = type(sensor).__name__.lower()
    if "camera" in cls_name or "raycaster" in cls_name and "camera" in cls_name:
        return robot.depth_camera(sensor_name)
    if "raycaster" in cls_name or "height" in cls_name:
        return robot.height_scan(sensor_name)
    if "contact" in cls_name or "force" in cls_name:
        return robot.force_sensor(sensor_name)
    # fallback: 尝试作为深度相机
    return robot.depth_camera(sensor_name)


def _build_obs_group_from_cfg(group_cfg: dict) -> type:
    """从 obs_cfg dict 动态生成 ObsGroup @configclass。

    Args:
        group_cfg: {"base_ang_vel": {"func": "mdp.base_ang_vel", "scale": 0.25, ...}, ...}

    Returns:
        动态生成的 @configclass 类，包含 ObsTerm 成员。
    """
    attrs = {}
    for term_name, term_def in group_cfg.items():
        func = _resolve_obs_func(term_def["func"])
        scale = term_def.get("scale", 1.0)
        kwargs = {}
        noise_cfg = term_def.get("noise")
        if noise_cfg:
            kwargs["noise"] = noise_cfg  # TODO: 转换为 NoiseCfg 对象
        attrs[term_name] = ObsTerm(func=func, scale=scale, **kwargs)

    # 动态创建 @configclass
    cls = type("PackagedObsGroupCfg", (ObsGroup,), attrs)
    return configclass(cls)


def _apply_obs_cfg(env_cfg, obs_cfg: dict) -> None:
    """将 obs_pipeline dict 应用到 env_cfg.observations。

    覆盖 policy（必须）和 critic（可选）观测组。
    """
    if "policy" in obs_cfg:
        policy_cls = _build_obs_group_from_cfg(obs_cfg["policy"])
        env_cfg.observations.policy = policy_cls()
    if "critic" in obs_cfg:
        critic_cls = _build_obs_group_from_cfg(obs_cfg["critic"])
        env_cfg.observations.critic = critic_cls()


# ── actuator / sensor 配置应用 ─────────────────────────────────────────────────

def _apply_actuator_cfg(env_cfg, actuator_cfg: dict) -> None:
    """将 actuator_cfg dict 应用到 env_cfg。

    支持 default_gains 和 joint_overrides。
    """
    robot_cfg = env_cfg.scene.robot
    if not hasattr(robot_cfg, "actuators"):
        return
    default_kp = actuator_cfg.get("default_gains", {}).get("kp")
    default_kd = actuator_cfg.get("default_gains", {}).get("kd")
    for act_name, act_cfg in robot_cfg.actuators.items():
        if default_kp is not None and hasattr(act_cfg, "stiffness"):
            act_cfg.stiffness = default_kp
        if default_kd is not None and hasattr(act_cfg, "damping"):
            act_cfg.damping = default_kd
    # joint_overrides
    for joint_name, overrides in actuator_cfg.get("joint_overrides", {}).items():
        for act_name, act_cfg in robot_cfg.actuators.items():
            if hasattr(act_cfg, "joint_names_expr"):
                for expr in (act_cfg.joint_names_expr if isinstance(act_cfg.joint_names_expr, list)
                             else [act_cfg.joint_names_expr]):
                    if re.match(expr, joint_name):
                        if "kp" in overrides and hasattr(act_cfg, "stiffness"):
                            act_cfg.stiffness = overrides["kp"]
                        if "kd" in overrides and hasattr(act_cfg, "damping"):
                            act_cfg.damping = overrides["kd"]


def _apply_sensor_cfg(env_cfg, sensor_cfg: dict) -> None:
    """将 sensor_cfg dict 应用到 env_cfg。

    主要设置 contact sensor 的 history_length 和 track_air_time。
    """
    sensors = sensor_cfg.get("sensors", [])
    for sensor_def in sensors:
        sensor_type = sensor_def.get("type")
        sensor_name = sensor_def.get("name")
        if sensor_type == "contact" and hasattr(env_cfg.scene, sensor_name):
            scene_sensor = getattr(env_cfg.scene, sensor_name)
            if "history_length" in sensor_def and hasattr(scene_sensor, "history_length"):
                scene_sensor.history_length = sensor_def["history_length"]
            if "track_air_time" in sensor_def and hasattr(scene_sensor, "track_air_time"):
                scene_sensor.track_air_time = sensor_def["track_air_time"]


# ── 环境配置 ──────────────────────────────────────────────────────────────────

@configclass
class G1FlatPackagedEnvCfg(G1FlatEnvCfg):
    """打包环境配置：保留 G1FlatEnvCfg 场景，替换奖励为 PackagedRewardsCfg。

    obs / actuator / sensor 从 ExperimentComposer 注入的配置覆盖。
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards = PackagedRewardsCfg()
        if _COMPOSER_OBS_CFG:
            _apply_obs_cfg(self, _COMPOSER_OBS_CFG)
        if _COMPOSER_ACTUATOR_CFG:
            _apply_actuator_cfg(self, _COMPOSER_ACTUATOR_CFG)
        if _COMPOSER_SENSOR_CFG:
            _apply_sensor_cfg(self, _COMPOSER_SENSOR_CFG)


# ExperimentComposer 通过此属性读取 env_cfg 类
__myrl_env_cfg__ = G1FlatPackagedEnvCfg
