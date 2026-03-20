"""ExperimentComposer — 从 .myrlpkg 组合并启动训练环境。

流程（7 步）：
    1. 读 package.yaml + experiment.yaml
    2. os.environ["MYRL_ASSETS_DIR"] = pkg.assets_dir
    3. sys.path.insert(0, pkg/reward_fns/) → import modules → @reward_fn 注册
    4. 运行 terrain/generators/*.py（注册生成 term）
    5. load reward_pipeline YAML → 构建 RewardBuilder（deferred params 暂存）
    6. import env_script → 注入 _COMPOSER_REWARD_BUILDER 等
    7. gym.register + gym.make → 返回 (env, runner_cfg_dict)

注意：必须在 AppLauncher 之后调用 compose()。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import yaml

from .packager import PackageReader, PackageManifest


class ExperimentComposer:
    """从 .myrlpkg 包组合实验环境。"""

    def __init__(self, package_path: str):
        self._pkg = PackageReader(package_path)

    @property
    def manifest(self) -> PackageManifest:
        return self._pkg.manifest

    def compose(
        self,
        num_envs: int | None = None,
        device: str | None = None,
    ) -> tuple[Any, dict]:
        """组合环境，返回 (env, runner_cfg_dict)。

        必须在 AppLauncher 之后调用（Isaac Sim 运行时可用）。
        """
        import gymnasium as gym

        exp = self._pkg.experiment_yaml
        manifest = self._pkg.manifest

        # ── Step 2: 设置 MYRL_ASSETS_DIR ────────────────────────────────────
        os.environ["MYRL_ASSETS_DIR"] = self._pkg.assets_dir

        # ── Step 3: 注入 reward_fn 模块 → @reward_fn 注册 ───────────────────
        for rfn_dir in self._pkg.list_reward_fn_dirs():
            if rfn_dir not in sys.path:
                sys.path.insert(0, rfn_dir)
            # import 所有 .py 文件
            for py_file in Path(rfn_dir).glob("*.py"):
                _import_file(py_file)

        # ── Step 4: terrain generators ───────────────────────────────────────
        for gen_script in self._pkg.get_terrain_generators_dir():
            _import_file(Path(gen_script))

        # ── Step 5: RewardBuilder from reward_pipeline YAML ─────────────────
        reward_builder = None
        pipeline_path = self._pkg.get_reward_pipeline_path()
        if pipeline_path and os.path.exists(pipeline_path):
            reward_builder = _build_reward_builder_from_yaml(pipeline_path)

        # ── Step 6: import env_script，注入 _COMPOSER_* ──────────────────────
        env_script_path = self._pkg.get_env_script_path()
        env_module = None
        if env_script_path and os.path.exists(env_script_path):
            env_module = _import_file(Path(env_script_path))
            if reward_builder is not None:
                env_module._COMPOSER_REWARD_BUILDER = reward_builder
            # 注入 actuator / sensor / terrain 配置
            actuator_path = self._pkg.get_asset_path("actuator_cfg",
                                                       manifest.asset_checksums.get(
                                                           "actuator_cfg", {}).get("name", ""))
            sensor_path = self._pkg.get_asset_path("sensor_cfg",
                                                     manifest.asset_checksums.get(
                                                         "sensor_cfg", {}).get("name", ""))
            if actuator_path and os.path.exists(actuator_path):
                with open(actuator_path) as f:
                    env_module._COMPOSER_ACTUATOR_CFG = yaml.safe_load(f)
            if sensor_path and os.path.exists(sensor_path):
                with open(sensor_path) as f:
                    env_module._COMPOSER_SENSOR_CFG = yaml.safe_load(f)

        # ── Step 7: gym.register + gym.make ──────────────────────────────────
        pkg_id = manifest.package_id
        task_id = f"myrl/Package-{pkg_id}-v0"

        if env_module is not None and hasattr(env_module, "__myrl_env_cfg__"):
            env_cfg_cls = env_module.__myrl_env_cfg__
        else:
            # 回退：无 env_script，使用 experiment_cfg 中的 env_script 字段查找
            # 或者直接抛错（env_script 是必须的）
            raise RuntimeError(
                f"env_script not found or missing __myrl_env_cfg__ in package '{pkg_id}'"
            )

        # 注册任务
        gym.register(
            id=task_id,
            entry_point="instinctlab.envs:InstinctRlEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_cfg_cls,
                "instinct_rl_cfg_entry_point": _make_runner_cfg_fn(
                    self._pkg.get_algo_cfg_path(), manifest
                ),
            },
        )

        # 构建 env_cfg（覆盖 num_envs / device）
        env_cfg = env_cfg_cls()
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
        if device is not None:
            env_cfg.sim.device = device

        # gym.make
        env = gym.make(task_id, cfg=env_cfg)

        # ── Step 7b: 解析 deferred params（__query_sensor__ / __query_pattern__）──
        if reward_builder is not None:
            _resolve_deferred_params(reward_builder, env.unwrapped)

        # 构建 runner_cfg_dict
        runner_cfg_dict = _load_runner_cfg_dict(
            self._pkg.get_algo_cfg_path(),
            manifest,
            experiment_name=manifest.experiment_name,
        )
        if num_envs is not None:
            runner_cfg_dict["num_envs"] = num_envs

        return env, runner_cfg_dict


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _import_file(path: Path) -> types.ModuleType:
    """动态 import 一个 .py 文件，返回 module。"""
    module_name = f"_myrlpkg_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None:
        raise ImportError(f"Cannot load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_reward_builder_from_yaml(pipeline_path: str):
    """从 reward_pipeline YAML 构建 RewardBuilder（deferred params 暂存）。"""
    from myrl.core.task.reward_builder import RewardBuilder

    with open(pipeline_path) as f:
        cfg = yaml.safe_load(f)

    builder = RewardBuilder()
    builder._deferred_params = []  # 存放含 __query_sensor__ 的 term

    for term in cfg.get("terms", []):
        name = term["name"]
        weight = term.get("weight", 1.0)
        params = term.get("params", {})

        # 检查是否有 deferred params
        has_deferred = any(
            isinstance(v, dict) and "__query_sensor__" in v
            for v in params.values()
        )
        if has_deferred:
            builder._deferred_params.append({
                "name": name, "weight": weight, "params": params
            })
        else:
            builder.add_from_lib(name, weight=weight, **params)

    return builder


def _resolve_deferred_params(reward_builder, env) -> None:
    """解析 __query_sensor__ / __query_pattern__ → 更新 body_ids，然后注册 term。"""
    if not hasattr(reward_builder, "_deferred_params"):
        return

    for item in reward_builder._deferred_params:
        name = item["name"]
        weight = item["weight"]
        params = dict(item["params"])

        # 解析每个 deferred 字段
        resolved = {}
        for key, val in params.items():
            if isinstance(val, dict) and "__query_sensor__" in val:
                sensor_name = val["__query_sensor__"]
                pattern = val.get("__query_pattern__", ".*")
                try:
                    sensor = env.scene.sensors[sensor_name]
                    body_ids = list(sensor.find_bodies([pattern])[0])
                    resolved[key] = body_ids
                except Exception as e:
                    print(f"[WARN] deferred param '{key}' resolution failed: {e}")
                    resolved[key] = []
            else:
                resolved[key] = val

        reward_builder.add_from_lib(name, weight=weight, **resolved)

    reward_builder._deferred_params = []


def _make_runner_cfg_fn(algo_cfg_path: str | None, manifest: PackageManifest):
    """返回 runner cfg entry point（字符串或可调用）。"""
    # 直接返回 None，用 runner_cfg_dict 代替（ExperimentComposer 路径不走 entry point）
    return None


def _load_runner_cfg_dict(
    algo_cfg_path: str | None,
    manifest: PackageManifest,
    experiment_name: str = "",
) -> dict:
    """从 algo_cfg YAML 构建 runner_cfg_dict。"""
    if algo_cfg_path and os.path.exists(algo_cfg_path):
        with open(algo_cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # 重构为 instinct_rl OnPolicyRunner 期望的格式
    runner_cfg = {
        "experiment_name": experiment_name or manifest.experiment_name,
        "num_steps_per_env": cfg.get("num_steps_per_env", 24),
        "max_iterations": cfg.get("max_iterations", 10000),
        "save_interval": cfg.get("save_interval", 1000),
        "device": cfg.get("device", "cuda:0"),
        "policy": cfg.get("policy", {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
        }),
        "algorithm": cfg.get("algorithm", {
            "class_name": "PPO",
            "learning_rate": 1e-3,
        }),
    }
    return runner_cfg
