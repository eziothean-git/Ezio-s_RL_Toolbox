"""双层环境 Patch — 注入调试工具钩子。

层 1：VecEnv wrapper 的 step()
    - ActionMux pre_step（修改 actions）
    - TimeScale pause/sleep
    - BodyAnchor post_step（复位锚定体）
    - ContactViz post_step（记录轨迹）

层 2：ManagerBasedRLEnv 的 scene.write_data_to_sim()
    - ForceApplicator 在每个 decimation 子步前设置外力

遵循 env_wrapper.py 的 monkey-patch 模式：
保存原始方法 → 替换为带钩子的版本 → 标记 _debug_patched。
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


def enable_debug_patch(env, ctx: DebugContext) -> None:
    """对 env 实例注入调试工具钩子。幂等。

    Args:
        env: VecEnv wrapper（IsaacLabBackend 或 InstinctRlVecEnvWrapper）。
        ctx: DebugContext 实例。
    """
    if getattr(env, "_debug_patched", False):
        return

    unwrapped = env.unwrapped
    step_dt = getattr(unwrapped, "step_dt", 1.0 / 60.0)

    # ── 层 1：patch VecEnv wrapper step() ─────────────────────────────
    _orig_step = env.step

    def _debug_step(actions: Tensor):
        # pre_step：所有插件可修改 actions（ActionMux 在此覆盖）
        for plugin in ctx._plugins.values():
            actions = plugin.pre_step(ctx, actions)

        # pause 支持：暂停时自旋等待
        while ctx.paused and not ctx.single_step_requested:
            time.sleep(0.01)
        if ctx.single_step_requested:
            ctx.single_step_requested = False

        # 执行原始 step
        obs, rew, dones, extras = _orig_step(actions)

        # post_step：所有插件处理（锚点复位、轨迹记录等）
        for plugin in ctx._plugins.values():
            plugin.post_step(ctx, obs, rew, dones, extras)

        # 时间流速：慢放 sleep
        if 0 < ctx.time_scale < 1.0:
            sleep_time = step_dt * (1.0 / ctx.time_scale - 1.0)
            time.sleep(sleep_time)

        return obs, rew, dones, extras

    env.step = _debug_step

    # ── 层 2：patch scene.write_data_to_sim() ─────────────────────────
    # 外力需要在每个 decimation 子步前应用。
    # set_external_force_and_torque() 缓冲力，write_data_to_sim() 消费。
    if hasattr(unwrapped, "scene"):
        scene = unwrapped.scene
        _orig_write = scene.write_data_to_sim

        def _debug_write():
            # 在物理写入前应用调试外力
            if ctx.force_active and ctx._articulation is not None:
                ctx._articulation.set_external_force_and_torque(
                    ctx.external_forces,
                    ctx.external_torques,
                    is_global=True,
                )
            _orig_write()

        scene.write_data_to_sim = _debug_write

    # ── 层 3：patch sim.render() 用于 on_render 回调 ──────────────────
    sim_ctx = getattr(unwrapped, "sim", None)
    if sim_ctx is not None and hasattr(sim_ctx, "render"):
        _orig_render = sim_ctx.render

        def _debug_render(*args, **kwargs):
            result = _orig_render(*args, **kwargs)
            for plugin in ctx._plugins.values():
                try:
                    plugin.on_render(ctx)
                except Exception:
                    pass  # on_render 失败不阻断仿真
            return result

        sim_ctx.render = _debug_render

    env._debug_patched = True
