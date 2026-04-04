"""enable_databus_on_env — 非侵入式 DataBus 集成。

不包装 env，直接 patch step/reset/get_observations 方法。
env 对象身份不变，isinstance/type/property 全部正常工作。

用法（train.py）:
    env = EnvWrapper(env)
    if _bus:
        from myrl.core.databus.env_wrapper import enable_databus_on_env
        enable_databus_on_env(env, _bus)
    runner = OnPolicyRunner(env, ...)
"""
from __future__ import annotations

from myrl.core.databus.bus import DataBus


def enable_databus_on_env(env, bus: DataBus) -> None:
    """Patch env 的 step/reset/get_observations，注入 DataBus publish。

    直接修改 env 实例，无返回值。幂等：重复调用跳过。
    """
    if getattr(env, "_databus_patched", False):
        return

    # ── 缓存 obs_format slices ────────────────────────────────────
    obs_slices: dict[str, dict[str, slice]] = {}
    try:
        for group, segments in env.get_obs_format().items():
            offset = 0
            sl_dict: dict[str, slice] = {}
            for name, dim in segments.items():
                size = dim if isinstance(dim, int) else dim[0]
                sl_dict[name] = slice(offset, offset + size)
                offset += size
            obs_slices[group] = sl_dict
        # 为多维 obs term 注册索引标签
        for group, sl_dict in obs_slices.items():
            for name, sl in sl_dict.items():
                size = sl.stop - sl.start
                if size > 1:
                    bus.set_labels(f"obs/{group}/{name}", [[f"d{i}" for i in range(size)]])
    except Exception:
        pass

    # ── 内部发布辅助 ──────────────────────────────────────────────
    def _publish_obs(extras: dict) -> None:
        obs_dict = extras.get("observations", {})
        for group, tensor in obs_dict.items():
            bus.publish(f"obs/{group}", tensor)
            for name, sl in obs_slices.get(group, {}).items():
                bus.publish(f"obs/{group}/{name}", tensor[:, sl])

    # ── 保存原始方法（bound method） ──────────────────────────────
    _orig_step = env.step
    _orig_reset = env.reset
    _orig_get_obs = env.get_observations

    # ── 带 publish 的替换方法 ─────────────────────────────────────
    def step(actions):
        bus.publish("action/raw", actions)
        obs, rew, dones, extras = _orig_step(actions)
        _publish_obs(extras)
        if rew.dim() > 1:
            bus.publish("reward/total", rew.sum(dim=1))
            if rew.shape[1] > 1:
                for i in range(rew.shape[1]):
                    bus.publish(f"reward/group_{i}", rew[:, i])
        else:
            bus.publish("reward/total", rew)
        bus.publish("episode/dones", dones)
        if "time_outs" in extras:
            bus.publish("episode/time_outs", extras["time_outs"])
        return obs, rew, dones, extras

    def reset():
        obs, extras = _orig_reset()
        _publish_obs(extras)
        return obs, extras

    def get_observations():
        result = _orig_get_obs()
        if isinstance(result, tuple) and len(result) == 2:
            _publish_obs(result[1])
        return result

    # ── 替换实例方法 ──────────────────────────────────────────────
    env.step = step
    env.reset = reset
    env.get_observations = get_observations
    env._databus_patched = True


# ── Deprecated 兼容层 ─────────────────────────────────────────────

class DataBusEnvWrapper:
    """Deprecated: 使用 enable_databus_on_env() 代替。

    保留此类仅为向后兼容。内部调用 enable_databus_on_env 后
    将所有属性访问转发给原始 env。
    """

    def __init__(self, env, bus: DataBus) -> None:
        enable_databus_on_env(env, bus)
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name == "_env":
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)
