"""adapters.py — 将 RewardLibrary term 适配为 instinctlab RewTerm 格式。

instinctlab 的奖励系统用 RewardTermCfg：
    func: Callable[[ManagerBasedRLEnv, **kwargs], Tensor]
    weight: float

这里提供辅助函数，把 RewardLibrary 中的 term 直接转换为该格式，
供不使用 RewardBuilder 但需要复用 Library term 的场景使用。
"""
from __future__ import annotations

from typing import Callable

from torch import Tensor


def make_instinctlab_rew_func(
    term_name: str,
    robot_name: str = "robot",
    **params_kwargs,
) -> Callable:
    """从 RewardLibrary 构建兼容 instinctlab RewardTermCfg.func 的函数。

    返回的函数签名：(env: ManagerBasedRLEnv, **kwargs) -> Tensor

    用法示例::

        from myrl.core.task.reward_lib.adapters import make_instinctlab_rew_func
        from isaaclab.envs.mdp import RewardTermCfg

        track_vel_fn = make_instinctlab_rew_func(
            "track_lin_vel_xy_exp", std=0.3
        )

        @configclass
        class RewardsCfg:
            track_vel = RewardTermCfg(func=track_vel_fn, weight=1.5)
    """
    from myrl.core.task.reward_lib import get_reward_library

    wrapped = get_reward_library().build(term_name, robot_name=robot_name, **params_kwargs)
    return wrapped


def list_available_terms() -> dict[str, dict]:
    """列出 RewardLibrary 中所有已注册的 term，返回 {name: meta.to_dict()}。"""
    from myrl.core.task.reward_lib import get_reward_library

    return get_reward_library().to_dict()
