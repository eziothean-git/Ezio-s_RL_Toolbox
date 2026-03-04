"""SimBackend ABC — 未来 IsaacGym / MuJoCo 后端的统一接口骨架。"""

from abc import ABC, abstractmethod

import torch


class SimBackend(ABC):
    """所有仿真后端的抽象基类。

    具体后端（IsaacLab、IsaacGym、MuJoCo 等）继承此类并实现所有抽象方法。
    此类不直接实现 instinct_rl.env.VecEnv；具体后端类负责同时继承两者。
    """

    # ── 必须由子类设置的属性 ─────────────────────────────────────────────────
    num_envs: int
    num_actions: int
    num_rewards: int
    device: torch.device

    # ── 抽象方法（子类必须实现）─────────────────────────────────────────────

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """推进一步仿真。

        Args:
            actions: 动作张量，形状 [num_envs, num_actions]。

        Returns:
            (obs, rewards, dones, extras)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """重置所有环境。

        Returns:
            (obs, extras)
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """返回当前观测（不推进仿真）。

        Returns:
            (obs, extras)
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """关闭仿真环境，释放资源。"""
        raise NotImplementedError
