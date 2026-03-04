"""IsaacLabBackend — Isaac Lab 仿真后端，实现 instinct_rl VecEnv ABC。

这是 Phase B 的核心文件，替换 instinctlab 的 InstinctRlVecEnvWrapper。
实现完整的 VecEnv 接口，同时继承 myrl 的 SimBackend 骨架。

Phase A → B 切换（train.py / play.py 只改一行 import）:
    # Phase A:
    from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper as EnvWrapper
    # Phase B:
    from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
"""

from __future__ import annotations

import gymnasium as gym
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

from instinct_rl.env import VecEnv

from myrl.core.compat.backends.base import SimBackend


class IsaacLabBackend(SimBackend, VecEnv):
    """Isaac Lab 仿真后端，同时实现 instinct_rl.VecEnv 和 myrl.SimBackend。

    直接替换 InstinctRlVecEnvWrapper，行为完全一致：
    - obs 每个 group 输出 [num_envs, flat_dim]（各 term 展平后拼接）
    - rewards: Tensor[num_envs, num_rewards]（单奖励也保留第二维）
    - time_outs: Tensor[num_envs]（bool）
    - dones: Tensor[num_envs]（long）
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """初始化 IsaacLabBackend。

        Args:
            env: Isaac Lab 环境实例（ManagerBasedRLEnv 或 DirectRLEnv）。

        Raises:
            ValueError: 如果 env 不是 ManagerBasedRLEnv 或 DirectRLEnv 的实例。
        """
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "IsaacLabBackend 要求环境继承自 ManagerBasedRLEnv 或 DirectRLEnv，"
                f"当前类型: {type(env)}"
            )

        self.env = env

        # ── VecEnv 必需属性 ────────────────────────────────────────────────
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # num_actions
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # num_obs（policy group）
        if hasattr(self.unwrapped, "observation_manager"):
            policy_dims = self.unwrapped.observation_manager.group_obs_term_dim["policy"]
            flat_dims = [torch.prod(torch.tensor(d, device="cpu")).item() for d in policy_dims]
            self.num_obs = int(sum(flat_dims))
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        # num_critic_obs（可选 privileged obs）
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            critic_dims = self.unwrapped.observation_manager.group_obs_term_dim["critic"]
            flat_dims = [torch.prod(torch.tensor(d, device="cpu")).item() for d in critic_dims]
            self.num_critic_obs = int(sum(flat_dims))
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_critic_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_critic_obs = None

        # OnPolicyRunner 不调用 reset，需要在此手动 reset
        self.env.reset()

    # ── 标准 Gym Wrapper 属性 ──────────────────────────────────────────────

    def __str__(self) -> str:
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self) -> str:
        return str(self)

    @property
    def cfg(self) -> ManagerBasedRLEnvCfg:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    @property
    def num_rewards(self) -> int:
        return self.unwrapped.num_rewards

    # ── VecEnv ABC 实现 ────────────────────────────────────────────────────

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """返回所有观测 group 的结构描述。

        返回格式:
            {
                "policy": {"term_name": (dim, ...), ...},
                "critic": {"term_name": (dim, ...), ...},  # 可选
            }
        """
        obs_format = {}
        for group_name in self.unwrapped.observation_manager.active_terms.keys():
            obs_format[group_name] = self._get_obs_segments(group_name)
        return obs_format

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """返回当前观测，不推进仿真。

        Returns:
            (policy_obs [num_envs, num_obs], {"observations": obs_pack})
        """
        if hasattr(self.unwrapped, "observation_manager"):
            obs_pack = self.unwrapped.observation_manager.compute()
        else:
            obs_pack = self.unwrapped._get_observations()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        return obs_pack["policy"], {"observations": obs_pack}

    def reset(self) -> tuple[torch.Tensor, dict]:
        """重置所有环境。

        Returns:
            (policy_obs [num_envs, num_obs], {"observations": obs_pack})
        """
        obs_pack, _ = self.env.reset()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        return obs_pack["policy"], {"observations": obs_pack}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """推进一步仿真。

        Args:
            actions: 动作张量，形状 [num_envs, num_actions]。

        Returns:
            - obs: policy 观测，形状 [num_envs, num_obs]
            - rewards: 奖励，形状 [num_envs, num_rewards]
            - dones: 终止标志，形状 [num_envs]，dtype=long
            - extras: 额外信息字典，包含 "observations"、"time_outs"（可选）等
        """
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)
        obs_pack = self._flatten_all_obs_groups(obs_pack)

        # dones：terminated 或 truncated
        dones = (terminated | truncated).to(dtype=torch.long)

        # 将观测写入 extras
        extras["observations"] = obs_pack

        # time_outs：仅用于无限时域任务（infinite horizon）
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # policy obs
        obs = obs_pack["policy"]

        # 奖励统一为 [num_envs, num_rewards]
        if isinstance(rew, dict):
            rew = self._stack_rewards(rew)
        else:
            rew = rew.unsqueeze(1)

        return obs, rew, dones, extras

    def close(self) -> None:
        """关闭环境。"""
        self.env.close()

    # ── Instinct-RL 兼容接口 ───────────────────────────────────────────────

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_obs_segments(self, group_name: str = "policy") -> dict[str, tuple]:
        """返回指定 group 的观测 term 名称 → 形状 映射。"""
        return self._get_obs_segments(group_name)

    # ── 内部工具方法 ───────────────────────────────────────────────────────

    def _get_obs_segments(self, group_name: str) -> dict[str, tuple]:
        """构造 {term_name: shape} 的有序字典。"""
        obs_manager = self.unwrapped.observation_manager
        term_names = obs_manager.active_terms[group_name]
        term_dims = obs_manager.group_obs_term_dim[group_name]
        return {name: dim for name, dim in zip(term_names, term_dims)}

    def _flatten_obs_group(self, obs_group: dict) -> torch.Tensor:
        """将一个 group 内所有 term 展平并拼接为 [num_envs, flat_dim]。"""
        parts = [v.flatten(start_dim=1) for v in obs_group.values()]
        return torch.cat(parts, dim=1)

    def _flatten_all_obs_groups(self, obs_pack: dict) -> dict:
        """对 obs_pack 中所有 group 做展平处理。

        如果 group 值已经是 Tensor（非 dict），则直接保留。
        """
        return {
            name: self._flatten_obs_group(group) if isinstance(group, dict) else group
            for name, group in obs_pack.items()
        }

    def _stack_rewards(self, rewards_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """将多奖励 dict 堆叠为 [num_envs, num_rewards]。"""
        return torch.stack(list(rewards_dict.values()), dim=-1)
