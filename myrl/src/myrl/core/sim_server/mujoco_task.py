"""MuJoCoTask ABC — 用户自定义任务的接口骨架。

实现者负责：
1. 使 obs_format() 返回的 shape 与 Isaac Lab 训练时完全一致（sim2sim 核心约束）
2. 使 apply_action() 语义与 Isaac Lab ActionCfg 对齐（position / torque / velocity）
3. reset_env() 随机化程度与训练时 EventCfg 对齐

内置 DummyTask 用于协议层冒烟测试，无需真实 MJCF 文件。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MuJoCoTask(ABC):
    """MuJoCo 任务抽象基类。

    MuJoCoSimServer 通过此接口与具体机器人任务解耦。
    每个任务实例对应一个 MJCF 模型的 N 个并行环境。
    """

    # ── 必须实现的属性 ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """动作空间维度（与训练时 num_actions 完全一致）。"""
        raise NotImplementedError

    @property
    def num_rewards(self) -> int:
        """奖励维度（默认 1，多 critic 任务可覆盖）。"""
        return 1

    @property
    @abstractmethod
    def max_episode_length(self) -> int:
        """单个 episode 最大步数（超出后触发 truncation）。"""
        raise NotImplementedError

    # ── 必须实现的方法 ─────────────────────────────────────────────────────

    @abstractmethod
    def obs_format(self) -> dict[str, dict[str, tuple]]:
        """返回观测格式描述（必须与 Isaac Lab 训练时 obs_format 一致）。

        Returns:
            格式示例::

                {
                    "policy": {
                        "base_ang_vel": (24,),   # 8 timesteps * 3 dims
                        "joint_pos": (87,),
                    },
                    "critic": {                  # 可选 privileged obs
                        "motion_reference": (50,),
                    },
                }
        """
        raise NotImplementedError

    @abstractmethod
    def compute_obs(
        self,
        model,  # mujoco.MjModel
        datas: list,  # list[mujoco.MjData], len == num_envs
    ) -> dict[str, np.ndarray]:
        """计算所有环境的观测。

        Args:
            model: MuJoCo 模型（只读，所有 env 共享）。
            datas: 所有 env 的 MjData 列表。

        Returns:
            ``{"policy": np.ndarray[N, policy_dim], "critic": np.ndarray[N, critic_dim]}``
            各 group 已展平为 float32。
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(
        self,
        model,
        datas: list,
        actions: np.ndarray,  # [N, num_actions]
    ) -> np.ndarray:
        """计算所有环境的奖励。

        Args:
            model: MuJoCo 模型。
            datas: 所有 env 的 MjData 列表。
            actions: 当前 step 传入的动作，形状 [N, num_actions]。

        Returns:
            奖励数组，形状 [N, num_rewards]，dtype float32。
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminated(
        self,
        model,
        datas: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """判断各环境是否结束。

        Args:
            model: MuJoCo 模型。
            datas: 所有 env 的 MjData 列表。

        Returns:
            ``(terminated[N], truncated[N])``，dtype bool。
            - terminated: 任务失败（如跌倒）
            - truncated: 超出 max_episode_length（由 MuJoCoSimServer 统一处理，
              此处可全部返回 False）
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(
        self,
        model,
        data,  # mujoco.MjData
        action: np.ndarray,  # [num_actions]
    ) -> None:
        """将动作写入单个环境的 data.ctrl / qfrc_applied 等。

        Args:
            model: MuJoCo 模型。
            data: 单个环境的 MjData（就地修改）。
            action: 该环境的动作，形状 [num_actions]。
        """
        raise NotImplementedError

    @abstractmethod
    def reset_env(
        self,
        model,
        data,  # mujoco.MjData
        env_id: int,
    ) -> None:
        """随机化重置单个环境的 qpos / qvel。

        Args:
            model: MuJoCo 模型。
            data: 单个环境的 MjData（就地修改）。
            env_id: 环境索引（用于确定性随机化种子）。
        """
        raise NotImplementedError


class DummyTask(MuJoCoTask):
    """协议层冒烟测试用占位任务（无需真实 MJCF）。

    观测：随机噪声 policy obs [N, 6]
    奖励：全 1
    终止：永不
    动作：忽略

    用法::

        python start_mujoco_server.py --task dummy --num_envs 4
    """

    _NUM_OBS = 6
    _NUM_ACTIONS = 3

    @property
    def num_actions(self) -> int:
        return self._NUM_ACTIONS

    @property
    def max_episode_length(self) -> int:
        return 1000

    def obs_format(self) -> dict[str, dict[str, tuple]]:
        return {"policy": {"dummy_obs": (self._NUM_OBS,)}}

    def compute_obs(self, model, datas: list) -> dict[str, np.ndarray]:
        n = len(datas)
        return {"policy": np.random.randn(n, self._NUM_OBS).astype(np.float32)}

    def compute_reward(self, model, datas: list, actions: np.ndarray) -> np.ndarray:
        return np.ones((len(datas), 1), dtype=np.float32)

    def is_terminated(self, model, datas: list) -> tuple[np.ndarray, np.ndarray]:
        n = len(datas)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    def apply_action(self, model, data, action: np.ndarray) -> None:
        pass  # DummyTask 无真实物理

    def reset_env(self, model, data, env_id: int) -> None:
        pass  # DummyTask 无真实物理
