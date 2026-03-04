"""MuJoCoSimServer — MuJoCo 物理仿真 TCP 服务端。

- 加载 MJCF，创建 N×MjData（每个 env 独立物理状态）
- 向量化 step / reset，批量处理 obs / reward / termination
- 维护 episode_length_buf，超出 max_episode_length 时自动 truncate + reset
- 实现 SimServer 的四个 handle_* 方法

使用 DummyTask（无 MJCF 文件）可直接验证协议层，无需真实机器人资产。

ROS 模式（--ros）：
- obs_callback 在 TCP STEP_RESP **之前**调用（保证 bridge Event.wait 不会超时）
- include_obs_in_response=False 时，STEP_RESP 不含 obs 字段（节省带宽）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from myrl.core.sim_server.base_server import SimServer
from myrl.core.sim_server.mujoco_task import MuJoCoTask

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MuJoCoSimServer(SimServer):
    """MuJoCo N 个并行环境的同步仿真服务端。

    Args:
        task: 任务实例（实现 MuJoCoTask ABC）。
        mjcf_path: MJCF 文件路径；DummyTask 可传 None。
        num_envs: 并行环境数量。
        sim_steps_per_ctrl: 每个控制帧的物理子步数（默认 4）。
        host: TCP 监听地址（默认 0.0.0.0）。
        port: TCP 监听端口（默认 7777）。
        include_obs_in_response: STEP_RESP / RESET_RESP 是否包含 obs 字段（默认 True）。
            ROS 模式下设为 False，obs 通过 obs_callback 经 ROS topic 传递。
    """

    def __init__(
        self,
        task: MuJoCoTask,
        mjcf_path: str | None,
        num_envs: int,
        sim_steps_per_ctrl: int = 4,
        host: str = "0.0.0.0",
        port: int = 7777,
        include_obs_in_response: bool = True,
    ) -> None:
        super().__init__(host=host, port=port)

        self.task = task
        self.num_envs = num_envs
        self.sim_steps_per_ctrl = sim_steps_per_ctrl
        self._include_obs = include_obs_in_response

        # obs_callback：在 TCP 响应之前调用（ROS 模式用于发布 obs topic）
        # 签名：fn(obs_all: dict[str, np.ndarray]) -> None
        self._obs_callback: Callable[[dict], None] | None = None

        # 加载 MuJoCo 模型（DummyTask 无 MJCF 时跳过）
        if mjcf_path is not None:
            import mujoco

            self.model = mujoco.MjModel.from_xml_path(mjcf_path)
            self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
            logger.info("MJCF 加载完成: %s, %d envs", mjcf_path, num_envs)
        else:
            # DummyTask：model 和 datas 均为 None，handle_* 方法直接调用 task
            self.model = None
            self.datas = [None] * num_envs
            logger.info("DummyTask 模式（无 MJCF），%d envs", num_envs)

        # episode 步数计数器
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int32)

        # 初始重置
        self._reset_all()

    # ── 公共 API ──────────────────────────────────────────────────────────

    def register_obs_callback(self, fn: Callable[[dict], None]) -> None:
        """注册 obs 回调（ROS 模式）。

        回调在 TCP STEP_RESP 发送**之前**被调用，保证 bridge 的
        ``_new_obs_event.wait()`` 不会超时。

        Args:
            fn: 回调函数，签名 ``fn(obs_all: dict[str, np.ndarray]) -> None``。
        """
        self._obs_callback = fn

    # ── SimServer handle_* 实现 ────────────────────────────────────────────

    def handle_handshake(self, req: dict) -> dict:
        """响应握手：返回环境元数据。"""
        obs_fmt = self.task.obs_format()
        # 将 tuple 转为 list（msgpack 不序列化 tuple）
        obs_format_serializable = {
            group: {term: list(shape) for term, shape in terms.items()}
            for group, terms in obs_fmt.items()
        }
        return {
            "num_envs": self.num_envs,
            "num_actions": self.task.num_actions,
            "num_rewards": self.task.num_rewards,
            "max_episode_length": self.task.max_episode_length,
            "obs_format": obs_format_serializable,
        }

    def handle_step(self, req: dict) -> dict:
        """推进仿真一步，处理 termination + auto-reset。

        Args:
            req: 含 ``actions`` 字段（np.ndarray[N, num_actions] float32）。

        Returns:
            STEP_RESP payload（不含 type 字段，由 base_server 填充）。
            - 始终包含：rewards / dones / time_outs / log
            - 含 obs 取决于 include_obs_in_response
        """
        actions = np.asarray(req["actions"], dtype=np.float32)  # [N, num_actions]

        # 1. 逐 env apply_action + mj_step × sim_steps_per_ctrl
        self._step_physics(actions)

        # 2. 向量化 obs / reward / termination
        obs_all = self.task.compute_obs(self.model, self.datas)
        rewards = self.task.compute_reward(self.model, self.datas, actions)
        terminated, truncated_task = self.task.is_terminated(self.model, self.datas)

        # 3. 维护 episode_length_buf，检测超时 truncation
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.task.max_episode_length  # bool[N]
        truncated = truncated_task | time_outs

        # 4. dones
        dones = (terminated | truncated).astype(np.int64)

        # 5. 对 done 的 env 做 auto-reset
        done_ids = np.where(dones)[0]
        if len(done_ids) > 0:
            self._reset_envs(done_ids)
            # reset 后重新拿 obs（已更新为初始状态）
            obs_all = self.task.compute_obs(self.model, self.datas)

        # 6. obs_callback 必须在 TCP 响应之前调用（ROS 同步保障）
        if self._obs_callback is not None:
            self._obs_callback(obs_all)

        resp: dict = {
            "rewards": rewards,
            "dones": dones,
            "time_outs": time_outs,
            "log": {},
        }
        if self._include_obs:
            resp["obs"] = obs_all.get("policy")
            resp["obs_all"] = {k: v for k, v in obs_all.items()}

        return resp

    def handle_reset(self, req: dict) -> dict:
        """重置所有环境，返回初始 obs。"""
        self._reset_all()
        obs_all = self.task.compute_obs(self.model, self.datas)

        # obs_callback（ROS 模式：发布 reset 后的初始 obs）
        if self._obs_callback is not None:
            self._obs_callback(obs_all)

        resp: dict = {}
        if self._include_obs:
            resp["obs"] = obs_all.get("policy")
            resp["obs_all"] = {k: v for k, v in obs_all.items()}
        return resp

    def handle_get_obs(self, req: dict) -> dict:
        """返回当前观测（不推进仿真）。"""
        obs_all = self.task.compute_obs(self.model, self.datas)
        resp: dict = {}
        if self._include_obs:
            resp["obs"] = obs_all.get("policy")
            resp["obs_all"] = {k: v for k, v in obs_all.items()}
        return resp

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _step_physics(self, actions: np.ndarray) -> None:
        """逐 env 写入动作并推进物理（DummyTask 跳过）。"""
        if self.model is None:
            return  # DummyTask 无物理

        import mujoco

        for i, data in enumerate(self.datas):
            self.task.apply_action(self.model, data, actions[i])
            for _ in range(self.sim_steps_per_ctrl):
                mujoco.mj_step(self.model, data)

    def _reset_all(self) -> None:
        """重置所有环境并清零 episode_length_buf。"""
        self.episode_length_buf[:] = 0
        for i, data in enumerate(self.datas):
            self.task.reset_env(self.model, data, env_id=i)
            if data is not None:
                import mujoco
                mujoco.mj_forward(self.model, data)

    def _reset_envs(self, env_ids: np.ndarray) -> None:
        """重置指定环境并清零其 episode_length_buf。"""
        for i in env_ids:
            self.episode_length_buf[i] = 0
            self.task.reset_env(self.model, self.datas[i], env_id=int(i))
            if self.datas[i] is not None:
                import mujoco
                mujoco.mj_forward(self.model, self.datas[i])
