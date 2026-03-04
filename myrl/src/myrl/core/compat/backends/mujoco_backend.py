"""MuJoCoBackend — MuJoCo socket 仿真后端（TCP 客户端）。

实现 instinct_rl.VecEnv ABC + myrl.SimBackend，使训练侧代码无需感知
仿真在本地（Isaac Lab）还是远端（MuJoCo / 真机）运行。

切换方式（play_mujoco.py / 任意 runner 只改一行）：
    # Isaac Lab 本地
    from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as Env
    env = Env(isaac_env)

    # MuJoCo 远端
    from myrl.core.compat.backends.mujoco_backend import MuJoCoBackend as Env
    env = Env(host="localhost", port=7777)
"""

from __future__ import annotations

import logging
import socket

import numpy as np
import torch

from instinct_rl.env import VecEnv

from myrl.core.compat.backends.base import SimBackend
from myrl.core.sim_server.protocol import MsgType, SimProto

logger = logging.getLogger(__name__)


class MuJoCoBackend(SimBackend, VecEnv):
    """MuJoCo TCP 客户端后端，实现 instinct_rl.VecEnv。

    握手完成后自动触发 reset，与 IsaacLabBackend 行为对齐。

    Args:
        host: MuJoCoSimServer（或 RealRobotServer）的 IP 地址。
        port: TCP 端口号。
        device: 输出 Tensor 所在设备（默认 "cpu"）。
        timeout: socket 超时秒数（默认 30）。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7777,
        device: str | torch.device = "cpu",
        timeout: float = 30.0,
    ) -> None:
        self.device = torch.device(device)

        # TCP 连接
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((host, port))
        logger.info("已连接到 MuJoCoSimServer %s:%d", host, port)

        # 握手：获取元数据
        SimProto.send(self._sock, {"type": int(MsgType.HANDSHAKE_REQ)})
        resp = SimProto.recv(self._sock)
        assert resp["type"] == int(MsgType.HANDSHAKE_RESP), f"握手失败: {resp}"

        self.num_envs: int = int(resp["num_envs"])
        self.num_actions: int = int(resp["num_actions"])
        self.num_rewards: int = int(resp["num_rewards"])
        self.max_episode_length: int = int(resp["max_episode_length"])
        self._obs_format: dict[str, dict[str, tuple]] = {
            group: {term: tuple(shape) for term, shape in terms.items()}
            for group, terms in resp["obs_format"].items()
        }

        # 计算 policy obs 总维度
        policy_terms = self._obs_format.get("policy", {})
        self.num_obs: int = sum(
            int(np.prod(shape)) for shape in policy_terms.values()
        )

        logger.info(
            "握手成功: num_envs=%d, num_actions=%d, num_obs=%d, num_rewards=%d",
            self.num_envs,
            self.num_actions,
            self.num_obs,
            self.num_rewards,
        )

        # 对齐 IsaacLabBackend：初始化时自动 reset
        self.reset()

    # ── VecEnv ABC 实现 ────────────────────────────────────────────────────

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """返回握手时从服务端获取的 obs_format。"""
        return self._obs_format

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """返回当前观测（不推进仿真）。"""
        resp = self._send_recv({"type": int(MsgType.GET_OBS_REQ)})
        obs_all = self._unpack_obs_all(resp)
        return obs_all["policy"], {"observations": obs_all}

    def reset(self) -> tuple[torch.Tensor, dict]:
        """重置所有环境。"""
        resp = self._send_recv({"type": int(MsgType.RESET_REQ)})
        obs_all = self._unpack_obs_all(resp)
        return obs_all["policy"], {"observations": obs_all}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """推进一步仿真。

        Args:
            actions: 动作张量，形状 [num_envs, num_actions]。

        Returns:
            (obs, rewards, dones, extras) — 格式与 IsaacLabBackend 完全一致。
        """
        actions_np = actions.cpu().numpy().astype(np.float32)
        resp = self._send_recv({"type": int(MsgType.STEP_REQ), "actions": actions_np})

        obs_all = self._unpack_obs_all(resp)
        obs = obs_all["policy"]

        rewards = torch.from_numpy(
            np.asarray(resp["rewards"], dtype=np.float32)
        ).to(self.device)
        # 统一为 [num_envs, num_rewards]
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)

        dones = torch.from_numpy(
            np.asarray(resp["dones"], dtype=np.int64)
        ).to(self.device)

        time_outs = torch.from_numpy(
            np.asarray(resp["time_outs"], dtype=bool)
        ).to(self.device)

        extras: dict = {
            "observations": obs_all,
            "time_outs": time_outs,
            "log": resp.get("log", {}),
        }

        return obs, rewards, dones, extras

    def close(self) -> None:
        """关闭连接（发送 CLOSE 消息后关闭 socket）。"""
        try:
            SimProto.send(self._sock, {"type": int(MsgType.CLOSE)})
        except Exception:
            pass
        finally:
            self._sock.close()
            logger.info("MuJoCoBackend 连接已关闭")

    # ── 内部工具方法 ───────────────────────────────────────────────────────

    def _send_recv(self, payload: dict) -> dict:
        """发送请求并等待响应。"""
        SimProto.send(self._sock, payload)
        return SimProto.recv(self._sock)

    def _unpack_obs_all(self, resp: dict) -> dict[str, torch.Tensor]:
        """将响应中的 obs_all（numpy）转为 Tensor dict。

        如果响应只有顶层 "obs"（policy），也构造出 obs_all。
        """
        obs_all_raw: dict = resp.get("obs_all") or {}
        obs_all: dict[str, torch.Tensor] = {}

        for group_name, arr in obs_all_raw.items():
            obs_all[group_name] = torch.from_numpy(
                np.asarray(arr, dtype=np.float32)
            ).to(self.device)

        # 保证 "policy" 存在（兼容只返回顶层 obs 的旧 server）
        if "policy" not in obs_all and resp.get("obs") is not None:
            obs_all["policy"] = torch.from_numpy(
                np.asarray(resp["obs"], dtype=np.float32)
            ).to(self.device)

        return obs_all
