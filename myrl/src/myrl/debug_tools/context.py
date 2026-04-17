"""DebugContext — 调试工具中心状态。

所有调试插件通过 DebugContext 读写状态。
HTTP handler 线程写入 scalar/tensor，仿真线程在 step() 中消费。
线程安全依赖：CPython GIL 保证 scalar 写入原子性，
tensor 元素级写入与整体读取不冲突（不同 env_id 写，整体 tensor 读）。
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from myrl.core.databus.bus import DataBus


class DebugContext:
    """调试工具中心状态持有者。

    生命周期：由 enable_debug_tools() 创建，存储于 env._debug_ctx。
    插件通过 register_plugin() 注册自身。
    """

    def __init__(self, env, bus: DataBus | None = None) -> None:
        self.env = env
        self.bus = bus
        self.num_envs: int = env.num_envs
        self.device: torch.device = env.device
        self._plugins: dict[str, object] = {}
        self._lock = threading.Lock()

        # ── 外力施加状态 ─────────────────────────────────────────────
        self.external_forces: Tensor | None = None   # (num_envs, num_bodies, 3)
        self.external_torques: Tensor | None = None  # (num_envs, num_bodies, 3)
        self.force_active: bool = False

        # ── 动作 MUX 状态 ────────────────────────────────────────────
        self.action_override: Tensor | None = None   # (num_envs, num_actions)
        self.action_mask: Tensor | None = None       # (num_envs, num_actions) bool
        self.mux_active_envs: set[int] = set()

        # ── 锚点状态 ────────────────────────────────────────────────
        # env_id → {body_id → pose Tensor(7,)}
        self.anchor_poses: dict[int, dict[int, Tensor]] = {}
        self.anchor_active: bool = False

        # ── 时间流速 ────────────────────────────────────────────────
        self.time_scale: float = 1.0       # 1.0=正常, 0.1=10倍慢放
        self.paused: bool = False
        self.single_step_requested: bool = False

        # ── 可视化标志 ──────────────────────────────────────────────
        self.viz_contact_forces: bool = False
        self.viz_trajectories: bool = False
        self.trajectory_buffer: dict[int, list[Tensor]] = {}
        self.trajectory_max_len: int = 200

        # ── Articulation 引用（由 ForceApplicator attach 时填充）────
        self._articulation = None
        self._body_names: list[str] = []
        self._joint_names: list[str] = []
        self._num_bodies: int = 0
        self._num_actions: int = 0

    def register_plugin(self, plugin) -> None:
        """注册一个 DebugPlugin 实例。"""
        self._plugins[plugin.name] = plugin

    def get_plugin(self, name: str):
        """按名称获取已注册的插件。"""
        return self._plugins.get(name)

    @property
    def plugin_names(self) -> list[str]:
        return list(self._plugins.keys())

    def state_snapshot(self) -> dict:
        """返回当前状态快照（JSON-safe），供 HTTP GET /debug/state 使用。"""
        snapshot = {
            "num_envs": self.num_envs,
            "time_scale": self.time_scale,
            "paused": self.paused,
            "force_active": self.force_active,
            "anchor_active": self.anchor_active,
            "mux_active_envs": sorted(self.mux_active_envs),
            "viz_contact_forces": self.viz_contact_forces,
            "viz_trajectories": self.viz_trajectories,
            "plugins": self.plugin_names,
            "body_names": self._body_names,
            "joint_names": self._joint_names,
            "anchored_bodies": {
                str(eid): list(bmap.keys())
                for eid, bmap in self.anchor_poses.items()
            },
        }
        return snapshot
