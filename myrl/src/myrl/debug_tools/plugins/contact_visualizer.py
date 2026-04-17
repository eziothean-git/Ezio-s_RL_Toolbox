"""ContactVisualizer 插件 — 接触力/轨迹可视化。

使用 Isaac Lab VisualizationMarkers 绘制接触力箭头（CylinderCfg），
使用 debug_draw API 绘制运动轨迹线（polyline）。

所有绘制在 on_render() 中执行（sim.render() 期间），不影响物理步进性能。
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from myrl.debug_tools.plugin_base import DebugPlugin

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext


class ContactVisualizer(DebugPlugin):
    name = "contact_visualizer"

    def __init__(self) -> None:
        self._force_markers = None
        self._draw = None
        self._viz_env_id: int = 0
        self._max_arrows: int = 32  # 最多同时显示的力箭头数

    def attach(self, ctx: DebugContext) -> None:
        ctx.viz_contact_forces = False
        ctx.viz_trajectories = False
        ctx.trajectory_buffer = {}
        ctx.trajectory_max_len = 200

        # 尝试创建 VisualizationMarkers（仅 Isaac Sim 非 headless）
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

            cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/myrl_debug/contact_forces",
                markers={
                    "arrow": sim_utils.CylinderCfg(
                        radius=0.008,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.2, 0.0),
                        ),
                    ),
                },
            )
            self._force_markers = VisualizationMarkers(cfg)
            self._force_markers.set_visibility(False)
        except Exception as e:
            print(f"[contact_visualizer] VisualizationMarkers init failed: {e}")
            self._force_markers = None

        # 尝试获取 debug_draw（用于轨迹线）
        try:
            from isaacsim.util.debug_drawing import _debug_drawing
            self._draw = _debug_drawing.acquire_debug_draw_interface()
        except Exception:
            self._draw = None

    def detach(self) -> None:
        if self._force_markers is not None:
            self._force_markers.set_visibility(False)

    def post_step(self, ctx: DebugContext, obs, rew, dones, extras) -> None:
        """记录轨迹数据。"""
        if not ctx.viz_trajectories or ctx._articulation is None:
            return
        try:
            pos = ctx._articulation.data.root_pos_w[self._viz_env_id].cpu().clone()
            buf = ctx.trajectory_buffer.setdefault(self._viz_env_id, [])
            buf.append(pos)
            if len(buf) > ctx.trajectory_max_len:
                buf.pop(0)
        except Exception:
            pass

    def on_render(self, ctx: DebugContext) -> None:
        """在 sim.render() 期间绘制可视化。"""
        if ctx.viz_contact_forces:
            self._render_contact_forces(ctx)
        else:
            if self._force_markers is not None:
                self._force_markers.set_visibility(False)

        if ctx.viz_trajectories and self._draw is not None:
            self._render_trajectories(ctx)

    # ── 接触力箭头（VisualizationMarkers）────────────────────────────

    def _render_contact_forces(self, ctx: DebugContext) -> None:
        """使用 VisualizationMarkers 绘制接触力箭头。"""
        if self._force_markers is None or ctx._articulation is None:
            return

        try:
            contact_sensor = ctx.env.scene.sensors.get("contact_forces")
            if contact_sensor is None:
                return

            forces = contact_sensor.data.net_forces_w[self._viz_env_id]  # (num_bodies, 3)
            art = ctx._articulation

            # 过滤有效力（magnitude > 1N）
            magnitudes = forces.norm(dim=1)  # (num_bodies,)
            valid = magnitudes > 1.0
            valid_indices = torch.where(valid)[0]

            if valid_indices.numel() == 0:
                self._force_markers.set_visibility(False)
                return

            # 限制箭头数量
            if valid_indices.numel() > self._max_arrows:
                valid_indices = valid_indices[:self._max_arrows]

            n = valid_indices.numel()
            device = ctx.device

            # body 位置作为箭头起点
            # contact_sensor body indices 对应 sensor 配置的 body subset
            body_pos = contact_sensor.data.pos_w[self._viz_env_id, valid_indices]  # (n, 3)

            # 力方向 → 箭头朝向（四元数）
            f_valid = forces[valid_indices]  # (n, 3)
            m_valid = magnitudes[valid_indices]  # (n,)

            directions = f_valid / m_valid.unsqueeze(1).clamp(min=1e-6)
            orientations = self._direction_to_quat(directions)  # (n, 4)

            # 缩放：长度 ∝ 力大小，宽度固定
            scale_factor = 0.002  # 1N = 0.002m 长度
            lengths = m_valid * scale_factor
            scales = torch.stack([
                torch.ones(n, device=device) * 0.008,  # radius x
                torch.ones(n, device=device) * 0.008,  # radius y
                lengths,                                # height z
            ], dim=1)

            # 偏移：箭头从 body 中心沿力方向延伸
            translations = body_pos + directions * (lengths * 0.5).unsqueeze(1)

            self._force_markers.set_visibility(True)
            self._force_markers.visualize(
                translations=translations,
                orientations=orientations,
                scales=scales,
            )

        except Exception:
            pass

    @staticmethod
    def _direction_to_quat(directions: torch.Tensor) -> torch.Tensor:
        """将方向向量转换为四元数（z 轴对齐到方向）。

        Args:
            directions: (N, 3) 归一化方向向量

        Returns:
            (N, 4) 四元数 (w, x, y, z)
        """
        n = directions.shape[0]
        device = directions.device
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

        # 旋转轴 = z × direction
        cross = torch.cross(z_axis, directions, dim=1)
        dot = (z_axis * directions).sum(dim=1)  # cos(angle)

        # 处理平行/反平行
        quats = torch.zeros(n, 4, device=device)
        sin_half = torch.sqrt((1.0 - dot).clamp(min=0) * 0.5)
        cos_half = torch.sqrt((1.0 + dot).clamp(min=0) * 0.5)

        cross_norm = cross.norm(dim=1, keepdim=True).clamp(min=1e-8)
        axis = cross / cross_norm

        quats[:, 0] = cos_half        # w
        quats[:, 1] = axis[:, 0] * sin_half  # x
        quats[:, 2] = axis[:, 1] * sin_half  # y
        quats[:, 3] = axis[:, 2] * sin_half  # z

        # 反平行情况（dot ≈ -1）：绕 x 轴旋转 180 度
        antiparallel = dot < -0.999
        if antiparallel.any():
            quats[antiparallel, 0] = 0.0
            quats[antiparallel, 1] = 1.0
            quats[antiparallel, 2] = 0.0
            quats[antiparallel, 3] = 0.0

        return quats

    # ── 运动轨迹线（debug_draw）──────────────────────────────────────

    def _render_trajectories(self, ctx: DebugContext) -> None:
        """使用 debug_draw 绘制根体运动轨迹。"""
        buf = ctx.trajectory_buffer.get(self._viz_env_id, [])
        if len(buf) < 2 or self._draw is None:
            return

        try:
            self._draw.clear_lines()
            n = len(buf) - 1
            starts = [buf[i].tolist() for i in range(n)]
            ends = [buf[i + 1].tolist() for i in range(n)]

            colors = []
            for i in range(n):
                alpha = 0.3 + 0.7 * (i / n)
                colors.append((0.0, 0.7, 1.0, alpha))
            sizes = [1.5] * n

            self._draw.draw_lines(starts, ends, colors, sizes)
        except Exception:
            pass
