"""Isaac Sim 调试面板 — omni.ui 窗口 + viewport 鼠标交互。

仅在非 headless 模式下创建。提供：
    - 时间流速滑条 + 暂停/单步按钮
    - 外力施加控件（选中体 + 力向量 + Ctrl+拖拽施力）
    - 动作 MUX 关节滑条
    - 锚点切换
    - 可视化开关
    - 视口选择集成（点击 body 自动更新选中状态）
    - Ctrl+左键拖拽施力（水平→X力，垂直→Z力）

依赖 omni.ui + carb.input（Isaac Sim Kit SDK），仅在容器内非 headless 时可用。
"""
from __future__ import annotations

import math
import re
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myrl.debug_tools.context import DebugContext

try:
    import carb
    import carb.input
    import omni.appwindow
    import omni.ui as ui
    import omni.usd

    _OMNI_AVAILABLE = True
except ImportError:
    _OMNI_AVAILABLE = False


class DebugToolsPanel:
    """Isaac Sim omni.ui 调试工具面板。"""

    WINDOW_NAME = "myrl Debug Tools"

    def __init__(self, ctx: DebugContext) -> None:
        if not _OMNI_AVAILABLE:
            raise ImportError("omni.ui not available (headless mode?)")

        self._ctx = ctx
        self._selected_env_id: int = 0
        self._selected_body_id: int = 0
        self._selected_body_name: str = ""

        # 力输入缓存
        self._force_x = 0.0
        self._force_y = 0.0
        self._force_z = 1.0
        self._force_magnitude = 100.0

        # 拖拽状态
        self._dragging = False
        self._drag_start = None  # (x, y) pixel
        self._drag_body_id: int = 0
        self._drag_env_id: int = 0

        # 创建窗口
        self._window = ui.Window(self.WINDOW_NAME, width=380, height=700,
                                 dock_preference=ui.DockPreference.RIGHT_BOTTOM)
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    self._build_time_section()
                    self._build_force_section()
                    self._build_mux_section()
                    self._build_anchor_section()
                    self._build_viz_section()

        # 订阅视口选择事件
        self._setup_viewport_selection()
        # 订阅鼠标事件（Ctrl+拖拽施力）
        self._setup_mouse_drag()

    def __del__(self) -> None:
        """清理订阅。"""
        if hasattr(self, "_mouse_sub") and self._mouse_sub is not None:
            try:
                self._input.unsubscribe_to_mouse_events(self._mouse, self._mouse_sub)
            except Exception:
                pass
        if hasattr(self, "_window") and self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None

    # ══════════════════════════════════════════════════════════════════
    #  鼠标拖拽施力（Ctrl+左键）
    # ══════════════════════════════════════════════════════════════════

    def _setup_mouse_drag(self) -> None:
        """订阅 carb.input 鼠标事件，实现 Ctrl+拖拽施力。"""
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._mouse = self._appwindow.get_mouse()
            self._keyboard = self._appwindow.get_keyboard()
            self._mouse_sub = self._input.subscribe_to_mouse_events(
                self._mouse,
                lambda event, *args, obj=weakref.proxy(self): obj._on_mouse_event(event, *args),
            )
        except Exception as e:
            print(f"[debug_panel] Mouse drag setup failed: {e}")
            self._mouse_sub = None

    def _is_ctrl_held(self) -> bool:
        """检查 Ctrl 键是否按下。"""
        try:
            return (
                self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT_CONTROL) > 0
                or self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT_CONTROL) > 0
            )
        except Exception:
            return False

    def _on_mouse_event(self, event, *args, **kwargs) -> bool:
        """处理鼠标事件：Ctrl+左键拖拽 = 施力。"""
        try:
            if event.type == carb.input.MouseEventType.LEFT_BUTTON_DOWN:
                if self._is_ctrl_held() and self._selected_body_id is not None:
                    self._dragging = True
                    self._drag_start = (event.normalized_coords.x, event.normalized_coords.y)
                    self._drag_body_id = self._selected_body_id
                    self._drag_env_id = self._selected_env_id
                    return True  # 消费事件，阻止 viewport 操作

            elif event.type == carb.input.MouseEventType.MOVE:
                if self._dragging and self._drag_start is not None:
                    dx = event.normalized_coords.x - self._drag_start[0]
                    dy = event.normalized_coords.y - self._drag_start[1]
                    # 映射：水平 dx → 世界 X，垂直 dy → 世界 Z（向上为正）
                    sensitivity = self._force_magnitude
                    force = [dx * sensitivity, 0.0, -dy * sensitivity]
                    plugin = self._ctx.get_plugin("force_applicator")
                    if plugin:
                        plugin.set_force(
                            self._ctx,
                            env_id=self._drag_env_id,
                            body_id=self._drag_body_id,
                            force_w=force,
                        )
                    # 更新 UI 显示
                    mag = math.sqrt(force[0]**2 + force[1]**2 + force[2]**2)
                    if hasattr(self, "_drag_label"):
                        self._drag_label.text = f"Dragging: [{force[0]:.0f}, {force[1]:.0f}, {force[2]:.0f}] ({mag:.0f}N)"
                    return True

            elif event.type == carb.input.MouseEventType.LEFT_BUTTON_UP:
                if self._dragging:
                    self._dragging = False
                    self._drag_start = None
                    # 非 Hold 模式时松开清除力
                    hold = self._hold_cb.model.as_bool if hasattr(self, "_hold_cb") else False
                    if not hold:
                        plugin = self._ctx.get_plugin("force_applicator")
                        if plugin:
                            plugin.clear_forces(self._ctx)
                    if hasattr(self, "_drag_label"):
                        self._drag_label.text = "Ctrl+drag on viewport to apply force"
                    return True
        except Exception:
            pass

        return False  # 不消费事件，让 viewport 正常处理

    # ══════════════════════════════════════════════════════════════════
    #  时间控制
    # ══════════════════════════════════════════════════════════════════

    def _build_time_section(self) -> None:
        with ui.CollapsableFrame("Time Control", collapsed=False):
            with ui.VStack(spacing=4):
                with ui.HStack(height=24):
                    ui.Label("Time Scale:", width=80)
                    self._time_slider = ui.FloatSlider(min=0.01, max=1.0, step=0.01)
                    self._time_slider.model.set_value(1.0)
                    self._time_slider.model.add_value_changed_fn(self._on_time_scale)

                with ui.HStack(height=28, spacing=4):
                    self._pause_btn = ui.Button("Pause", height=24)
                    self._pause_btn.set_clicked_fn(self._on_pause)
                    step_btn = ui.Button("Single Step", height=24)
                    step_btn.set_clicked_fn(self._on_single_step)

    def _on_time_scale(self, model) -> None:
        plugin = self._ctx.get_plugin("time_scale")
        if plugin:
            plugin.set_scale(self._ctx, model.as_float)

    def _on_pause(self) -> None:
        plugin = self._ctx.get_plugin("time_scale")
        if plugin:
            plugin.toggle_pause(self._ctx)
            self._pause_btn.text = "Resume" if self._ctx.paused else "Pause"

    def _on_single_step(self) -> None:
        plugin = self._ctx.get_plugin("time_scale")
        if plugin:
            plugin.single_step(self._ctx)

    # ══════════════════════════════════════════════════════════════════
    #  外力施加
    # ══════════════════════════════════════════════════════════════════

    def _build_force_section(self) -> None:
        with ui.CollapsableFrame("Force Applicator", collapsed=False):
            with ui.VStack(spacing=4):
                with ui.HStack(height=20):
                    ui.Label("Selected:", width=60)
                    self._body_label = ui.Label("(click body in viewport)")

                with ui.HStack(height=20):
                    ui.Label("Env ID:", width=60)
                    self._env_field = ui.IntField(width=60)
                    self._env_field.model.set_value(0)

                # 力向量输入
                self._force_fields = {}
                for axis, attr, default in [("X", "_force_x", 0.0), ("Y", "_force_y", 0.0), ("Z", "_force_z", 1.0)]:
                    with ui.HStack(height=20):
                        ui.Label(f"F.{axis}:", width=40)
                        field = ui.FloatField()
                        field.model.set_value(default)
                        field.model.add_value_changed_fn(
                            lambda m, a=attr: setattr(self, a, m.as_float)
                        )
                        self._force_fields[axis] = field

                with ui.HStack(height=20):
                    ui.Label("Magnitude:", width=70)
                    mag_slider = ui.FloatSlider(min=0, max=1000, step=10)
                    mag_slider.model.set_value(100.0)
                    mag_slider.model.add_value_changed_fn(
                        lambda m: setattr(self, "_force_magnitude", m.as_float)
                    )

                with ui.HStack(height=28, spacing=4):
                    apply_btn = ui.Button("Apply Force", height=24)
                    apply_btn.set_clicked_fn(self._on_apply_force)
                    clear_btn = ui.Button("Clear All", height=24)
                    clear_btn.set_clicked_fn(self._on_clear_forces)

                with ui.HStack(height=20):
                    self._hold_cb = ui.CheckBox(width=20)
                    ui.Label("Hold (continuous force)")

                # 拖拽状态提示
                ui.Spacer(height=2)
                self._drag_label = ui.Label(
                    "Ctrl+drag on viewport to apply force",
                    style={"color": 0xFF888888, "font_size": 11},
                )

    def _on_apply_force(self) -> None:
        plugin = self._ctx.get_plugin("force_applicator")
        if not plugin:
            return

        dx, dy, dz = self._force_x, self._force_y, self._force_z
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag < 1e-6:
            dz = 1.0
            mag = 1.0
        scale = self._force_magnitude / mag
        force = [dx * scale, dy * scale, dz * scale]

        hold = self._hold_cb.model.as_bool if self._hold_cb else False
        env_id = self._env_field.model.as_int

        plugin.set_force(
            self._ctx,
            env_id=env_id,
            body_id=self._selected_body_id,
            force_w=force,
            impulse=not hold,
        )

    def _on_clear_forces(self) -> None:
        plugin = self._ctx.get_plugin("force_applicator")
        if plugin:
            plugin.clear_forces(self._ctx)

    # ══════════════════════════════════════════════════════════════════
    #  动作 MUX
    # ══════════════════════════════════════════════════════════════════

    def _build_mux_section(self) -> None:
        with ui.CollapsableFrame("Action MUX", collapsed=True):
            with ui.VStack(spacing=4):
                with ui.HStack(height=20):
                    ui.Label("Env ID:", width=60)
                    self._mux_env_field = ui.IntField(width=60)
                    self._mux_env_field.model.set_value(0)

                ui.Label("Check joint to enable override, drag slider to set value:", height=16,
                         style={"color": 0xFF888888, "font_size": 11})

                self._mux_sliders: list = []
                for i, name in enumerate(self._ctx._joint_names[:32]):
                    with ui.HStack(height=18, spacing=2):
                        cb = ui.CheckBox(width=16)
                        ui.Label(name[:20], width=100, style={"font_size": 11})
                        slider = ui.FloatSlider(min=-3.14, max=3.14, step=0.01)
                        slider.model.set_value(0.0)
                        slider.model.add_value_changed_fn(
                            lambda m, idx=i, checkbox=cb: self._on_mux_slider(idx, m.as_float, checkbox)
                        )
                        self._mux_sliders.append((cb, slider))

                with ui.HStack(height=28, spacing=4):
                    clear_btn = ui.Button("Clear All MUX", height=24)
                    clear_btn.set_clicked_fn(self._on_clear_mux)

    def _on_mux_slider(self, joint_idx: int, value: float, checkbox) -> None:
        plugin = self._ctx.get_plugin("action_mux")
        if not plugin:
            return
        env_id = self._mux_env_field.model.as_int
        if checkbox.model.as_bool:
            plugin.set_override(self._ctx, env_id, joint_idx, value)

    def _on_clear_mux(self) -> None:
        plugin = self._ctx.get_plugin("action_mux")
        if plugin:
            plugin.clear_all(self._ctx)
        for cb, _ in self._mux_sliders:
            cb.model.set_value(False)

    # ══════════════════════════════════════════════════════════════════
    #  锚点
    # ══════════════════════════════════════════════════════════════════

    def _build_anchor_section(self) -> None:
        with ui.CollapsableFrame("Body Anchor", collapsed=True):
            with ui.VStack(spacing=4):
                with ui.HStack(height=28, spacing=4):
                    anchor_btn = ui.Button("Toggle Anchor (selected body)", height=24)
                    anchor_btn.set_clicked_fn(self._on_toggle_anchor)
                    clear_btn = ui.Button("Clear All", height=24)
                    clear_btn.set_clicked_fn(self._on_clear_anchors)
                self._anchor_label = ui.Label("No anchors active")

    def _on_toggle_anchor(self) -> None:
        plugin = self._ctx.get_plugin("body_anchor")
        if plugin:
            env_id = self._env_field.model.as_int
            anchored = plugin.toggle_anchor(self._ctx, env_id, self._selected_body_id)
            name = self._selected_body_name or f"body_{self._selected_body_id}"
            status = "ANCHORED" if anchored else "released"
            self._anchor_label.text = f"env {env_id} / {name}: {status}"

    def _on_clear_anchors(self) -> None:
        plugin = self._ctx.get_plugin("body_anchor")
        if plugin:
            plugin.clear_all(self._ctx)
            self._anchor_label.text = "No anchors active"

    # ══════════════════════════════════════════════════════════════════
    #  可视化
    # ══════════════════════════════════════════════════════════════════

    def _build_viz_section(self) -> None:
        with ui.CollapsableFrame("Visualization", collapsed=False):
            with ui.VStack(spacing=4):
                with ui.HStack(height=20):
                    self._viz_contacts_cb = ui.CheckBox(width=20)
                    self._viz_contacts_cb.model.add_value_changed_fn(
                        lambda m: setattr(self._ctx, "viz_contact_forces", m.as_bool)
                    )
                    ui.Label("Contact Forces")

                with ui.HStack(height=20):
                    self._viz_traj_cb = ui.CheckBox(width=20)
                    self._viz_traj_cb.model.add_value_changed_fn(
                        lambda m: setattr(self._ctx, "viz_trajectories", m.as_bool)
                    )
                    ui.Label("Root Trajectory")

    # ══════════════════════════════════════════════════════════════════
    #  视口选择
    # ══════════════════════════════════════════════════════════════════

    def _setup_viewport_selection(self) -> None:
        """监听 Isaac Sim 视口中的 prim 选择事件。"""
        try:
            usd_ctx = omni.usd.get_context()
            events = usd_ctx.get_stage_event_stream()
            self._selection_sub = events.create_subscription_to_pop(
                self._on_selection_changed
            )
        except Exception:
            pass

    def _on_selection_changed(self, event) -> None:
        """视口选择变化时更新选中体。"""
        try:
            selection = omni.usd.get_context().get_selection()
            paths = selection.get_selected_prim_paths()
            if not paths:
                return
            env_id, body_id, body_name = self._resolve_prim_path(paths[0])
            if body_id is not None:
                self._selected_env_id = env_id
                self._selected_body_id = body_id
                self._selected_body_name = body_name
                self._body_label.text = f"env {env_id} / {body_name} (id={body_id})"
                self._env_field.model.set_value(env_id)
        except Exception:
            pass

    def _resolve_prim_path(self, prim_path: str) -> tuple[int, int | None, str]:
        """将 USD prim 路径解析为 (env_id, body_id, body_name)。

        路径格式：/World/envs/env_0/Robot/torso_link
        """
        env_match = re.search(r"/env_(\d+)/", prim_path)
        env_id = int(env_match.group(1)) if env_match else 0

        parts = prim_path.rstrip("/").split("/")
        body_name = parts[-1] if parts else ""

        body_id = None
        for i, name in enumerate(self._ctx._body_names):
            if name == body_name:
                body_id = i
                break

        return env_id, body_id, body_name
