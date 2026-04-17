"""myrl Debug Tools — 交互式仿真调试工具集。

入口：
    from myrl.debug_tools import enable_debug_tools
    ctx = enable_debug_tools(env, bus, headless=False)

功能：
    - 外力施加（ForceApplicator）：选中刚体施加力
    - 动作覆盖（ActionMux）：MUX 覆盖特定 env 的策略输出
    - 刚体锚定（BodyAnchor）：固定刚体位姿
    - 接触力可视化（ContactVisualizer）：debug draw 接触力箭头/轨迹
    - 时间流速（TimeScale）：慢放/暂停/单步
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myrl.core.databus.bus import DataBus
    from myrl.core.databus.signal_server import SignalServer
    from myrl.debug_tools.context import DebugContext


def enable_debug_tools(
    env,
    bus: DataBus | None = None,
    plugins: list[str] | None = None,
    signal_server: SignalServer | None = None,
    headless: bool = True,
) -> DebugContext:
    """在环境上启用调试工具。

    Args:
        env: VecEnv wrapper（IsaacLabBackend 或 InstinctRlVecEnvWrapper）。
        bus: DataBus 实例。None 则不发布调试 channel。
        plugins: 要加载的插件名称列表。None = 全部加载。
        signal_server: 如提供，注册 HTTP 调试端点。
        headless: False 时创建 Isaac Sim omni.ui 面板。

    Returns:
        DebugContext 实例，可用于编程式控制。
    """
    from myrl.debug_tools.context import DebugContext
    from myrl.debug_tools.env_patch import enable_debug_patch
    from myrl.debug_tools.plugins import load_plugins

    # 创建中心状态
    ctx = DebugContext(env.unwrapped, bus)
    env._debug_ctx = ctx

    # 加载并注册插件
    loaded = load_plugins(plugins)
    for plugin in loaded:
        try:
            plugin.attach(ctx)
            ctx.register_plugin(plugin)
            print(f"[debug_tools] Plugin loaded: {plugin.name}")
        except Exception as e:
            print(f"[debug_tools] Plugin '{plugin.name}' attach failed: {e}")

    # 注入 env 钩子
    enable_debug_patch(env, ctx)

    # 注册 HTTP 调试端点
    if signal_server is not None:
        try:
            from myrl.debug_tools.http_routes import register_debug_routes
            register_debug_routes(signal_server, ctx)
            print("[debug_tools] HTTP debug routes registered")
        except Exception as e:
            print(f"[debug_tools] HTTP routes registration failed: {e}")

    # 创建 Isaac Sim UI 面板（非 headless）
    if not headless:
        try:
            from myrl.debug_tools.ui.debug_panel import DebugToolsPanel
            ctx._ui_panel = DebugToolsPanel(ctx)
            print("[debug_tools] Isaac Sim debug panel created")
        except Exception as e:
            print(f"[debug_tools] UI panel creation skipped: {e}")

    print(f"[debug_tools] Enabled with {len(ctx._plugins)} plugins: {ctx.plugin_names}")
    return ctx
