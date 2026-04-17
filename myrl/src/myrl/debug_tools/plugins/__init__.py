"""调试插件注册表。

用法：
    from myrl.debug_tools.plugins import load_plugins
    plugins = load_plugins()              # 加载全部
    plugins = load_plugins(["time_scale", "action_mux"])  # 按名称加载
"""
from __future__ import annotations

from myrl.debug_tools.plugin_base import DebugPlugin

# 已知插件映射：name → (module_path, class_name)
_PLUGIN_REGISTRY: dict[str, tuple[str, str]] = {
    "time_scale":         ("myrl.debug_tools.plugins.time_scale",        "TimeScale"),
    "action_mux":         ("myrl.debug_tools.plugins.action_mux",        "ActionMux"),
    "force_applicator":   ("myrl.debug_tools.plugins.force_applicator",  "ForceApplicator"),
    "body_anchor":        ("myrl.debug_tools.plugins.body_anchor",       "BodyAnchor"),
    "contact_visualizer": ("myrl.debug_tools.plugins.contact_visualizer", "ContactVisualizer"),
}


def load_plugins(names: list[str] | None = None) -> list[DebugPlugin]:
    """按名称加载插件实例。names=None 则加载全部。

    加载失败的插件会打印警告并跳过（不阻断启动）。
    """
    import importlib

    target_names = names if names is not None else list(_PLUGIN_REGISTRY.keys())
    plugins: list[DebugPlugin] = []

    for name in target_names:
        entry = _PLUGIN_REGISTRY.get(name)
        if entry is None:
            print(f"[debug_tools] Unknown plugin: {name}")
            continue
        module_path, class_name = entry
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            plugins.append(cls())
        except Exception as e:
            print(f"[debug_tools] Failed to load plugin '{name}': {e}")

    return plugins
