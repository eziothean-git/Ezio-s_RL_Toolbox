"""RewardLibrary + TransformLibrary — 两个全局单例注册表。"""
from __future__ import annotations

import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import yaml

from .meta import RewardTermMeta, TransformMeta


class RewardLibrary:
    """全局 reward term 注册表（单例）。"""

    def __init__(self):
        self._registry: dict[str, RewardTermMeta] = {}

    # ── 注册 ──────────────────────────────────────────────────────

    def register(self, meta: RewardTermMeta) -> None:
        if meta.name in self._registry:
            raise ValueError(f"RewardLibrary: term '{meta.name}' 已注册（来自 "
                             f"{self._registry[meta.name].source_file}）")
        self._registry[meta.name] = meta

    # ── 查询 ──────────────────────────────────────────────────────

    def get(self, name: str) -> RewardTermMeta:
        if name not in self._registry:
            available = sorted(self._registry)
            raise KeyError(
                f"RewardLibrary: 未找到 term '{name}'。"
                f"已注册: {available}"
            )
        return self._registry[name]

    def list_names(self) -> list[str]:
        return sorted(self._registry)

    def list_by_tag(self, tag: str) -> list[str]:
        return sorted(n for n, m in self._registry.items() if tag in m.tags)

    # ── 实例化 ────────────────────────────────────────────────────

    def build(self, name: str, robot_name: str = "robot", **params_kwargs) -> Callable:
        """验证参数、绑定 params，返回 make_rew 包装后的 (env)->Tensor 函数。"""
        # 延迟导入，避免循环（make_rew 依赖 isaaclab，只在 AppLauncher 后可用）
        from myrl.core.compat.views.robot import make_rew

        meta = self.get(name)
        params_obj = meta.params(**params_kwargs)       # Pydantic 验证，失败抛 ValidationError
        bound_fn = partial(meta._func, params=params_obj)
        return make_rew(bound_fn, robot_name)

    # ── 导出 ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {n: m.to_dict() for n, m in sorted(self._registry.items())}

    def export_yaml(self, path: str | Path) -> None:
        """导出全量 Schema 为 YAML（供前端 / 文档消费）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "1",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "terms": self.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"[RewardLibrary] 导出 {len(self._registry)} 个 term → {path}")


class TransformLibrary:
    """全局 RewardTransform 算子注册表（单例）。"""

    def __init__(self):
        self._registry: dict[str, TransformMeta] = {}

    # ── 注册 ──────────────────────────────────────────────────────

    def register(self, meta: TransformMeta) -> None:
        if meta.name in self._registry:
            raise ValueError(f"TransformLibrary: transform '{meta.name}' 已注册")
        self._registry[meta.name] = meta

    # ── 查询 ──────────────────────────────────────────────────────

    def get(self, name: str) -> TransformMeta:
        if name not in self._registry:
            available = sorted(self._registry)
            raise KeyError(
                f"TransformLibrary: 未找到 transform '{name}'。"
                f"已注册: {available}"
            )
        return self._registry[name]

    def list_names(self) -> list[str]:
        return sorted(self._registry)

    # ── 实例化 ────────────────────────────────────────────────────

    def build(self, name: str, **params_kwargs):
        """验证参数，返回实例化的 RewardTransform 对象。"""
        meta = self.get(name)
        params_obj = meta.params(**params_kwargs)
        return meta._cls(params_obj)

    # ── 导出 ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {n: m.to_dict() for n, m in sorted(self._registry.items())}

    def export_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "1",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "transforms": self.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"[TransformLibrary] 导出 {len(self._registry)} 个 transform → {path}")


# ── 全局单例 ──────────────────────────────────────────────────────────
_reward_library: RewardLibrary | None = None
_transform_library: TransformLibrary | None = None


def get_reward_library() -> RewardLibrary:
    global _reward_library
    if _reward_library is None:
        _reward_library = RewardLibrary()
    return _reward_library


def get_transform_library() -> TransformLibrary:
    global _transform_library
    if _transform_library is None:
        _transform_library = TransformLibrary()
    return _transform_library
