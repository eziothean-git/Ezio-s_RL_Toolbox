"""RewardTermMeta + TransformMeta — 奖励函数与算子的元数据描述。"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


# ── 预训练 reward 分布估计元数据 ────────────────────────────────────

@dataclass
class RewardSignature:
    """静态可推导的 reward 特性（预训练期估计用）。

    让 RewardEstimator 不启动 Isaac Sim 就能给出每个 term 的值域、形状提示、
    以及"分析上界公式"（基于 URDF joint limits）。

    Attributes:
        shape_hint:
            - "exp_kernel"    : exp(-err²/σ²) 形式，值域 (0, 1]
            - "l2_sum"        : Σ xᵢ² 形式，值域 [0, +∞)
            - "bounded_quad"  : 明确 [0, 1] 范围的二次形式（如 orientation）
            - "clamped_sum"   : Σ max(0, ...) 类，分析无上界（需 runtime）
            - "linear"        : 线性组合
        value_range:
            分析值域 (min, max)，None 表示开区间。独立于 weight。
        input_views:
            term 访问的 RobotHandle 视图字段（如 "joints.applied_torque"），
            用于前端展示依赖关系。
        deps:
            运行时外部依赖的类别：
            - "command"        : 需要 env.command_manager
            - "history"        : 需要 Contact/Joint 历史累积
            - "sensor:contact" : 需要 ContactSensor
            - "reference"      : 需要 motion reference dataset
            这些 term 的 value_range 无法分析估计（标记为 requires_runtime）。
        max_expr:
            基于 URDF 的分析最大值表达式（字符串），供 RewardEstimator AST 白
            名单 eval。允许访问 `urdf` 对象和 sum/max/min/abs 函数。
            例：`"sum(j.limits.effort**2 for j in urdf.joints.values() if j.limits)"`。
            None 表示无法分析（依赖运行时）。
        notes:
            自由文本，用于前端 tooltip 解释。
    """
    shape_hint: str
    value_range: tuple[float | None, float | None] | None = None
    input_views: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    max_expr: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "shape_hint": self.shape_hint,
            "value_range": list(self.value_range) if self.value_range else None,
            "input_views": list(self.input_views),
            "deps": list(self.deps),
            "max_expr": self.max_expr,
            "notes": self.notes,
        }

# repo root — 用于计算 source_file 相对路径
_REPO_ROOT = Path(__file__).parents[7]  # myrl/src/myrl/core/task/reward_lib/meta.py -> repo


def _rel_path(fn: Callable) -> str:
    try:
        p = Path(inspect.getfile(fn))
        return str(p.relative_to(_REPO_ROOT))
    except (ValueError, TypeError):
        return inspect.getfile(fn)


def _source_line(fn: Callable) -> int:
    try:
        return inspect.getsourcelines(fn)[1]
    except (OSError, TypeError):
        return 0


@dataclass
class RewardTermMeta:
    """单个 reward term 的完整元数据（含 Pydantic 参数类）。"""

    name: str
    module: str
    source_file: str
    source_line: int
    version: str
    description: str
    long_description: str
    tags: list[str]
    params: type  # type[BaseModel]
    output_description: str
    author: str
    added_in: str
    signature: RewardSignature | None = None
    _func: Callable = field(default=None, repr=False)

    @classmethod
    def from_fn(
        cls,
        fn: Callable,
        *,
        name: str,
        version: str,
        description: str,
        long_description: str,
        tags: list[str],
        params: type,
        output_description: str,
        author: str,
        added_in: str,
        signature: RewardSignature | None = None,
    ) -> RewardTermMeta:
        mod = inspect.getmodule(fn)
        return cls(
            name=name,
            module=mod.__name__ if mod else "<unknown>",
            source_file=_rel_path(fn),
            source_line=_source_line(fn),
            version=version,
            description=description,
            long_description=long_description,
            tags=list(tags),
            params=params,
            output_description=output_description,
            author=author,
            added_in=added_in,
            signature=signature,
            _func=fn,
        )

    def params_json_schema(self) -> dict:
        """返回 Pydantic 模型的标准 JSON Schema（前端直接消费）。"""
        return self.params.model_json_schema()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "version": self.version,
            "description": self.description,
            "long_description": self.long_description,
            "tags": self.tags,
            "params_schema": self.params_json_schema(),
            "output_description": self.output_description,
            "author": self.author,
            "added_in": self.added_in,
            "signature": self.signature.to_dict() if self.signature else None,
        }


@dataclass
class TransformMeta:
    """单个奖励后处理算子的元数据。"""

    name: str
    module: str
    source_file: str
    source_line: int
    version: str
    description: str
    tags: list[str]
    params: type  # type[BaseModel]
    _cls: type = field(default=None, repr=False)

    @classmethod
    def from_cls(
        cls,
        transform_cls: type,
        *,
        name: str,
        version: str,
        description: str,
        tags: list[str],
        params: type,
    ) -> TransformMeta:
        mod = inspect.getmodule(transform_cls)
        return cls(
            name=name,
            module=mod.__name__ if mod else "<unknown>",
            source_file=_rel_path(transform_cls),
            source_line=_source_line(transform_cls),
            version=version,
            description=description,
            tags=list(tags),
            params=params,
            _cls=transform_cls,
        )

    def params_json_schema(self) -> dict:
        return self.params.model_json_schema()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "params_schema": self.params_json_schema(),
        }
