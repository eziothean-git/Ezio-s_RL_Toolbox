"""RewardTermMeta + TransformMeta — 奖励函数与算子的元数据描述。"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

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
