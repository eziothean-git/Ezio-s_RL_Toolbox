"""myrl.core.task.reward_lib — 奖励函数资产化系统。

核心装饰器：
    @reward_fn(...)   — 将函数注册为 RewardLibrary term
    @transform_fn(...) — 将类注册为 TransformLibrary 算子

全局访问器：
    get_reward_library()    → RewardLibrary 单例
    get_transform_library() → TransformLibrary 单例

使用示例：
    from myrl.core.task.reward_lib import reward_fn, get_reward_library
    from pydantic import BaseModel, Field

    class MyParams(BaseModel):
        std: float = Field(0.25, ge=0.01)

    @reward_fn(
        description="示例奖励",
        tags=["locomotion"],
        params=MyParams,
    )
    def my_reward(robot, params: MyParams):
        ...
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from .library import get_reward_library, get_transform_library, RewardLibrary, TransformLibrary
from .meta import RewardTermMeta, TransformMeta
from .transform import RewardTransform

if TYPE_CHECKING:
    pass

__all__ = [
    "reward_fn",
    "transform_fn",
    "get_reward_library",
    "get_transform_library",
    "RewardLibrary",
    "TransformLibrary",
    "RewardTermMeta",
    "TransformMeta",
    "RewardTransform",
]


def reward_fn(
    *,
    description: str,
    tags: list[str],
    params: type,                     # type[BaseModel]
    version: str = "1.0.0",
    long_description: str = "",
    output_description: str = "scalar reward per environment",
    author: str = "ezio",
    added_in: str = "",
    name: str | None = None,          # None → 使用函数名
) -> Callable:
    """将函数注册为 RewardLibrary term 的装饰器。

    Args:
        description:       简短描述（1 行）
        tags:              分类标签列表
        params:            Pydantic BaseModel 子类（描述超参数）
        version:           语义版本号
        long_description:  多行详细说明（可含公式）
        output_description: 输出说明
        author:            作者名
        added_in:          添加日期（ISO 格式字符串）
        name:              注册名，默认用函数名
    """
    def decorator(fn: Callable) -> Callable:
        term_name = name or fn.__name__
        meta = RewardTermMeta.from_fn(
            fn,
            name=term_name,
            version=version,
            description=description,
            long_description=long_description,
            tags=tags,
            params=params,
            output_description=output_description,
            author=author,
            added_in=added_in,
        )
        get_reward_library().register(meta)
        fn.__reward_meta__ = meta
        return fn
    return decorator


def transform_fn(
    *,
    name: str,
    description: str,
    tags: list[str],
    params: type,                     # type[BaseModel]
    version: str = "1.0.0",
) -> Callable:
    """将 RewardTransform 子类注册为 TransformLibrary 算子的装饰器。

    Args:
        name:        注册名（必填，确保唯一）
        description: 简短描述
        tags:        分类标签
        params:      Pydantic BaseModel 子类
        version:     语义版本号
    """
    def decorator(cls: type) -> type:
        meta = TransformMeta.from_cls(
            cls,
            name=name,
            version=version,
            description=description,
            tags=tags,
            params=params,
        )
        get_transform_library().register(meta)
        cls.__transform_meta__ = meta
        return cls
    return decorator


# ── 自动注册内置 transform 算子 ──────────────────────────────────────
# 导入 transform 模块触发 @transform_fn 装饰器注册
# （内置算子用 Params 内嵌类，不用 @transform_fn，需手动注册）

def _register_builtin_transforms() -> None:
    from .transform import (
        RunningNormalize, RelativeRebalance, ClipReward, WeightSchedule,
    )
    lib = get_transform_library()

    _builtins = [
        (RunningNormalize, "running_normalize",
         "按 term 的运行期标准差归一化，消除量纲差异，稳定 PPO 训练",
         ["normalization", "stateful"]),
        (RelativeRebalance, "relative_rebalance",
         "调整各 term 权重使实际贡献占比趋近目标，同时保持总奖励幅度恒定",
         ["rebalance", "stateful"]),
        (ClipReward, "clip_reward",
         "裁剪奖励范围 + 可选 EMA 平滑",
         ["clip", "smooth"]),
        (WeightSchedule, "weight_schedule",
         "根据训练步数线性/余弦插值调整 term 权重，实现课程学习",
         ["curriculum", "schedule"]),
    ]

    for cls, reg_name, desc, tags in _builtins:
        if reg_name not in lib._registry:
            meta = TransformMeta.from_cls(
                cls,
                name=reg_name,
                version="1.0.0",
                description=desc,
                tags=tags,
                params=cls.Params,
            )
            lib.register(meta)


_register_builtin_transforms()
