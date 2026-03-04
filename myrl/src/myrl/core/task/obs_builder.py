from __future__ import annotations
from collections import OrderedDict
from typing import Callable
from torch import Tensor


class ObsGroup:
    """单个 obs 分组（如 policy / critic）。"""

    def __init__(self, name: str):
        self._name = name
        self._terms: OrderedDict[str, tuple[Callable, tuple[int, ...]]] = OrderedDict()

    def add(self, term_name: str, func: Callable, shape: tuple[int, ...]) -> ObsGroup:
        """注册一个 obs 项。shape 是单环境输出的维度（不含 num_envs）。"""
        self._terms[term_name] = (func, shape)
        return self

    def remove(self, term_name: str) -> ObsGroup:
        self._terms.pop(term_name, None)
        return self

    def obs_format(self) -> dict[str, tuple[int, ...]]:
        """返回 instinct_rl obs_format 的子层。"""
        return {name: shape for name, (_, shape) in self._terms.items()}

    def compute(self, env) -> Tensor:
        """拼接本组所有 obs 项 → (num_envs, flat_dim)。"""
        import torch
        parts = [func(env) for func, _ in self._terms.values()]
        return torch.cat(parts, dim=-1)


class ObsBuilder:
    """管理多个 obs 分组，兼容 instinct_rl obs_format 格式。"""

    def __init__(self):
        self._groups: dict[str, ObsGroup] = {}

    def __getattr__(self, group_name: str) -> ObsGroup:
        if group_name.startswith("_"):
            raise AttributeError(group_name)
        if group_name not in self._groups:
            self._groups[group_name] = ObsGroup(group_name)
        return self._groups[group_name]

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """返回完整 obs_format（兼容 instinct_rl VecEnv 合约）。"""
        return {name: group.obs_format() for name, group in self._groups.items()}

    def compute(self, env) -> dict[str, Tensor]:
        """返回 {group_name: flat_tensor} 字典。"""
        return {name: group.compute(env) for name, group in self._groups.items()}
