from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor


class RewardBuilder:
    """管理多个 reward 项，计算加权和并记录 breakdown。"""

    def __init__(self):
        self._terms: dict[str, tuple[Callable, float, bool]] = {}  # name -> (func, weight, active)

    def add(self, term_name: str, func: Callable, weight: float,
            active: bool = True) -> RewardBuilder:
        self._terms[term_name] = (func, weight, active)
        return self

    def remove(self, term_name: str) -> RewardBuilder:
        self._terms.pop(term_name, None)
        return self

    def set_weight(self, term_name: str, weight: float) -> RewardBuilder:
        func, _, active = self._terms[term_name]
        self._terms[term_name] = (func, weight, active)
        return self

    def set_active(self, term_name: str, active: bool) -> RewardBuilder:
        func, weight, _ = self._terms[term_name]
        self._terms[term_name] = (func, weight, active)
        return self

    def compute(self, env, return_per_term: bool = False
                ) -> tuple[Tensor, dict[str, Tensor]]:
        """计算 reward。

        Returns:
            total:    (num_envs,) 加权总奖励
            per_term: {name: unweighted_reward} breakdown（始终返回，用于日志）
        """
        total = None
        per_term: dict[str, Tensor] = {}
        for name, (func, weight, active) in self._terms.items():
            r = func(env)              # (num_envs,)
            per_term[name] = r
            if active:
                weighted = weight * r
                total = weighted if total is None else total + weighted
        if total is None:
            # 没有 active 项
            device = next(iter(per_term.values())).device if per_term else "cpu"
            total = torch.zeros(env.num_envs, device=device)
        return total, per_term
