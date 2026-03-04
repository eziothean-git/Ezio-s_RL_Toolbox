from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from myrl.core.task.reward_lib.transform import RewardTransform


class RewardBuilder:
    """管理多个 reward 项，计算加权和并记录 breakdown。

    支持 RewardLibrary 集成（add_from_lib）和 Transform 后处理流水线。
    """

    def __init__(self):
        self._terms: dict[str, tuple[Callable, float, bool]] = {}  # name -> (func, weight, active)
        self._transforms: list[RewardTransform] = []               # 后处理算子流水线

    # ── 基础操作（不变） ─────────────────────────────────────────────

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

    # ── RewardLibrary 集成 ────────────────────────────────────────

    def add_from_lib(
        self,
        term_name: str,
        weight: float,
        *,
        lib_name: str | None = None,
        robot_name: str = "robot",
        active: bool = True,
        **params,
    ) -> RewardBuilder:
        """从 RewardLibrary 查找 term，验证参数后添加。

        Args:
            term_name:  本 builder 内的注册名（可与 lib_name 不同）
            weight:     奖励权重
            lib_name:   RewardLibrary 中的注册名，None 时用 term_name
            robot_name: 机器人资产名
            active:     是否参与求和
            **params:   传给 Pydantic Params 的超参（自动验证）
        """
        from myrl.core.task.reward_lib import get_reward_library
        func = get_reward_library().build(lib_name or term_name,
                                          robot_name=robot_name, **params)
        return self.add(term_name, func, weight, active)

    # ── Transform 流水线 ──────────────────────────────────────────

    def add_transform(self, transform: RewardTransform) -> RewardBuilder:
        """追加后处理算子（按 add_transform 顺序执行）。"""
        self._transforms.append(transform)
        return self

    def add_transform_from_lib(self, name: str, **params_kwargs) -> RewardBuilder:
        """从 TransformLibrary 实例化算子并追加。"""
        from myrl.core.task.reward_lib import get_transform_library
        transform = get_transform_library().build(name, **params_kwargs)
        return self.add_transform(transform)

    # ── 计算（含 transform 流水线） ───────────────────────────────

    def compute(
        self,
        env,
        step: int = 0,
        return_per_term: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """计算 reward，并顺序应用所有 transform 算子。

        Args:
            env:            环境对象（含 num_envs / device）
            step:           全局训练步数（课程调度算子需要）
            return_per_term: 保留参数（per_term 始终返回）

        Returns:
            total:    (num_envs,) 加权总奖励（transform 后）
            per_term: {name: unweighted_reward} breakdown（transform 后，用于日志）
        """
        # 1. 计算所有 term（含 inactive，transform 可感知全部 term）
        per_term: dict[str, Tensor] = {}
        for name, (func, _, _) in self._terms.items():
            per_term[name] = func(env)

        # 2. 提取 active weights
        weights: dict[str, float] = {
            name: w for name, (_, w, active) in self._terms.items() if active
        }

        # 3. 顺序应用 transform 算子
        for t in self._transforms:
            per_term, weights = t.apply(per_term, weights, step)

        # 4. 加权求和（只含 active term）
        total = None
        for name, w in weights.items():
            if name in per_term:
                weighted = w * per_term[name]
                total = weighted if total is None else total + weighted

        if total is None:
            device = next(iter(per_term.values())).device if per_term else "cpu"
            total = torch.zeros(env.num_envs, device=device)

        return total, per_term

    # ── 元数据 ────────────────────────────────────────────────────

    def list_terms(self) -> dict[str, dict]:
        """列出所有 term 的当前状态（name: {weight, active, has_meta}）。"""
        result = {}
        for name, (func, weight, active) in self._terms.items():
            meta = getattr(func, "__myrl_fn__", None) and getattr(
                getattr(func, "__myrl_fn__", None), "__reward_meta__", None
            )
            # 检查 make_rew 包装的内部函数
            inner = getattr(func, "__myrl_fn__", None)
            has_meta = hasattr(inner, "__reward_meta__") if inner else False
            result[name] = {
                "weight": weight,
                "active": active,
                "has_lib_meta": has_meta,
            }
        return result

    # ── Checkpoint 支持 ───────────────────────────────────────────

    def state_dict(self) -> dict:
        """收集所有有状态算子的 state，用于 checkpoint。"""
        return {str(i): t.state_dict() for i, t in enumerate(self._transforms)}

    def load_state_dict(self, d: dict) -> None:
        for i, t in enumerate(self._transforms):
            if str(i) in d:
                t.load_state_dict(d[str(i)])
