"""RewardTransform ABC + 5 个内置后处理算子。

流水线顺序建议：RunningNormalize → RelativeRebalance → ClipReward → WeightSchedule → RewardMetrics
（前 4 个算子无顺序依赖；RewardMetrics 为纯观察器，必须放在最后以观测最终加权贡献）
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from myrl.core.databus.bus import get_databus as _get_databus

# 模块级 bus 懒加载（与 reward_builder.py 同模式）
_bus = None


# ── 基类 ─────────────────────────────────────────────────────────────

class RewardTransform(ABC):
    """有状态的奖励后处理算子。RewardBuilder._transforms 中顺序执行。"""

    @abstractmethod
    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        """
        Args:
            per_term: {name: unweighted Tensor[num_envs]}
            weights:  当前激活 term 的权重字典（仅含 active=True 的项）
            step:     全局训练步数（课程调度需要）
        Returns:
            (变换后 per_term, 更新后 weights)
        """
        ...

    def state_dict(self) -> dict:
        """用于 checkpoint 持久化（默认空）。"""
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


# ── 内置算子 1：RunningNormalize ─────────────────────────────────────

class _WelfordAccumulator:
    """在线 Welford 均值 / 方差累积器（标量版）。"""

    def __init__(self, window: int):
        self.window = window
        self.count = 0
        self._mean = 0.0
        self._M2 = 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return math.sqrt(max(self._M2 / (self.count - 1), 0.0))

    def update(self, value: float) -> None:
        self.count = min(self.count + 1, self.window)
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        # 指数移动平均近似：window 满后衰减旧样本
        decay = 1.0 / self.window
        self._M2 = self._M2 * (1 - decay) + delta * delta2 * decay * self.window

    def to_dict(self) -> dict:
        return {"count": self.count, "mean": self._mean, "M2": self._M2}

    @classmethod
    def from_dict(cls, d: dict, window: int) -> _WelfordAccumulator:
        acc = cls(window)
        acc.count = d["count"]
        acc._mean = d["mean"]
        acc._M2 = d["M2"]
        return acc


class RunningNormalize(RewardTransform):
    """按 term 的运行期标准差归一化，消除量纲差异，稳定 PPO 训练。

    只改变 per_term tensor，不修改 weights。
    """

    class Params(BaseModel):
        window: int = Field(1000, ge=10, description="Welford 窗口大小（样本数）")
        min_std: float = Field(1e-3, ge=1e-8, description="数值稳定最小标准差")
        terms: list[str] | None = Field(None, description="限定归一化的 term 名，None=全部")

    def __init__(self, params: Params | None = None):
        self.params = params or self.Params()
        self._stats: dict[str, _WelfordAccumulator] = {}

    def _get_acc(self, name: str) -> _WelfordAccumulator:
        if name not in self._stats:
            self._stats[name] = _WelfordAccumulator(self.params.window)
        return self._stats[name]

    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        new_per_term: dict[str, Tensor] = {}
        for name, r in per_term.items():
            target = (self.params.terms is None) or (name in self.params.terms)
            if target:
                acc = self._get_acc(name)
                acc.update(r.mean().item())
                std = max(acc.std, self.params.min_std)
                new_per_term[name] = r / std
            else:
                new_per_term[name] = r
        return new_per_term, weights

    def state_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self._stats.items()}

    def load_state_dict(self, d: dict) -> None:
        self._stats = {
            k: _WelfordAccumulator.from_dict(v, self.params.window)
            for k, v in d.items()
        }


# ── 内置算子 2：RelativeRebalance ────────────────────────────────────

class RelativeRebalance(RewardTransform):
    """调整各 term 权重，使实际贡献占比趋近目标，同时保持总奖励幅度恒定。

    只改变 weights，不修改 per_term tensor。

    原理：EMA 追踪 |w_i * r_i| / sum(|w_j * r_j|)，按比例误差微调权重。
    """

    class Params(BaseModel):
        target_ratios: dict[str, float] = Field(
            description="term名→目标占比（内部归一化，无需和=1）"
        )
        window: int = Field(500, ge=10, description="EMA 窗口大小")
        lr: float = Field(0.01, ge=1e-4, le=1.0, description="权重更新步长")
        total_scale: float = Field(1.0, description="总奖励目标幅度")

    def __init__(self, params: Params):
        self.params = params
        # EMA 估计的当前贡献（E[|w_i * r_i|]）
        self._ema_contrib: dict[str, float] = {}
        self._decay = None  # 在 apply 首次调用时初始化

    def _ema(self, name: str, val: float) -> float:
        if self._decay is None:
            self._decay = 1.0 / self.params.window
        if name not in self._ema_contrib:
            self._ema_contrib[name] = val
        else:
            self._ema_contrib[name] = (1 - self._decay) * self._ema_contrib[name] + self._decay * val
        return self._ema_contrib[name]

    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        if not weights:
            return per_term, weights

        # 归一化目标占比
        target_keys = [k for k in self.params.target_ratios if k in weights]
        if not target_keys:
            return per_term, weights

        target_sum = sum(self.params.target_ratios[k] for k in target_keys)
        if target_sum <= 0:
            return per_term, weights
        target = {k: self.params.target_ratios[k] / target_sum for k in target_keys}

        # 更新 EMA 贡献估计
        for name in target_keys:
            r = per_term.get(name)
            if r is None:
                continue
            contrib = abs(weights[name]) * r.abs().mean().item()
            self._ema(name, contrib)

        ema_sum = sum(self._ema_contrib.get(k, 1e-8) for k in target_keys)
        if ema_sum <= 0:
            return per_term, weights

        # 按比例误差微调权重
        new_weights = dict(weights)
        for k in target_keys:
            actual_ratio = self._ema_contrib.get(k, 1e-8) / ema_sum
            desired_ratio = target[k]
            error = desired_ratio - actual_ratio
            new_weights[k] = max(1e-8, weights[k] * (1 + self.params.lr * error))

        # 重新归一化使总幅度 ≈ total_scale
        # 用 EMA 贡献估计归一化
        new_total = sum(
            new_weights.get(k, 0.0) * (self._ema_contrib.get(k, 0.0) / max(abs(weights.get(k, 1.0)), 1e-8))
            for k in target_keys
        )
        if new_total > 0:
            scale_factor = self.params.total_scale / new_total
            for k in target_keys:
                new_weights[k] *= scale_factor

        return per_term, new_weights

    def state_dict(self) -> dict:
        return {"ema_contrib": dict(self._ema_contrib)}

    def load_state_dict(self, d: dict) -> None:
        self._ema_contrib = d.get("ema_contrib", {})


# ── 内置算子 3：ClipReward ────────────────────────────────────────────

class ClipReward(RewardTransform):
    """裁剪奖励范围 + 可选 EMA 平滑。"""

    class Params(BaseModel):
        min_val: float | None = Field(None, description="下界，None=不限")
        max_val: float | None = Field(None, description="上界，None=不限")
        per_term: bool = Field(False, description="True=按 term 裁剪，False=总奖励裁剪")
        ema_alpha: float | None = Field(
            None, ge=0.0, le=1.0, description="EMA 平滑系数，None=不平滑"
        )

    def __init__(self, params: Params | None = None):
        self.params = params or self.Params()
        self._ema_state: dict[str, Tensor] = {}

    def _clip(self, t: Tensor) -> Tensor:
        if self.params.min_val is not None:
            t = t.clamp(min=self.params.min_val)
        if self.params.max_val is not None:
            t = t.clamp(max=self.params.max_val)
        return t

    def _smooth(self, name: str, t: Tensor) -> Tensor:
        alpha = self.params.ema_alpha
        if alpha is None:
            return t
        if name not in self._ema_state:
            self._ema_state[name] = t.clone()
        else:
            self._ema_state[name] = alpha * self._ema_state[name] + (1 - alpha) * t
        return self._ema_state[name]

    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        if self.params.per_term:
            new_per_term = {
                name: self._smooth(name, self._clip(r))
                for name, r in per_term.items()
            }
            return new_per_term, weights
        else:
            # 对加权总量裁剪：scale 各 term
            if not weights:
                return per_term, weights
            total = sum(
                weights.get(n, 0.0) * r for n, r in per_term.items() if n in weights
            )
            clipped = self._smooth("_total", self._clip(total))
            # 按比例缩放各 term
            ratio = clipped / (total.abs() + 1e-8) * total.sign()
            # 只缩放 active term
            new_per_term = {
                name: r * ratio if name in weights else r
                for name, r in per_term.items()
            }
            return new_per_term, weights


# ── 内置算子 4：WeightSchedule ───────────────────────────────────────

class WeightSchedule(RewardTransform):
    """根据训练步数线性/余弦插值调整 term 权重，实现课程学习。

    只改变 weights，不修改 per_term tensor。
    """

    class Params(BaseModel):
        schedules: dict[str, list[tuple[int, float]]] = Field(
            description=(
                "{term名: [(step, weight), ...]}"
                "例: {'forward': [(0, 0.0), (5000, 1.0)]}"
            )
        )
        interpolation: Literal["linear", "cosine"] = Field(
            "linear", description="插值方式"
        )

    def __init__(self, params: Params):
        self.params = params

    def _interpolate(self, keyframes: list[tuple[int, float]], step: int) -> float:
        if not keyframes:
            return 0.0
        if step <= keyframes[0][0]:
            return keyframes[0][1]
        if step >= keyframes[-1][0]:
            return keyframes[-1][1]
        for i in range(len(keyframes) - 1):
            s0, w0 = keyframes[i]
            s1, w1 = keyframes[i + 1]
            if s0 <= step <= s1:
                t = (step - s0) / max(s1 - s0, 1)
                if self.params.interpolation == "cosine":
                    t = 0.5 * (1 - math.cos(math.pi * t))
                return w0 + (w1 - w0) * t
        return keyframes[-1][1]

    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        new_weights = dict(weights)
        for name, keyframes in self.params.schedules.items():
            if name in new_weights:
                new_weights[name] = self._interpolate(keyframes, step)
        return per_term, new_weights


# ── 内置算子 5：RewardMetricsTransform（纯观察器） ────────────────────

class RewardMetricsTransform(RewardTransform):
    """纯观察器：沿时间轴发布各 term 的 magnitude fraction 到 DataBus。

    不修改 per_term 和 weights。推荐放在 transform 流水线末端，观察的是真正
    进入 PPO 优化器的加权贡献。

    发布到 DataBus 的 channel：
      - reward/metrics/mag_frac/{term}  : 标量 |w*r| / Σ|w*r|
      - reward/metrics/step             : 标量 float(step)
      - reward/metrics/wall_clock       : 标量 time.time() - t0 (秒)

    Variance / Herfindahl / pos-neg 分解的扩展点已预留（self._welford 字段）。
    """

    class Params(BaseModel):
        publish_every: int = Field(
            1, ge=1, description="step 降采样：1=每步发布"
        )
        track_terms: list[str] | None = Field(
            None, description="None=发布全部 active term，否则只发布指定 term"
        )

    def __init__(self, params: Params | None = None):
        self.params = params or self.Params()
        self._step_local = 0       # 内部计数，用于 publish_every 降采样
        self._t0: float | None = None   # wall clock baseline

    def apply(
        self,
        per_term: dict[str, Tensor],
        weights: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        if self._t0 is None:
            self._t0 = time.time()
        self._step_local += 1
        if self._step_local % self.params.publish_every != 0:
            return per_term, weights

        global _bus
        if _bus is None:
            _bus = _get_databus()
        if _bus is None:
            return per_term, weights

        # magnitude fraction: |w_i * r_i| / Σ_j |w_j * r_j|
        mags: dict[str, float] = {}
        for name, r in per_term.items():
            if name not in weights:
                continue
            mags[name] = abs(weights[name]) * r.abs().mean().item()
        total = sum(mags.values()) + 1e-12

        # 发布 shape=[1] 张量（非 0-d），与 SignalServer Tap env_id=0 兼容
        for name, m in mags.items():
            if self.params.track_terms is None or name in self.params.track_terms:
                _bus.publish(
                    f"reward/metrics/mag_frac/{name}",
                    torch.tensor([m / total]),
                )

        # X 轴索引：step（env 传入）+ wall_clock（iteration 由客户端 step/N_steps 推导）
        _bus.publish("reward/metrics/step", torch.tensor([float(step)]))
        _bus.publish("reward/metrics/wall_clock",
                     torch.tensor([time.time() - self._t0]))

        return per_term, weights  # 关键：原值不变

    def state_dict(self) -> dict:
        return {"step_local": self._step_local, "t0": self._t0}

    def load_state_dict(self, d: dict) -> None:
        self._step_local = d.get("step_local", 0)
        self._t0 = d.get("t0", None)
