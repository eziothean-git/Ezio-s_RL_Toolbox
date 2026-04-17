"""RewardEstimator — 预训练 reward 分布分析估计。

给定 reward pipeline（YAML dict） + URDF 路径，估计每个 term 的：
  - analytical_max     : 基于形状公式的理论上界（不考虑权重）
  - weighted_max       : abs(weight) * analytical_max
  - shape              : shape_hint
  - deps               : 依赖类别（command/history/sensor 等）
  - status             : "ok" / "unannotated" / "requires_runtime"

以及整条 pipeline 的：
  - weighted_positive_max / weighted_negative_max
  - warnings         : 自动检测的设计陷阱（如 penalty/reward 量纲失衡）

**安全性**：`max_expr` 通过 AST 白名单 eval，禁止 `__import__` / `exec` / 任意属性
访问 / 函数调用（仅允许 sum/max/min/abs/len）。任何越界 AST 节点立刻 raise。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from myrl.core.robot.urdf_parser import parse_urdf


# AST 白名单：允许的节点类型
_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Constant,
    ast.Name, ast.Attribute, ast.Subscript, ast.Load, ast.Store,
    ast.Call,
    ast.GeneratorExp, ast.comprehension,
    ast.ListComp, ast.SetComp, ast.DictComp,
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
    ast.IfExp,
    ast.BoolOp, ast.And, ast.Or,
    ast.Tuple, ast.List,
    # Python 3.9+：Index 节点已被 Constant 等替代；为兼容旧版仍收录
    getattr(ast, "Index", type(None)),
)

# 白名单函数（按 id 匹配）
_ALLOWED_CALLS: frozenset[str] = frozenset({"sum", "max", "min", "abs", "len"})


def _assert_safe(tree: ast.AST) -> None:
    """遍历 AST，遇到白名单外的节点/调用立即 raise ValueError。"""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            # 只允许 Name 形式的调用（不允许 a.b() 等属性方法调用）
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"Method / attribute call not allowed: {ast.dump(node.func)}"
                )
            if node.func.id not in _ALLOWED_CALLS:
                raise ValueError(
                    f"Disallowed function: {node.func.id} "
                    f"(allowed: {sorted(_ALLOWED_CALLS)})"
                )
        if isinstance(node, ast.Attribute):
            # 禁止访问 dunder 属性（__class__ / __globals__ 等）
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValueError(f"Disallowed dunder attribute access: {node.attr}")
        if isinstance(node, ast.Name):
            # 禁止引用 __builtins__ / __import__ 等
            if node.id.startswith("__"):
                raise ValueError(f"Disallowed name: {node.id}")


def safe_eval(expr: str, urdf: Any) -> Any:
    """AST 白名单 eval。

    暴露的符号：
      joints   : list[URDFJoint]（等价于 list(urdf.joints.values())）
      links    : list[URDFLink]
      root     : str | None (根 link 名)
      sum/max/min/abs/len : 白名单数学函数

    禁止：任何方法调用（Attribute→Call）、dunder 访问、lambda、import、
    隐含全局（__builtins__ 被清空）。
    """
    tree = ast.parse(expr, mode="eval")
    _assert_safe(tree)
    globals_ = {"__builtins__": {}}
    joints_list = []
    links_list = []
    root = None
    if urdf is not None:
        if hasattr(urdf, "joints") and isinstance(urdf.joints, dict):
            joints_list = list(urdf.joints.values())
        elif hasattr(urdf, "joints"):
            joints_list = list(urdf.joints)
        if hasattr(urdf, "links") and isinstance(urdf.links, dict):
            links_list = list(urdf.links.values())
        if hasattr(urdf, "root_link"):
            try:
                root = urdf.root_link()
            except TypeError:
                root = None
    locals_ = {
        "joints": joints_list,
        "links": links_list,
        "root": root,
        "sum": sum, "max": max, "min": min, "abs": abs, "len": len,
    }
    return eval(compile(tree, "<max_expr>", "eval"), globals_, locals_)


class RewardEstimator:
    """给定 reward pipeline + URDF，估计各 term 的分析上下界及整体量纲比。"""

    def __init__(self, urdf_path: str | Path | None = None):
        self.urdf = None
        if urdf_path is not None:
            self.urdf = parse_urdf(urdf_path)

    def estimate_term(
        self,
        term_name: str,
        params: dict | None,
        weight: float,
    ) -> dict:
        """返回单 term 的估计结果。"""
        from myrl.core.task.reward_lib import get_reward_library
        lib = get_reward_library()
        try:
            meta = lib.get(term_name)
        except KeyError:
            return {
                "status": "unknown_term",
                "weight": weight,
                "error": f"term '{term_name}' not in RewardLibrary",
            }

        sig = meta.signature
        if sig is None:
            return {
                "status": "unannotated",
                "weight": weight,
                "shape": None,
                "deps": [],
                "notes": "no RewardSignature annotated",
            }

        result: dict = {
            "status": "ok",
            "weight": weight,
            "shape": sig.shape_hint,
            "deps": list(sig.deps),
            "input_views": list(sig.input_views),
            "notes": sig.notes,
        }

        # 运行时依赖（history / sensor / reference）→ 分析不可估计
        runtime_deps = {"history", "reference"}
        has_sensor_dep = any(d.startswith("sensor:") for d in sig.deps)
        if any(d in runtime_deps for d in sig.deps) or has_sensor_dep:
            result["status"] = "requires_runtime"

        # 形状依赖的分析范围
        if sig.shape_hint == "exp_kernel":
            # exp(-err²/σ²) ∈ (0, 1]
            result["analytical_max"] = 1.0
            result["analytical_min"] = 0.0
            result["weighted_max"] = abs(weight) * 1.0
            if params and "std" in params:
                result["sigma"] = params["std"]
        elif sig.shape_hint == "bounded_quad":
            result["analytical_max"] = 1.0
            result["analytical_min"] = 0.0
            result["weighted_max"] = abs(weight) * 1.0
        elif sig.shape_hint == "l2_sum":
            if sig.max_expr and self.urdf is not None:
                try:
                    max_val = float(safe_eval(sig.max_expr, self.urdf))
                    result["analytical_max"] = max_val
                    result["weighted_max"] = abs(weight) * max_val
                except Exception as e:
                    result["status"] = "eval_error"
                    result["error"] = str(e)
            elif sig.max_expr and self.urdf is None:
                result["status"] = "urdf_required"
            else:
                # l2_sum 无 max_expr → 无法估计
                result["status"] = "requires_runtime"
        elif sig.shape_hint == "clamped_sum":
            # 依赖累积时间或 policy 行为，无分析解
            if sig.max_expr and self.urdf is not None:
                try:
                    max_val = float(safe_eval(sig.max_expr, self.urdf))
                    result["analytical_max"] = max_val
                    result["weighted_max"] = abs(weight) * max_val
                except Exception as e:
                    result["status"] = "eval_error"
                    result["error"] = str(e)
            else:
                result["status"] = "requires_runtime"
        else:
            result["notes"] = (result.get("notes") or "") + \
                f" (unsupported shape_hint={sig.shape_hint})"

        return result

    def estimate_pipeline(self, pipeline: dict) -> dict:
        """估计整条 pipeline；pipeline 格式见 reward_pipeline YAML。"""
        term_estimates: dict[str, dict] = {}
        pos_sum = 0.0
        neg_sum = 0.0
        num_requires_runtime = 0
        num_unannotated = 0

        for t in pipeline.get("terms", []):
            name = t["name"]
            w = float(t.get("weight", 1.0))
            # 过滤掉 deferred param（含 __query_sensor__）—估计端不解析
            p = t.get("params") or {}
            p_clean = {
                k: v for k, v in p.items()
                if not (isinstance(v, dict)
                        and ("__query_sensor__" in v or "__query_pattern__" in v))
            }
            est = self.estimate_term(name, p_clean, w)
            term_estimates[name] = est

            if est.get("status") == "ok" and "weighted_max" in est:
                if w >= 0:
                    pos_sum += est["weighted_max"]
                else:
                    neg_sum += est["weighted_max"]
            elif est.get("status") == "requires_runtime":
                num_requires_runtime += 1
            elif est.get("status") == "unannotated":
                num_unannotated += 1

        warnings: list[dict] = []
        if pos_sum > 1e-9 and (neg_sum / pos_sum) > 10.0:
            warnings.append({
                "type": "scale_imbalance",
                "severity": "severe",
                "message": (
                    f"Penalty/reward weighted-max ratio = {neg_sum / pos_sum:.1f}× > 10× "
                    f"(|neg|={neg_sum:.2f}, pos={pos_sum:.2f})。"
                    "Training 可能被 penalty 主导，policy 仅学会躲避惩罚而非追踪指令。"
                ),
            })
        elif pos_sum > 1e-9 and (neg_sum / pos_sum) > 3.0:
            warnings.append({
                "type": "scale_imbalance",
                "severity": "warn",
                "message": (
                    f"Penalty/reward weighted-max ratio = {neg_sum / pos_sum:.1f}× > 3×；"
                    "建议检查 penalty 权重是否过大。"
                ),
            })
        if num_unannotated > 0:
            warnings.append({
                "type": "missing_signature",
                "severity": "info",
                "message": f"{num_unannotated} term(s) 未注解 RewardSignature，"
                           "无法静态估计。",
            })

        return {
            "terms": term_estimates,
            "overall": {
                "weighted_positive_max": pos_sum,
                "weighted_negative_max": neg_sum,
                "ratio_neg_over_pos": (neg_sum / pos_sum) if pos_sum > 1e-9 else None,
                "num_requires_runtime": num_requires_runtime,
                "num_unannotated": num_unannotated,
                "warnings": warnings,
            },
            "urdf_loaded": self.urdf is not None,
            "urdf_name": self.urdf.name if self.urdf else None,
        }
