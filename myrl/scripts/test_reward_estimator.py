"""RewardEstimator 单元测试（含 AST 白名单安全测试）。

用法：
    /home/eziothean/myrl_work/.mamba/envs/myrl-train/bin/python3 \
        myrl/scripts/test_reward_estimator.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# 先加载 rewards 模块（触发 @reward_fn 注册）
for modname, path in [
    ("locomotion_rewards",
     str(_ROOT / "src/myrl/tasks/locomotion/mdp/rewards/locomotion.py")),
    ("regularization_rewards",
     str(_ROOT / "src/myrl/tasks/locomotion/mdp/rewards/regularization.py")),
]:
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)

from myrl.core.task.reward_lib.estimator import (  # noqa: E402
    RewardEstimator, safe_eval,
)

G1_URDF = _ROOT / "assets" / "robots" / "g1" / "g1_29dof.urdf"


# ── 安全：AST 白名单测试 ──────────────────────────────────────────────

class _FakeURDF:
    name = "fake"
    joints = {}


def test_safe_eval_basic_arith():
    assert safe_eval("1 + 2 * 3", _FakeURDF()) == 7
    assert safe_eval("2 ** 10", _FakeURDF()) == 1024
    assert safe_eval("sum([1, 2, 3, 4])", _FakeURDF()) == 10
    assert safe_eval("abs(-5)", _FakeURDF()) == 5
    print("✓ test_safe_eval_basic_arith")


def test_safe_eval_rejects_import():
    """__import__(...).system(...) 必须被拒绝。"""
    try:
        safe_eval("__import__('os').system('echo hacked')", _FakeURDF())
        assert False, "should have raised"
    except ValueError as e:
        # 任何一种拒绝理由都 OK：dunder / Method call / Disallowed name
        msg = str(e).lower()
        assert any(k in msg for k in ("disallowed", "method", "attribute", "dunder")), \
            f"unexpected msg={e}"
    print("✓ test_safe_eval_rejects_import")


def test_safe_eval_rejects_open_file():
    """open('/etc/passwd') 必须被拒绝（open 不在白名单）。"""
    try:
        safe_eval("open('/etc/passwd').read()", _FakeURDF())
        assert False, "should have raised"
    except ValueError as e:
        assert "Disallowed function" in str(e) or "Method" in str(e), f"msg={e}"
    print("✓ test_safe_eval_rejects_open_file")


def test_safe_eval_rejects_exec():
    try:
        safe_eval("exec('print(1)')", _FakeURDF())
        assert False
    except ValueError:
        pass
    print("✓ test_safe_eval_rejects_exec")


def test_safe_eval_rejects_method_call():
    """禁止方法调用（属性链 + ()）。"""
    try:
        safe_eval("''.join(['a', 'b'])", _FakeURDF())
        assert False
    except ValueError as e:
        assert "Method" in str(e) or "attribute" in str(e).lower(), f"msg={e}"
    print("✓ test_safe_eval_rejects_method_call")


def test_safe_eval_rejects_dunder():
    try:
        safe_eval("__builtins__", _FakeURDF())
        assert False
    except ValueError as e:
        assert "Disallowed name" in str(e), f"msg={e}"
    print("✓ test_safe_eval_rejects_dunder")


def test_safe_eval_rejects_class_escape():
    """典型 sandbox 逃逸：通过 .__class__.__bases__[0].__subclasses__()"""
    try:
        safe_eval(
            "(1).__class__.__bases__[0].__subclasses__()",
            _FakeURDF(),
        )
        assert False
    except ValueError as e:
        assert "dunder" in str(e).lower() or "Method" in str(e) or "Disallowed" in str(e)
    print("✓ test_safe_eval_rejects_class_escape")


def test_safe_eval_rejects_lambda():
    try:
        safe_eval("(lambda: 1)()", _FakeURDF())
        assert False
    except ValueError:
        pass
    print("✓ test_safe_eval_rejects_lambda")


# ── URDF 访问测试 ─────────────────────────────────────────────────────

def test_safe_eval_urdf_genexp():
    """典型 max_expr：sum over joints 列表（预暴露）。"""
    est = RewardEstimator(str(G1_URDF))
    expr = (
        "sum((j.limits.effort or 0.0)**2 "
        "for j in joints "
        "if j.limits is not None and j.type in ('revolute', 'continuous', 'prismatic'))"
    )
    val = safe_eval(expr, est.urdf)
    assert val > 100_000, f"G1 Σ(effort²) should exceed 100k, got {val}"
    assert val < 200_000, f"G1 Σ(effort²) should be under 200k, got {val}"
    print(f"✓ test_safe_eval_urdf_genexp (Σ={val:.1f})")


# ── Estimator 核心逻辑 ────────────────────────────────────────────────

def test_estimate_torque_penalty():
    est = RewardEstimator(str(G1_URDF))
    r = est.estimate_term("penalize_joint_torque_l2", params={}, weight=-0.001)
    assert r["status"] == "ok", f"status={r['status']}"
    assert r["shape"] == "l2_sum"
    assert r["analytical_max"] > 100_000
    assert abs(r["weighted_max"] - 0.001 * r["analytical_max"]) < 1e-6
    print(f"✓ test_estimate_torque_penalty (weighted_max={r['weighted_max']:.2f})")


def test_estimate_orientation():
    est = RewardEstimator(str(G1_URDF))
    r = est.estimate_term("penalize_orientation", params={}, weight=-1.0)
    assert r["status"] == "ok"
    assert r["shape"] == "bounded_quad"
    assert r["analytical_max"] == 1.0
    assert r["weighted_max"] == 1.0
    print("✓ test_estimate_orientation")


def test_estimate_track_vel():
    est = RewardEstimator(str(G1_URDF))
    r = est.estimate_term(
        "track_lin_vel_xy_exp", params={"std": 0.25}, weight=1.5,
    )
    assert r["status"] == "ok"
    assert r["shape"] == "exp_kernel"
    assert r["analytical_max"] == 1.0
    assert r["weighted_max"] == 1.5
    assert r["deps"] == ["command"]
    print("✓ test_estimate_track_vel")


def test_estimate_feet_air_time_requires_runtime():
    est = RewardEstimator(str(G1_URDF))
    r = est.estimate_term(
        "feet_air_time_biped",
        params={"threshold": 0.35, "foot_body_ids": [0, 1]},
        weight=0.5,
    )
    assert r["status"] == "requires_runtime", f"got {r['status']}"
    assert "history" in r["deps"]
    print("✓ test_estimate_feet_air_time_requires_runtime")


def test_estimate_pipeline_full_g1():
    """模拟完整 g1_loco_v1 pipeline（含 deferred params）。"""
    pipeline = {
        "name": "g1_loco_v1",
        "terms": [
            {"name": "track_lin_vel_xy_exp", "weight": 1.5, "params": {"std": 0.25}},
            {"name": "track_ang_vel_z_exp", "weight": 0.75, "params": {"std": 0.25}},
            {"name": "feet_air_time_biped", "weight": 0.5,
             "params": {"threshold": 0.35,
                        "foot_body_ids": {"__query_sensor__": "contact_forces",
                                          "__query_pattern__": ".*ankle_roll_link"}}},
            {"name": "penalize_joint_torque_l2", "weight": -0.001, "params": {}},
            {"name": "penalize_orientation", "weight": -1.0, "params": {}},
            {"name": "penalize_lin_accel", "weight": -0.01, "params": {}},
        ],
    }
    est = RewardEstimator(str(G1_URDF))
    out = est.estimate_pipeline(pipeline)

    # 正向最大：track_lin (1.5) + track_ang (0.75) = 2.25（feet_air 不计）
    assert abs(out["overall"]["weighted_positive_max"] - 2.25) < 1e-6, \
        f"pos_max={out['overall']['weighted_positive_max']}"

    # 负向最大：0.001 * ~114200 + 1.0 * 1.0 + 0.01 * 27 ≈ 114 + 1 + 0.27 = ~115.5
    neg = out["overall"]["weighted_negative_max"]
    assert 100 < neg < 150, f"neg_max={neg}"

    # ratio > 10 → severe warning
    warn_types = [w["type"] for w in out["overall"]["warnings"]]
    assert "scale_imbalance" in warn_types, f"warnings={out['overall']['warnings']}"

    # feet_air_time 标记为 requires_runtime
    assert out["terms"]["feet_air_time_biped"]["status"] == "requires_runtime"

    # URDF 元数据
    assert out["urdf_loaded"] is True
    assert out["urdf_name"] is not None

    print(f"✓ test_estimate_pipeline_full_g1 "
          f"(pos={out['overall']['weighted_positive_max']:.2f}, "
          f"neg={out['overall']['weighted_negative_max']:.2f}, "
          f"ratio={out['overall']['ratio_neg_over_pos']:.1f}x)")


def test_estimate_balanced_pipeline_no_warning():
    """量纲平衡的 pipeline → 无 scale_imbalance 警告。"""
    pipeline = {
        "terms": [
            {"name": "track_lin_vel_xy_exp", "weight": 1.0, "params": {"std": 0.25}},
            {"name": "penalize_orientation", "weight": -0.5, "params": {}},
        ],
    }
    est = RewardEstimator(str(G1_URDF))
    out = est.estimate_pipeline(pipeline)
    assert abs(out["overall"]["weighted_positive_max"] - 1.0) < 1e-6
    assert abs(out["overall"]["weighted_negative_max"] - 0.5) < 1e-6
    warns = [w["type"] for w in out["overall"]["warnings"]
             if w["severity"] in ("severe", "warn")]
    assert "scale_imbalance" not in warns, f"unexpected: {out['overall']['warnings']}"
    print("✓ test_estimate_balanced_pipeline_no_warning")


def test_estimate_without_urdf():
    """无 URDF 时：能分析的 term（exp_kernel/bounded_quad）仍估计，l2_sum 标记 urdf_required。"""
    pipeline = {
        "terms": [
            {"name": "track_lin_vel_xy_exp", "weight": 1.0, "params": {"std": 0.25}},
            {"name": "penalize_orientation", "weight": -1.0, "params": {}},
            {"name": "penalize_joint_torque_l2", "weight": -0.001, "params": {}},
        ],
    }
    est = RewardEstimator()   # no URDF
    out = est.estimate_pipeline(pipeline)
    assert out["urdf_loaded"] is False
    assert out["terms"]["track_lin_vel_xy_exp"]["status"] == "ok"
    assert out["terms"]["penalize_orientation"]["status"] == "ok"
    assert out["terms"]["penalize_joint_torque_l2"]["status"] == "urdf_required"
    print("✓ test_estimate_without_urdf")


def test_estimate_unknown_term():
    est = RewardEstimator()
    r = est.estimate_term("no_such_term", params={}, weight=1.0)
    assert r["status"] == "unknown_term"
    print("✓ test_estimate_unknown_term")


def main():
    tests = [
        # AST safety
        test_safe_eval_basic_arith,
        test_safe_eval_rejects_import,
        test_safe_eval_rejects_open_file,
        test_safe_eval_rejects_exec,
        test_safe_eval_rejects_method_call,
        test_safe_eval_rejects_dunder,
        test_safe_eval_rejects_class_escape,
        test_safe_eval_rejects_lambda,
        test_safe_eval_urdf_genexp,
        # Estimator logic
        test_estimate_torque_penalty,
        test_estimate_orientation,
        test_estimate_track_vel,
        test_estimate_feet_air_time_requires_runtime,
        test_estimate_pipeline_full_g1,
        test_estimate_balanced_pipeline_no_warning,
        test_estimate_without_urdf,
        test_estimate_unknown_term,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
