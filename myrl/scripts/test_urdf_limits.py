"""URDF parser joint limits 单元测试。

用法：
    /home/eziothean/myrl_work/.mamba/envs/myrl-train/bin/python3 \
        myrl/scripts/test_urdf_limits.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from myrl.core.robot.urdf_parser import parse_urdf, JointLimits  # noqa: E402


def _write_urdf(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


def test_limits_full_parsed():
    path = _write_urdf("""<?xml version="1.0"?>
<robot name="test">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.5" upper="2.5" velocity="32.0" effort="88.0"/>
  </joint>
</robot>
""")
    model = parse_urdf(path)
    j = model.joints["joint1"]
    assert j.limits is not None, "limits 应被解析"
    assert j.limits.lower == -2.5
    assert j.limits.upper == 2.5
    assert j.limits.velocity == 32.0
    assert j.limits.effort == 88.0
    print("✓ test_limits_full_parsed")


def test_limits_missing_attrs():
    """continuous 关节常省略 lower/upper，只有 velocity/effort。"""
    path = _write_urdf("""<?xml version="1.0"?>
<robot name="test">
  <link name="a"/>
  <link name="b"/>
  <joint name="wheel" type="continuous">
    <parent link="a"/>
    <child link="b"/>
    <limit effort="100" velocity="50"/>
  </joint>
</robot>
""")
    model = parse_urdf(path)
    j = model.joints["wheel"]
    assert j.limits is not None
    assert j.limits.lower is None
    assert j.limits.upper is None
    assert j.limits.effort == 100.0
    assert j.limits.velocity == 50.0
    print("✓ test_limits_missing_attrs")


def test_no_limit_element():
    """fixed joint 通常无 <limit>。"""
    path = _write_urdf("""<?xml version="1.0"?>
<robot name="test">
  <link name="a"/>
  <link name="b"/>
  <joint name="mount" type="fixed">
    <parent link="a"/>
    <child link="b"/>
  </joint>
</robot>
""")
    model = parse_urdf(path)
    assert model.joints["mount"].limits is None
    print("✓ test_no_limit_element")


def test_g1_29dof_torque_distribution():
    """加载真实 G1 URDF，验证 29 个关节 effort 按预期分布。"""
    g1 = _ROOT / "assets" / "robots" / "g1" / "g1_29dof.urdf"
    if not g1.exists():
        print(f"⊘ test_g1_29dof_torque_distribution (skip: {g1} not found)")
        return
    model = parse_urdf(g1)

    # 统计含 limits.effort 的 revolute joint
    efforts = [
        j.limits.effort
        for j in model.joints.values()
        if j.type in ("revolute", "continuous", "prismatic")
        and j.limits is not None
        and j.limits.effort is not None
    ]
    print(f"  G1 29dof: {len(efforts)} joints with effort, "
          f"range [{min(efforts):.1f}, {max(efforts):.1f}] N·m")
    # 按调研报告，G1 29dof 力矩分布应包含 5 / 25 / 50 / 88 / 139 等档
    assert len(efforts) >= 20, f"至少 20 个 joint 应有 effort，got {len(efforts)}"
    assert max(efforts) >= 88.0, f"最大力矩应 >= 88, got {max(efforts)}"

    # 计算分析最大值 Σ(effort²)
    max_torque_sq_sum = sum(e ** 2 for e in efforts)
    print(f"  Σ(effort²) = {max_torque_sq_sum:.1f}")
    assert max_torque_sq_sum > 10000, "G1 torque sum should be significant"
    print("✓ test_g1_29dof_torque_distribution")


def test_to_dict_includes_limits():
    path = _write_urdf("""<?xml version="1.0"?>
<robot name="t">
  <link name="a"/>
  <link name="b"/>
  <joint name="j1" type="revolute">
    <parent link="a"/><child link="b"/>
    <limit lower="-1" upper="1" velocity="10" effort="50"/>
  </joint>
</robot>
""")
    model = parse_urdf(path)
    d = model.to_dict()
    j = d["joints"][0]
    assert j["limits"] is not None
    assert j["limits"]["effort"] == 50.0
    print("✓ test_to_dict_includes_limits")


def main():
    tests = [
        test_limits_full_parsed,
        test_limits_missing_attrs,
        test_no_limit_element,
        test_g1_29dof_torque_distribution,
        test_to_dict_includes_limits,
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
