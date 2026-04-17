"""RewardMetricsTransform 单元测试（无 Isaac Sim）。

用法：
    /home/eziothean/myrl_work/.mamba/envs/myrl-train/bin/python3 \
        myrl/scripts/test_reward_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 src 模块可导入（绕过 myrl/tasks 对 isaaclab 的依赖）
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from myrl.core.databus.bus import DataBus  # noqa: E402
import myrl.core.task.reward_lib.transform as tm  # noqa: E402
from myrl.core.task.reward_lib.transform import RewardMetricsTransform  # noqa: E402


class _BusWithTaps:
    """测试 harness：预先 tap 已知 channel 名，publish 后读回最新值。

    Channel 内部不保留 last tensor；必须先 tap 再 publish 才能读到。
    """
    def __init__(self):
        self.bus = DataBus()
        self._taps: dict[str, object] = {}

    def prewire(self, *channels: str) -> None:
        for c in channels:
            self._taps[c] = self.bus.tap(c, buffer_len=4)

    def latest(self, channel: str) -> float:
        assert channel in self._taps, f"prewire 时未加 {channel}"
        buf = self._taps[channel].buffer
        assert buf.numel() > 0, f"channel {channel} 未收到 publish"
        return float(buf[-1].flatten()[0].item())

    def close(self) -> None:
        for t in self._taps.values():
            t.close()


def _reset_bus(*prewire_channels: str) -> _BusWithTaps:
    h = _BusWithTaps()
    tm._bus = h.bus  # 注入 module-level 懒加载槽
    if prewire_channels:
        h.prewire(*prewire_channels)
    return h


def test_magnitude_fraction_sums_to_one():
    h = _reset_bus(
        "reward/metrics/mag_frac/a",
        "reward/metrics/mag_frac/b",
        "reward/metrics/mag_frac/c",
    )
    tr = RewardMetricsTransform()
    per_term = {
        "a": torch.tensor([1.0, 1.0]),
        "b": torch.tensor([2.0, 2.0]),
        "c": torch.tensor([3.0, 3.0]),
    }
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    tr.apply(per_term, weights, step=0)

    fracs = [
        h.latest("reward/metrics/mag_frac/a"),
        h.latest("reward/metrics/mag_frac/b"),
        h.latest("reward/metrics/mag_frac/c"),
    ]
    total = sum(fracs)
    assert abs(total - 1.0) < 1e-4, f"Σ mag_frac = {total}, 应 ≈ 1.0"
    assert abs(fracs[0] - 1 / 6) < 1e-4
    assert abs(fracs[1] - 2 / 6) < 1e-4
    assert abs(fracs[2] - 3 / 6) < 1e-4
    h.close()
    print("✓ test_magnitude_fraction_sums_to_one")


def test_single_term_dominance():
    """单 term 权重极大 → 对应 mag_frac → 1。"""
    h = _reset_bus(
        "reward/metrics/mag_frac/dominant",
        "reward/metrics/mag_frac/noise1",
        "reward/metrics/mag_frac/noise2",
    )
    tr = RewardMetricsTransform()
    per_term = {
        "dominant": torch.tensor([1.0]),
        "noise1": torch.tensor([1.0]),
        "noise2": torch.tensor([1.0]),
    }
    weights = {"dominant": 1000.0, "noise1": 0.001, "noise2": 0.001}
    tr.apply(per_term, weights, step=0)

    dom = h.latest("reward/metrics/mag_frac/dominant")
    assert dom > 0.99, f"dominant mag_frac={dom}, 应 → 1.0"
    h.close()
    print("✓ test_single_term_dominance")


def test_mixed_sign_weights_use_abs():
    """权重负号（penalty） → mag_frac 仍正（用 |w|*|r|）。"""
    h = _reset_bus(
        "reward/metrics/mag_frac/pos",
        "reward/metrics/mag_frac/neg",
    )
    tr = RewardMetricsTransform()
    per_term = {
        "pos": torch.tensor([2.0]),
        "neg": torch.tensor([2.0]),
    }
    weights = {"pos": 1.0, "neg": -1.0}
    tr.apply(per_term, weights, step=0)

    pos_frac = h.latest("reward/metrics/mag_frac/pos")
    neg_frac = h.latest("reward/metrics/mag_frac/neg")
    assert abs(pos_frac - 0.5) < 1e-4
    assert abs(neg_frac - 0.5) < 1e-4
    h.close()
    print("✓ test_mixed_sign_weights_use_abs")


def test_publish_every_downsampling():
    """publish_every=3 → 仅第 3/6/9... 步 publish。"""
    h = _reset_bus("reward/metrics/mag_frac/a")
    tr = RewardMetricsTransform(RewardMetricsTransform.Params(publish_every=3))
    per_term = {"a": torch.tensor([1.0])}
    weights = {"a": 1.0}

    def pc() -> int:
        # ChannelInfo 是快照，每次重新拿
        return h.bus.channel_info("reward/metrics/mag_frac/a").publish_count

    tr.apply(per_term, weights, step=1)
    tr.apply(per_term, weights, step=2)
    assert pc() == 0, f"前 2 步应不发布, got {pc()}"

    tr.apply(per_term, weights, step=3)
    assert pc() == 1, f"第 3 步应发布 1 次, got {pc()}"

    tr.apply(per_term, weights, step=4)
    tr.apply(per_term, weights, step=5)
    assert pc() == 1, f"4/5 步不发布, got {pc()}"

    tr.apply(per_term, weights, step=6)
    assert pc() == 2, f"第 6 步应发布第 2 次, got {pc()}"
    h.close()
    print("✓ test_publish_every_downsampling")


def test_returns_per_term_unchanged():
    """Transform 是纯观察器：per_term 和 weights 不变。"""
    h = _reset_bus()
    tr = RewardMetricsTransform()
    t_a = torch.tensor([1.0, 2.0])
    t_b = torch.tensor([3.0, 4.0])
    per_term = {"a": t_a, "b": t_b}
    weights = {"a": 0.5, "b": -2.0}
    out_pt, out_w = tr.apply(per_term, weights, step=0)

    assert out_pt is per_term, "per_term 应按原引用返回"
    assert out_w is weights, "weights 应按原引用返回"
    assert torch.equal(out_pt["a"], t_a)
    assert torch.equal(out_pt["b"], t_b)
    assert out_w == weights
    h.close()
    print("✓ test_returns_per_term_unchanged")


def test_inactive_term_excluded():
    """不在 weights dict 中的 term（inactive）不发布。"""
    h = _reset_bus()
    tr = RewardMetricsTransform()
    per_term = {
        "active": torch.tensor([1.0]),
        "inactive": torch.tensor([5.0]),
    }
    weights = {"active": 1.0}
    tr.apply(per_term, weights, step=0)

    assert h.bus.channel_info("reward/metrics/mag_frac/active") is not None
    assert h.bus.channel_info("reward/metrics/mag_frac/inactive") is None
    h.close()
    print("✓ test_inactive_term_excluded")


def test_track_terms_filter():
    """track_terms 过滤：只发布白名单 term。"""
    h = _reset_bus()
    tr = RewardMetricsTransform(
        RewardMetricsTransform.Params(track_terms=["watch_me"])
    )
    per_term = {
        "watch_me": torch.tensor([1.0]),
        "ignore_me": torch.tensor([1.0]),
    }
    weights = {"watch_me": 1.0, "ignore_me": 1.0}
    tr.apply(per_term, weights, step=0)

    assert h.bus.channel_info("reward/metrics/mag_frac/watch_me") is not None
    assert h.bus.channel_info("reward/metrics/mag_frac/ignore_me") is None
    h.close()
    print("✓ test_track_terms_filter")


def test_step_and_wall_clock_published():
    h = _reset_bus("reward/metrics/step", "reward/metrics/wall_clock")
    tr = RewardMetricsTransform()
    per_term = {"a": torch.tensor([1.0])}
    weights = {"a": 1.0}
    tr.apply(per_term, weights, step=42)

    step_val = h.latest("reward/metrics/step")
    assert step_val == 42.0
    wc_val = h.latest("reward/metrics/wall_clock")
    assert wc_val >= 0.0
    h.close()
    print("✓ test_step_and_wall_clock_published")


def test_library_registration():
    """验证 @_register_builtin_transforms 已注册 reward_metrics。"""
    from myrl.core.task.reward_lib import get_transform_library
    lib = get_transform_library()
    assert "reward_metrics" in lib.list_names(), \
        f"reward_metrics 未注册，现有 {lib.list_names()}"
    meta = lib.get("reward_metrics")
    assert meta.params is RewardMetricsTransform.Params
    # 通过 library 实例化
    instance = lib.build("reward_metrics", publish_every=2)
    assert isinstance(instance, RewardMetricsTransform)
    assert instance.params.publish_every == 2
    print("✓ test_library_registration")


def test_state_dict_roundtrip():
    tr = RewardMetricsTransform()
    tr._step_local = 123
    tr._t0 = 456.789
    sd = tr.state_dict()
    assert sd == {"step_local": 123, "t0": 456.789}

    tr2 = RewardMetricsTransform()
    tr2.load_state_dict(sd)
    assert tr2._step_local == 123
    assert tr2._t0 == 456.789
    print("✓ test_state_dict_roundtrip")


def main():
    tests = [
        test_magnitude_fraction_sums_to_one,
        test_single_term_dominance,
        test_mixed_sign_weights_use_abs,
        test_publish_every_downsampling,
        test_returns_per_term_unchanged,
        test_inactive_term_excluded,
        test_track_terms_filter,
        test_step_and_wall_clock_published,
        test_library_registration,
        test_state_dict_roundtrip,
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
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
