"""SignalServer buffer 前缀扩展单元测试（无网络）。

用法：
    /home/eziothean/myrl_work/.mamba/envs/myrl-train/bin/python3 \
        myrl/scripts/test_signal_server_buffer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from myrl.core.databus.bus import DataBus  # noqa: E402
from myrl.core.databus.signal_server import SignalServer  # noqa: E402


def test_default_buffer_len():
    bus = DataBus()
    srv = SignalServer(bus, port=0)  # port=0: 不真正 start
    assert srv._buffer_len_for("robot/joints/pos") == 256
    assert srv._buffer_len_for("obs/policy/joint_pos") == 256
    print("✓ test_default_buffer_len")


def test_reward_metrics_buffer_len():
    bus = DataBus()
    srv = SignalServer(bus, port=0)
    assert srv._buffer_len_for("reward/metrics/mag_frac/torque") == 4096
    assert srv._buffer_len_for("reward/metrics/step") == 4096
    assert srv._buffer_len_for("reward/metrics/wall_clock") == 4096
    print("✓ test_reward_metrics_buffer_len")


def test_non_metrics_reward_still_default():
    """reward/total 不是 metrics/，应该 fallback 默认。"""
    bus = DataBus()
    srv = SignalServer(bus, port=0)
    assert srv._buffer_len_for("reward/total") == 256
    assert srv._buffer_len_for("reward/track_lin_vel_xy_exp") == 256
    print("✓ test_non_metrics_reward_still_default")


def test_persistent_tap_uses_prefix_buffer():
    """persistent tap 实际创建时使用前缀决定的容量。"""
    bus = DataBus()
    srv = SignalServer(bus, port=0)

    # 先 publish 一些帧，之后 tap 读取
    for i in range(300):
        bus.publish("reward/metrics/mag_frac/foo", torch.tensor(float(i)))

    tap = srv._get_or_create_tap("reward/metrics/mag_frac/foo")
    # buffer 开辟的容量 = 4096，但只收到 tap 创建之后的 publish
    # （Channel 不保留 tap 创建前的历史）。这里验证的是 buffer 属性的容量。
    # Tap.buffer_len 属性应为 4096（如果 Tap 暴露）。若不暴露，直接查内部属性。
    assert tap._buffer_len == 4096, f"期望 4096, got {tap._buffer_len}"

    tap2 = srv._get_or_create_tap("robot/joints/pos")
    assert tap2._buffer_len == 256
    print("✓ test_persistent_tap_uses_prefix_buffer")


def main():
    tests = [
        test_default_buffer_len,
        test_reward_metrics_buffer_len,
        test_non_metrics_reward_still_default,
        test_persistent_tap_uses_prefix_buffer,
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
