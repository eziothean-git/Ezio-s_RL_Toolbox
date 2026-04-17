"""Mock SignalServer for Part A 端到端验证。

独立启动一个 SignalServer（默认端口 7002），模拟 3 个伪 reward term 每
100ms publish 一次 magnitude fraction（加上 step + wall_clock 索引）。
用户浏览器打开 editor 就能看到 timeline 滚动。

用法：
    /home/eziothean/myrl_work/.mamba/envs/myrl-train/bin/python3 \
        myrl/scripts/test_timeline_mock_server.py [--port 7002]

Ctrl+C 退出。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from myrl.core.databus.bus import DataBus  # noqa: E402
from myrl.core.databus.signal_server import SignalServer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7002)
    p.add_argument("--hz", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    bus = DataBus()
    srv = SignalServer(bus, port=args.port)
    srv.start()
    print(f"[mock] SignalServer listening on 0.0.0.0:{args.port}")
    print(f"[mock] Publishing 3 fake terms @ {args.hz} Hz")
    print("[mock] Open editor → 展开 Reward Timeline 查看")
    print("[mock] Ctrl+C 退出")

    terms = ["velocity_track", "orientation_penalty", "torque_penalty"]
    interval = 1.0 / args.hz
    t0 = time.time()
    step = 0

    try:
        while True:
            # 模拟 implicit curriculum：前期 orientation 占主导，中期 velocity 上升，
            # 后期 torque penalty 显著
            phase = (time.time() - t0) / 30.0  # 每 30 秒走完一次 phase
            phase = min(phase, 3.0)
            if phase < 1.0:
                # 早期：orientation 主导（0.5 → 0.3）
                raw = [0.2 + 0.3 * phase, 0.6 - 0.3 * phase, 0.2]
            elif phase < 2.0:
                # 中期：velocity 上升
                p = phase - 1.0
                raw = [0.5 + 0.2 * p, 0.3 - 0.1 * p, 0.2 - 0.1 * p + 0.15 * p]
            else:
                # 后期：torque penalty 占比上升
                p = min(phase - 2.0, 1.0)
                raw = [0.7 - 0.15 * p, 0.15 - 0.05 * p, 0.15 + 0.2 * p]

            # 添加小噪声
            import random
            raw = [max(0.01, v + random.uniform(-0.03, 0.03)) for v in raw]
            total = sum(raw)
            fracs = [v / total for v in raw]
            for name, f in zip(terms, fracs):
                bus.publish(f"reward/metrics/mag_frac/{name}",
                            torch.tensor([float(f)]))
            bus.publish("reward/metrics/step", torch.tensor([float(step)]))
            bus.publish("reward/metrics/wall_clock",
                        torch.tensor([time.time() - t0]))

            step += 24  # 模拟 num_steps_per_env=24 的推进
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[mock] Stopped")


if __name__ == "__main__":
    main()
