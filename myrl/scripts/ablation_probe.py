"""消融实验：对比有/无 DataBus 探针下的训练行为。

容器内运行：
    python3 /workspace/myrl/scripts/ablation_probe.py \
        --task Instinct-Locomotion-Flat-G1-v0 \
        --num_envs 64 --steps 200

输出：两轮各 N 步的 episode 长度直方图 + 均值对比。
如果两轮一致 → 探针无影响，重置是任务本身行为。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Instinct-Locomotion-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200, help="每轮步数")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_known_args()[0]
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import instinctlab.tasks  # noqa: F401 — 触发 gym.register
import myrl.tasks          # noqa: F401

from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper as EnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def run_steps(env, n_steps: int, label: str):
    """跑 n_steps 步，统计 episode 长度。"""
    env.reset()
    obs, _ = env.get_observations()

    ep_lengths = []
    cur_len = torch.zeros(env.num_envs, device=env.device)

    for i in range(n_steps):
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
        obs, rew, dones, extras = env.step(actions)
        cur_len += 1
        done_ids = dones.nonzero(as_tuple=True)[0]
        if len(done_ids) > 0:
            ep_lengths.extend(cur_len[done_ids].cpu().tolist())
            cur_len[done_ids] = 0

    if not ep_lengths:
        ep_lengths = cur_len.cpu().tolist()  # 没结束的算进去

    mean_len = sum(ep_lengths) / len(ep_lengths)
    min_len = min(ep_lengths)
    max_len = max(ep_lengths)
    n_resets = len(ep_lengths)

    print(f"\n{'='*50}")
    print(f"  [{label}]  {n_steps} steps, {env.num_envs} envs")
    print(f"  Resets:     {n_resets}")
    print(f"  Ep length:  mean={mean_len:.1f}  min={min_len:.0f}  max={max_len:.0f}")
    print(f"{'='*50}")
    return {"mean": mean_len, "min": min_len, "max": max_len, "resets": n_resets}


def main():
    # 创建环境
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, use_fabric=True)
    env = gym.make(args.task, cfg=env_cfg)
    env = EnvWrapper(env)

    torch.manual_seed(args.seed)

    # ── Round 1: 无探针 ──────────────────────────────────────────
    print("\n>>> Round 1: 无探针（baseline）")
    r1 = run_steps(env, args.steps, "NO PROBE")

    # ── Round 2: 有探针 ──────────────────────────────────────────
    print("\n>>> Round 2: 有探针（enable_databus_on_env）")
    from myrl.core.databus.bus import enable_databus
    from myrl.core.databus.env_wrapper import enable_databus_on_env

    bus = enable_databus()
    enable_databus_on_env(env, bus)

    torch.manual_seed(args.seed)
    r2 = run_steps(env, args.steps, "WITH PROBE")

    # ── 对比 ─────────────────────────────────────────────────────
    diff = abs(r1["mean"] - r2["mean"])
    print(f"\n>>> 对比结果:")
    print(f"  Ep length 均值差: {diff:.2f}")
    print(f"  Reset 次数差:     {abs(r1['resets'] - r2['resets'])}")
    if diff < 1.0 and abs(r1["resets"] - r2["resets"]) < r1["resets"] * 0.1:
        print(f"  结论: ✅ 探针无影响，重置是任务本身行为")
    else:
        print(f"  结论: ⚠️ 有差异，需进一步调查")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
