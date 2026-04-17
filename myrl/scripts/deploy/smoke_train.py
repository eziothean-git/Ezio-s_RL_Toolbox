#!/usr/bin/env python3
"""无 Isaac Lab 的冒烟训练脚本 — 合成 VecEnv + instinct_rl PPO。

用于验证远程服务器上的完整训练管控链路：
  instinct_rl OnPolicyRunner → PPO → GPU 训练 → checkpoint 保存
  + DataBus SignalServer (Oscilloscope)
  + SSE Log Server

无需 Isaac Sim / Isaac Lab / instinctlab，只需 torch + instinct_rl。

用法:
    python3 smoke_train.py [--num_envs 256] [--max_iterations 100] \
        [--signal_server_port 7002] [--log_server_port 7000]
"""
import argparse
import os
import signal
import sys
import time
from collections import OrderedDict

import torch
from torch import Tensor

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Smoke training without Isaac Lab")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_actions", type=int, default=29)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--logdir", type=str, default=None)
parser.add_argument("--signal_server_port", type=int, default=None)
parser.add_argument("--log_server_port", type=int, default=None)
parser.add_argument("--log_server_host", type=str, default="0.0.0.0")
# train_manager 会传这些参数，接受但忽略
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--headless", action="store_true", default=False)
# Random policy probe（B6）：不训练，仅按 pipeline signature 发布伪造 mag_frac
# 到 DataBus，供 editor reward-timeline 作"预训练粗估"观察。
parser.add_argument("--random_policy", action="store_true", default=False,
                    help="跳过 PPO，仅按 reward pipeline 做随机策略 probe")
parser.add_argument("--pipeline", type=str, default="g1_flat_walk_reward",
                    help="random policy 模式下使用的 reward pipeline 名")
parser.add_argument("--robot", type=str, default="g1_29dof",
                    help="random policy 模式下用于估计 URDF limits 的机器人名")
parser.add_argument("--preview_output", type=str, default=None,
                    help="random policy 结束后导出 mag_frac 历史 JSON")
args, _ = parser.parse_known_args()


# ── 合成 VecEnv（模拟 G1 行走） ──────────────────────────────────────
class SyntheticWalkEnv:
    """模拟 29-DOF 人形机器人行走的合成 VecEnv。

    实现 instinct_rl VecEnv 接口：
      num_envs, num_actions, num_rewards, max_episode_length, device
      get_obs_format(), get_observations(), step(actions), reset()
    """

    OBS_FORMAT = OrderedDict({
        "policy": OrderedDict({
            "base_ang_vel": (3,),
            "projected_gravity": (3,),
            "commands": (3,),
            "joint_pos": (29,),
            "joint_vel": (29,),
            "last_actions": (29,),
        }),
    })

    def __init__(self, num_envs: int, num_actions: int, device: str):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.num_rewards = 1
        self.max_episode_length = 500
        self.device = torch.device(device)

        self._obs_size = sum(d[0] for d in self.OBS_FORMAT["policy"].values())
        self._step_count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._joint_pos = torch.zeros(num_envs, 29, device=self.device)
        self._last_actions = torch.zeros(num_envs, num_actions, device=self.device)

    def get_obs_format(self):
        return self.OBS_FORMAT

    def _make_obs(self) -> tuple[Tensor, dict]:
        obs = torch.randn(self.num_envs, self._obs_size, device=self.device) * 0.1
        # 填入有意义的信号
        obs[:, 6:9] = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # commands
        obs[:, 9:38] = self._joint_pos
        obs[:, 38:67] = torch.randn(self.num_envs, 29, device=self.device) * 0.5  # vel
        obs[:, 67:96] = self._last_actions
        extras = {"observations": {"policy": obs}}
        return obs, extras

    def get_observations(self):
        return self._make_obs()

    def reset(self):
        self._step_count.zero_()
        self._joint_pos.zero_()
        self._last_actions.zero_()
        return self._make_obs()

    def step(self, actions: Tensor):
        self._last_actions = actions.clone()
        self._step_count += 1

        # 简单动力学：关节位置 += 动作 * dt
        self._joint_pos += actions[:, :29] * 0.02
        self._joint_pos.clamp_(-3.14, 3.14)

        obs, extras = self._make_obs()

        # 奖励：鼓励前进命令跟踪 + 惩罚大动作
        reward = (
            1.0
            - 0.01 * actions.pow(2).sum(dim=1)
            + 0.1 * torch.randn(self.num_envs, device=self.device)
        )

        # episode 终止
        dones = (self._step_count >= self.max_episode_length).float()
        # 10% 随机早停（模拟摔倒）
        dones = torch.max(dones, (torch.rand(self.num_envs, device=self.device) < 0.002).float())
        time_outs = (self._step_count >= self.max_episode_length).float()

        # 重置 done 的 env
        reset_ids = dones.nonzero(as_tuple=True)[0]
        if len(reset_ids) > 0:
            self._step_count[reset_ids] = 0
            self._joint_pos[reset_ids] = 0
            self._last_actions[reset_ids] = 0

        extras["time_outs"] = time_outs
        extras["log"] = {
            "reward_mean": reward.mean().item(),
            "ep_len_mean": self._step_count.float().mean().item(),
        }

        return obs, reward.unsqueeze(1), dones, extras


# ── Random policy probe（B6） ──────────────────────────────────────
def run_random_policy_probe():
    """不做 PPO 训练，仅按 reward pipeline 估计值 + 噪声发布 mag_frac 到 DataBus。

    用于 editor 在正式训练前快速观察期望分布（"预训练粗估 Sim smoke"）。
    输出 JSON cache 文件，editor 可离线 overlay。
    """
    # 3 层上：myrl/scripts/deploy/smoke_train.py → repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "myrl", "src"))
    sys.path.insert(0, os.path.join(repo_root, "myrl", "scripts"))
    from myrl.core.databus.bus import enable_databus
    from myrl.core.databus.signal_server import SignalServer

    # 加载 pipeline + estimator
    pipeline_path = os.path.join(
        repo_root, "myrl", "assets", "reward_pipelines", f"{args.pipeline}.yaml"
    )
    if not os.path.exists(pipeline_path):
        print(f"[smoke] pipeline YAML not found: {pipeline_path}")
        sys.exit(1)
    import yaml as _yaml
    with open(pipeline_path) as f:
        pipeline = _yaml.safe_load(f) or {}

    urdf_path = os.path.join(
        repo_root, "myrl", "assets", "robots", "g1", f"{args.robot}.urdf"
    )
    if not os.path.exists(urdf_path):
        urdf_path = None

    # 触发 reward_lib 注册
    import importlib.util
    for rel in [
        "myrl/src/myrl/tasks/locomotion/mdp/rewards/locomotion.py",
        "myrl/src/myrl/tasks/locomotion/mdp/rewards/regularization.py",
    ]:
        p = os.path.join(repo_root, rel)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(
                "_smoke_reward_" + os.path.basename(p), p
            )
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"[smoke] reward module import warning: {e}")

    from myrl.core.task.reward_lib.estimator import RewardEstimator
    est = RewardEstimator(urdf_path)
    analysis = est.estimate_pipeline(pipeline)
    term_est = analysis["terms"]

    # 解析每个 term 的 "analytical magnitude"，构造 probe 时用的期望值
    term_mags: dict[str, float] = {}
    for name, e in term_est.items():
        if e.get("status") == "ok" and isinstance(e.get("weighted_max"), (int, float)):
            term_mags[name] = float(e["weighted_max"])
        elif e.get("status") == "requires_runtime":
            # 典型值估计：feet_air_time ~ 0.05 * weight, etc.
            w = e.get("weight", 0.0)
            term_mags[name] = abs(w) * 0.05
        else:
            term_mags[name] = 0.0

    terms_sorted = sorted(term_mags.keys())
    total_mag = sum(term_mags.values()) + 1e-12

    # DataBus + SignalServer
    bus = enable_databus()
    port = args.signal_server_port or 7002
    SignalServer(bus, port=port).start()
    print(f"[smoke] random-policy probe: SignalServer on :{port}")
    print(f"[smoke] pipeline={args.pipeline} robot={args.robot} terms={terms_sorted}")

    # 加载 RewardMetricsTransform 用于正规发布（保证 channel name 完全一致）
    from myrl.core.task.reward_lib.transform import RewardMetricsTransform
    tr = RewardMetricsTransform()

    import time as _time
    import random

    total_steps = args.num_steps_per_env
    history_for_dump: dict[str, list[float]] = {n: [] for n in terms_sorted}
    step_history: list[int] = []

    for step in range(total_steps):
        # 合成 per_term 张量（num_envs=4，每 env scalar → 注入 transform）
        per_term = {}
        weights = {}
        for name in terms_sorted:
            base = term_mags[name]
            # 正值基础 + 噪声；RewardMetricsTransform 只用 |w|·|r|, 方向无关
            noise = 1.0 + 0.15 * (random.random() - 0.5)
            per_term[name] = torch.full((args.num_envs,), base * noise / max(abs(_weight_of(pipeline, name)), 1e-8))
            weights[name] = _weight_of(pipeline, name)
        # 调用 transform → 自动发布 reward/metrics/*
        tr.apply(per_term, weights, step=step * 24)
        # 同步记录到 cache history
        step_history.append(step * 24)
        cur_total = sum(
            abs(weights[n]) * per_term[n].abs().mean().item() for n in terms_sorted
        ) + 1e-12
        for name in terms_sorted:
            frac = abs(weights[name]) * per_term[name].abs().mean().item() / cur_total
            history_for_dump[name].append(frac)
        _time.sleep(0.02)    # 50 Hz

    # 导出 cache
    if args.preview_output:
        import json
        cache = {
            "pipeline": args.pipeline,
            "robot": args.robot,
            "urdf_path": urdf_path,
            "terms": terms_sorted,
            "step_history": step_history,
            "mag_frac_history": history_for_dump,
            "analytical": analysis,
            "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_steps": total_steps,
        }
        os.makedirs(os.path.dirname(args.preview_output) or ".", exist_ok=True)
        with open(args.preview_output, "w") as f:
            json.dump(cache, f, indent=2, default=str)
        print(f"[smoke] preview cache → {args.preview_output}")

    print("[smoke] ✓ Random policy probe complete!")


def _weight_of(pipeline: dict, name: str) -> float:
    for t in pipeline.get("terms", []):
        if t.get("name") == name:
            return float(t.get("weight", 1.0))
    return 1.0


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if args.random_policy:
        run_random_policy_probe()
        return

    from instinct_rl.runners import OnPolicyRunner

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}, num_envs={args.num_envs}, actions={args.num_actions}")
    print(f"[smoke] max_iterations={args.max_iterations}, steps_per_env={args.num_steps_per_env}")

    # 创建环境
    env = SyntheticWalkEnv(args.num_envs, args.num_actions, device)

    # DataBus + SignalServer
    if args.signal_server_port:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from myrl.core.databus.bus import enable_databus
        from myrl.core.databus.signal_server import SignalServer
        from myrl.core.databus.env_wrapper import enable_databus_on_env

        bus = enable_databus()
        SignalServer(bus, port=args.signal_server_port).start()
        enable_databus_on_env(env, bus)
        print(f"[smoke] SignalServer on :{args.signal_server_port}")

    # 训练配置
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "num_steps_per_env": args.num_steps_per_env,
        "max_iterations": args.max_iterations,
        "save_interval": 50,
        "experiment_name": "smoke_test",
    }

    # 日志目录
    log_dir = args.logdir or os.path.join(
        os.path.dirname(__file__), "..", "work", "logs", "smoke_test",
        time.strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[smoke] log_dir={log_dir}")

    # 创建 Runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)

    # SSE Log Server
    if args.log_server_port:
        try:
            from myrl.logging.server import SSELogServer

            sse_srv = SSELogServer(host=args.log_server_host, port=args.log_server_port)
            sse_srv.serve_background()
            runner.add_log_sink(sse_srv)
            print(f"[smoke] SSELogServer on :{args.log_server_port}")
        except Exception as e:
            print(f"[smoke] SSELogServer 跳过: {e}")

    # 训练
    print(f"[smoke] Starting training...")
    runner.learn(num_learning_iterations=args.max_iterations)

    # 验证 checkpoint
    ckpt = os.path.join(log_dir, f"model_{args.max_iterations}.pt")
    if os.path.exists(ckpt):
        print(f"[smoke] ✓ Checkpoint saved: {ckpt} ({os.path.getsize(ckpt) / 1024:.0f} KB)")
    else:
        print(f"[smoke] ✗ Checkpoint not found: {ckpt}")
        sys.exit(1)

    print("[smoke] ✓ Training complete!")


if __name__ == "__main__":
    main()
