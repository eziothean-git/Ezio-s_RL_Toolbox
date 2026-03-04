"""myrl 训练入口脚本（Phase A）。

调用链:
    AppLauncher(args_cli)
    → gym.make(task, cfg=env_cfg)          # InstinctRlEnv (from instinctlab)
    → InstinctRlVecEnvWrapper(env)          # Phase A: 直接用 instinctlab 的 wrapper
    → OnPolicyRunner(env, cfg, log_dir)     # from instinct_rl
    → runner.learn(num_iterations)
    → env.close()

Phase B 切换: 把 InstinctRlVecEnvWrapper 换成
    from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
"""

"""必须先启动 Isaac Sim，后续 import 才可用。"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ── CLI 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train an RL agent with myrl + Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="训练过程中录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="单段视频帧数。")
parser.add_argument("--video_interval", type=int, default=2000, help="录制间隔（步数）。")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称，例如 Instinct-G1Locomotion-Flat-v0。")
parser.add_argument("--seed", type=int, default=None, help="随机种子。")
parser.add_argument("--logroot", type=str, default=None, help="日志根目录（覆盖默认值）。")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO 训练迭代数。")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名称（用于日志目录）。")
parser.add_argument("--run_name", type=str, default=None, help="运行名称后缀。")
parser.add_argument("--resume", default=None, action="store_true", help="是否从 checkpoint 恢复训练。")
parser.add_argument("--load_run", type=str, default=None, help="要恢复的运行目录名。")
parser.add_argument("--checkpoint", type=str, default=None, help="要加载的 checkpoint 文件。")
parser.add_argument("--debug", action="store_true", default=False, help="启用 debugpy 调试模式。")
# AppLauncher 参数（--headless 等）
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录制视频需要开摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 hydra 参数（如有）
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim（必须在所有 isaacsim 相关 import 之前）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下 import 在 Isaac Sim 启动后才可用。"""

import gymnasium as gym
import torch
from datetime import datetime

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

# Phase A: 直接用 instinctlab 的 wrapper
# Phase B: 换成 from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper as EnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# 等待调试器附加
if args_cli.debug:
    import debugpy
    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print(f"Is waiting for attach at address: {ip_address[0]}:{ip_address[1]}", flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

# 注册 instinctlab 任务（触发 gym.register）
import instinctlab.tasks  # noqa: F401
# 注册 myrl 自有任务（优先级：myrl/assets/ > instinctlab 内置）
import myrl.tasks          # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _update_agent_cfg(agent_cfg: InstinctRlOnPolicyRunnerCfg, args_cli: argparse.Namespace) -> InstinctRlOnPolicyRunnerCfg:
    """用 CLI 参数覆盖 agent 配置。"""
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    return agent_cfg


def main():
    """myrl 主训练函数。"""
    # 解析环境和 agent 配置
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "instinct_rl_cfg_entry_point")
    agent_cfg = _update_agent_cfg(agent_cfg, args_cli)

    # 同步种子和设备
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # 覆盖环境数量
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # 确定日志目录
    if args_cli.logroot is not None:
        log_root_path = args_cli.logroot
    else:
        log_root_path = os.path.join("logs", "myrl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 处理 resume 路径
    resume_path = None
    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run),
                os.path.basename(agent_cfg.load_run),
                agent_cfg.load_checkpoint,
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from: {resume_path}")

    # 创建 Isaac Lab 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 可选：录制视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 多智能体任务转单智能体
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Phase A: 用 instinctlab 的 wrapper 包装环境
    env = EnvWrapper(env)

    # 创建 OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # 加载 checkpoint（resume 模式）
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # 保存配置到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 开始训练
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
