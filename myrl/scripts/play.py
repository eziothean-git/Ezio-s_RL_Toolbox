"""myrl 推理/可视化脚本（Phase A）。

调用链:
    AppLauncher → gym.make → InstinctRlVecEnvWrapper
    → OnPolicyRunner → runner.load(checkpoint)
    → policy = runner.get_inference_policy(device)
    → rollout loop: obs, _, _, _ = env.step(policy(obs))

用法示例:
    # 加载 checkpoint 推理
    python play.py --task Instinct-G1Locomotion-Flat-v0 \\
                   --load_run 20240101_120000 --num_envs 1

    # 不加载 checkpoint，直接随机策略
    python play.py --task Instinct-G1Locomotion-Flat-v0 --no_resume
"""

"""必须先启动 Isaac Sim。"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ── CLI 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play an RL agent with myrl + Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="录制推理视频。")
parser.add_argument("--video_length", type=int, default=3000, help="单段视频帧数。")
parser.add_argument("--video_start_step", type=int, default=0, help="从第几步开始录制。")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名称（用于定位日志目录）。")
parser.add_argument("--load_run", type=str, default=None, help="要加载的运行目录名。")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 文件名。")
parser.add_argument("--no_resume", action="store_true", default=False, help="不加载 checkpoint，从头开始。")
parser.add_argument("--debug", action="store_true", default=False, help="启用 debugpy 调试。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下 import 在 Isaac Sim 启动后才可用。"""

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Phase A: 直接用 instinctlab 的 wrapper
# Phase B: 换成 from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper as EnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# 等待调试器附加
if args_cli.debug:
    import debugpy
    ip_address = ("0.0.0.0", 6789)
    print(f"Is waiting for attach at address: {ip_address[0]}:{ip_address[1]}", flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

# 注册 instinctlab 任务
import instinctlab.tasks  # noqa: F401
# 注册 myrl 自有任务（优先级：myrl/assets/ > instinctlab 内置）
import myrl.tasks          # noqa: F401


def main():
    """myrl 推理主函数。"""
    # 解析配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "instinct_rl_cfg_entry_point")

    # 覆盖 CLI 参数
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    # 确定日志目录和 checkpoint 路径
    log_root_path = os.path.join("logs", "myrl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run

    if args_cli.load_run is not None:
        if os.path.isabs(args_cli.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(args_cli.load_run),
                os.path.basename(args_cli.load_run),
                agent_cfg.load_checkpoint,
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif args_cli.no_resume:
        print(f"[INFO] No checkpoint specified, running with untrained policy.")
        log_dir = os.path.join(log_root_path, "play_scratch")
        resume_path = None
    else:
        raise RuntimeError(
            "[ERROR] No checkpoint specified. Use --load_run to specify a checkpoint, "
            "or --no_resume to run without loading."
        )

    # 创建 Isaac Lab 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 可选：录制视频
    if args_cli.video:
        checkpoint_tag = resume_path.split("_")[-1].split(".")[0] if resume_path else "scratch"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{checkpoint_tag}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 多智能体转单智能体
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 用 wrapper 包装
    env = EnvWrapper(env)

    # 初始化 runner（log_dir=None 表示不写日志）
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # 加载 checkpoint
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 获取初始观测
    obs, _ = env.get_observations()
    timestep = 0

    # 推理循环
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rewards, _dones, _infos = env.step(actions)
        timestep += 1

        # 录制完成后退出
        if args_cli.video and timestep == args_cli.video_length:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
