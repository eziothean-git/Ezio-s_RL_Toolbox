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
parser.add_argument("--package", type=str, default=None, help="Path to .myrlpkg（替代 --task，由 ExperimentComposer 加载）。")
parser.add_argument("--seed", type=int, default=None, help="随机种子。")
parser.add_argument("--logroot", type=str, default=None, help="日志根目录（覆盖默认值）。")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO 训练迭代数。")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名称（用于日志目录）。")
parser.add_argument("--run_name", type=str, default=None, help="运行名称后缀。")
parser.add_argument("--resume", default=None, action="store_true", help="是否从 checkpoint 恢复训练。")
parser.add_argument("--load_run", type=str, default=None, help="要恢复的运行目录名。")
parser.add_argument("--checkpoint", type=str, default=None, help="要加载的 checkpoint 文件。")
parser.add_argument("--debug", action="store_true", default=False, help="启用 debugpy 调试模式。")
# ── 日志扩展参数 ───────────────────────────────────────────────────────────
parser.add_argument("--wandb", action="store_true", default=False, help="启用 Weights & Biases 日志（需要 pip install wandb）。")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project 名（默认=experiment_name）。")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity（默认=None）。")
parser.add_argument("--log_server_port", type=int, default=None, help="启动 SSE log server 的端口（不指定则不启动）。")
parser.add_argument("--log_server_host", type=str, default="0.0.0.0", help="SSE log server 绑定地址。")
parser.add_argument("--no_jsonl", action="store_true", default=False, help="禁用 JSONL 结构化日志（默认启用）。")
parser.add_argument("--no_registry", action="store_true", default=False, help="禁用训练结束后的 Experiment Registry 写入。")
parser.add_argument("--signal_server_port", type=int, default=None, help="启动 DataBus SignalServer 的端口（不指定则不启动）。")
parser.add_argument("--debug_tools", action="store_true", default=False, help="启用交互式调试工具（力施加/MUX/锚点/可视化/时间控制）。非 headless 时自动创建 omni.ui 面板。")
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

# DataBus：如果设置了 MYRL_OSCILLOSCOPE=1 或 --signal_server_port，自动启用
from myrl.core.databus.bus import auto_enable_databus, enable_databus
_bus = auto_enable_databus()
if _bus is None and getattr(args_cli, "signal_server_port", None):
    _bus = enable_databus()
_signal_srv = None
if _bus and getattr(args_cli, "signal_server_port", None):
    from myrl.core.databus.signal_server import SignalServer
    _signal_srv = SignalServer(_bus, port=args_cli.signal_server_port)
    _signal_srv.start()
    print(f"[myrl] SignalServer started on :{args_cli.signal_server_port}")

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

# Phase A/B 切换：设置环境变量 MYRL_USE_ISAACLAB_BACKEND=1 启用 Phase B
if os.environ.get("MYRL_USE_ISAACLAB_BACKEND"):
    from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
else:
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
    global _bus  # 模块级变量，debug_tools 路径可能重新赋值

    # ── 确定是 --package 路径还是 --task 路径 ──────────────────────────────
    _package_id = None  # 用于 registry 记录

    if args_cli.package:
        # ExperimentComposer 路径
        from myrl.assets.composer import ExperimentComposer
        composer = ExperimentComposer(args_cli.package)
        env, runner_cfg_dict = composer.compose(
            num_envs=args_cli.num_envs,
            device=args_cli.device,
        )
        _package_id = composer.manifest.package_id
        experiment_name = runner_cfg_dict.get("experiment_name", _package_id)
        max_iterations = runner_cfg_dict.get("max_iterations", 10000)
        device = runner_cfg_dict.get("device", "cuda:0")

        if args_cli.max_iterations is not None:
            max_iterations = args_cli.max_iterations
            runner_cfg_dict["max_iterations"] = max_iterations

        # 确定日志目录
        log_root_path = (
            os.path.abspath(args_cli.logroot)
            if args_cli.logroot
            else os.path.abspath(os.path.join("logs", "myrl", experiment_name))
        )
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y%m%d_%H%M%S"))

        # 包装环境
        env = EnvWrapper(env)
        if _bus:
            from myrl.core.databus.env_wrapper import enable_databus_on_env
            enable_databus_on_env(env, _bus)

        # 调试工具（--debug_tools）
        if getattr(args_cli, "debug_tools", False):
            if _bus is None:
                from myrl.core.databus.bus import enable_databus
                _bus = enable_databus()
            from myrl.debug_tools import enable_debug_tools
            _is_headless = getattr(args_cli, "headless", True)
            enable_debug_tools(env, _bus, signal_server=_signal_srv, headless=_is_headless)

        # wandb
        if args_cli.wandb:
            import wandb
            wandb.init(
                project=args_cli.wandb_project or experiment_name,
                name=os.path.basename(log_dir),
                entity=args_cli.wandb_entity,
                config=runner_cfg_dict,
                dir=log_dir,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="thread"),
            )

        runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device=device)
        runner.add_git_repo_to_log(__file__)

        from myrl.logging import build_sinks
        for sink in build_sinks(args_cli, log_dir, run_name=""):
            runner.add_log_sink(sink)

        runner.learn(num_learning_iterations=max_iterations)

        # 写入 Experiment Registry
        if not getattr(args_cli, "no_registry", False):
            try:
                from myrl.registry import RunRegistry, RunManifest

                class _AgentCfgProxy:
                    """给 RunManifest.from_train_run 用的轻量代理。"""
                    def __init__(self, d, exp_name):
                        self.experiment_name = exp_name
                        self.max_iterations = d.get("max_iterations", 0)
                        self.num_envs = d.get("num_envs", 0)
                        self.seed = d.get("seed", 0)
                        self.device = d.get("device", "cuda:0")
                        self.run_name = ""

                proxy = _AgentCfgProxy(runner_cfg_dict, experiment_name)
                manifest = RunManifest.from_train_run(
                    log_dir=log_dir,
                    task_id=f"package:{_package_id}",
                    agent_cfg=proxy,
                    runner=runner,
                )
                manifest.package_id = _package_id
                reg = RunRegistry()
                run_id = reg.save(manifest)
                print(f"[INFO] Experiment manifest saved: {run_id}")
            except Exception as e:
                print(f"[WARN] Registry save failed (non-fatal): {e}")

        # 关闭日志
        for sink in runner._log_sinks:
            sink.close()
        if args_cli.wandb:
            import wandb
            wandb.finish()

        env.close()
        return

    # ── 标准 --task 路径 ────────────────────────────────────────────────────
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
    if _bus:
        from myrl.core.databus.env_wrapper import enable_databus_on_env
        enable_databus_on_env(env, _bus)

    # 调试工具（--debug_tools）
    if getattr(args_cli, "debug_tools", False):
        if _bus is None:
            from myrl.core.databus.bus import enable_databus
            _bus = enable_databus()
        from myrl.debug_tools import enable_debug_tools
        _is_headless = getattr(args_cli, "headless", True)
        enable_debug_tools(env, _bus, signal_server=_signal_srv, headless=_is_headless)

    # wandb：在 OnPolicyRunner 创建（即 SummaryWriter 初始化）之前调用 wandb.init，
    # sync_tensorboard=True 让 wandb 自动同步所有 TensorBoard scalar，无需额外侵入。
    if args_cli.wandb:
        import wandb
        wandb.init(
            project=args_cli.wandb_project or agent_cfg.experiment_name,
            name=os.path.basename(log_dir),
            entity=args_cli.wandb_entity,
            config=agent_cfg.to_dict(),
            dir=log_dir,
            sync_tensorboard=True,
            settings=wandb.Settings(start_method="thread"),  # Isaac Sim 兼容
        )

    # 创建 OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # 挂载 myrl 日志 sinks（JSONL always-on，wandb/SSE server opt-in）
    from myrl.logging import build_sinks
    for sink in build_sinks(args_cli, log_dir, run_name=agent_cfg.run_name or ""):
        runner.add_log_sink(sink)

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

    # 写入 Experiment Registry（除非 --no_registry）
    if not getattr(args_cli, "no_registry", False):
        try:
            from myrl.registry import RunRegistry, RunManifest
            manifest = RunManifest.from_train_run(
                log_dir=log_dir,
                task_id=args_cli.task,
                agent_cfg=agent_cfg,
                runner=runner,
            )
            reg = RunRegistry()
            run_id = reg.save(manifest)
            print(f"[INFO] Experiment manifest saved: {run_id}")
        except Exception as e:
            print(f"[WARN] Registry save failed (non-fatal): {e}")

    # 关闭所有日志后端
    for sink in runner._log_sinks:
        sink.close()
    if args_cli.wandb:
        import wandb
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
