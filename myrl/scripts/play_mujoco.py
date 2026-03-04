"""play_mujoco.py — 基于 MuJoCo socket 后端的策略推理脚本。

不依赖 Isaac Sim / AppLauncher，适用于：
- sim2sim 验证（策略在 MuJoCo 中推理）
- 真机部署验证（MuJoCoSimServer 替换为 RealRobotServer 时协议不变）

用法::

    # 先在另一个终端启动 server：
    python scripts/start_mujoco_server.py --task dummy --num_envs 4 --port 7777

    # 然后运行推理（不加载 checkpoint，随机权重冒烟）：
    python scripts/play_mujoco.py --num_envs 4 --num_steps 10

    # 加载真实 checkpoint：
    python scripts/play_mujoco.py \\
        --load_run logs/G1Smoke/2026-03-04_00-00-00 \\
        --checkpoint model_5000.pt \\
        --host localhost --port 7777
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保 myrl 包可被导入
_repo_src = Path(__file__).resolve().parents[1] / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("play_mujoco")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo socket 后端策略推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="localhost", help="MuJoCoSimServer 地址")
    parser.add_argument("--port", type=int, default=7777, help="MuJoCoSimServer 端口")
    parser.add_argument("--device", type=str, default="cpu", help="Tensor 设备")
    parser.add_argument("--num_steps", type=int, default=100, help="推理步数")
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="训练日志目录（含 model_*.pt）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 文件名（默认加载最新）",
    )
    args = parser.parse_args()

    import torch
    from myrl.core.compat.backends.mujoco_backend import MuJoCoBackend

    # 1. 建立 socket 连接（握手 + 自动 reset）
    logger.info("连接到 MuJoCoSimServer %s:%d ...", args.host, args.port)
    env = MuJoCoBackend(host=args.host, port=args.port, device=args.device)
    logger.info(
        "连接成功: num_envs=%d, num_actions=%d, num_obs=%d",
        env.num_envs,
        env.num_actions,
        env.num_obs,
    )

    # 2. 加载策略（如果提供了 load_run）
    policy = None
    if args.load_run is not None:
        from instinct_rl.runners import OnPolicyRunner

        train_cfg: dict = {}  # play 模式不需要完整 train_cfg
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=args.device)
        runner.load(args.load_run, load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
        logger.info("策略已加载: %s", args.load_run)
    else:
        logger.info("未指定 --load_run，使用随机动作（冒烟模式）")

    # 3. 推理循环
    obs, extras = env.reset()
    logger.info("开始推理，共 %d 步", args.num_steps)

    for step in range(args.num_steps):
        with torch.no_grad():
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=torch.device(args.device))

        obs, rewards, dones, extras = env.step(actions)

        if (step + 1) % 10 == 0:
            logger.info(
                "step %d/%d | reward_mean=%.4f | dones=%d",
                step + 1,
                args.num_steps,
                rewards.mean().item(),
                dones.sum().item(),
            )

    logger.info("推理完成，obs.shape=%s", tuple(obs.shape))
    env.close()


if __name__ == "__main__":
    main()
