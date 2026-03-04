"""start_ros2_bridge.py — 启动 MuJoCo ↔ ROS2 双向桥接节点（目标架构）。

需要在 source /opt/ros/humble/setup.bash 后运行，或容器 entrypoint 已配置。

数据流：
    ROS /myrl/{task_id}/obs/{group}  ← 传感器总线（server 发布 obs）
    TCP STEP_RESP                    ← 只含 rewards / dones / time_outs / log

用法::

    # 先启动 server（ROS 模式）：
    python scripts/start_mujoco_server.py --task dummy --num_envs 4 \\
        --port 7777 --ros --task_id test

    # 再启动 bridge（另一终端，source ROS2 后）：
    python scripts/start_ros2_bridge.py --port 7777 --task_id test \\
        --history_cfg '{"policy": {"dummy_obs": 1}}' --hz 50

    # 验证 obs topic 有数据：
    ros2 topic list | grep myrl
    ros2 topic echo /myrl/test/obs/policy --once

    # 注入动作并验证回路：
    ros2 topic pub /myrl/test/action std_msgs/msg/Float32MultiArray \\
        "{data: [0.1, 0.0, -0.1]}" --once
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("start_ros2_bridge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo ↔ ROS2 双向桥接（目标架构：ROS 为传感器总线）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="localhost", help="MuJoCoSimServer IP")
    parser.add_argument("--port", type=int, default=7777, help="MuJoCoSimServer 端口")
    parser.add_argument("--hz", type=float, default=50.0, help="控制循环频率（Hz）")
    parser.add_argument(
        "--task_id",
        type=str,
        default="default",
        help="topic 命名空间（/myrl/{task_id}/...），必须与 server --task_id 一致",
    )
    parser.add_argument(
        "--history_cfg",
        type=str,
        default=None,
        help=(
            "观测历史配置 JSON，例如 '{\"policy\": {\"base_ang_vel\": 8}}' "
            "或 '{\"policy\": 8}'（group 粒度）。为空则不启用历史管理。"
        ),
    )
    parser.add_argument("--node_name", type=str, default="myrl_ros2_bridge", help="ROS2 节点名")
    parser.add_argument(
        "--obs_wait_timeout",
        type=float,
        default=0.1,
        help="等待 ROS obs 的超时秒数（server 保证先发 ROS 再发 TCP，通常无需修改）",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch 设备（cpu / cuda）")
    args = parser.parse_args()

    # 解析 history_cfg JSON
    history_cfg = None
    if args.history_cfg:
        try:
            history_cfg = json.loads(args.history_cfg)
        except json.JSONDecodeError as e:
            logger.error("history_cfg JSON 解析失败: %s", e)
            sys.exit(1)

    from myrl.core.sim_server.ros2_bridge import Ros2Bridge

    bridge = Ros2Bridge(
        host=args.host,
        port=args.port,
        task_id=args.task_id,
        control_hz=args.hz,
        history_cfg=history_cfg,
        node_name=args.node_name,
        obs_wait_timeout=args.obs_wait_timeout,
        device=args.device,
    )
    bridge.spin()


if __name__ == "__main__":
    main()
