"""start_mujoco_server.py — 启动 MuJoCo 仿真服务器（独立进程，无需 AppLauncher）。

用法::

    # 内置 DummyTask（协议层冒烟测试，无需真实 MJCF）
    python scripts/start_mujoco_server.py --task dummy --num_envs 4 --port 7777

    # ROS 模式（obs 经 ROS topic 发布，TCP 只传 rewards/dones）
    python scripts/start_mujoco_server.py --task dummy --num_envs 4 \\
        --port 7777 --ros --task_id test

    # 真实任务（实现了 MuJoCoTask ABC 的自定义类）
    python scripts/start_mujoco_server.py \\
        --task myrl.tasks.mujoco.g1_walk:G1WalkTask \\
        --mjcf assets/robots/humanoid_x/robot.xml \\
        --num_envs 16 --port 7777

该脚本不依赖 Isaac Sim / AppLauncher，可在轻量环境（纯 CPU / 本地）直接运行。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import threading
from pathlib import Path

# 确保 myrl 包可被导入（容器内由 entrypoint PYTHONPATH 处理，脚本直接运行时需要手动加）
_repo_src = Path(__file__).resolve().parents[1] / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from myrl.core.sim_server.mujoco_server import MuJoCoSimServer
from myrl.core.sim_server.mujoco_task import DummyTask, MuJoCoTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("start_mujoco_server")


def load_task(task_spec: str) -> MuJoCoTask:
    """根据 task 规格字符串加载任务实例。

    支持格式：
    - ``"dummy"`` — 使用内置 DummyTask（无需 MJCF）
    - ``"module.path:ClassName"`` — 动态导入并实例化指定类

    Args:
        task_spec: 任务规格字符串。

    Returns:
        MuJoCoTask 实例。
    """
    if task_spec.lower() == "dummy":
        logger.info("使用内置 DummyTask（协议层冒烟测试）")
        return DummyTask()

    if ":" not in task_spec:
        raise ValueError(
            f"task 格式错误: {task_spec!r}，应为 'module.path:ClassName' 或 'dummy'"
        )

    module_path, class_name = task_spec.rsplit(":", 1)
    logger.info("加载任务: %s.%s", module_path, class_name)
    module = importlib.import_module(module_path)
    task_cls = getattr(module, class_name)
    return task_cls()


def setup_ros_publisher(server: MuJoCoSimServer, task_id: str) -> None:
    """初始化 ROS2 发布者并注册到 server 的 obs_callback。

    obs_callback 在 TCP STEP_RESP 之前被调用（同步保障）。
    rclpy.spin 在 daemon 线程，不阻塞主线程。

    Args:
        server: 已初始化的 MuJoCoSimServer 实例。
        task_id: topic 命名空间（/myrl/{task_id}/obs/{group}）。
    """
    import rclpy
    from std_msgs.msg import Float32MultiArray

    rclpy.init()
    node = rclpy.create_node(f"mujoco_sim_{task_id}")

    # 为每个 obs group 创建发布者
    obs_fmt = server.task.obs_format()
    publishers: dict[str, object] = {}
    for group in obs_fmt:
        topic = f"/myrl/{task_id}/obs/{group}"
        publishers[group] = node.create_publisher(Float32MultiArray, topic, 10)
        logger.info("ROS obs 发布者: %s", topic)

    def obs_callback(obs_all: dict) -> None:
        """将 obs_all 发布到各 group 的 ROS topic。"""
        for group, arr in obs_all.items():
            if group not in publishers:
                continue
            import numpy as np
            arr_np = np.asarray(arr, dtype=np.float32)
            msg = Float32MultiArray()
            msg.data = arr_np.flatten().tolist()
            publishers[group].publish(msg)

    server.register_obs_callback(obs_callback)

    # rclpy.spin 在 daemon 线程（server.serve_forever 在主线程阻塞）
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    logger.info("ROS2 spin 线程已启动 (task_id=%s)", task_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="启动 MuJoCo 仿真服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dummy",
        help="任务规格：'dummy' 或 'module.path:ClassName'",
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        default=None,
        help="MJCF 文件路径（DummyTask 不需要）",
    )
    parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
    parser.add_argument("--sim_steps", type=int, default=4, help="每控制帧物理子步数")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7777, help="监听端口")
    # ROS 模式参数
    parser.add_argument(
        "--ros",
        action="store_true",
        help="启用 ROS obs 发布模式：obs 经 /myrl/{task_id}/obs/{group} 发布",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="default",
        help="topic 命名空间（/myrl/{task_id}/...），与 bridge --task_id 保持一致",
    )
    parser.add_argument(
        "--no_obs_in_resp",
        action="store_true",
        help="STEP_RESP 不含 obs（--ros 时自动推荐，节省 TCP 带宽）",
    )
    args = parser.parse_args()

    # ROS 模式下默认不在 TCP resp 中包含 obs
    include_obs = not (args.no_obs_in_resp or args.ros)

    task = load_task(args.task)

    server = MuJoCoSimServer(
        task=task,
        mjcf_path=args.mjcf,
        num_envs=args.num_envs,
        sim_steps_per_ctrl=args.sim_steps,
        host=args.host,
        port=args.port,
        include_obs_in_response=include_obs,
    )

    # 启用 ROS 发布模式
    if args.ros:
        setup_ros_publisher(server, task_id=args.task_id)
        logger.info("ROS 模式已启用: task_id=%s, TCP obs=%s", args.task_id, include_obs)

    logger.info(
        "服务器启动: task=%s, num_envs=%d, port=%d, ros=%s",
        args.task,
        args.num_envs,
        args.port,
        args.ros,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
