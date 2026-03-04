"""MuJoCo Socket 服务端包。

提供 TCP socket 通信协议，使训练好的策略可无缝切换到 MuJoCo 运行（sim2sim 验证）。
协议与真机部署完全一致——真机只需替换 MuJoCoSimServer 为 RealRobotServer。

ROS2 桥接（双向）由 Ros2Bridge 提供，需要容器内已安装 ros-humble-ros-base。
"""

from myrl.core.sim_server.protocol import MsgType, SimProto
from myrl.core.sim_server.base_server import SimServer
from myrl.core.sim_server.mujoco_task import MuJoCoTask, DummyTask
from myrl.core.sim_server.mujoco_server import MuJoCoSimServer

__all__ = [
    "MsgType",
    "SimProto",
    "SimServer",
    "MuJoCoTask",
    "DummyTask",
    "MuJoCoSimServer",
]
