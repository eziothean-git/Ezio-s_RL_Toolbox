"""ros2_bridge.py — MuJoCo socket ↔ ROS2 双向桥接节点（目标架构）。

数据流（与旧版相反）：
    ROS topics (/myrl/{task_id}/obs/{group})  ← 传感器总线（server 发布 obs）
    TCP STEP_RESP                             ← 只含 rewards / dones / time_outs / log

控制循环：
    1. 读取最新动作（外部 ROS /myrl/{task_id}/action 订阅注入，或策略调用）
    2. TCP: send STEP_REQ(actions) → recv STEP_RESP(rewards/dones)
    3. _new_obs_event.wait()  — 等 ROS obs 到位
       （server 保证先 obs_callback 发布 ROS，再发 TCP resp，故 Event 必然已 set）
    4. 读 _obs_cache → torch tensors
    5. history_mgr.push(obs_pack) → obs_with_history（若 history_mgr 存在）
    6. 对 done envs 调用 history_mgr.reset()
    7. 发布动作到 /myrl/{task_id}/action（真机兼容接口）

同步保障：
    server side: obs_callback（ROS publish） → TCP STEP_RESP
    bridge side: TCP STEP_RESP recv → _new_obs_event.wait(timeout)

topic 命名：
    subscribe: /myrl/{task_id}/obs/{group}   Float32MultiArray（server 发布 obs）
    publish:   /myrl/{task_id}/action        Float32MultiArray（bridge 发布动作）
    publish:   /myrl/{task_id}/reward        Float32（平均奖励，调试用）

用法::

    # 先启动 server（ROS 模式）：
    python scripts/start_mujoco_server.py --task dummy --num_envs 4 \\
        --port 7777 --ros --task_id test

    # 再启动 bridge：
    python scripts/start_ros2_bridge.py --port 7777 --task_id test \\
        --history_cfg '{"policy": {"dummy_obs": 1}}' --hz 50

    # 验证 obs topic：
    ros2 topic echo /myrl/test/obs/policy --once

    # 注入动作：
    ros2 topic pub /myrl/test/action std_msgs/msg/Float32MultiArray \\
        "{data: [0.1, 0.0, -0.1]}" --once
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable

import numpy as np

from myrl.core.sim_server.protocol import MsgType, SimProto

logger = logging.getLogger(__name__)


class Ros2Bridge:
    """MuJoCo socket ↔ ROS2 双向桥接节点（目标架构：ROS 为传感器总线）。

    Args:
        host: MuJoCoSimServer 地址。
        port: MuJoCoSimServer TCP 端口。
        task_id: topic 命名空间（/myrl/{task_id}/...）。
        control_hz: 控制循环频率（Hz）。
        history_cfg: 观测历史配置，格式同 ObsHistoryManager。
            e.g. ``{"policy": {"base_ang_vel": 8, "joint_pos": 1}}``
            或 ``{"policy": 8}``（group 粒度）。
            为 None 时不启用历史管理。
        node_name: ROS2 节点名称。
        obs_wait_timeout: 等待 ROS obs 的超时秒数（默认 0.1s）。
        device: torch 设备（"cpu" 或 "cuda"）。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7777,
        task_id: str = "default",
        control_hz: float = 50.0,
        history_cfg: dict | None = None,
        node_name: str = "myrl_ros2_bridge",
        obs_wait_timeout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.control_hz = control_hz
        self.task_id = task_id
        self._node_name = node_name
        self._obs_wait_timeout = obs_wait_timeout
        self._device = device
        self._history_cfg = history_cfg

        # ── 连接 TCP server（握手） ────────────────────────────────────────
        self._sock = socket.create_connection((host, port), timeout=10.0)
        SimProto.send(self._sock, {"type": int(MsgType.HANDSHAKE_REQ)})
        resp = SimProto.recv(self._sock)
        assert resp["type"] == int(MsgType.HANDSHAKE_RESP), f"握手失败: {resp}"

        self.num_envs: int = int(resp["num_envs"])
        self.num_actions: int = int(resp["num_actions"])
        self._obs_format: dict = resp["obs_format"]  # {group: {term: [dim]}}
        logger.info(
            "TCP 握手成功: num_envs=%d num_actions=%d obs_groups=%s",
            self.num_envs,
            self.num_actions,
            list(self._obs_format.keys()),
        )

        # 初始 reset（触发 server obs_callback，但此时 ROS 节点未启动，obs 可能丢失）
        # bridge 控制循环开始前会等第一帧 ROS obs
        SimProto.send(self._sock, {"type": int(MsgType.RESET_REQ)})
        _reset_resp = SimProto.recv(self._sock)
        logger.info("初始 reset 完成（ROS obs 将在控制循环启动后到达）")

        # ── ObsHistoryManager（可选） ─────────────────────────────────────
        self._history_mgr = None
        if history_cfg is not None:
            from myrl.core.obs.history_manager import ObsHistoryManager
            # obs_format 从 server 获取（list shape → tuple shape）
            fmt = {
                group: {term: (dims[0],) for term, dims in terms.items()}
                for group, terms in self._obs_format.items()
            }
            self._history_mgr = ObsHistoryManager(
                obs_format=fmt,
                history_cfg=history_cfg,
                num_envs=self.num_envs,
                device=device,
            )
            logger.info("ObsHistoryManager 已初始化: history_cfg=%s", history_cfg)

        # ── 线程安全的 obs 缓存（ROS subscriber 写，控制循环读） ──────────
        # {group: np.ndarray[N, flat_dim]}
        self._obs_cache: dict[str, np.ndarray] = {}
        self._obs_lock = threading.Lock()
        self._new_obs_event = threading.Event()

        # ── 线程安全的最新动作缓存（外部 /action subscriber 写，控制循环读） ─
        # 或由策略调用 set_actions() 写入
        self._latest_actions = np.zeros(
            (self.num_envs, self.num_actions), dtype=np.float32
        )
        self._action_lock = threading.Lock()

        # ── ROS2 节点占位（spin() 中初始化） ─────────────────────────────
        self._node = None
        self._obs_subs: dict[str, object] = {}   # group → subscriber
        self._action_pub = None
        self._reward_pub = None

        self._stop_event = threading.Event()

    # ── 公共 API ──────────────────────────────────────────────────────────

    def set_actions(self, actions: np.ndarray) -> None:
        """策略外部注入动作（无需 ROS，适合离线评测）。

        Args:
            actions: float32 数组，形状 [N, num_actions] 或 [num_actions]（广播）。
        """
        if actions.ndim == 1:
            actions = np.tile(actions, (self.num_envs, 1))
        with self._action_lock:
            self._latest_actions = actions.astype(np.float32)

    def spin(self) -> None:
        """初始化 ROS2 节点并启动控制循环（阻塞）。

        在 source /opt/ros/humble/setup.bash 后调用。
        """
        import rclpy
        from rclpy.node import Node

        rclpy.init()
        self._node = Node(self._node_name)
        self._setup_ros_interfaces()

        # 控制循环在独立线程，rclpy.spin 在主线程
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()
        logger.info(
            "Ros2Bridge 已启动: control_hz=%.1f, task_id=%s, node=%s",
            self.control_hz,
            self.task_id,
            self._node_name,
        )

        try:
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            ctrl_thread.join(timeout=2.0)
            self._close()
            self._node.destroy_node()
            rclpy.shutdown()

    # ── ROS2 接口初始化 ────────────────────────────────────────────────────

    def _setup_ros_interfaces(self) -> None:
        """创建发布者和订阅者。"""
        from std_msgs.msg import Float32, Float32MultiArray

        ns = f"/myrl/{self.task_id}"

        # 订阅每个 obs group 的 topic（server 发布）
        for group in self._obs_format:
            topic = f"{ns}/obs/{group}"
            sub = self._node.create_subscription(
                Float32MultiArray,
                topic,
                self._make_obs_callback(group),
                10,
            )
            self._obs_subs[group] = sub
            logger.info("订阅 obs topic: %s", topic)

        # 发布动作（真机兼容接口）
        self._action_pub = self._node.create_publisher(
            Float32MultiArray, f"{ns}/action", 10
        )

        # 订阅外部动作注入（来自策略或手动发布）
        self._ext_action_sub = self._node.create_subscription(
            Float32MultiArray,
            f"{ns}/action",
            self._on_ext_action,
            10,
        )

        # 发布平均奖励（调试用）
        self._reward_pub = self._node.create_publisher(
            Float32, f"{ns}/reward", 10
        )

        logger.info(
            "ROS2 接口就绪 | obs groups: %s | action: %s/action",
            list(self._obs_format.keys()),
            ns,
        )

    # ── ROS2 回调 ─────────────────────────────────────────────────────────

    def _make_obs_callback(self, group: str) -> Callable:
        """工厂：生成指定 group 的 obs subscriber 回调。"""
        def _callback(msg) -> None:
            data = np.asarray(msg.data, dtype=np.float32)
            # 期望形状：[N * flat_dim]，reshape 到 [N, flat_dim]
            group_terms = self._obs_format[group]
            flat_dim = sum(dims[0] for dims in group_terms.values())
            if data.size == self.num_envs * flat_dim:
                arr = data.reshape(self.num_envs, flat_dim)
            else:
                logger.warning(
                    "obs group=%s 尺寸不匹配: got %d, expect %d*%d=%d，忽略",
                    group, data.size, self.num_envs, flat_dim, self.num_envs * flat_dim,
                )
                return
            with self._obs_lock:
                self._obs_cache[group] = arr
                # 所有 group 都到齐后才 set Event
                if set(self._obs_cache.keys()) >= set(self._obs_format.keys()):
                    self._new_obs_event.set()
        return _callback

    def _on_ext_action(self, msg) -> None:
        """接收外部注入的动作（与 set_actions 语义相同）。"""
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size == self.num_actions:
            actions = np.tile(data, (self.num_envs, 1))
        elif data.size == self.num_envs * self.num_actions:
            actions = data.reshape(self.num_envs, self.num_actions)
        else:
            logger.warning(
                "外部动作尺寸不匹配: got %d, expect %d or %d，忽略",
                data.size, self.num_actions, self.num_envs * self.num_actions,
            )
            return
        with self._action_lock:
            self._latest_actions = actions

    # ── 控制循环 ──────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        """以固定频率驱动仿真步进（在独立线程中运行）。"""
        dt = 1.0 / self.control_hz
        step_count = 0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            # 1. 读取最新动作
            with self._action_lock:
                actions = self._latest_actions.copy()

            # 2. TCP step：发送动作，接收 rewards/dones（obs 不在 TCP 里）
            try:
                resp = self._tcp_step(actions)
            except Exception as e:
                logger.error("TCP step 失败: %s", e)
                break

            rewards = np.asarray(resp.get("rewards", []), dtype=np.float32)
            dones = np.asarray(resp.get("dones", []))
            time_outs = np.asarray(resp.get("time_outs", []))

            # 3. 等待 ROS obs（server 保证先 obs_callback 再发 TCP resp）
            got_obs = self._new_obs_event.wait(timeout=self._obs_wait_timeout)
            self._new_obs_event.clear()
            if not got_obs:
                logger.warning("step=%d 未收到 ROS obs（超时 %.2fs），用缓存值", step_count, self._obs_wait_timeout)

            # 4. 读 obs cache → torch tensors
            obs_pack_np: dict[str, np.ndarray] = {}
            with self._obs_lock:
                obs_pack_np = {k: v.copy() for k, v in self._obs_cache.items()}

            # 5. history_mgr.push()
            if self._history_mgr is not None and obs_pack_np:
                import torch
                obs_pack_torch = {
                    k: torch.from_numpy(v).to(self._device)
                    for k, v in obs_pack_np.items()
                }
                obs_with_history = self._history_mgr.push(obs_pack_torch)

                # 6. done envs 清零 history
                done_ids = np.where(dones)[0].tolist()
                if done_ids:
                    self._history_mgr.reset(done_ids)

                # （obs_with_history 已供策略使用，此处只打印调试信息）
                _ = obs_with_history

            # 7. 发布动作到 /myrl/{task_id}/action（真机兼容接口）
            self._publish_action(actions)

            # 发布平均奖励（调试用）
            if rewards.size > 0:
                self._publish_reward(float(rewards.mean()))

            step_count += 1
            if step_count % 100 == 0:
                logger.debug(
                    "step=%d reward_mean=%.4f dones=%d",
                    step_count,
                    float(rewards.mean()) if rewards.size > 0 else 0.0,
                    int(dones.sum()) if dones.size > 0 else 0,
                )

            # 精确频率控制
            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── 发布方法 ──────────────────────────────────────────────────────────

    def _publish_action(self, actions: np.ndarray) -> None:
        """发布当前动作到 /myrl/{task_id}/action（env 0，真机兼容接口）。"""
        from std_msgs.msg import Float32MultiArray

        if self._action_pub is None:
            return
        msg = Float32MultiArray()
        msg.data = actions[0].tolist()  # 发布 env 0 的动作
        self._action_pub.publish(msg)

    def _publish_reward(self, reward: float) -> None:
        """发布平均奖励（调试用）。"""
        from std_msgs.msg import Float32

        if self._reward_pub is None:
            return
        msg = Float32()
        msg.data = reward
        self._reward_pub.publish(msg)

    # ── TCP 工具 ──────────────────────────────────────────────────────────

    def _tcp_step(self, actions: np.ndarray) -> dict:
        SimProto.send(self._sock, {"type": int(MsgType.STEP_REQ), "actions": actions})
        return SimProto.recv(self._sock)

    def _close(self) -> None:
        try:
            SimProto.send(self._sock, {"type": int(MsgType.CLOSE)})
        except Exception:
            pass
        finally:
            self._sock.close()
