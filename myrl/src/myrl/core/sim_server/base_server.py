"""SimServer ABC — TCP 监听 + 消息分发基类。

子类（MuJoCoSimServer、未来的 RealRobotServer）只需实现
handle_handshake / handle_step / handle_reset / handle_get_obs 四个方法。
协议层（SimProto）与业务逻辑完全解耦。
"""

from __future__ import annotations

import logging
import socket
import traceback
from abc import ABC, abstractmethod

from myrl.core.sim_server.protocol import MsgType, SimProto

logger = logging.getLogger(__name__)


class SimServer(ABC):
    """同步单连接 TCP 服务基类。

    - 单线程、单连接：RL runner 只有一个客户端，无需并发
    - 每次 accept 后进入消息循环，直到客户端发 CLOSE 或断线
    - 子类实现四个 handle_* 方法，返回值直接作为响应 payload 发回
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 7777) -> None:
        self.host = host
        self.port = port
        self._server_sock: socket.socket | None = None

    # ── 子类必须实现的业务方法 ─────────────────────────────────────────────

    @abstractmethod
    def handle_handshake(self, req: dict) -> dict:
        """处理握手请求，返回元数据（num_envs / obs_format 等）。"""
        raise NotImplementedError

    @abstractmethod
    def handle_step(self, req: dict) -> dict:
        """处理 STEP_REQ，推进仿真，返回 obs / rewards / dones / extras。"""
        raise NotImplementedError

    @abstractmethod
    def handle_reset(self, req: dict) -> dict:
        """处理 RESET_REQ，重置所有环境，返回初始 obs。"""
        raise NotImplementedError

    @abstractmethod
    def handle_get_obs(self, req: dict) -> dict:
        """处理 GET_OBS_REQ，返回当前观测（不推进仿真）。"""
        raise NotImplementedError

    # ── 服务主循环 ────────────────────────────────────────────────────────

    def serve_forever(self) -> None:
        """启动 TCP 服务，阻塞直到进程终止。

        每次连接结束后重新等待下一个客户端（支持 play 脚本多次连接）。
        """
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        logger.info("SimServer 监听 %s:%d", self.host, self.port)

        try:
            while True:
                logger.info("等待客户端连接...")
                conn, addr = self._server_sock.accept()
                logger.info("客户端已连接: %s", addr)
                try:
                    self._handle_connection(conn)
                except Exception:
                    logger.error("连接处理异常:\n%s", traceback.format_exc())
                finally:
                    conn.close()
                    logger.info("客户端已断开: %s", addr)
        finally:
            self._server_sock.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        """处理单个客户端连接的消息循环。"""
        while True:
            try:
                msg = SimProto.recv(conn)
            except ConnectionError:
                break  # 客户端断开

            msg_type = msg.get("type")
            try:
                resp = self._dispatch(msg_type, msg)
            except Exception as e:
                logger.error("消息处理异常 type=%s: %s", msg_type, e)
                resp = {"type": int(MsgType.ERROR), "error": str(e)}

            if resp is None:
                break  # CLOSE 消息，退出循环

            SimProto.send(conn, resp)

    def _dispatch(self, msg_type: int | None, msg: dict) -> dict | None:
        """根据消息类型分发到对应处理方法。

        Returns:
            响应 dict，或 None 表示连接应关闭。
        """
        if msg_type == MsgType.HANDSHAKE_REQ:
            resp = self.handle_handshake(msg)
            resp["type"] = int(MsgType.HANDSHAKE_RESP)
            return resp

        if msg_type == MsgType.STEP_REQ:
            resp = self.handle_step(msg)
            resp["type"] = int(MsgType.STEP_RESP)
            return resp

        if msg_type == MsgType.RESET_REQ:
            resp = self.handle_reset(msg)
            resp["type"] = int(MsgType.RESET_RESP)
            return resp

        if msg_type == MsgType.GET_OBS_REQ:
            resp = self.handle_get_obs(msg)
            resp["type"] = int(MsgType.GET_OBS_RESP)
            return resp

        if msg_type == MsgType.CLOSE:
            logger.info("收到 CLOSE，断开连接")
            return None

        # 未知消息类型
        return {"type": int(MsgType.ERROR), "error": f"未知消息类型: {msg_type}"}
