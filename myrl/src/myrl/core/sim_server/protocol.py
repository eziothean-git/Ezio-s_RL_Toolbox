"""SimProto — TCP socket 通信协议（msgpack-numpy 编解码）。

每帧格式：
    [4 bytes: frame_len uint32 big-endian] + [msgpack body]

msgpack body 用 msgpack_numpy.packb 序列化，numpy 数组直接内嵌（高效传输）。
客户端（MuJoCoBackend）和服务端（MuJoCoSimServer）共用同一协议层。
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum

import msgpack_numpy


class MsgType(IntEnum):
    """Socket 协议消息类型枚举。"""

    HANDSHAKE_REQ = 0   # 客户端握手请求
    HANDSHAKE_RESP = 1  # 服务端握手响应（携带 num_envs / obs_format 等元数据）
    STEP_REQ = 2        # 推进一步：传入 actions
    STEP_RESP = 3       # step 响应：obs / rewards / dones / extras
    RESET_REQ = 4       # 重置所有环境
    RESET_RESP = 5      # reset 响应：初始 obs
    GET_OBS_REQ = 6     # 获取当前观测（不推进仿真）
    GET_OBS_RESP = 7    # get_obs 响应
    CLOSE = 8           # 客户端关闭连接
    ERROR = 9           # 服务端错误响应


class SimProto:
    """TCP socket 编解码工具类（客户端/服务端共用）。

    帧格式：
        [4 bytes uint32 big-endian: body_len] + [msgpack body]

    numpy 数组通过 msgpack_numpy 扩展高效序列化（零拷贝 dtype/shape 保留）。
    """

    _HEADER = struct.Struct(">I")  # big-endian uint32

    @classmethod
    def send(cls, sock: socket.socket, payload: dict) -> None:
        """将 payload 序列化后通过 socket 发送。

        Args:
            sock: 已连接的 TCP socket。
            payload: 待发送的 dict（可含 numpy 数组）。
        """
        body = msgpack_numpy.packb(payload, use_bin_type=True)
        header = cls._HEADER.pack(len(body))
        # 合并发送减少系统调用
        sock.sendall(header + body)

    @classmethod
    def recv(cls, sock: socket.socket) -> dict:
        """从 socket 接收并反序列化一个完整帧。

        Args:
            sock: 已连接的 TCP socket。

        Returns:
            反序列化后的 dict。

        Raises:
            ConnectionError: 连接断开（recv 返回空）。
        """
        # 读取 4 字节长度头
        header_data = cls._recv_exactly(sock, cls._HEADER.size)
        (body_len,) = cls._HEADER.unpack(header_data)

        # 读取 body
        body = cls._recv_exactly(sock, body_len)
        return msgpack_numpy.unpackb(body, raw=False)

    @staticmethod
    def _recv_exactly(sock: socket.socket, n: int) -> bytes:
        """从 socket 精确读取 n 字节。

        Args:
            sock: 已连接的 TCP socket。
            n: 要读取的字节数。

        Returns:
            恰好 n 字节的数据。

        Raises:
            ConnectionError: 连接断开或数据不足。
        """
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError(
                    f"Socket 连接断开：预期 {n} 字节，已收到 {len(data)} 字节"
                )
            data.extend(chunk)
        return bytes(data)
