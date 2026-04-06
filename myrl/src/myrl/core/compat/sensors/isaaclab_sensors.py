"""Isaac Lab 传感器驱动——适配 Isaac Lab 传感器对象到 myrl 协议。

每个 adapter 是一个薄包装，读取上游 sensor.data 并转换为
myrl DepthCameraProto / HeightScanProto / ForceSensorProto 格式。
"""

from __future__ import annotations

import torch
from torch import Tensor


class IsaacLabDepthCamera:
    """Isaac Lab 深度相机驱动。

    适配 RayCasterCamera / TiledCamera / GroupedRayCasterCamera /
    NoisyGroupedRayCasterCamera。

    数据源：sensor.data.output[data_type] → (N, H, W, 1) 或 (N, T, H, W, 1)。
    """

    def __init__(self, sensor, *, data_type: str = "distance_to_image_plane"):
        self._sensor = sensor
        self._data_type = data_type

    @property
    def depth(self) -> Tensor:
        """深度图 (num_envs, H, W)，inf 替换为 0。"""
        raw = self._sensor.data.output[self._data_type]  # (N, H, W, 1)
        d = raw.squeeze(-1)  # (N, H, W)
        return torch.where(torch.isinf(d), torch.zeros_like(d), d)

    @property
    def resolution(self) -> tuple[int, int]:
        """(H, W) 分辨率。"""
        shape = self._sensor.data.output[self._data_type].shape
        return (shape[1], shape[2])

    @property
    def depth_history(self) -> Tensor | None:
        """深度图历史 (num_envs, T, H, W)，无历史时返回 None。

        依次尝试：
        1. data_type + "_history" key
        2. data_histories 配置中声明的 key（如 "distance_to_image_plane_noised"）
        """
        output = self._sensor.data.output

        # 尝试标准 history key
        hist_key = self._data_type + "_history"
        if hist_key in output:
            return output[hist_key].squeeze(-1)  # (N, T, H, W, 1) → (N, T, H, W)

        # 尝试从 data_histories 配置读取
        cfg = self._sensor.cfg
        if hasattr(cfg, "data_histories") and cfg.data_histories:
            for key in cfg.data_histories:
                if key in output and output[key].ndim == 5:
                    return output[key].squeeze(-1)

        return None


class IsaacLabHeightScanner:
    """Isaac Lab 高度扫描驱动。

    适配 RayCaster (GridPattern)。
    数据源：sensor.data.ray_hits_w → (N, num_rays, 3)。
    """

    def __init__(self, sensor):
        self._sensor = sensor

    @property
    def ray_hits_w(self) -> Tensor:
        """射线击中点世界坐标 (num_envs, num_rays, 3)。"""
        return self._sensor.data.ray_hits_w

    @property
    def pos_w(self) -> Tensor:
        """传感器挂载点世界坐标 (num_envs, 3)。"""
        return self._sensor.data.pos_w


class IsaacLabForceSensor:
    """Isaac Lab 力传感器驱动。

    适配 ContactSensor 的 net_forces_w 输出。
    ContactSensor 只提供力，不提供力矩。
    """

    def __init__(self, sensor, *, body_ids: list[int] | None = None):
        self._sensor = sensor
        self._body_ids = body_ids

    @property
    def net_forces_w(self) -> Tensor:
        """接触净力 (num_envs, num_sensors, 3)。"""
        d = self._sensor.data.net_forces_w  # (N, num_bodies, 3)
        return d[:, self._body_ids] if self._body_ids is not None else d

    @property
    def net_torques_w(self) -> Tensor | None:
        """ContactSensor 无力矩输出。"""
        return None
