"""ObsHistoryManager — 环境无关的观测历史 ring buffer。

职责：
- 按 term 粒度维护历史观测（deque maxlen=H_i）
- push() 输入当前帧 {group: Tensor[N, flat_dim]}，输出历史展开后的 obs_pack
- reset() 对 done 的 env 清零 history，防止跨 episode 污染
- get_output_format() 返回含历史展开后的 obs_format，供 policy 构造使用

设计约束：
- 纯 torch，无仿真依赖
- 每 term 独立 deque，history_cfg 精细到 term 粒度
- 初始化：全零填充到 maxlen（冷启动安全）
- 归属于模型层：环境只推送当前帧，历史由此类维护
"""

from __future__ import annotations

from collections import deque
from typing import Union

import torch
from torch import Tensor


# history_cfg 格式：
#   {group: int}  — 该 group 所有 term 均使用相同 history length
#   {group: {term: int}}  — 精细到 term 粒度
HistoryCfg = dict[str, Union[int, dict[str, int]]]


class ObsHistoryManager:
    """环境无关的观测历史 ring buffer。

    Args:
        obs_format: 与 instinct_rl 约定相同的格式描述，形如
            ``{"policy": {"base_ang_vel": (24,), ...}, ...}``。
            注意：此处的 shape 是**当前帧**（无历史）的展平维度。
        history_cfg: 每 group / term 的历史长度配置，支持两种格式：
            - ``{"policy": 8}`` — policy group 所有 term 均保留 8 帧历史
            - ``{"policy": {"base_ang_vel": 8, "joint_pos": 1}}`` — 精细配置
            未出现在 history_cfg 中的 group/term 默认 history_length=1（只保留当前帧）。
        num_envs: 并行环境数量。
        device: torch 设备（"cpu" 或 "cuda"）。
    """

    def __init__(
        self,
        obs_format: dict[str, dict[str, tuple]],
        history_cfg: HistoryCfg,
        num_envs: int,
        device: str = "cpu",
    ) -> None:
        self.obs_format = obs_format
        self.num_envs = num_envs
        self.device = torch.device(device)

        # 解析每个 (group, term) 的 history length 和 term flat dim
        # _term_meta[(group, term)] = (history_len, flat_dim)
        self._term_meta: dict[tuple[str, str], tuple[int, int]] = {}
        for group, terms in obs_format.items():
            group_cfg = history_cfg.get(group, 1)
            for term, shape in terms.items():
                if isinstance(group_cfg, int):
                    h = group_cfg
                else:
                    h = group_cfg.get(term, 1)
                flat_dim = int(shape[0]) if len(shape) == 1 else int(shape[0])
                self._term_meta[(group, term)] = (h, flat_dim)

        # 初始化 deque，每 term 一个，元素为 Tensor[num_envs, flat_dim]
        # deque maxlen=h，全零初始化（冷启动安全）
        self._buffers: dict[tuple[str, str], deque] = {}
        for (group, term), (h, flat_dim) in self._term_meta.items():
            zero = torch.zeros(num_envs, flat_dim, device=self.device)
            buf: deque = deque(maxlen=h)
            for _ in range(h):
                buf.append(zero.clone())
            self._buffers[(group, term)] = buf

        # 预计算每个 group 的输出 dim（用于 get_output_format）
        self._output_format: dict[str, dict[str, tuple]] = {}
        for group, terms in obs_format.items():
            self._output_format[group] = {}
            for term, shape in terms.items():
                h, flat_dim = self._term_meta[(group, term)]
                self._output_format[group][term] = (h * flat_dim,)

    # ── 公共 API ──────────────────────────────────────────────────────────

    def push(self, obs_pack: dict[str, Tensor]) -> dict[str, Tensor]:
        """将当前帧 obs 入队，返回历史展开后的 obs_pack。

        Args:
            obs_pack: 当前帧 ``{group: Tensor[N, flat_dim]}``。
                      flat_dim 必须等于 sum(obs_format[group][term][0] for term)。

        Returns:
            历史展开后的 obs_pack ``{group: Tensor[N, sum(H_i * term_i_dim)]}``。
        """
        result: dict[str, Tensor] = {}

        for group, terms in self.obs_format.items():
            if group not in obs_pack:
                continue

            group_tensor = obs_pack[group]  # [N, group_flat_dim]
            if group_tensor.device != self.device:
                group_tensor = group_tensor.to(self.device)

            # 按 obs_format 顺序 split 出各 term
            term_dims = [terms[t][0] for t in terms]
            term_tensors = torch.split(group_tensor, term_dims, dim=-1)

            out_parts: list[Tensor] = []
            for term, t_tensor in zip(terms.keys(), term_tensors):
                buf = self._buffers[(group, term)]
                buf.append(t_tensor)  # deque 自动滚动（oldest 出队）

                # 按时序 oldest→newest concat
                # buf 中 index 0 是最旧的，-1 是最新的
                h_tensors = list(buf)  # len == maxlen
                # 若 deque 未满（理论上不会，因为初始化已填满），用零补
                # 此处直接 cat 所有元素
                out_parts.append(torch.cat(h_tensors, dim=-1))  # [N, H * flat_dim]

            result[group] = torch.cat(out_parts, dim=-1)  # [N, sum(H_i * dim_i)]

        return result

    def reset(self, env_ids: list[int] | None = None) -> None:
        """清零指定 env 的历史，防止跨 episode 污染。

        Args:
            env_ids: 需要清零的 env 索引列表。
                     若为 None，清零所有 env。
        """
        if env_ids is None:
            # 重置所有 env：直接重建全零 deque
            for (group, term), (h, flat_dim) in self._term_meta.items():
                zero = torch.zeros(self.num_envs, flat_dim, device=self.device)
                buf: deque = deque(maxlen=h)
                for _ in range(h):
                    buf.append(zero.clone())
                self._buffers[(group, term)] = buf
        else:
            # 只清零指定 env 的历史帧
            env_ids_list = list(env_ids)
            for (group, term), (h, flat_dim) in self._term_meta.items():
                buf = self._buffers[(group, term)]
                for frame in buf:
                    frame[env_ids_list] = 0.0

    def get_output_format(self) -> dict[str, dict[str, tuple]]:
        """返回历史展开后的 obs_format，用于 policy 网络构造。

        Returns:
            ``{"policy": {"base_ang_vel": (H * 3,), ...}, ...}``
        """
        return self._output_format
