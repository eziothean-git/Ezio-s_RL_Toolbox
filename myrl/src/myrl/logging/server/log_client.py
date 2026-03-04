"""SSELogClient — 连接 SSE 服务端的客户端库（纯 stdlib，无额外依赖）。

设计原则：metrics 解析与展示逻辑分离，
textual TUI 版本可直接复用 SSEClient.stream() 和 parse_event()，
替换 format_event_text() 为 Rich 渲染即可。

使用示例：
    from myrl.logging.server.log_client import SSEClient, format_event_text

    client = SSEClient("localhost", 7000)
    for event in client.stream():
        print(format_event_text(event))
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterator


def parse_event(raw: str) -> dict | None:
    """解析单条 SSE 原始字符串，返回 dict 或 None（心跳/注释行）。

    Args:
        raw: SSE 原始字符串，如 "data: {...}" 或 ": heartbeat"

    Note:
        stream() 现在使用逐行解析，parse_event() 主要供外部调用者使用。
    """
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                return json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                return None
    return None


def format_event_text(
    event: dict,
    filter_keys: list[str] | None = None,
    width: int = 80,
    pad: int = 35,
) -> str:
    """将 LogEvent dict 格式化为可读文本（复用 OnPolicyRunner 的视觉风格）。

    此函数是纯函数，textual TUI 版本可直接调用或替换为 Rich 渲染。

    Args:
        event:       parse_event() 返回的 dict
        filter_keys: 只显示包含这些子串的 metric key，None 表示全部显示
        width:       输出宽度
        pad:         左对齐填充宽度
    """
    lines = [
        "#" * width,
        f" Learning iteration {event.get('iter', '?')} ".center(width),
        "",
    ]

    metrics: dict = event.get("metrics", {})
    extras: dict = event.get("extras", {})

    def _show(key: str) -> bool:
        return filter_keys is None or any(fk in key for fk in filter_keys)

    # 性能信息
    fps = metrics.get("Perf/total_fps")
    if fps is not None and _show("Perf"):
        coll = extras.get("collection_time") or 0.0
        lrn = extras.get("learn_time") or 0.0
        lines.append(
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {coll:.3f}s, learning {lrn:.3f}s)"
        )

    # Loss
    for k, v in sorted(metrics.items()):
        if k.startswith("Loss/") and _show(k):
            lines.append(f"{k.replace('Loss/', ''):>{pad}} {v:.4f}")

    # Reward / episode length
    for i in range(8):  # 最多 8 个 reward group
        key = f"Train/mean_reward_{i}"
        if key in metrics and _show(key):
            lines.append(f"{f'Mean reward {i}:':>{pad}} {metrics[key]:.2f}")
        else:
            break
    ep_len = metrics.get("Train/mean_episode_length")
    if ep_len is not None and _show("episode_length"):
        lines.append(f"{'Mean episode length:':>{pad}} {ep_len:.2f}")

    # Episode metrics（task 自定义指标）
    for k, v in sorted(metrics.items()):
        if k.startswith("Episode/") and _show(k):
            display = k.replace("Episode/", "")
            lines.append(f"{f'Mean episode {display}:':>{pad}} {v:.4f}")

    # 底部统计
    lines.append("-" * width)
    if extras.get("tot_timesteps") is not None:
        lines.append(f"{'Total timesteps:':>{pad}} {extras['tot_timesteps']}")
    tot_time = extras.get("tot_time")
    tot_iter = extras.get("tot_iter")
    start_iter = extras.get("start_iter", 0)
    cur_iter = event.get("iter", 0)
    if tot_time is not None:
        lines.append(f"{'Total time:':>{pad}} {tot_time:.2f}s")
    if tot_time and tot_iter and cur_iter > start_iter:
        eta = tot_time / (cur_iter + 1 - start_iter) * (tot_iter - cur_iter)
        lines.append(f"{'ETA:':>{pad}} {eta:.1f}s")

    return "\n".join(lines)


class SSEClient:
    """SSE 流客户端（stdlib only），可迭代，供 CLI 和 textual TUI 复用。

    Args:
        host:        服务端主机名或 IP
        port:        服务端端口
        timeout:     连接超时（秒）
        retry_delay: 断连后重试间隔（秒）
        max_retries: 最大重试次数，-1 为无限
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7000,
        timeout: float = 60.0,
        retry_delay: float = 2.0,
        max_retries: int = -1,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries

    def stream(self) -> Iterator[dict]:
        """迭代 SSE 事件流，自动重连。

        使用逐行读取（readline），每收到 \\n 立即处理，
        不等待缓冲区满，适合 SSE 长连接。

        Yields:
            parse_event() 解析后的 LogEvent dict
        """
        retries = 0
        while self.max_retries < 0 or retries <= self.max_retries:
            try:
                req = urllib.request.Request(
                    self.base_url + "/stream",
                    headers={
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    retries = 0
                    event_lines: list[str] = []
                    while True:
                        raw = resp.readline()
                        if not raw:
                            break  # 服务端关闭连接
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if line.startswith(":"):
                            continue  # heartbeat comment
                        if not line:
                            # 空行 = 事件分隔符，处理积累的 data 行
                            for l in event_lines:
                                if l.startswith("data:"):
                                    try:
                                        yield json.loads(l[len("data:"):].strip())
                                    except json.JSONDecodeError:
                                        pass
                            event_lines = []
                        else:
                            event_lines.append(line)
            except (OSError, urllib.error.URLError) as e:
                retries += 1
                print(f"[连接断开] {e}，{self.retry_delay:.0f}s 后重试 ({retries})...")
                time.sleep(self.retry_delay)

    def fetch_history(self, n: int = 100) -> list[dict]:
        """同步拉取最近 N 条历史事件。"""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/history?n={n}", timeout=10
            ) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"[WARN] fetch_history 失败: {e}")
            return []

    def fetch_metrics(self) -> dict[str, float]:
        """同步拉取最新指标快照。"""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/metrics", timeout=10
            ) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"[WARN] fetch_metrics 失败: {e}")
            return {}

    def health_check(self) -> bool:
        """检查服务端是否可达。"""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=3
            ) as resp:
                return json.loads(resp.read().decode()).get("status") == "ok"
        except Exception:
            return False
