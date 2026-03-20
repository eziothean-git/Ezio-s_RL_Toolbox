#!/usr/bin/env python3
"""reward_inspector：实时显示每个 reward term 的均值（via SSE log server）。

用法（两个终端）：
    # 终端 1：启动训练
    python scripts/train.py --task myrl/Locomotion-Flat-G1Native-v0 \\
        --num_envs 4096 --headless --log_server_port 7000

    # 终端 2：启动 inspector
    python scripts/reward_inspector.py --host localhost --port 7000
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import OrderedDict
from urllib.error import URLError


def _fetch_metrics(host: str, port: int) -> dict | None:
    """从 /metrics 端点拉取当前指标快照。"""
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _format_table(metrics: dict, weights: dict[str, float] | None = None) -> list[str]:
    """构造文本表格行。"""
    # 筛选 rew/ 前缀的 term
    rew_keys = sorted(k for k in metrics if k.startswith("Episode/rew/"))
    total_key = "Train/mean_reward"
    iteration = metrics.get("Train/iteration", 0)
    max_iter = metrics.get("Train/max_iterations", 0)

    col_w = [36, 8, 10, 10]
    sep = "─" * (sum(col_w) + 3 * 3 + 2)

    lines = [
        "┌" + "─" * (sum(col_w) + 3 * 3 + 2) + "┐",
        f"│ {'Term':<{col_w[0]}} │ {'Weight':>{col_w[1]}} │ {'Raw Mean':>{col_w[2]}} │ {'Wtd Mean':>{col_w[3]}} │",
        "├" + "─" * (sum(col_w) + 3 * 3 + 2) + "┤",
    ]

    total_wtd = 0.0
    for key in rew_keys:
        name = key.replace("Episode/rew/", "")
        raw = metrics[key]
        w = (weights or {}).get(name, float("nan"))
        wtd = raw * w if not (isinstance(w, float) and w != w) else float("nan")
        if not (isinstance(wtd, float) and wtd != wtd):
            total_wtd += wtd
        w_str = f"{w:+.3f}" if not (isinstance(w, float) and w != w) else "  N/A"
        wtd_str = f"{wtd:+.4f}" if not (isinstance(wtd, float) and wtd != wtd) else "  N/A"
        lines.append(
            f"│ {name:<{col_w[0]}} │ {w_str:>{col_w[1]}} │ {raw:>{col_w[2]}.4f} │ {wtd_str:>{col_w[3]}} │"
        )

    lines.append("├" + "─" * (sum(col_w) + 3 * 3 + 2) + "┤")
    mean_rew = metrics.get(total_key, float("nan"))
    mean_str = f"{mean_rew:.4f}" if not (isinstance(mean_rew, float) and mean_rew != mean_rew) else "N/A"
    lines.append(
        f"│ {'TOTAL (mean_reward)':<{col_w[0]}} │ {'':>{col_w[1]}} │ {'':>{col_w[2]}} │ {mean_str:>{col_w[3]}} │"
    )
    lines.append("└" + "─" * (sum(col_w) + 3 * 3 + 2) + "┘")

    iter_str = f"{int(iteration)}/{int(max_iter)}" if max_iter else str(int(iteration))
    lines.append(f"  Iter: {iter_str}   Connected: ● {host}:{port}")
    return lines


def _stream_events(host: str, port: int, refresh: float):
    """订阅 SSE /stream，每次收到事件时刷新显示。"""
    url = f"http://{host}:{port}/stream"
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
        resp = urllib.request.urlopen(req, timeout=60)
    except URLError as e:
        print(f"[ERROR] Cannot connect to log server at {host}:{port}: {e}")
        return

    metrics_cache: dict = {}
    last_refresh = 0.0
    event_data_lines = []

    while True:
        try:
            raw = resp.readline()
        except Exception:
            break

        if not raw:
            break

        line = raw.decode(errors="replace").rstrip("\n\r")

        if line.startswith("data:"):
            event_data_lines.append(line[5:].strip())
        elif line == "":
            # 事件结束
            if event_data_lines:
                try:
                    payload = json.loads(" ".join(event_data_lines))
                    metrics_cache.update(payload.get("metrics", {}))
                except Exception:
                    pass
                event_data_lines = []

            now = time.monotonic()
            if now - last_refresh >= refresh:
                last_refresh = now
                lines = _format_table(metrics_cache)
                # 清屏并输出
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="myrl reward_inspector")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--refresh_interval", type=float, default=1.0,
                        help="最小刷新间隔（秒）")
    args = parser.parse_args()

    print(f"Connecting to log server at {args.host}:{args.port} ...")
    # 先拉一次快照确认连通性
    snap = _fetch_metrics(args.host, args.port)
    if snap is None:
        print(f"[WARN] /metrics 不可达，将在 SSE 流中等待数据...")

    _stream_events(args.host, args.port, args.refresh_interval)
    print("Connection closed.")


if __name__ == "__main__":
    main()
