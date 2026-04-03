#!/usr/bin/env python3
"""signal_viewer — DataBus 信号实时查看器（纯终端，无外部依赖）。

通过 SignalServer 的 SSE 端点订阅 channel 数据，
以 ASCII 表格 + sparkline 实时显示。

用法（两个终端）：
    # 终端 1：启动训练（带 DataBus + SignalServer）
    MYRL_OSCILLOSCOPE=1 python scripts/train.py \\
        --task myrl/Locomotion-Flat-G1Smoke-v0 \\
        --num_envs 4 --max_iterations 100 \\
        --signal_server_port 7002

    # 终端 2：启动查看器
    python scripts/signal_viewer.py --host localhost --port 7002
    python scripts/signal_viewer.py --channels reward/total,robot/joints/pos
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import deque
from urllib.error import URLError

# ── sparkline ─────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 32) -> str:
    """生成 ASCII sparkline。"""
    if not values:
        return ""
    # 取最近 width 个值
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1.0
    return "".join(_SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in vals)


# ── SSE 读取（stdlib only） ───────────────────────────────────────────────────

def _stream_sse(host: str, port: int, channels: str, hz: float):
    """连接 /stream SSE 端点，yield 每条事件的 dict。"""
    url = f"http://{host}:{port}/stream?channels={channels}&hz={hz}"
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
    except URLError as e:
        print(f"无法连接 {url}: {e}", file=sys.stderr)
        sys.exit(1)

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if line.startswith("data: "):
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                pass


# ── 渲染 ──────────────────────────────────────────────────────────────────────

def _clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _render(data: dict, histories: dict[str, deque], term_width: int):
    """渲染一帧。"""
    _clear()

    # 表头
    hdr = f"{'Channel':<36} {'Shape':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>8}"
    print(f"\033[1m{hdr}\033[0m")
    print("─" * min(len(hdr), term_width))

    # 表格
    for ch, stats in sorted(data.items()):
        shape_str = str(stats.get("shape", "?"))
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        mn = stats.get("min", 0.0)
        mx = stats.get("max", 0.0)
        count = stats.get("count", 0)

        # 更新历史
        if ch not in histories:
            histories[ch] = deque(maxlen=64)
        histories[ch].append(mean)

        print(f"{ch:<36} {shape_str:<12} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {mx:>10.4f} {count:>8}")

    # sparkline 区域
    print()
    spark_width = min(64, term_width - 4)
    for ch, hist in sorted(histories.items()):
        if len(hist) < 2:
            continue
        spark = _sparkline(list(hist), spark_width)
        # 截断 channel 名到 30 字符
        label = ch if len(ch) <= 30 else "..." + ch[-27:]
        print(f"  {spark}  {label}")

    print()
    print(f"\033[2m[Ctrl+C 退出] 连接: {len(data)} channels\033[0m")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DataBus 信号实时查看器")
    parser.add_argument("--host", default="localhost", help="SignalServer 地址")
    parser.add_argument("--port", type=int, default=7002, help="SignalServer 端口")
    parser.add_argument("--channels", default="",
                        help="要订阅的 channel（逗号分隔，空=全部）")
    parser.add_argument("--hz", type=float, default=10, help="刷新频率 (Hz)")
    args = parser.parse_args()

    # health check
    try:
        resp = urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=3)
        if resp.status != 200:
            raise Exception(f"status={resp.status}")
    except Exception as e:
        print(f"SignalServer 不可达 ({args.host}:{args.port}): {e}", file=sys.stderr)
        print("请确认训练进程已启动 --signal_server_port", file=sys.stderr)
        sys.exit(1)

    # 获取终端宽度
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 100

    # 列出可用 channels
    if not args.channels:
        try:
            resp = urllib.request.urlopen(
                f"http://{args.host}:{args.port}/channels", timeout=3)
            all_channels = json.loads(resp.read().decode())
            args.channels = ",".join(all_channels)
            if not args.channels:
                print("暂无 channel（训练可能还未开始 step）", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"获取 channel 列表失败: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"连接 {args.host}:{args.port}，订阅: {args.channels}")
    print("等待数据...")

    histories: dict[str, deque] = {}

    try:
        for data in _stream_sse(args.host, args.port, args.channels, args.hz):
            if data:
                _render(data, histories, term_width)
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
