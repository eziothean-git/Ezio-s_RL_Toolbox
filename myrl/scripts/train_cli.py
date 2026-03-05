#!/usr/bin/env python3
"""train_cli.py — myrl 训练管控 CLI

stdlib only，无外部依赖。通过 HTTP 与 train_manager.py 通信。

用法:
    python myrl/scripts/train_cli.py [--host HOST] [--port PORT] <command>

环境变量覆盖: MYRL_HOST, MYRL_PORT
"""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_BASE = ""  # 由 main() 初始化


def _url(path: str, **params) -> str:
    u = _BASE + path
    if params:
        u += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    return u


def _get(path: str, **params) -> dict:
    try:
        with urlopen(_url(path, **params), timeout=10) as r:
            return json.loads(r.read())
    except HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"连接失败 ({_BASE}): {e.reason}", file=sys.stderr)
        sys.exit(1)


def _post(path: str, body: dict = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = Request(
        _BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"连接失败 ({_BASE}): {e.reason}", file=sys.stderr)
        sys.exit(1)


def _fmt_eta(eta_s: float) -> str:
    if eta_s <= 0:
        return "—"
    h, r = divmod(int(eta_s), 3600)
    m = r // 60
    return f"{h}h{m:02d}m"


# ── 命令 ───────────────────────────────────────────────────────────────────────

def cmd_status(_args) -> None:
    d = _get("/status")
    state = d.get("state", "?")
    pid   = d.get("pid")
    task  = d.get("task") or "—"
    cur   = d.get("iteration", 0)
    tot   = d.get("tot_iter", 0)
    eta   = _fmt_eta(d.get("eta_s", 0))

    print(f"State:   {state}")
    if pid:
        print(f"PID:     {pid}")
    print(f"Task:    {task}")
    if tot:
        print(f"Iter:    {cur} / {tot}")
    if eta != "—":
        print(f"ETA:     {eta}")
    metrics = d.get("metrics", {})
    fps = metrics.get("Perf/total_fps")
    if fps:
        print(f"FPS:     {int(fps)}")
    for g in d.get("gpus", []):
        mu, mt = g["mem_used"] / 1024, g["mem_total"] / 1024
        print(f"GPU {g['idx']}:   {g['util']}%  util  |  "
              f"{mu:.1f} / {mt:.1f} GB  |  {g['temp']}°C  |  {g['power']:.0f}W")


def cmd_start(args) -> None:
    body = {
        "task": args.task,
        "num_envs": args.num_envs,
        "extra_args": args.extra or [],
    }
    r = _post("/start", body)
    print("OK" if r.get("ok") else f"FAILED: {r.get('msg', r)}")


def cmd_stream(args) -> None:
    """tail -f 风格，逐行打印 SSE 事件。"""
    url = _url("/stream")
    if args.filter:
        url += f"&filter={args.filter}"
    print(f"[stream] {url}  (Ctrl+C 停止)", flush=True)
    try:
        with urlopen(url, timeout=None) as resp:
            for raw in resp:
                line = raw.decode("utf-8").rstrip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    try:
                        print(json.dumps(json.loads(line[6:]), ensure_ascii=False))
                    except json.JSONDecodeError:
                        print(line)
    except KeyboardInterrupt:
        pass
    except URLError as e:
        print(f"连接失败: {e.reason}", file=sys.stderr)


def cmd_console(args) -> None:
    """打印最近 N 行后持续追踪（订阅 SSE console 事件）。"""
    d = _get("/console", n=args.n)
    for line in d.get("lines", []):
        print(line)
    # 持续追踪
    url = _url("/stream") + "&filter=console"
    try:
        with urlopen(url, timeout=None) as resp:
            for raw in resp:
                line = raw.decode("utf-8").rstrip()
                if line.startswith("data: "):
                    try:
                        print(json.loads(line[6:]).get("line", ""))
                    except json.JSONDecodeError:
                        pass
    except KeyboardInterrupt:
        pass
    except URLError as e:
        print(f"连接失败: {e.reason}", file=sys.stderr)


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _BASE
    parser = argparse.ArgumentParser(description="myrl 训练管控 CLI")
    parser.add_argument("--host", default=os.environ.get("MYRL_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYRL_PORT", "7001")))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="打印当前训练状态 + GPU 快照")

    p_start = sub.add_parser("start", help="启动训练")
    p_start.add_argument("--task", required=True, help="任务 ID，例如 myrl/Locomotion-Flat-G1Smoke-v0")
    p_start.add_argument("--num_envs", type=int, default=16, help="并行环境数（默认 16）")
    p_start.add_argument("extra", nargs="*", help="传递给 train.py 的额外参数")

    _helps = {
        "stop":       "优雅停止（SIGTERM，等 checkpoint 保存后退出）",
        "kill":       "强制终止（SIGKILL）",
        "halt":       "迭代级暂停（SIGUSR1）",
        "resume":     "恢复训练（SIGUSR2）",
        "checkpoint": "保存 checkpoint 后继续训练（SIGUSR1 + 10s + SIGUSR2）",
    }
    for cmd, help_str in _helps.items():
        sub.add_parser(cmd, help=help_str)

    p_stream = sub.add_parser("stream", help="tail -f 风格 SSE 事件流")
    p_stream.add_argument("--filter", default="",
                          help="事件类型过滤：system | train | console | status")

    p_console = sub.add_parser("console", help="打印控制台输出后持续追踪")
    p_console.add_argument("--n", type=int, default=200, help="初始输出行数（默认 200）")

    args = parser.parse_args()
    _BASE = f"http://{args.host}:{args.port}"

    simple_cmds = {
        "stop":       "/stop",
        "kill":       "/kill",
        "halt":       "/halt",
        "resume":     "/resume",
        "checkpoint": "/checkpoint",
    }
    if args.command in simple_cmds:
        r = _post(simple_cmds[args.command])
        print("OK" if r.get("ok") else f"FAILED: {r}")
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "stream":
        cmd_stream(args)
    elif args.command == "console":
        cmd_console(args)


if __name__ == "__main__":
    main()
