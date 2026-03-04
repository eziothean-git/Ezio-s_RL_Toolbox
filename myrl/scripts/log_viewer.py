"""myrl log viewer — 远程实时查看训练日志。

连接 SSE log server，格式化输出训练指标。

用法：
    python scripts/log_viewer.py --host localhost --port 7000
    python scripts/log_viewer.py --host <ip> --port 7000 --metrics Loss,reward
    python scripts/log_viewer.py --host <ip> --port 7000 --history 50 --no_stream

或直接用 curl：
    curl http://<ip>:7000/metrics
    curl http://<ip>:7000/stream
    curl "http://<ip>:7000/history?n=10"

与 textual TUI 的接口对齐：
    SSEClient.stream() 产生标准 dict，TUI 版本直接在 worker 中 for 循环即可。
    format_event_text() 是纯函数，TUI 版本替换为 Rich 渲染而无需改变数据结构。
"""
import argparse
import os
import sys

# 支持直接运行（不需要 pip install myrl）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myrl.logging.server.log_client import SSEClient, format_event_text


def main():
    parser = argparse.ArgumentParser(
        description="远程查看 myrl 训练日志（SSE 实时流）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 查看所有指标（实时）
  python scripts/log_viewer.py --host localhost --port 7000

  # 只看 loss 和 reward
  python scripts/log_viewer.py --host 192.168.1.10 --port 7000 --metrics Loss,reward

  # 先拉 50 条历史再实时跟踪
  python scripts/log_viewer.py --host localhost --port 7000 --history 50

  # 只查看当前指标快照（不订阅实时）
  python scripts/log_viewer.py --host localhost --port 7000 --no_stream
        """,
    )
    parser.add_argument("--host", default="localhost", help="服务端 IP 或主机名")
    parser.add_argument("--port", type=int, default=7000, help="服务端端口")
    parser.add_argument(
        "--metrics",
        default=None,
        help="只显示包含这些子串的指标（逗号分隔），例如：Loss,reward,fps",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="启动时先拉取并显示最近 N 条历史（0=不拉取）",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        default=False,
        help="只拉取历史/当前指标，不订阅实时流",
    )
    parser.add_argument("--width", type=int, default=80, help="输出宽度")
    parser.add_argument("--pad", type=int, default=35, help="左对齐填充")
    args = parser.parse_args()

    filter_keys = [k.strip() for k in args.metrics.split(",")] if args.metrics else None

    client = SSEClient(
        host=args.host,
        port=args.port,
        timeout=60.0,
        retry_delay=3.0,
        max_retries=-1,
    )

    # 健康检查
    if not client.health_check():
        print(f"[ERROR] 无法连接到 {args.host}:{args.port}，请确认 --log_server_port 已启动")
        sys.exit(1)

    # 拉取历史
    if args.history > 0:
        print(f"[INFO] 拉取最近 {args.history} 条历史...")
        history = client.fetch_history(n=args.history)
        for event in history:
            print(format_event_text(event, filter_keys, args.width, args.pad))
            print()
        print(f"[INFO] 历史加载完毕（{len(history)} 条）\n")

    if args.no_stream:
        # 只显示当前指标快照
        metrics = client.fetch_metrics()
        if metrics:
            print("[当前指标快照]")
            for k, v in sorted(metrics.items()):
                if filter_keys is None or any(fk in k for fk in filter_keys):
                    print(f"  {k}: {v}")
        return

    # 订阅实时流
    print(f"[INFO] 连接 SSE 流: http://{args.host}:{args.port}/stream")
    print("[INFO] 按 Ctrl+C 退出\n")
    try:
        for event in client.stream():
            print(format_event_text(event, filter_keys, args.width, args.pad))
            print()
    except KeyboardInterrupt:
        print("\n[INFO] 已退出")


if __name__ == "__main__":
    main()
