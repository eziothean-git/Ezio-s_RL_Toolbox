"""myrl.logging — 三层日志体系。

提供 build_sinks() 工厂函数，根据 CLI 参数组装 LogSink 列表，
由 train.py 调用后挂载到 OnPolicyRunner。

架构：
    OnPolicyRunner._dispatch_log_sinks()
        → LogEvent
            → JSONLSink          （always-on，结构化文件日志）
            → WandbSink          （opt-in，extras 补充；metrics 由 sync_tensorboard 处理）
            → SSELogServer       （opt-in，HTTP SSE 实时流）
"""
from __future__ import annotations

from myrl.logging.sinks.base import LogSink, LogEvent

__all__ = ["LogSink", "LogEvent", "build_sinks"]


def build_sinks(args_cli, log_dir: str | None, run_name: str = "") -> list[LogSink]:
    """根据 CLI 参数组装 LogSink 列表。

    Args:
        args_cli:  argparse.Namespace，包含 --no_jsonl / --wandb / --log_server_port 等
        log_dir:   训练日志目录（None 时不创建文件类 sink）
        run_name:  运行名称（wandb 显示用）

    Returns:
        list[LogSink]，按顺序: JSONLSink? → WandbSink? → SSELogServer?
    """
    sinks: list[LogSink] = []

    # ── JSONL（默认开启，--no_jsonl 可关闭）────────────────────────────────
    if log_dir and not getattr(args_cli, "no_jsonl", False):
        from myrl.logging.sinks.jsonl_sink import JSONLSink
        sinks.append(JSONLSink(log_dir=log_dir))

    # ── wandb extras sink（仅在 --wandb 时挂载，metrics 由 sync_tensorboard 同步）
    if getattr(args_cli, "wandb", False):
        from myrl.logging.sinks.wandb_sink import WandbSink
        sinks.append(WandbSink(log_extras=["tot_timesteps", "tot_time", "eta_sec"]))

    # ── SSE log server（--log_server_port 指定时启动）─────────────────────
    log_server_port = getattr(args_cli, "log_server_port", None)
    if log_server_port is not None:
        from myrl.logging.server.log_server import SSELogServer
        host = getattr(args_cli, "log_server_host", "0.0.0.0")
        srv = SSELogServer(host=host, port=log_server_port)
        sinks.append(srv)
        print(f"[INFO] SSE log server 已启动: http://{host}:{log_server_port}/stream")
        print(f"[INFO] 远程查看：python scripts/log_viewer.py --host <ip> --port {log_server_port}")

    return sinks
