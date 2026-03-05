#!/usr/bin/env python3
"""train_tui.py — myrl 训练管控 TUI（btop 风格）

依赖: pip install textual httpx

用法:
    python myrl/scripts/train_tui.py --host 100.x.x.x --port 7001
    python myrl/scripts/train_tui.py --mock          # 用模拟数据测试布局

键盘: [S]tart [T]erm [H]alt [R]esume [K]ill [C]heckpoint [Q]uit
"""

import argparse
import asyncio
import json
import os
import sys

try:
    import httpx
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.widgets import DataTable, Footer, Header, RichLog, Static
    from textual import work
except ImportError:
    print("缺少依赖，请运行: pip install textual httpx", file=sys.stderr)
    sys.exit(1)


# ── 自定义消息（App ↔ Worker 通信）────────────────────────────────────────────

class StreamEvent(Message):
    def __init__(self, data: dict) -> None:
        super().__init__()
        self.data = data


class ConnStatus(Message):
    def __init__(self, connected: bool, error: str = "") -> None:
        super().__init__()
        self.connected = connected
        self.error = error


# ── TUI App ────────────────────────────────────────────────────────────────────

class TrainTUI(App):
    CSS = """
    Screen { background: $surface; }

    #left {
        width: 44;
        border-right: solid $primary-darken-2;
        padding: 0;
    }
    #gpu_panel {
        height: auto;
        padding: 1 1 0 1;
    }
    #status_panel {
        height: auto;
        padding: 0 1;
        border-top: solid $primary-darken-2;
    }
    #metrics_table {
        height: 1fr;
        border-top: solid $primary-darken-2;
    }
    #console {
        height: 1fr;
        padding: 0 1;
    }
    Footer {
        background: $primary-darken-3;
    }
    """

    BINDINGS = [
        Binding("s", "start_training", "Start",      show=True),
        Binding("t", "stop_training",  "Term",        show=True),
        Binding("h", "halt_training",  "Halt",        show=True),
        Binding("r", "resume_training","Resume",      show=True),
        Binding("k", "kill_training",  "Kill",        show=True),
        Binding("c", "checkpoint",     "Ckpt",        show=True),
        Binding("q", "quit",           "Quit",        show=True),
    ]

    def __init__(self, host: str, port: int, task: str = "",
                 num_envs: int = 16, mock: bool = False) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.task = task
        self.num_envs = num_envs
        self.mock = mock
        self._base = f"http://{host}:{port}"
        self._connected = False
        self._prev_metrics: dict = {}
        self._cur_task: str = task
        self._cur_state: str = "stopped"
        self._cur_iter: int = 0
        self._tot_iter: int = 0
        self._eta_s: float = 0.0

    # ── 布局 ──────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Static("", id="gpu_panel")
                yield Static("", id="status_panel")
                yield DataTable(id="metrics_table", show_cursor=False)
            yield RichLog(id="console", highlight=True, markup=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "myrl TrainTUI"
        self.sub_title = f"{self.host}:{self.port}"
        tbl = self.query_one("#metrics_table", DataTable)
        tbl.add_columns("Metric", "Value", "Δ")
        self._refresh_status_panel()
        if self.mock:
            self._mock_worker()
        else:
            self._stream_worker()

    # ── SSE Worker ────────────────────────────────────────────────────────────

    @work(exclusive=True)
    async def _stream_worker(self) -> None:
        url = f"{self._base}/stream"
        while True:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(None, connect=5.0)
                ) as client:
                    async with client.stream("GET", url) as resp:
                        self.post_message(ConnStatus(connected=True))
                        async for line in resp.aiter_lines():
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data: "):
                                try:
                                    self.post_message(StreamEvent(json.loads(line[6:])))
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                self.post_message(ConnStatus(connected=False, error=str(e)))
                await asyncio.sleep(3.0)

    # ── Mock Worker ───────────────────────────────────────────────────────────

    @work(exclusive=True)
    async def _mock_worker(self) -> None:
        import math, random
        self.post_message(ConnStatus(connected=True))
        it = 1000
        while True:
            self.post_message(StreamEvent({
                "type": "system",
                "gpus": [
                    {"idx": 0, "util": 70 + int(10 * math.sin(it * 0.05)),
                     "mem_used": 12000, "mem_total": 40960, "temp": 72, "power": 280.0},
                    {"idx": 1, "util": 68 + int(10 * math.cos(it * 0.05)),
                     "mem_used": 11800, "mem_total": 40960, "temp": 70, "power": 275.0},
                ],
                "cpu": random.uniform(18, 32),
                "ram_used": 46000 + random.randint(-500, 500),
                "ram_total": 131072,
            }))
            self.post_message(StreamEvent({
                "type": "train",
                "iteration": it,
                "metrics": {
                    "Loss/total_loss":    round(0.025 + random.gauss(0, 0.001), 5),
                    "Loss/surrogate_loss":round(0.013 + random.gauss(0, 0.0005), 5),
                    "Loss/value_loss":    round(0.012 + random.gauss(0, 0.0005), 5),
                    "Train/mean_reward_0":round(3.0 + random.gauss(0, 0.15), 3),
                    "Train/mean_episode_length": round(850 + random.gauss(0, 30), 1),
                },
                "extras": {"tot_iter": 30000, "tot_time": (it - 1000) * 1.8,
                           "start_iter": 1000},
            }))
            self.post_message(StreamEvent({
                "type": "console",
                "line": f"[{it}] mock iteration — fps: {random.randint(43000, 47000)}",
            }))
            it += 1
            await asyncio.sleep(2.0)

    # ── 消息处理 ──────────────────────────────────────────────────────────────

    def on_conn_status(self, msg: ConnStatus) -> None:
        self._connected = msg.connected
        console = self.query_one("#console", RichLog)
        if msg.connected:
            console.write("[green]● 已连接到管控服务器[/green]")
        else:
            err = f" ({msg.error})" if msg.error else ""
            console.write(f"[yellow]⟳ 重连中...{err}[/yellow]")
        self._refresh_status_panel()

    def on_stream_event(self, msg: StreamEvent) -> None:
        d = msg.data
        t = d.get("type")
        if t == "system":
            self._update_gpu(d)
        elif t == "train":
            self._update_metrics(d)
        elif t == "console":
            line = d.get("line", "")
            if line:
                self.query_one("#console", RichLog).write(line)
        elif t == "status":
            self._cur_state = d.get("state", self._cur_state)
            self._cur_task = d.get("task") or self._cur_task
            self._refresh_status_panel()
            self.query_one("#console", RichLog).write(
                f"[dim][status → {self._cur_state}][/dim]"
            )

    # ── 面板更新 ──────────────────────────────────────────────────────────────

    def _update_gpu(self, d: dict) -> None:
        gpus = d.get("gpus", [])
        cpu = d.get("cpu", 0.0)
        ram_used = d.get("ram_used", 0)
        ram_total = d.get("ram_total", 1)
        lines = []
        for g in gpus:
            bar_u = _bar(g["util"], 100)
            bar_m = _bar(g["mem_used"], g["mem_total"])
            mu, mt = g["mem_used"] / 1024, g["mem_total"] / 1024
            lines += [
                f"[bold]GPU {g['idx']}[/bold]",
                f"  Util [{bar_u}] {g['util']}%",
                f"  VRAM [{bar_m}] {mu:.1f}/{mt:.0f}G",
                f"  Temp {g['temp']}°C   Power {g['power']:.0f}W",
                "",
            ]
        cpu_bar = _bar(int(cpu), 100)
        ram_pct = int(100 * ram_used / ram_total) if ram_total else 0
        ram_bar = _bar(ram_pct, 100)
        lines += [
            f"CPU [{cpu_bar}] {cpu:.1f}%",
            f"RAM [{ram_bar}] {ram_used // 1024:.1f}/{ram_total // 1024:.0f}G",
        ]
        self.query_one("#gpu_panel", Static).update("\n".join(lines))

    def _update_metrics(self, d: dict) -> None:
        metrics: dict = d.get("metrics", {})
        extras: dict = d.get("extras", {})
        self._cur_iter = d.get("iteration", self._cur_iter)
        self._tot_iter = extras.get("tot_iter", self._tot_iter)
        tot_time = extras.get("tot_time", 0)
        start_iter = extras.get("start_iter", 0)
        elapsed = self._cur_iter - start_iter
        self._eta_s = (
            (tot_time / elapsed * (self._tot_iter - self._cur_iter))
            if elapsed > 0 and self._tot_iter > self._cur_iter else 0.0
        )
        self._refresh_status_panel()
        # 更新指标表
        tbl = self.query_one("#metrics_table", DataTable)
        tbl.clear()
        for key, val in metrics.items():
            prev = self._prev_metrics.get(key, val)
            delta = val - prev
            if abs(delta) > 1e-7:
                delta_str = f"[green]▲{abs(delta):.4f}[/green]" if delta > 0 else \
                            f"[red]▼{abs(delta):.4f}[/red]"
            else:
                delta_str = ""
            tbl.add_row(key, f"{val:.4f}", delta_str)
        self._prev_metrics = dict(metrics)

    def _refresh_status_panel(self) -> None:
        state_color = {
            "running": "green", "halted": "yellow",
            "stopping": "red", "starting": "cyan",
        }.get(self._cur_state, "dim")
        conn = "[green]●[/green]" if self._connected else "[red]○[/red]"
        lines = [
            f"─── Training ──────────────────",
            f"Task   {self._cur_task or '—'}",
            f"State  [{state_color}]{self._cur_state}[/{state_color}]  "
            f"{self._cur_iter}/{self._tot_iter}",
            f"ETA    {_fmt_eta(self._eta_s)}   {conn}",
        ]
        self.query_one("#status_panel", Static).update("\n".join(lines))

    # ── 动作 ──────────────────────────────────────────────────────────────────

    def action_start_training(self) -> None:
        if not self.task:
            self.query_one("#console", RichLog).write(
                "[red]请指定 --task 参数后再按 [S][/red]"
            )
            return
        self._api_post("start", {"task": self.task, "num_envs": self.num_envs})

    def action_stop_training(self) -> None:
        self._api_post("stop")

    def action_halt_training(self) -> None:
        self._api_post("halt")

    def action_resume_training(self) -> None:
        self._api_post("resume")

    def action_kill_training(self) -> None:
        self._api_post("kill")

    def action_checkpoint(self) -> None:
        self._api_post("checkpoint")

    def _api_post(self, cmd: str, body: dict = None) -> None:
        self.run_worker(self._do_post(cmd, body or {}), exclusive=False)

    @work
    async def _do_post(self, cmd: str, body: dict) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(f"{self._base}/{cmd}", json=body)
        except Exception as e:
            self.query_one("#console", RichLog).write(
                f"[red]命令 /{cmd} 失败: {e}[/red]"
            )


# ── 辅助 ───────────────────────────────────────────────────────────────────────

def _bar(val: int, total: int, width: int = 10) -> str:
    filled = max(0, min(width, int(width * val / total))) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _fmt_eta(eta_s: float) -> str:
    if eta_s <= 0:
        return "—"
    h, r = divmod(int(eta_s), 3600)
    return f"{h}h{r // 60:02d}m"


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="myrl 训练管控 TUI")
    parser.add_argument("--host", default=os.environ.get("MYRL_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYRL_PORT", "7001")))
    parser.add_argument("--task", default="", help="默认启动任务（供 [S] 键使用）")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--mock", action="store_true", help="用模拟数据测试 TUI 布局（不连服务器）")
    args = parser.parse_args()
    TrainTUI(args.host, args.port, args.task, args.num_envs, args.mock).run()


if __name__ == "__main__":
    main()
