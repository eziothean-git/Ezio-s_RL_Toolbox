#!/usr/bin/env python3
"""myrl Asset Manager TUI — 类 Unity Project 窗口的终端资产管理器。

用法：
    python scripts/asset_tui.py [--experiments-dir myrl/assets/experiments]

依赖：
    pip install textual>=0.47
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# 将 myrl src 加入 path（不依赖 pip install）
_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ── 资产状态检查 ──────────────────────────────────────────────────────────────

def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "CLAUDE.md").exists():
            return p
    return start


def check_source(source: str, repo_root: Path) -> tuple[bool, Path]:
    """返回 (exists, abs_path)。"""
    p = (repo_root / source).resolve()
    return p.exists(), p


def load_experiment(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def save_experiment(yaml_path: Path, cfg: dict) -> None:
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)


def collect_asset_rows(cfg: dict, repo_root: Path) -> list[dict]:
    """将 experiment cfg 的 assets 展开为可显示的行列表。"""
    rows: list[dict] = []
    assets = cfg.get("assets", {})

    def _row(atype: str, name: str, version: str, source: str, key_path: list) -> dict:
        exists, abs_p = check_source(source, repo_root)
        return {
            "type": atype,
            "name": name,
            "version": version,
            "source": source,
            "exists": exists,
            "key_path": key_path,   # 用于定位回 cfg dict 的键路径
        }

    scalar_types = [
        ("robot_model",    "robot_model"),
        ("actuator_cfg",   "actuator_cfg"),
        ("sensor_cfg",     "sensor_cfg"),
        ("terrain",        "terrain"),
        ("reward_pipeline","reward_pipeline"),
        ("obs_pipeline",   "obs_pipeline"),
        ("algo_cfg",       "algo_cfg"),
        ("env_script",     "env_script"),
    ]
    for key, atype in scalar_types:
        if key in assets:
            a = assets[key]
            rows.append(_row(atype, a.get("name","?"), a.get("version","?"),
                             a.get("source",""), ["assets", key, "source"]))

    for i, a in enumerate(assets.get("reward_fns", [])):
        rows.append(_row("reward_fn", a.get("name","?"), a.get("version","?"),
                         a.get("source",""), ["assets", "reward_fns", i, "source"]))

    return rows


def readiness(rows: list[dict]) -> tuple[int, int]:
    ok = sum(1 for r in rows if r["exists"])
    return ok, len(rows)


# ── Textual TUI ───────────────────────────────────────────────────────────────

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import (
        Footer, Header, Input, Label, ListItem, ListView, Static
    )
    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False


if _HAS_TEXTUAL:

    ASSET_TYPE_ABBR = {
        "robot_model":    "robot  ",
        "actuator_cfg":   "actcfg ",
        "sensor_cfg":     "sencfg ",
        "terrain":        "terrain",
        "reward_fn":      "rew_fn ",
        "reward_pipeline":"rewpipe",
        "obs_pipeline":   "obspipe",
        "algo_cfg":       "algocfg",
        "env_script":     "envscpt",
    }

    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: horizontal;
        height: 1fr;
    }
    #left {
        width: 30;
        border: solid $primary;
        padding: 0 1;
    }
    #right {
        width: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    #exp-list {
        height: 1fr;
    }
    #detail-panel {
        height: 1fr;
        overflow-y: auto;
    }
    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    .asset-row {
        layout: horizontal;
    }
    .asset-exists {
        color: $success;
    }
    .asset-missing {
        color: $error;
    }
    .asset-selected {
        background: $accent 20%;
    }
    #source-input {
        dock: bottom;
        height: 3;
        display: none;
    }
    #pack-input {
        dock: bottom;
        height: 3;
        display: none;
    }
    """

    class AssetManagerApp(App):
        CSS = CSS
        BINDINGS = [
            Binding("q", "quit", "退出"),
            Binding("n", "new_experiment", "新建"),
            Binding("p", "pack", "打包"),
            Binding("e", "edit_source", "编辑 source"),
            Binding("r", "refresh", "刷新"),
            Binding("o", "open_editor", "编辑 YAML"),
            Binding("d", "deploy", "部署"),
            Binding("tab", "focus_next", "切换面板"),
        ]

        # 当前选中实验
        selected_exp: reactive[Path | None] = reactive(None)
        # 当前选中资产行 index
        selected_row: reactive[int] = reactive(0)
        # 所有实验 yaml 路径
        exp_paths: list[Path] = []
        # 当前实验的资产行
        asset_rows: list[dict] = []
        # repo root
        repo_root: Path = Path.cwd()

        def __init__(self, experiments_dir: Path, **kw):
            super().__init__(**kw)
            self.experiments_dir = experiments_dir
            self.repo_root = _find_repo_root(experiments_dir)

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Label("[b]EXPERIMENTS[/b]")
                    yield ListView(id="exp-list")
                with Vertical(id="right"):
                    yield Label("[b]Assets[/b]", id="exp-title")
                    yield Static("", id="detail-panel")
                    yield Input(placeholder="New source path... (Enter=save, Esc=cancel)",
                                id="source-input")
                    yield Input(placeholder="Output dir for package... (Enter=pack, Esc=cancel)",
                                id="pack-input")
            yield Static("", id="status-bar")
            yield Footer()

        def on_mount(self) -> None:
            self._reload_experiments()

        def _reload_experiments(self) -> None:
            self.exp_paths = sorted(self.experiments_dir.glob("*.yaml"))
            lv = self.query_one("#exp-list", ListView)
            lv.clear()
            for p in self.exp_paths:
                cfg = load_experiment(p)
                rows = collect_asset_rows(cfg, self.repo_root)
                ok, total = readiness(rows)
                bullet = "●" if ok == total else "○"
                lv.append(ListItem(Label(f"{bullet} {p.stem}  {ok}/{total}")))
            if self.exp_paths:
                lv.index = 0
                self.selected_exp = self.exp_paths[0]
                self._refresh_detail()

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            lv = self.query_one("#exp-list", ListView)
            idx = lv.index or 0
            if 0 <= idx < len(self.exp_paths):
                self.selected_exp = self.exp_paths[idx]
                self.selected_row = 0
                self._refresh_detail()

        def _refresh_detail(self) -> None:
            if not self.selected_exp:
                return
            cfg = load_experiment(self.selected_exp)
            self.asset_rows = collect_asset_rows(cfg, self.repo_root)
            ok, total = readiness(self.asset_rows)

            title = self.query_one("#exp-title", Label)
            title.update(
                f"[b]{cfg.get('name','?')}[/b]  v{cfg.get('version','?')}\n"
                f"{cfg.get('description','')}"
            )

            lines: list[str] = [
                f"{'TYPE':<9}  {'NAME':<20}  {'VER':<8}  STATUS  SOURCE",
                "─" * 80,
            ]
            for i, row in enumerate(self.asset_rows):
                abbr = ASSET_TYPE_ABBR.get(row["type"], row["type"][:7])
                status = "[green]✓ ok    [/green]" if row["exists"] else "[red]✗ miss  [/red]"
                src_short = row["source"][-38:] if len(row["source"]) > 38 else row["source"]
                prefix = "▶ " if i == self.selected_row else "  "
                lines.append(
                    f"{prefix}{abbr}  {row['name'][:20]:<20}  {row['version'][:8]:<8}  "
                    f"{status}  {src_short}"
                )

            lines += ["─" * 80, f"{ok}/{total} ready   [e] edit source  [p] pack  [o] open YAML"]
            self.query_one("#detail-panel", Static).update("\n".join(lines))
            self.query_one("#status-bar", Static).update(
                f"  {self.selected_exp.name}  |  repo: {self.repo_root}"
            )

        # ── 键位动作 ────────────────────────────────────────────────────────

        def action_refresh(self) -> None:
            self._reload_experiments()

        def action_edit_source(self) -> None:
            if not self.asset_rows:
                return
            row = self.asset_rows[self.selected_row]
            inp = self.query_one("#source-input", Input)
            inp.value = row["source"]
            inp.display = True
            inp.focus()

        def action_pack(self) -> None:
            inp = self.query_one("#pack-input", Input)
            inp.value = "/tmp/"
            inp.display = True
            inp.focus()

        def action_new_experiment(self) -> None:
            self._show_status("新建实验：请手动创建 YAML 后按 r 刷新")

        def action_open_editor(self) -> None:
            if not self.selected_exp:
                return
            editor = os.environ.get("EDITOR", "nano")
            # 在后台打开（TUI 暂停）
            self.suspend()
            subprocess.run([editor, str(self.selected_exp)])
            self.resume()
            self._refresh_detail()

        def action_deploy(self) -> None:
            self._show_status("部署：请用 asset_cli.py deploy <pkg> <dest> 命令")

        def on_key(self, event) -> None:
            # 左右面板焦点切换外，上下键导航资产行
            focused = self.focused
            if focused and focused.id in ("exp-list",):
                return  # 由 ListView 处理
            if event.key == "up":
                self.selected_row = max(0, self.selected_row - 1)
                self._refresh_detail()
                event.stop()
            elif event.key == "down":
                self.selected_row = min(len(self.asset_rows) - 1, self.selected_row + 1)
                self._refresh_detail()
                event.stop()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "source-input":
                self._save_source(event.value)
                event.input.display = False
                self.query_one("#detail-panel").focus()
            elif event.input.id == "pack-input":
                event.input.display = False
                self._do_pack(event.value)
                self.query_one("#detail-panel").focus()

        def on_input_changed(self, event) -> None:
            pass  # 不需要实时校验

        def _on_key_escape(self) -> None:
            for inp_id in ("#source-input", "#pack-input"):
                inp = self.query_one(inp_id, Input)
                if inp.display:
                    inp.display = False
                    self.query_one("#detail-panel").focus()
                    return

        def _save_source(self, new_source: str) -> None:
            if not self.selected_exp or not self.asset_rows:
                return
            row = self.asset_rows[self.selected_row]
            cfg = load_experiment(self.selected_exp)
            # 按 key_path 修改 cfg
            obj: Any = cfg
            for k in row["key_path"][:-1]:
                if isinstance(k, int):
                    obj = obj[k]
                else:
                    obj = obj[k]
            obj[row["key_path"][-1]] = new_source
            save_experiment(self.selected_exp, cfg)
            self._show_status(f"[OK] source 已保存: {new_source}")
            self._refresh_detail()

        def _do_pack(self, output_dir: str) -> None:
            if not self.selected_exp:
                return
            from myrl.assets.packager import PackageBuilder
            try:
                builder = PackageBuilder.from_yaml_file(
                    str(self.selected_exp), repo_root=str(self.repo_root)
                )
                pkg_path = builder.build(output_dir)
                self._show_status(f"[OK] 打包完成: {pkg_path}")
            except Exception as e:
                self._show_status(f"[ERR] 打包失败: {e}")

        def _show_status(self, msg: str) -> None:
            self.query_one("#status-bar", Static).update(f"  {msg}")


# ── 终端降级模式（无 textual 时）────────────────────────────────────────────

def _fallback_cli(experiments_dir: Path, repo_root: Path) -> None:
    """无 textual 时的简单 CLI 展示。"""
    exp_paths = sorted(experiments_dir.glob("*.yaml"))
    if not exp_paths:
        print("[WARN] No experiment YAMLs found in", experiments_dir)
        return

    print(f"\nmyrl Asset Manager (fallback CLI — install textual for TUI)")
    print(f"Repo root: {repo_root}\n")

    for yaml_path in exp_paths:
        cfg = load_experiment(yaml_path)
        rows = collect_asset_rows(cfg, repo_root)
        ok, total = readiness(rows)
        bullet = "●" if ok == total else "○"
        print(f"{bullet} {cfg.get('name','?')}  v{cfg.get('version','?')}  ({ok}/{total} ok)")
        print(f"  {yaml_path}")
        for r in rows:
            status = "✓" if r["exists"] else "✗"
            print(f"    [{status}] {r['type']:<14} {r['name']:<22} {r['source']}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="myrl Asset Manager TUI")
    parser.add_argument(
        "--experiments-dir", default=None,
        help="experiments YAML 目录（默认: <repo_root>/myrl/assets/experiments/）",
    )
    args = parser.parse_args()

    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir).resolve()
    else:
        repo_root = _find_repo_root(Path.cwd())
        exp_dir = repo_root / "myrl" / "assets" / "experiments"

    if not exp_dir.exists():
        print(f"[ERROR] experiments dir not found: {exp_dir}")
        sys.exit(1)

    repo_root = _find_repo_root(exp_dir)

    if _HAS_TEXTUAL:
        app = AssetManagerApp(experiments_dir=exp_dir)
        app.run()
    else:
        print("[WARN] textual not installed, falling back to CLI mode")
        print("       Install: pip install textual>=0.47")
        _fallback_cli(exp_dir, repo_root)


if __name__ == "__main__":
    main()
