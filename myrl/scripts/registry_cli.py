#!/usr/bin/env python3
"""Experiment Registry CLI。

用法：
    python scripts/registry_cli.py list [--experiment g1_native]
    python scripts/registry_cli.py show <run_id>
    python scripts/registry_cli.py verify <run_id>
    python scripts/registry_cli.py import <log_dir>
"""
import argparse
import os
import sys

# 确保 myrl 可导入
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "..", "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

from myrl.registry import RunManifest, RunRegistry


def cmd_list(args):
    reg = RunRegistry()
    runs = reg.list_runs(experiment_name=args.experiment)
    if not runs:
        print("(no runs found)")
        return
    print(f"{'Run ID':<40} {'Experiment':<20} {'Iter':>6} {'Created'}")
    print("-" * 90)
    for m in runs:
        print(f"{m.run_id:<40} {m.experiment_name:<20} {m.checkpoint_iteration:>6}  {m.created_at[:19]}")


def cmd_show(args):
    reg = RunRegistry()
    try:
        m = reg.load(args.run_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(m.to_yaml())


def cmd_verify(args):
    reg = RunRegistry()
    try:
        m = reg.load(args.run_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = reg.verify(m)
    if not results:
        print("(nothing to verify — no checksummed assets or checkpoint)")
        return

    all_pass = True
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {key}")
        if not passed:
            all_pass = False

    if all_pass:
        print("[OK] All checksums verified.")
    else:
        print("[FAIL] Some checksums do not match.")
        sys.exit(1)


def cmd_import(args):
    """从旧 log_dir 补录 manifest（需 metrics.jsonl 存在）。"""
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        print(f"Error: '{log_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # 最小化 manifest：只读 metrics.jsonl + checkpoint
    import json
    import yaml

    metrics = {}
    jsonl = os.path.join(log_dir, "metrics.jsonl")
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            last = None
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last:
            for key in ("Loss/surrogate_loss", "Loss/value_loss", "Train/mean_reward"):
                if key in last:
                    metrics[key.split("/")[-1]] = last[key]

    manifest = RunManifest(
        experiment_name=os.path.basename(os.path.dirname(log_dir)),
        created_at=os.path.basename(log_dir),
        metrics_snapshot=metrics,
    )
    from datetime import datetime, timezone
    manifest.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_imported"

    reg = RunRegistry()
    run_id = reg.save(manifest)
    print(f"Imported as run_id: {run_id}")


def main():
    parser = argparse.ArgumentParser(description="myrl Experiment Registry CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出所有运行")
    p_list.add_argument("--experiment", default=None, help="按实验名过滤")

    p_show = sub.add_parser("show", help="显示运行详情")
    p_show.add_argument("run_id")

    p_verify = sub.add_parser("verify", help="校验 checksum")
    p_verify.add_argument("run_id")

    p_import = sub.add_parser("import", help="补录旧 log_dir")
    p_import.add_argument("log_dir")

    args = parser.parse_args()
    {"list": cmd_list, "show": cmd_show, "verify": cmd_verify, "import": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
