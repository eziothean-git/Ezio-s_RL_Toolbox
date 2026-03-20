"""Phase B 验证脚本：比较 InstinctRlVecEnvWrapper 与 IsaacLabBackend 的训练 loss。

用法：
    python scripts/test_phase_b.py [--num_envs 4] [--iters 5] [--task TASK]
    (需在容器内或有 Isaac Lab 环境中运行)

验证标准：surrogate_loss 和 value_loss 的最大差异 < 1e-4。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def run_train(task: str, num_envs: int, iters: int, log_dir: str, use_phase_b: bool) -> str:
    """运行训练，返回 metrics.jsonl 路径。"""
    script = os.path.join(os.path.dirname(__file__), "train.py")
    cmd = [
        sys.executable, script,
        f"--task={task}",
        f"--num_envs={num_envs}",
        f"--max_iterations={iters}",
        "--seed=42",
        "--headless",
        "--no_registry",
        f"--logroot={log_dir}",
    ]
    env = os.environ.copy()
    if use_phase_b:
        env["MYRL_USE_ISAACLAB_BACKEND"] = "1"
    else:
        env.pop("MYRL_USE_ISAACLAB_BACKEND", None)

    label = "Phase B (IsaacLabBackend)" if use_phase_b else "Phase A (InstinctRlVecEnvWrapper)"
    print(f"[INFO] Running {label} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {label} failed:\n{result.stderr[-2000:]}")
        sys.exit(1)

    # 找到生成的 metrics.jsonl
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f == "metrics.jsonl":
                return os.path.join(root, f)
    raise FileNotFoundError(f"metrics.jsonl not found in {log_dir}")


def load_last_metrics(jsonl_path: str) -> dict:
    """读取最后一行 metrics。"""
    last = None
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def main():
    parser = argparse.ArgumentParser(description="Phase B 验证：比较两个 wrapper 的 loss。")
    parser.add_argument("--task", default="myrl/Locomotion-Flat-G1Smoke-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="loss 差异允许上限")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="myrl_phase_b_") as tmpdir:
        dir_a = os.path.join(tmpdir, "phase_a")
        dir_b = os.path.join(tmpdir, "phase_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        jsonl_a = run_train(args.task, args.num_envs, args.iters, dir_a, use_phase_b=False)
        jsonl_b = run_train(args.task, args.num_envs, args.iters, dir_b, use_phase_b=True)

        metrics_a = load_last_metrics(jsonl_a)
        metrics_b = load_last_metrics(jsonl_b)

    keys_to_compare = ["Loss/surrogate_loss", "Loss/value_loss"]
    max_delta = 0.0
    all_pass = True

    print("\n── Phase B Comparison ──────────────────────────────────")
    print(f"{'Metric':<35} {'Phase A':>12} {'Phase B':>12} {'|delta|':>12}")
    print("-" * 75)
    for key in keys_to_compare:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None or vb is None:
            print(f"{key:<35} {'N/A':>12} {'N/A':>12} {'MISSING':>12}")
            all_pass = False
            continue
        delta = abs(va - vb)
        max_delta = max(max_delta, delta)
        status = "OK" if delta < args.threshold else "FAIL"
        print(f"{key:<35} {va:>12.6f} {vb:>12.6f} {delta:>12.2e}  {status}")
        if status == "FAIL":
            all_pass = False

    print("-" * 75)
    if all_pass:
        print(f"[PASS] Phase B: max_delta={max_delta:.2e} (threshold={args.threshold:.2e})")
        sys.exit(0)
    else:
        print(f"[FAIL] Phase B: max_delta={max_delta:.2e} (threshold={args.threshold:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
