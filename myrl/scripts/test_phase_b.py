"""Phase B 收敛验证：比较 InstinctRlVecEnvWrapper 与 IsaacLabBackend 的训练收敛特性。

用法：
    # 默认 35 分钟对比运行（需在容器内或有 Isaac Lab 环境中运行）
    python scripts/test_phase_b.py

    # 指定时长
    python scripts/test_phase_b.py --minutes 40 --num_envs 64

    # 快速冒烟（仅验证脚本流程）
    python scripts/test_phase_b.py --iters 10

验证内容：
    1. Loss 曲线 Pearson 相关系数（默认 > 0.85）
    2. 关键指标最终检查点相对差异（默认 < 30%）
    3. 奖励收敛方向一致性（线性回归斜率符号）
    4. Per-term 奖励相对差异（< 50%）
    5. YAML 实验定义 task ID 一致性
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B 收敛验证：对比 Phase A / Phase B 训练曲线。"
    )
    p.add_argument("--task", default="myrl/Locomotion-Flat-G1Native-v0",
                   help="任务 ID")
    p.add_argument("--num_envs", type=int, default=4,
                   help="并行环境数量")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（两次运行共享）")
    p.add_argument("--minutes", type=float, default=None,
                   help="每次运行的墙钟时长（分钟），默认 35")
    p.add_argument("--iters", type=int, default=None,
                   help="替代 --minutes，指定迭代数")
    p.add_argument("--checkpoint_interval", type=int, default=500,
                   help="比较检查点间隔（迭代）")
    p.add_argument("--window", type=int, default=50,
                   help="滑动平均窗口大小")
    p.add_argument("--threshold_corr", type=float, default=0.85,
                   help="主要指标 Pearson 相关系数下限")
    p.add_argument("--threshold_rel", type=float, default=0.30,
                   help="主要指标最终检查点相对差异上限")
    p.add_argument("--threshold_term_rel", type=float, default=0.50,
                   help="Per-term 奖励相对差异上限")
    p.add_argument("--reverse", action="store_true", default=False,
                   help="先跑 Phase B 再跑 Phase A（用于排除顺序偏差）")
    p.add_argument("--output", type=str, default=None,
                   help="保存对比结果 JSON 的路径")
    p.add_argument("--yaml_config", type=str, default=None,
                   help="YAML 实验定义路径（用于一致性验证）")
    return p.parse_args()


# ── 训练执行 ─────────────────────────────────────────────────────────────────


def run_train(
    task: str,
    num_envs: int,
    seed: int,
    max_iters: int,
    log_dir: str,
    use_phase_b: bool,
    timeout_sec: float | None = None,
) -> str:
    """运行训练，返回 metrics.jsonl 路径。

    当 timeout_sec 不为 None 时，使用 SIGTERM 定时停止。
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    cmd = [
        sys.executable, script,
        f"--task={task}",
        f"--num_envs={num_envs}",
        f"--max_iterations={max_iters}",
        f"--seed={seed}",
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
    print(f"\n[INFO] ── {label} ──")
    print(f"[INFO] Task: {task} | envs: {num_envs} | seed: {seed} | max_iters: {max_iters}")
    if timeout_sec:
        print(f"[INFO] 时间模式：{timeout_sec / 60:.1f} 分钟后 SIGTERM 停止")

    t0 = time.time()

    if timeout_sec is not None:
        # 时间模式：Popen + SIGTERM 定时器
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def _stop():
            if proc.poll() is None:
                print(f"[INFO] 时间到，发送 SIGTERM → PID {proc.pid}")
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except OSError:
                    pass

        timer = threading.Timer(timeout_sec, _stop)
        timer.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec + 60)
        except subprocess.TimeoutExpired:
            # SIGTERM 后 60s 仍未退出，强制 SIGKILL
            print(f"[WARN] SIGTERM 后 60s 未退出，发送 SIGKILL")
            proc.kill()
            stdout, stderr = proc.communicate()
        finally:
            timer.cancel()

        returncode = proc.returncode
    else:
        # 迭代模式：直接 subprocess.run
        result = subprocess.run(cmd, env=env, capture_output=True, text=False)
        stdout, stderr = result.stdout, result.stderr
        returncode = result.returncode

    elapsed = time.time() - t0
    print(f"[INFO] {label} 完成：{elapsed:.1f}s，exit_code={returncode}")

    # SIGTERM 优雅退出时 returncode 可能非零（-15 或 143）
    if returncode not in (0, -15, -signal.SIGTERM, 128 + signal.SIGTERM):
        stderr_text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        print(f"[ERROR] {label} 失败 (code={returncode}):\n{(stderr_text or '')[-2000:]}")
        sys.exit(1)

    # 查找 metrics.jsonl
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f == "metrics.jsonl":
                return os.path.join(root, f)

    raise FileNotFoundError(f"metrics.jsonl not found in {log_dir}")


# ── 指标加载与提取 ───────────────────────────────────────────────────────────


def load_metrics(jsonl_path: str) -> list[dict]:
    """解析 metrics.jsonl 为 list[dict]。"""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_series(records: list[dict], key: str) -> list[tuple[int, float]]:
    """提取单指标时序 [(iter, value), ...]，跳过缺失的迭代。"""
    series = []
    for r in records:
        it = r.get("iter", 0)
        metrics = r.get("metrics", {})
        if key in metrics:
            val = metrics[key]
            if val is not None and math.isfinite(val):
                series.append((it, val))
    return series


def discover_reward_keys(records: list[dict]) -> list[str]:
    """从 metrics 中发现所有 Episode/rew/ 开头的 key。"""
    keys = set()
    for r in records:
        for k in r.get("metrics", {}):
            if k.startswith("Episode/rew/"):
                keys.add(k)
    return sorted(keys)


# ── 统计工具 ─────────────────────────────────────────────────────────────────


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson 相关系数（stdlib 实现）。"""
    n = len(xs)
    if n < 2:
        return 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 1.0  # 两者都接近常数
    return cov / (sx * sy)


def linear_slope(series: list[tuple[int, float]]) -> float:
    """最小二乘线性回归斜率。"""
    n = len(series)
    if n < 2:
        return 0.0
    xs = [s[0] for s in series]
    ys = [s[1] for s in series]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def trailing_mean(series: list[tuple[int, float]], up_to_iter: int, window: int) -> float | None:
    """取 iter <= up_to_iter 的最后 window 个值的均值。"""
    vals = [v for it, v in series if it <= up_to_iter]
    if not vals:
        return None
    tail = vals[-window:]
    return sum(tail) / len(tail)


# ── 检查点比较 ───────────────────────────────────────────────────────────────


def compute_checkpoints(
    series_a: list[tuple[int, float]],
    series_b: list[tuple[int, float]],
    interval: int,
    window: int,
) -> list[dict]:
    """在共同迭代范围内按 interval 采样检查点，返回比较记录。"""
    if not series_a or not series_b:
        return []

    max_iter = min(series_a[-1][0], series_b[-1][0])
    if max_iter <= 0:
        return []

    # 生成检查点迭代
    checkpoints_iters = list(range(interval, max_iter + 1, interval))
    # 确保包含最终迭代
    if not checkpoints_iters or checkpoints_iters[-1] != max_iter:
        checkpoints_iters.append(max_iter)

    records = []
    for cp in checkpoints_iters:
        ma = trailing_mean(series_a, cp, window)
        mb = trailing_mean(series_b, cp, window)
        if ma is None or mb is None:
            continue
        diff = abs(ma - mb)
        denom = max(abs(ma), abs(mb), 1e-8)
        rel = diff / denom
        records.append({
            "iter": cp,
            "phase_a": ma,
            "phase_b": mb,
            "abs_diff": diff,
            "rel_diff": rel,
        })
    return records


def compare_series(
    checkpoints: list[dict],
    threshold_corr: float,
    threshold_rel: float,
) -> tuple[float, float, bool]:
    """比较两条曲线的检查点：返回 (corr, final_rel, passed)。"""
    if len(checkpoints) < 2:
        # 检查点不足，只比较相对差异
        if checkpoints:
            final_rel = checkpoints[-1]["rel_diff"]
            return 1.0, final_rel, final_rel <= threshold_rel
        return 1.0, 0.0, True

    vals_a = [c["phase_a"] for c in checkpoints]
    vals_b = [c["phase_b"] for c in checkpoints]
    corr = pearson(vals_a, vals_b)
    final_rel = checkpoints[-1]["rel_diff"]

    passed = corr >= threshold_corr and final_rel <= threshold_rel
    return corr, final_rel, passed


def compare_reward_trend(
    series_a: list[tuple[int, float]],
    series_b: list[tuple[int, float]],
) -> tuple[float, float, bool]:
    """比较 reward 后半段线性回归斜率的符号。"""
    def second_half(s):
        if len(s) < 4:
            return s
        return s[len(s) // 2:]

    slope_a = linear_slope(second_half(series_a))
    slope_b = linear_slope(second_half(series_b))

    # 两者斜率符号一致（或都接近零）
    if abs(slope_a) < 1e-8 and abs(slope_b) < 1e-8:
        ok = True
    elif slope_a * slope_b >= 0:
        ok = True
    else:
        ok = False

    return slope_a, slope_b, ok


# ── YAML 验证 ────────────────────────────────────────────────────────────────


def validate_yaml_config(yaml_path: str | None, task: str) -> bool | None:
    """验证 YAML 实验定义中的 task ID 包含测试使用的 task。

    返回 True=通过 / False=不通过 / None=跳过。
    """
    if yaml_path is None:
        # 尝试默认路径
        default = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "assets", "experiments", "g1_flat_walk.yaml",
        )
        if os.path.exists(default):
            yaml_path = default
        else:
            return None

    if not os.path.exists(yaml_path):
        print(f"[WARN] YAML 配置文件不存在: {yaml_path}")
        return None

    # 尝试用 PyYAML 解析
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        task_ids = [t.get("id", "") for t in cfg.get("tasks", [])]
    except ImportError:
        # 回退：正则提取
        import re
        with open(yaml_path) as f:
            content = f.read()
        task_ids = re.findall(r"id:\s*(.+)", content)
        task_ids = [t.strip() for t in task_ids]

    found = task in task_ids
    return found


# ── 输出格式化 ───────────────────────────────────────────────────────────────


def print_summary(
    args: argparse.Namespace,
    records_a: list[dict],
    records_b: list[dict],
    primary_results: dict,
    slope_a: float,
    slope_b: float,
    trend_ok: bool,
    term_results: dict,
    yaml_ok: bool | None,
    all_pass: bool,
):
    W = 70

    iters_a = records_a[-1]["iter"] if records_a else 0
    iters_b = records_b[-1]["iter"] if records_b else 0
    time_a = records_a[-1].get("extras", {}).get("tot_time", 0) if records_a else 0
    time_b = records_b[-1].get("extras", {}).get("tot_time", 0) if records_b else 0

    print(f"\n{'=' * W}")
    print(f"  Phase B Convergence Verification")
    print(f"{'=' * W}")
    print(f"  Task:       {args.task}")
    print(f"  Seed:       {args.seed} | Num envs: {args.num_envs}")
    rate_a = iters_a / time_a if time_a > 0 else 0
    rate_b = iters_b / time_b if time_b > 0 else 0
    run_order = "B→A (--reverse)" if args.reverse else "A→B"
    print(f"  Phase A:    {iters_a} iters ({time_a / 60:.1f} min, {rate_a:.1f} iter/s)")
    print(f"  Phase B:    {iters_b} iters ({time_b / 60:.1f} min, {rate_b:.1f} iter/s)")
    print(f"  Run order:  {run_order}")
    if rate_a > 0 and rate_b > 0:
        slower = (max(rate_a, rate_b) / min(rate_a, rate_b) - 1) * 100
        who = "B" if rate_a > rate_b else "A"
        print(f"  Throughput:  Phase {who} is {slower:.1f}% slower (ordering bias?)")
    print(f"  Thresholds: corr>{args.threshold_corr}  rel<{args.threshold_rel:.0%}  term_rel<{args.threshold_term_rel:.0%}")

    # 主要指标
    for key, (checkpoints, corr, final_rel, passed) in primary_results.items():
        tag = "PASS" if passed else "FAIL"
        short_key = key.split("/", 1)[-1] if "/" in key else key
        print(f"\n{'─' * 3} {key} {'─' * max(1, W - len(key) - 5)}")
        print(f"  {'Iter':>8}  {'Phase A':>12}  {'Phase B':>12}  {'|diff|':>10}  {'rel%':>8}")
        for c in checkpoints:
            print(f"  {c['iter']:>8}  {c['phase_a']:>12.6f}  {c['phase_b']:>12.6f}  "
                  f"{c['abs_diff']:>10.2e}  {c['rel_diff']:>7.1%}")
        print(f"  Corr: {corr:.3f}  Final rel: {final_rel:.1%}  [{tag}]")

    # 奖励趋势
    print(f"\n{'─' * 3} Reward trend {'─' * max(1, W - 17)}")
    tag = "PASS" if trend_ok else "FAIL"
    print(f"  Slope (2nd half):  A={slope_a:+.6f}/iter  B={slope_b:+.6f}/iter  [{tag}]")

    # Per-term 奖励
    if term_results:
        print(f"\n{'─' * 3} Per-term rewards (final checkpoint) {'─' * max(1, W - 41)}")
        for key, (checkpoints, final_rel, passed) in term_results.items():
            tag = "PASS" if passed else "FAIL"
            short = key.replace("Episode/rew/", "")
            if checkpoints:
                c = checkpoints[-1]
                print(f"  {short:<30}  A:{c['phase_a']:>10.4f}  B:{c['phase_b']:>10.4f}"
                      f"  {final_rel:>6.1%}  [{tag}]")

    # YAML 验证
    print(f"\n{'─' * 3} YAML Config {'─' * max(1, W - 16)}")
    if yaml_ok is True:
        print(f"  YAML → {args.task}  [PASS]")
    elif yaml_ok is False:
        print(f"  YAML → {args.task}  [FAIL] task ID not found")
    else:
        print(f"  YAML validation skipped (file not found)")

    # 最终判定
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n{'=' * W}")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * W}\n")


def save_output(
    output_path: str,
    args: argparse.Namespace,
    records_a: list[dict],
    records_b: list[dict],
    primary_results: dict,
    slope_a: float,
    slope_b: float,
    trend_ok: bool,
    term_results: dict,
    yaml_ok: bool | None,
    all_pass: bool,
):
    """保存对比结果为 JSON。"""
    data = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "iters_a": records_a[-1]["iter"] if records_a else 0,
        "iters_b": records_b[-1]["iter"] if records_b else 0,
        "primary": {},
        "trend": {"slope_a": slope_a, "slope_b": slope_b, "ok": trend_ok},
        "terms": {},
        "yaml_ok": yaml_ok,
        "verdict": "PASS" if all_pass else "FAIL",
    }
    for key, (checkpoints, corr, final_rel, passed) in primary_results.items():
        data["primary"][key] = {
            "checkpoints": checkpoints,
            "correlation": corr,
            "final_rel_diff": final_rel,
            "passed": passed,
        }
    for key, (checkpoints, final_rel, passed) in term_results.items():
        data["terms"][key] = {
            "checkpoints": checkpoints,
            "final_rel_diff": final_rel,
            "passed": passed,
        }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 对比结果已保存: {output_path}")


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # 确定运行模式
    if args.iters is not None:
        max_iters = args.iters
        timeout_sec = None
        print(f"[INFO] 迭代模式：每次 {max_iters} 迭代")
    else:
        minutes = args.minutes if args.minutes is not None else 35.0
        max_iters = 999999
        timeout_sec = minutes * 60
        print(f"[INFO] 时间模式：每次 {minutes:.1f} 分钟")

    all_pass = True

    # 运行顺序（--reverse 用于排除 GPU 热降频等顺序偏差）
    run_order = [
        ("Phase B", True),
        ("Phase A", False),
    ] if args.reverse else [
        ("Phase A", False),
        ("Phase B", True),
    ]
    if args.reverse:
        print("[INFO] --reverse: Phase B 先跑，Phase A 后跑")

    with tempfile.TemporaryDirectory(prefix="myrl_phase_b_") as tmpdir:
        jsonl_paths = {}
        for label, use_b in run_order:
            tag = "phase_b" if use_b else "phase_a"
            d = os.path.join(tmpdir, tag)
            os.makedirs(d)
            jsonl_paths[tag] = run_train(
                args.task, args.num_envs, args.seed, max_iters, d,
                use_phase_b=use_b, timeout_sec=timeout_sec,
            )

        jsonl_a = jsonl_paths["phase_a"]
        jsonl_b = jsonl_paths["phase_b"]

        # ── 加载 metrics ─────────────────────────────────────────────────
        records_a = load_metrics(jsonl_a)
        records_b = load_metrics(jsonl_b)

        if not records_a or not records_b:
            print("[ERROR] metrics.jsonl 为空，无法比较")
            sys.exit(1)

        print(f"\n[INFO] Phase A: {len(records_a)} 条记录 (iter 0..{records_a[-1]['iter']})")
        print(f"[INFO] Phase B: {len(records_b)} 条记录 (iter 0..{records_b[-1]['iter']})")

        # ── YAML 验证 ────────────────────────────────────────────────────
        yaml_ok = validate_yaml_config(args.yaml_config, args.task)
        if yaml_ok is False:
            all_pass = False

        # ── 主要指标对比 ─────────────────────────────────────────────────
        primary_keys = [
            "Loss/surrogate_loss",
            "Loss/value_loss",
            "Train/mean_reward_0",
            "Train/mean_episode_length",
        ]

        primary_results = {}
        for key in primary_keys:
            sa = extract_series(records_a, key)
            sb = extract_series(records_b, key)
            if not sa or not sb:
                print(f"[WARN] 指标 {key} 数据不足，跳过")
                continue
            cps = compute_checkpoints(sa, sb, args.checkpoint_interval, args.window)
            corr, final_rel, passed = compare_series(cps, args.threshold_corr, args.threshold_rel)
            primary_results[key] = (cps, corr, final_rel, passed)
            if not passed:
                all_pass = False

        # ── 奖励趋势 ────────────────────────────────────────────────────
        reward_a = extract_series(records_a, "Train/mean_reward_0")
        reward_b = extract_series(records_b, "Train/mean_reward_0")
        if reward_a and reward_b:
            slope_a, slope_b, trend_ok = compare_reward_trend(reward_a, reward_b)
        else:
            slope_a, slope_b, trend_ok = 0.0, 0.0, True

        if not trend_ok:
            all_pass = False

        # ── Per-term 奖励 ────────────────────────────────────────────────
        rew_keys = sorted(set(discover_reward_keys(records_a)) | set(discover_reward_keys(records_b)))

        term_results = {}
        for key in rew_keys:
            sa = extract_series(records_a, key)
            sb = extract_series(records_b, key)
            if not sa or not sb:
                continue
            cps = compute_checkpoints(sa, sb, args.checkpoint_interval, args.window)
            if not cps:
                continue
            final_rel = cps[-1]["rel_diff"]
            passed = final_rel <= args.threshold_term_rel
            term_results[key] = (cps, final_rel, passed)
            if not passed:
                all_pass = False

        # ── 输出 ─────────────────────────────────────────────────────────
        print_summary(
            args, records_a, records_b,
            primary_results, slope_a, slope_b, trend_ok,
            term_results, yaml_ok, all_pass,
        )

        if args.output:
            save_output(
                args.output, args, records_a, records_b,
                primary_results, slope_a, slope_b, trend_ok,
                term_results, yaml_ok, all_pass,
            )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
