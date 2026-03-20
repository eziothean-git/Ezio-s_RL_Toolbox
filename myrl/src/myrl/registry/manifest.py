"""RunManifest：训练运行的内容寻址清单。"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import yaml

from .checksum import sha256_file, sha256_dict, sha256_bytes


@dataclass
class AssetEntry:
    path: str
    sha256: str
    role: str


@dataclass
class RunManifest:
    manifest_version: str = "1"
    run_id: str = ""
    experiment_name: str = ""
    created_at: str = ""

    # 任务
    task_id: str = ""
    env_cfg_sha256: str = ""

    # Agent
    max_iterations: int = 0
    num_envs: int = 0
    seed: int = 0
    device: str = "cuda:0"

    # 资产
    assets: list[AssetEntry] = field(default_factory=list)

    # Checkpoint
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    checkpoint_iteration: int = 0

    # 代码
    myrl_commit: str = ""

    # 包来源（由 ExperimentComposer 路径填充）
    package_id: str = ""

    # 指标快照
    metrics_snapshot: dict = field(default_factory=dict)

    # ── 序列化 ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assets"] = [asdict(a) if isinstance(a, AssetEntry) else a for a in self.assets]
        return d

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def content_sha256(self) -> str:
        return sha256_bytes(self.to_yaml().encode())

    @classmethod
    def from_yaml(cls, path: str) -> RunManifest:
        with open(path) as f:
            d = yaml.safe_load(f)
        assets = [AssetEntry(**a) for a in d.pop("assets", [])]
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.assets = assets
        return obj

    @classmethod
    def from_train_run(
        cls,
        log_dir: str,
        task_id: str,
        agent_cfg,
        runner,
    ) -> RunManifest:
        """从训练结果构造 RunManifest。"""
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%d_%H%M%S") + f"_{getattr(agent_cfg, 'experiment_name', 'unknown')}"

        # Git commit
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).decode().strip()
        except Exception:
            commit = "unknown"

        # 找最新 checkpoint
        checkpoint_path = ""
        checkpoint_sha256 = ""
        checkpoint_iteration = 0
        for root, dirs, files in os.walk(log_dir):
            for f in sorted(files):
                if f.endswith(".pt") and f.startswith("model_"):
                    cp = os.path.join(root, f)
                    try:
                        it = int(f.replace("model_", "").replace(".pt", ""))
                        if it >= checkpoint_iteration:
                            checkpoint_iteration = it
                            checkpoint_path = cp
                    except ValueError:
                        pass
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_sha256 = sha256_file(checkpoint_path)

        # 收集 myrl assets
        assets = []
        repo_root = _find_repo_root()
        for role, rel in [
            ("robot_model", "myrl/assets/robots/g1/g1.xml"),
        ]:
            abs_path = os.path.join(repo_root, rel) if repo_root else rel
            if os.path.exists(abs_path):
                assets.append(AssetEntry(
                    path=rel,
                    sha256=sha256_file(abs_path),
                    role=role,
                ))

        # 指标快照（从 metrics.jsonl 读最后一行）
        metrics = {}
        jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        if os.path.exists(jsonl_path):
            try:
                import json
                with open(jsonl_path) as f:
                    last = None
                    for line in f:
                        line = line.strip()
                        if line:
                            last = json.loads(line)
                if last:
                    for key in ("Loss/surrogate_loss", "Loss/value_loss", "Train/mean_reward"):
                        if key in last:
                            metrics[key.split("/")[-1]] = last[key]
            except Exception:
                pass

        return cls(
            run_id=run_id,
            experiment_name=getattr(agent_cfg, "experiment_name", "unknown"),
            created_at=now.isoformat(),
            task_id=task_id,
            env_cfg_sha256="",
            max_iterations=getattr(agent_cfg, "max_iterations", 0),
            num_envs=getattr(agent_cfg, "num_envs", 0),
            seed=getattr(agent_cfg, "seed", 0),
            device=getattr(agent_cfg, "device", "cuda:0"),
            assets=assets,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_iteration=checkpoint_iteration,
            myrl_commit=commit,
            metrics_snapshot=metrics,
        )


def _find_repo_root() -> str | None:
    """从当前文件向上查找包含 CLAUDE.md 的仓库根目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.exists(os.path.join(here, "CLAUDE.md")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None
