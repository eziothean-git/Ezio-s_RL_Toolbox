"""RunRegistry：内容寻址实验注册表。

目录结构：
    <registry_dir>/
    ├── manifests/<sha8>.yaml       # 内容寻址存储
    └── runs/<experiment>/<run_id>  # symlink → ../../manifests/<sha8>.yaml
"""
from __future__ import annotations

import os

from .manifest import RunManifest
from .checksum import sha256_file


def _repo_root() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.exists(os.path.join(here, "CLAUDE.md")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


class RunRegistry:
    """管理训练运行清单的内容寻址注册表。"""

    def __init__(self, registry_dir: str | None = None):
        if registry_dir is None:
            root = _repo_root()
            registry_dir = os.path.join(root, "myrl", "registry") if root else "registry"
        self._dir = os.path.abspath(registry_dir)
        os.makedirs(os.path.join(self._dir, "manifests"), exist_ok=True)
        os.makedirs(os.path.join(self._dir, "runs"), exist_ok=True)

    def save(self, manifest: RunManifest) -> str:
        """保存 manifest 并创建 symlink，返回 run_id。"""
        yaml_str = manifest.to_yaml()
        sha8 = manifest.content_sha256()[:8]
        manifest_path = os.path.join(self._dir, "manifests", f"{sha8}.yaml")

        with open(manifest_path, "w") as f:
            f.write(yaml_str)

        # 创建 runs/<experiment>/<run_id> symlink
        exp_dir = os.path.join(self._dir, "runs", manifest.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        link_path = os.path.join(exp_dir, manifest.run_id)
        rel_target = os.path.relpath(manifest_path, exp_dir)
        if os.path.lexists(link_path):
            os.unlink(link_path)
        os.symlink(rel_target, link_path)

        return manifest.run_id

    def load(self, run_id: str, experiment_name: str | None = None) -> RunManifest:
        """按 run_id 加载 manifest。"""
        # 先尝试在 runs/ 下搜索
        runs_dir = os.path.join(self._dir, "runs")
        if experiment_name:
            link = os.path.join(runs_dir, experiment_name, run_id)
            if os.path.exists(link):
                return RunManifest.from_yaml(link)

        # 搜索所有 experiment
        for exp in os.listdir(runs_dir):
            link = os.path.join(runs_dir, exp, run_id)
            if os.path.exists(link):
                return RunManifest.from_yaml(link)

        raise FileNotFoundError(f"run_id '{run_id}' not found in registry")

    def list_runs(self, experiment_name: str | None = None) -> list[RunManifest]:
        """列出所有（或指定 experiment 的）运行。"""
        runs_dir = os.path.join(self._dir, "runs")
        if not os.path.exists(runs_dir):
            return []

        result = []
        exps = [experiment_name] if experiment_name else os.listdir(runs_dir)
        for exp in exps:
            exp_dir = os.path.join(runs_dir, exp)
            if not os.path.isdir(exp_dir):
                continue
            for run_id in sorted(os.listdir(exp_dir)):
                link = os.path.join(exp_dir, run_id)
                try:
                    result.append(RunManifest.from_yaml(link))
                except Exception:
                    pass
        return result

    def verify(self, manifest: RunManifest) -> dict[str, bool]:
        """重新计算所有 sha256，返回 {key: pass/fail}。"""
        results = {}
        for asset in manifest.assets:
            if os.path.exists(asset.path):
                actual = sha256_file(asset.path)
                results[f"asset:{asset.role}"] = (actual == asset.sha256)
            else:
                results[f"asset:{asset.role}"] = False

        if manifest.checkpoint_path and manifest.checkpoint_sha256:
            if os.path.exists(manifest.checkpoint_path):
                actual = sha256_file(manifest.checkpoint_path)
                results["checkpoint"] = (actual == manifest.checkpoint_sha256)
            else:
                results["checkpoint"] = False

        return results
