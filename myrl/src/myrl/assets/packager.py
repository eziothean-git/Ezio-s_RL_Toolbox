"""PackageBuilder / PackageReader — 将 experiment_cfg 打包为 .myrlpkg。

新路径（from_yaml_file）直接读 source-path 格式的 experiment.yaml，无需 asset_store。
旧路径（from_experiment_cfg）读 asset_store，保持向后兼容。

包目录结构：
    g1_locomotion_v1.myrlpkg/
    ├── package.yaml              # PackageManifest
    ├── experiment.yaml           # experiment_cfg 副本
    ├── assets/                   # MYRL_ASSETS_DIR 指向此处
    │   ├── robots/g1/            # robot_model blob
    │   ├── terrains/flat_plane/
    │   ├── actuator_cfgs/
    │   └── sensor_cfgs/
    ├── reward_fns/
    │   └── locomotion/           # reward_fn Python 源码
    ├── reward_pipelines/
    ├── obs_pipelines/
    ├── algo_cfgs/
    └── env_scripts/
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .asset_store import AssetStore, AssetType, AssetRecord


def _sha256(path: Path) -> str:
    """计算文件或目录的 SHA256（目录则对所有文件内容排序后累计哈希）。"""
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    elif path.is_dir():
        for fp in sorted(path.rglob("*")):
            if fp.is_file():
                h.update(fp.read_bytes())
    return h.hexdigest()


def _find_repo_root(start: Path) -> Path:
    """从 start 向上查找含 CLAUDE.md 的目录（即 repo root）。"""
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "CLAUDE.md").exists():
            return parent
    return p


# ── PackageManifest ────────────────────────────────────────────────────────────

@dataclass
class AssetChecksum:
    name: str
    version: str
    content_hash: str


@dataclass
class PackageManifest:
    package_version: str = "1"
    package_id: str = ""
    experiment_name: str = ""
    created_at: str = ""
    source_experiment_cfg: str = ""
    asset_checksums: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> PackageManifest:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── PackageBuilder ─────────────────────────────────────────────────────────────

class PackageBuilder:
    """从 experiment_cfg 打包所有依赖资产为 .myrlpkg。

    两种初始化路径：
    - from_yaml_file(yaml_path)  — 新路径，直接读 source-path 格式 YAML，无需 asset_store
    - from_experiment_cfg(name, version)  — 旧路径，读 asset_store（向后兼容）
    """

    def __init__(self, store: AssetStore | None = None):
        self._store = store or AssetStore()
        self._exp_cfg: dict = {}
        self._exp_name: str = ""
        self._exp_version: str = ""
        self._yaml_mode: bool = False   # True → from_yaml_file 路径
        self._repo_root: Path | None = None

    # ── 新路径 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml_file(cls, yaml_path: str, repo_root: str | None = None) -> "PackageBuilder":
        """从 source-path 格式的 experiment YAML 创建 builder（无需 asset_store）。

        Args:
            yaml_path: experiment YAML 路径（绝对或相对 cwd）
            repo_root: repo 根目录（默认向上查找含 CLAUDE.md 的目录）
        """
        p = Path(yaml_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"YAML not found: {yaml_path}")

        with open(p) as f:
            cfg = yaml.safe_load(f)

        root = Path(repo_root).resolve() if repo_root else _find_repo_root(p)

        builder = cls(store=None)
        builder._exp_cfg = cfg
        builder._exp_name = cfg.get("name", p.stem)
        builder._exp_version = cfg.get("version", "1.0.0")
        builder._yaml_mode = True
        builder._repo_root = root
        return builder

    def _copy_source(self, source: str, dest_dir: Path, dest_name: str | None = None) -> tuple[Path, str]:
        """将 source（相对 repo_root）复制到 dest_dir，返回 (dest_path, sha256)。"""
        src = (self._repo_root / source).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = dest_name or src.name
        dst = dest_dir / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return dst, _sha256(src)

    def _build_from_yaml(self, pkg_dir: Path) -> dict:
        """新路径 build 实现，返回 checksums dict。"""
        assets = self._exp_cfg.get("assets", {})
        checksums: dict[str, Any] = {}
        assets_dir = pkg_dir / "assets"

        # robot_model
        if "robot_model" in assets:
            a = assets["robot_model"]
            _, h = self._copy_source(a["source"], assets_dir / "robots" / a["name"])
            checksums["robot_model"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # actuator_cfg
        if "actuator_cfg" in assets:
            a = assets["actuator_cfg"]
            _, h = self._copy_source(a["source"], assets_dir / "actuator_cfgs")
            checksums["actuator_cfg"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # sensor_cfg
        if "sensor_cfg" in assets:
            a = assets["sensor_cfg"]
            _, h = self._copy_source(a["source"], assets_dir / "sensor_cfgs")
            checksums["sensor_cfg"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # terrain
        if "terrain" in assets:
            a = assets["terrain"]
            _, h = self._copy_source(a["source"], assets_dir / "terrains" / a["name"])
            checksums["terrain"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # reward_fns（列表）
        rfn_checksums = []
        for a in assets.get("reward_fns", []):
            _, h = self._copy_source(a["source"], pkg_dir / "reward_fns" / a["name"])
            rfn_checksums.append({"name": a["name"], "version": a["version"], "content_hash": h})
        if rfn_checksums:
            checksums["reward_fns"] = rfn_checksums

        # reward_pipeline
        if "reward_pipeline" in assets:
            a = assets["reward_pipeline"]
            _, h = self._copy_source(a["source"], pkg_dir / "reward_pipelines")
            checksums["reward_pipeline"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # obs_pipeline
        if "obs_pipeline" in assets:
            a = assets["obs_pipeline"]
            _, h = self._copy_source(a["source"], pkg_dir / "obs_pipelines")
            checksums["obs_pipeline"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # algo_cfg
        if "algo_cfg" in assets:
            a = assets["algo_cfg"]
            _, h = self._copy_source(a["source"], pkg_dir / "algo_cfgs")
            checksums["algo_cfg"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        # env_script
        if "env_script" in assets:
            a = assets["env_script"]
            _, h = self._copy_source(a["source"], pkg_dir / "env_scripts")
            checksums["env_script"] = {"name": a["name"], "version": a["version"], "content_hash": h}

        return checksums

    # ── 旧路径 ─────────────────────────────────────────────────────────────

    def from_experiment_cfg(self, name: str, version: str) -> "PackageBuilder":
        """读取 experiment_cfg，准备打包（asset_store 路径，向后兼容）。"""
        rec = self._store.get(name, version, AssetType.EXPERIMENT_CFG)
        blob_abs = self._store._root / rec.blob_path
        with open(blob_abs) as f:
            self._exp_cfg = yaml.safe_load(f)
        self._exp_name = name
        self._exp_version = version
        return self

    def build(self, output_dir: str) -> str:
        """构建 .myrlpkg 目录，返回路径。"""
        if not self._exp_cfg:
            raise RuntimeError("call from_yaml_file() or from_experiment_cfg() first")

        pkg_id = self._exp_cfg.get("name", self._exp_name)
        pkg_dir = Path(output_dir) / f"{pkg_id}.myrlpkg"
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        pkg_dir.mkdir(parents=True)

        # experiment.yaml
        with open(pkg_dir / "experiment.yaml", "w") as f:
            yaml.dump(self._exp_cfg, f, sort_keys=False, allow_unicode=True)

        # 分发到两种 build 路径
        if self._yaml_mode:
            checksums = self._build_from_yaml(pkg_dir)
        else:
            checksums = self._build_from_store(pkg_dir)

        # package.yaml
        manifest = PackageManifest(
            package_id=pkg_id,
            experiment_name=self._exp_cfg.get("name", self._exp_name),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_experiment_cfg=f"{self._exp_name}@{self._exp_version}",
            asset_checksums=checksums,
        )
        with open(pkg_dir / "package.yaml", "w") as f:
            f.write(manifest.to_yaml())

        return str(pkg_dir)

    def _build_from_store(self, pkg_dir: Path) -> dict:
        """旧路径 build 实现（asset_store），返回 checksums dict。"""
        checksums: dict[str, Any] = {}
        assets_dir = pkg_dir / "assets"

        # robot_model
        if "robot_model" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["robot_model"], AssetType.ROBOT_MODEL)
            if rec:
                self._store.export(rec, str(assets_dir / "robots"))
                checksums["robot_model"] = {"name": rec.name, "version": rec.version,
                                             "content_hash": rec.content_hash}

        # actuator_cfg
        if "actuator_cfg" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["actuator_cfg"], AssetType.ACTUATOR_CFG)
            if rec:
                dest_d = assets_dir / "actuator_cfgs"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["actuator_cfg"] = {"name": rec.name, "version": rec.version,
                                              "content_hash": rec.content_hash}

        # sensor_cfg
        if "sensor_cfg" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["sensor_cfg"], AssetType.SENSOR_CFG)
            if rec:
                dest_d = assets_dir / "sensor_cfgs"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["sensor_cfg"] = {"name": rec.name, "version": rec.version,
                                            "content_hash": rec.content_hash}

        # terrain
        if "terrain" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["terrain"], AssetType.TERRAIN)
            if rec:
                self._store.export(rec, str(assets_dir / "terrains"))
                checksums["terrain"] = {"name": rec.name, "version": rec.version,
                                         "content_hash": rec.content_hash}

        # reward_fns（列表）
        reward_fns_checksums = []
        for rfn_ref in self._exp_cfg.get("reward_fns", []):
            rec = self._get_asset(rfn_ref, AssetType.REWARD_FN)
            if rec:
                dest_d = pkg_dir / "reward_fns"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                reward_fns_checksums.append({"name": rec.name, "version": rec.version,
                                              "content_hash": rec.content_hash})
        if reward_fns_checksums:
            checksums["reward_fns"] = reward_fns_checksums

        # reward_pipeline
        if "reward_pipeline" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["reward_pipeline"], AssetType.REWARD_PIPELINE)
            if rec:
                dest_d = pkg_dir / "reward_pipelines"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["reward_pipeline"] = {"name": rec.name, "version": rec.version,
                                                 "content_hash": rec.content_hash}

        # obs_pipeline
        if "obs_pipeline" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["obs_pipeline"], AssetType.OBS_PIPELINE)
            if rec:
                dest_d = pkg_dir / "obs_pipelines"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["obs_pipeline"] = {"name": rec.name, "version": rec.version,
                                              "content_hash": rec.content_hash}

        # algo_cfg
        if "algo_cfg" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["algo_cfg"], AssetType.ALGO_CFG)
            if rec:
                dest_d = pkg_dir / "algo_cfgs"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["algo_cfg"] = {"name": rec.name, "version": rec.version,
                                          "content_hash": rec.content_hash}

        # env_script
        if "env_script" in self._exp_cfg:
            rec = self._get_asset(self._exp_cfg["env_script"], AssetType.ENV_SCRIPT)
            if rec:
                dest_d = pkg_dir / "env_scripts"
                dest_d.mkdir(parents=True, exist_ok=True)
                self._store.export(rec, str(dest_d))
                checksums["env_script"] = {"name": rec.name, "version": rec.version,
                                            "content_hash": rec.content_hash}

        return checksums

    def _get_asset(self, ref: str, asset_type: AssetType) -> AssetRecord | None:
        """解析 'name@version' 引用，返回 AssetRecord（不存在则打印警告返回 None）。"""
        if "@" not in ref:
            print(f"[WARN] Invalid asset ref '{ref}', expected 'name@version'")
            return None
        name, version = ref.split("@", 1)
        try:
            return self._store.get(name, version, asset_type)
        except KeyError:
            print(f"[WARN] Asset not found in store: {ref} ({asset_type.value}), skipping")
            return None


# ── PackageReader ──────────────────────────────────────────────────────────────

class PackageReader:
    """读取 .myrlpkg 目录（或 .zip），提供资产路径访问。"""

    def __init__(self, package_path: str):
        pkg = Path(package_path)
        if pkg.is_file() and pkg.suffix == ".zip":
            # 解压到临时目录
            import tempfile
            self._tmp = tempfile.mkdtemp(prefix="myrlpkg_")
            with zipfile.ZipFile(pkg, "r") as zf:
                zf.extractall(self._tmp)
            # 找到内层 .myrlpkg 目录
            contents = list(Path(self._tmp).glob("*.myrlpkg"))
            self._root = contents[0] if contents else Path(self._tmp)
        else:
            self._tmp = None
            self._root = pkg.resolve()

        if not self._root.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

    @property
    def manifest(self) -> PackageManifest:
        return PackageManifest.from_yaml(str(self._root / "package.yaml"))

    @property
    def experiment_yaml(self) -> dict:
        exp_path = self._root / "experiment.yaml"
        if not exp_path.exists():
            return {}
        with open(exp_path) as f:
            return yaml.safe_load(f) or {}

    @property
    def assets_dir(self) -> str:
        """MYRL_ASSETS_DIR 应指向此目录。"""
        d = self._root / "assets"
        d.mkdir(exist_ok=True)
        return str(d)

    def get_asset_path(self, asset_type: str, name: str) -> str:
        """获取包内资产路径（文件或目录）。"""
        type_map = {
            "reward_fn": "reward_fns",
            "reward_pipeline": "reward_pipelines",
            "obs_pipeline": "obs_pipelines",
            "algo_cfg": "algo_cfgs",
            "env_script": "env_scripts",
            "robot_model": "assets/robots",
            "terrain": "assets/terrains",
            "actuator_cfg": "assets/actuator_cfgs",
            "sensor_cfg": "assets/sensor_cfgs",
        }
        subdir = type_map.get(asset_type, asset_type)
        base = self._root / subdir
        # 找匹配 name 的文件/目录
        for candidate in base.iterdir() if base.exists() else []:
            if candidate.name == name or candidate.stem == name:
                return str(candidate)
        return str(base / name)

    def list_reward_fn_dirs(self) -> list[str]:
        """列出所有 reward_fn 目录（用于 sys.path 注入）。"""
        rfn_dir = self._root / "reward_fns"
        if not rfn_dir.exists():
            return []
        return [str(rfn_dir)]

    def get_reward_pipeline_path(self) -> str | None:
        """返回 reward_pipeline YAML 路径（取第一个）。"""
        d = self._root / "reward_pipelines"
        if not d.exists():
            return None
        yamls = list(d.glob("*.yaml"))
        return str(yamls[0]) if yamls else None

    def get_obs_pipeline_path(self) -> str | None:
        """返回 obs_pipeline YAML 路径（取第一个）。"""
        d = self._root / "obs_pipelines"
        if not d.exists():
            return None
        yamls = list(d.glob("*.yaml"))
        return str(yamls[0]) if yamls else None

    def get_algo_cfg_path(self) -> str | None:
        d = self._root / "algo_cfgs"
        if not d.exists():
            return None
        yamls = list(d.glob("*.yaml"))
        return str(yamls[0]) if yamls else None

    def get_env_script_path(self) -> str | None:
        d = self._root / "env_scripts"
        if not d.exists():
            return None
        pys = list(d.glob("*.py"))
        return str(pys[0]) if pys else None

    def get_terrain_generators_dir(self) -> list[str]:
        """返回所有 terrain generators 脚本列表。"""
        terrain_dir = self._root / "assets" / "terrains"
        scripts = []
        if terrain_dir.exists():
            for gen in terrain_dir.rglob("generators/*.py"):
                scripts.append(str(gen))
        return scripts

    def __del__(self):
        if self._tmp and os.path.exists(self._tmp):
            shutil.rmtree(self._tmp, ignore_errors=True)
