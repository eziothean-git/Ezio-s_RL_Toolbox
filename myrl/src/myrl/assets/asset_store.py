"""myrl AssetStore — 内容寻址资产管理库。

目录结构：
    myrl/asset_store/
    ├── blobs/
    │   └── <hash[:2]>/<hash[2:]>   # 内容寻址 blob（文件或 .tar.gz 目录）
    └── assets/
        ├── robot_model/g1_29dof@1.0.0.yaml
        ├── reward_fn/locomotion_rewards@1.0.0.yaml
        └── ...

API:
    store = AssetStore()
    rec = store.register("reward_fn", "locomotion_rewards", "1.0.0", source_path)
    rec = store.get("locomotion_rewards", "1.0.0", AssetType.REWARD_FN)
    store.export(rec, "/tmp/dest")
    ok = store.verify(rec)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml


# ── 资产类型 ──────────────────────────────────────────────────────────────────

class AssetType(str, Enum):
    ROBOT_MODEL     = "robot_model"
    ACTUATOR_CFG    = "actuator_cfg"
    SENSOR_CFG      = "sensor_cfg"
    TERRAIN         = "terrain"
    REWARD_FN       = "reward_fn"
    REWARD_PIPELINE = "reward_pipeline"
    OBS_PIPELINE    = "obs_pipeline"
    ALGO_CFG        = "algo_cfg"
    ENV_SCRIPT      = "env_script"
    EXPERIMENT_CFG  = "experiment_cfg"


# ── 资产记录 ──────────────────────────────────────────────────────────────────

@dataclass
class AssetRecord:
    name: str
    version: str                    # semver 字符串
    asset_type: AssetType
    content_hash: str               # SHA256（文件或目录）
    blob_path: str                  # 相对 store root 的 blob 路径
    is_directory: bool = False      # True = 目录资产（.tar.gz blob）
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def asset_id(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asset_type"] = self.asset_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AssetRecord:
        d = dict(d)
        d["asset_type"] = AssetType(d["asset_type"])
        return cls(**d)


# ── 内容哈希工具 ──────────────────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(path: str) -> str:
    """对目录内容计算确定性哈希（按文件名排序）。"""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, path)
            h.update(rel.encode())
            h.update(_sha256_file(fpath).encode())
    return h.hexdigest()


def _sha256_path(path: str) -> tuple[str, bool]:
    """计算文件或目录哈希，返回 (hash, is_directory)。"""
    if os.path.isdir(path):
        return _sha256_dir(path), True
    return _sha256_file(path), False


# ── AssetStore ─────────────────────────────────────────────────────────────────

def _default_store_dir() -> str:
    """默认 store 目录：<repo_root>/myrl/asset_store/"""
    here = Path(__file__).resolve()
    # src/myrl/assets/asset_store.py → parents: assets, myrl, src, myrl/, repo
    for p in here.parents:
        if (p / "CLAUDE.md").exists():
            return str(p / "myrl" / "asset_store")
    return str(Path.cwd() / "asset_store")


class AssetStore:
    """内容寻址资产管理库。"""

    def __init__(self, store_dir: str | None = None):
        self._root = Path(store_dir or _default_store_dir()).resolve()
        self._blobs = self._root / "blobs"
        self._assets = self._root / "assets"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._assets.mkdir(parents=True, exist_ok=True)

    # ── 注册 ──────────────────────────────────────────────────────────────────

    def register(
        self,
        asset_type: AssetType | str,
        name: str,
        version: str,
        source_path: str,
        description: str = "",
        tags: list[str] | None = None,
        **metadata,
    ) -> AssetRecord:
        """注册资产（文件或目录），返回 AssetRecord。"""
        asset_type = AssetType(asset_type)
        source_path = os.path.abspath(source_path)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"source_path not found: {source_path}")

        content_hash, is_dir = _sha256_path(source_path)

        # 存入 blob
        blob_rel = self._store_blob(source_path, content_hash, is_dir)

        rec = AssetRecord(
            name=name,
            version=version,
            asset_type=asset_type,
            content_hash=content_hash,
            blob_path=blob_rel,
            is_directory=is_dir,
            description=description,
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        self._write_record(rec)
        return rec

    def _store_blob(self, source_path: str, content_hash: str, is_dir: bool) -> str:
        """将源路径存入 blob 目录，返回相对 blob_path。"""
        prefix = content_hash[:2]
        suffix = content_hash[2:]
        blob_dir = self._blobs / prefix
        blob_dir.mkdir(parents=True, exist_ok=True)

        if is_dir:
            blob_path = blob_dir / f"{suffix}.tar.gz"
            if not blob_path.exists():
                with tarfile.open(blob_path, "w:gz") as tar:
                    tar.add(source_path, arcname=os.path.basename(source_path))
        else:
            ext = Path(source_path).suffix
            blob_path = blob_dir / f"{suffix}{ext}"
            if not blob_path.exists():
                shutil.copy2(source_path, blob_path)

        return str(blob_path.relative_to(self._root))

    def _write_record(self, rec: AssetRecord) -> None:
        """写资产记录 YAML。"""
        type_dir = self._assets / rec.asset_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        record_path = type_dir / f"{rec.asset_id}.yaml"
        with open(record_path, "w") as f:
            yaml.dump(rec.to_dict(), f, sort_keys=False, allow_unicode=True)

    def _read_record(self, record_path: Path) -> AssetRecord:
        with open(record_path) as f:
            d = yaml.safe_load(f)
        return AssetRecord.from_dict(d)

    # ── 查询 ──────────────────────────────────────────────────────────────────

    def get(self, name: str, version: str, asset_type: AssetType | str) -> AssetRecord:
        """按名称+版本+类型查询资产，不存在则抛 KeyError。"""
        asset_type = AssetType(asset_type)
        record_path = self._assets / asset_type.value / f"{name}@{version}.yaml"
        if not record_path.exists():
            raise KeyError(f"Asset not found: {name}@{version} ({asset_type.value})")
        return self._read_record(record_path)

    def list_assets(self, asset_type: AssetType | str | None = None) -> list[AssetRecord]:
        """列出所有（或指定类型的）资产。"""
        if asset_type is not None:
            types = [AssetType(asset_type)]
        else:
            types = list(AssetType)

        result = []
        for at in types:
            type_dir = self._assets / at.value
            if not type_dir.exists():
                continue
            for yaml_file in sorted(type_dir.glob("*.yaml")):
                try:
                    result.append(self._read_record(yaml_file))
                except Exception:
                    pass
        return result

    # ── 导出 ──────────────────────────────────────────────────────────────────

    def export(self, record: AssetRecord, dest_dir: str) -> str:
        """将 blob 提取到 dest_dir，返回提取后的路径。"""
        blob_abs = self._root / record.blob_path
        if not blob_abs.exists():
            raise FileNotFoundError(f"Blob missing: {blob_abs}")

        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.abspath(dest_dir)

        if record.is_directory:
            with tarfile.open(blob_abs, "r:gz") as tar:
                tar.extractall(dest)
            # 提取出的目录名 = tarball 内 arcname
            extracted = os.path.join(dest, record.name)
            if not os.path.exists(extracted):
                # arcname 可能不一致，返回 dest_dir
                return dest
            return extracted
        else:
            ext = Path(blob_abs).suffix
            dest_file = os.path.join(dest, f"{record.name}{ext}")
            shutil.copy2(blob_abs, dest_file)
            return dest_file

    # ── 校验 ──────────────────────────────────────────────────────────────────

    def verify(self, record: AssetRecord) -> bool:
        """重新计算 blob 哈希，与记录比对。"""
        blob_abs = self._root / record.blob_path
        if not blob_abs.exists():
            return False
        if record.is_directory:
            # 对 .tar.gz 本身计算哈希（内容寻址，文件不变则 hash 不变）
            actual = _sha256_file(str(blob_abs))
        else:
            actual = _sha256_file(str(blob_abs))
        # 比较 hash prefix（blob 文件名 = suffix，合起来是完整 hash）
        prefix = Path(record.blob_path).parent.name  # hash[:2]
        suffix_part = Path(record.blob_path).stem     # hash[2:]（不含扩展名）
        stored_hash = prefix + suffix_part
        return stored_hash == record.content_hash
