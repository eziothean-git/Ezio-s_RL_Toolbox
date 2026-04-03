#!/usr/bin/env python3
"""myrl Asset Manager CLI — 资产注册/查询/打包/分发。

用法：
    python scripts/asset_cli.py register reward_fn locomotion_rewards:1.0.0 \\
        --source myrl/src/myrl/tasks/locomotion/mdp/rewards/

    python scripts/asset_cli.py register terrain flat_plane:1.0.0 \\
        --source myrl/assets/terrains/flat_plane/

    python scripts/asset_cli.py register actuator_cfg g1_default:1.0.0 \\
        --source myrl/assets/actuator_cfgs/g1_default.yaml

    python scripts/asset_cli.py register sensor_cfg g1_contact:1.0.0 \\
        --source myrl/assets/sensor_cfgs/g1_contact.yaml

    python scripts/asset_cli.py register experiment_cfg g1_locomotion_v1:1.0.0 \\
        --source myrl/assets/experiments/g1_locomotion_v1.yaml

    python scripts/asset_cli.py list [--type terrain]
    python scripts/asset_cli.py show g1_locomotion_v1:1.0.0

    python scripts/asset_cli.py pack g1_locomotion_v1:1.0.0 --output packages/
    python scripts/asset_cli.py info packages/g1_locomotion_v1.myrlpkg

    python scripts/asset_cli.py deploy packages/g1_locomotion_v1.myrlpkg \\
        user@server:/workspace/packages/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# 将 myrl src 加入 path（不依赖 pip install）
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _get_store():
    from myrl.assets.asset_store import AssetStore
    return AssetStore()


# ── register ──────────────────────────────────────────────────────────────────

def cmd_register(args: argparse.Namespace) -> int:
    from myrl.assets.asset_store import AssetStore, AssetType
    store = _get_store()

    if ":" not in args.name_version:
        print(f"[ERROR] name_version must be 'name:version', got: {args.name_version}")
        return 1
    name, version = args.name_version.split(":", 1)

    try:
        at = AssetType(args.asset_type)
    except ValueError:
        valid = [e.value for e in AssetType]
        print(f"[ERROR] Unknown asset_type '{args.asset_type}'. Valid: {valid}")
        return 1

    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        print(f"[ERROR] source path not found: {source}")
        return 1

    try:
        rec = store.register(at, name, version, source,
                             description=args.description or "")
        print(f"[OK] Registered: {rec.asset_id} ({at.value})")
        print(f"     hash  : {rec.content_hash[:16]}...")
        print(f"     blob  : {rec.blob_path}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> int:
    from myrl.assets.asset_store import AssetStore, AssetType
    store = _get_store()

    asset_type = None
    if args.type:
        try:
            asset_type = AssetType(args.type)
        except ValueError:
            print(f"[ERROR] Unknown asset_type '{args.type}'")
            return 1

    assets = store.list_assets(asset_type)
    if not assets:
        print("(no assets registered)")
        return 0

    # 按类型分组显示
    by_type: dict = {}
    for rec in assets:
        by_type.setdefault(rec.asset_type.value, []).append(rec)

    for at, recs in sorted(by_type.items()):
        print(f"\n  [{at}]")
        for r in sorted(recs, key=lambda x: x.asset_id):
            desc = f"  # {r.description}" if r.description else ""
            print(f"    {r.asset_id:<40}  {r.content_hash[:12]}...{desc}")

    print(f"\n  Total: {len(assets)} asset(s)")
    return 0


# ── show ──────────────────────────────────────────────────────────────────────

def cmd_show(args: argparse.Namespace) -> int:
    from myrl.assets.asset_store import AssetStore, AssetType
    import yaml
    store = _get_store()

    ref = args.asset_id
    if ":" not in ref:
        print(f"[ERROR] asset_id must be 'name:version', got: {ref}")
        return 1
    name, version = ref.split(":", 1)

    # 搜索所有类型
    found = None
    for at in AssetType:
        try:
            found = store.get(name, version, at)
            break
        except KeyError:
            pass

    if found is None:
        print(f"[ERROR] Asset not found: {ref}")
        return 1

    print(yaml.dump(found.to_dict(), sort_keys=False, allow_unicode=True))
    return 0


# ── pack ──────────────────────────────────────────────────────────────────────

def cmd_pack(args: argparse.Namespace) -> int:
    from myrl.assets.packager import PackageBuilder

    ref = args.experiment_id
    output_dir = os.path.abspath(args.output or ".")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 新路径：直接传 YAML 文件路径（含 / 或 .yaml 后缀）
        if "/" in ref or ref.endswith(".yaml"):
            builder = PackageBuilder.from_yaml_file(ref)
        else:
            # 旧路径：name:version，从 asset_store 读取
            from myrl.assets.asset_store import AssetStore
            if ":" not in ref:
                print(f"[ERROR] experiment_id must be 'name:version' or a YAML file path, got: {ref}")
                return 1
            name, version = ref.split(":", 1)
            store = _get_store()
            builder = PackageBuilder(store)
            builder.from_experiment_cfg(name, version)

        pkg_path = builder.build(output_dir)
        print(f"[OK] Package built: {pkg_path}")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 1
    return 0


# ── info ──────────────────────────────────────────────────────────────────────

def cmd_info(args: argparse.Namespace) -> int:
    from myrl.assets.packager import PackageReader
    import yaml

    try:
        reader = PackageReader(args.package_path)
        manifest = reader.manifest
        print(f"Package ID  : {manifest.package_id}")
        print(f"Experiment  : {manifest.experiment_name}")
        print(f"Created     : {manifest.created_at}")
        print(f"Source cfg  : {manifest.source_experiment_cfg}")
        print(f"Assets dir  : {reader.assets_dir}")
        print()
        print("Asset checksums:")
        print(yaml.dump(manifest.asset_checksums, sort_keys=False, allow_unicode=True))

        env_script = reader.get_env_script_path()
        algo_cfg = reader.get_algo_cfg_path()
        reward_pipeline = reader.get_reward_pipeline_path()
        print(f"env_script  : {env_script}")
        print(f"algo_cfg    : {algo_cfg}")
        print(f"reward_pipe : {reward_pipeline}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


# ── deploy ────────────────────────────────────────────────────────────────────

def cmd_deploy(args: argparse.Namespace) -> int:
    """用 rsync/scp 分发包到远程服务器。"""
    src = os.path.abspath(args.package_path)
    dst = args.destination

    if not os.path.exists(src):
        print(f"[ERROR] Package not found: {src}")
        return 1

    # 优先用 rsync，回退到 scp
    rsync = shutil.which("rsync")
    if rsync:
        cmd = ["rsync", "-avz", "--progress", src + "/", dst]
    else:
        cmd = ["scp", "-r", src, dst]

    print(f"[INFO] Deploying {src} → {dst}")
    try:
        subprocess.run(cmd, check=True)
        print("[OK] Deploy complete")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Deploy failed: {e}")
        return 1
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="myrl Asset Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # register
    p_reg = sub.add_parser("register", help="注册资产到 AssetStore")
    p_reg.add_argument("asset_type", help="资产类型（reward_fn/terrain/robot_model/...）")
    p_reg.add_argument("name_version", help="name:version（如 locomotion_rewards:1.0.0）")
    p_reg.add_argument("--source", required=True, help="源文件或目录路径")
    p_reg.add_argument("--description", default="", help="资产描述")

    # list
    p_list = sub.add_parser("list", help="列出所有资产")
    p_list.add_argument("--type", default=None, help="过滤资产类型")

    # show
    p_show = sub.add_parser("show", help="显示资产详情")
    p_show.add_argument("asset_id", help="name:version（如 locomotion_rewards:1.0.0）")

    # pack
    p_pack = sub.add_parser("pack", help="从 experiment YAML 或 asset_store 打包 .myrlpkg")
    p_pack.add_argument(
        "experiment_id",
        help="YAML 文件路径（如 myrl/assets/experiments/g1_locomotion_v1.yaml）"
             " 或 name:version（asset_store 旧路径）",
    )
    p_pack.add_argument("--output", "-o", default=".", help="输出目录")

    # info
    p_info = sub.add_parser("info", help="查看 .myrlpkg 包内容")
    p_info.add_argument("package_path", help=".myrlpkg 目录路径")

    # deploy
    p_dep = sub.add_parser("deploy", help="分发 .myrlpkg 到远程服务器")
    p_dep.add_argument("package_path", help=".myrlpkg 目录路径")
    p_dep.add_argument("destination", help="rsync/scp 目标（如 user@host:/path/）")

    args = parser.parse_args()

    dispatch = {
        "register": cmd_register,
        "list": cmd_list,
        "show": cmd_show,
        "pack": cmd_pack,
        "info": cmd_info,
        "deploy": cmd_deploy,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    import shutil
    sys.exit(main())
