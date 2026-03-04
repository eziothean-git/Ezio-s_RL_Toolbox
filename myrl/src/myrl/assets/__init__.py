"""myrl 资产解析器。

优先级：myrl/assets/{path}  >  回退（由调用方决定）

API:
    has_asset(relative_path)    → bool
    resolve_asset(relative_path) → str | None   (不存在返回 None)
    require_asset(relative_path) → str           (不存在抛 FileNotFoundError)
"""
from pathlib import Path

# src/myrl/assets/__init__.py 上 3 层 → myrl/ (repo root)
# parents[0] = src/myrl/assets
# parents[1] = src/myrl
# parents[2] = src
# parents[3] = myrl/  (容器内: /workspace/myrl)
_MYRL_ROOT = Path(__file__).resolve().parents[3]
MYRL_ASSETS_DIR = _MYRL_ROOT / "assets"


def has_asset(relative_path: str) -> bool:
    return (MYRL_ASSETS_DIR / relative_path).is_file()


def resolve_asset(relative_path: str) -> str | None:
    p = MYRL_ASSETS_DIR / relative_path
    return str(p) if p.is_file() else None


def resolve_asset_dir(relative_path: str) -> str | None:
    """解析资产目录（返回目录路径字符串），不存在则返回 None。"""
    p = MYRL_ASSETS_DIR / relative_path
    return str(p) if p.is_dir() else None


def require_asset(relative_path: str) -> str:
    p = resolve_asset(relative_path)
    if p is None:
        raise FileNotFoundError(
            f"[myrl] Required asset not found: {MYRL_ASSETS_DIR / relative_path}\n"
            "Place the file there or use resolve_asset() with a fallback."
        )
    return p
