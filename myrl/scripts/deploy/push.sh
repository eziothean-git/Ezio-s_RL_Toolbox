#!/usr/bin/env bash
set -euo pipefail
# ── push.sh — 推送代码到远程服务器 ──────────────────────────────────
#
# 用法:  bash myrl/scripts/deploy/push.sh user@server [remote_dir]
#
# 默认远程目录: ~/Ezio\'s_RL_Toolbox
# 排除: .git, work/, __pycache__, *.pyc, .env

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -z "${1:-}" ]; then
    echo "用法: bash $0 user@server [remote_dir]"
    exit 1
fi

TARGET="$1"
REMOTE_DIR="${2:-~/Ezios_RL_Toolbox}"

echo "[push] 源目录: $REPO_ROOT"
echo "[push] 目标:   $TARGET:$REMOTE_DIR"
echo ""

# 确保远程目录存在
ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='myrl/work/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='*.tar.gz' \
    --exclude='image.png' \
    "$REPO_ROOT/" "$TARGET:$REMOTE_DIR/"

echo ""
echo "[push] 完成。远程目录: $TARGET:$REMOTE_DIR"
echo "[push] 下一步在服务器上运行:"
echo "       ssh $TARGET"
echo "       cd $REMOTE_DIR && bash myrl/scripts/deploy/setup.sh"
