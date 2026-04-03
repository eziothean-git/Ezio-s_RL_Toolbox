#!/usr/bin/env bash
# quick_start.sh — 一键 build + 启动开发容器（带 X11 显示）
# 用法：bash myrl/scripts/quick_start.sh [--headless] [--no-build]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="$REPO_ROOT/myrl/docker"

HEADLESS=0
DO_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --headless)  HEADLESS=1 ;;
    --no-build)  DO_BUILD=0 ;;
  esac
done

echo "=== myrl quick_start ==="
echo "  repo:     $REPO_ROOT"
echo "  headless: $HEADLESS"
echo "  build:    $DO_BUILD"
echo ""

# ── X11 认证（Wayland+XWayland 兼容）──────────────────────────────────
if [ "$HEADLESS" = "0" ] && [ -n "${DISPLAY:-}" ]; then
  echo "[x11] 生成 xauth 文件..."
  DOCKER_XAUTH=/tmp/.docker-xauth
  touch "$DOCKER_XAUTH"
  if command -v xauth >/dev/null 2>&1; then
    xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | \
      xauth -f "$DOCKER_XAUTH" nmerge - 2>/dev/null || true
  fi
  chmod 644 "$DOCKER_XAUTH" 2>/dev/null || true
  xhost +SI:localuser:root 2>/dev/null || true
  echo "[x11] DISPLAY=$DISPLAY, xauth ready"
else
  echo "[x11] headless 模式或无 DISPLAY，跳过"
fi

# ── Docker build ──────────────────────────────────────────────────────
cd "$DOCKER_DIR"

if [ "$DO_BUILD" = "1" ]; then
  echo ""
  echo "[docker] building myrl/isaaclab-dev:2.3.2 ..."
  echo "[docker] 这可能需要几分钟（首次）或几秒（有缓存）"
  echo ""
  DOCKER_BUILDKIT=1 docker compose build
  echo ""
  echo "[docker] build 完成"
else
  echo "[docker] 跳过 build（--no-build）"
fi

# ── 启动容器 ──────────────────────────────────────────────────────────
echo ""
echo "[docker] 启动容器..."
echo "  GUI 模式: $([ "$HEADLESS" = "0" ] && echo "是（Isaac Lab viewport 应出现在屏幕上）" || echo "否")"
echo ""
echo "  容器内可运行："
echo "    # GUI 训练"
echo "    python3 scripts/train.py --task myrl/Locomotion-Flat-G1Smoke-v0 --num_envs 1 --max_iterations 10"
echo ""
echo "    # headless 训练 + signal viewer"
echo "    python3 scripts/train.py --task myrl/Locomotion-Flat-G1Smoke-v0 --headless --num_envs 4 --max_iterations 50 --signal_server_port 7002"
echo ""
echo "    # 另一个终端查看信号"
echo "    python3 scripts/signal_viewer.py --port 7002"
echo ""
echo "=========================================="

HEADLESS=$HEADLESS docker compose run --rm dev
