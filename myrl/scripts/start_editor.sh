#!/usr/bin/env bash
# start_editor.sh — 一键启动 myrl Editor WebUI
#
# 在宿主机直接运行 train_manager（不再进容器）。
# 容器的启停通过 WebUI 控制。
#
# 用法:
#   bash myrl/scripts/start_editor.sh [--port 7001] [--no-browser]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT=7001
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    *) shift ;;
  esac
done

log() { printf "\033[36m[editor]\033[0m %s\n" "$*"; }

# 检查是否已在运行
if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  log "train_manager already running on :$PORT"
else
  log "Starting train_manager on :$PORT..."
  cd "$REPO_ROOT"
  python3 myrl/scripts/train_manager.py --port "$PORT" &
  MANAGER_PID=$!
  disown $MANAGER_PID 2>/dev/null || true

  # 等待就绪
  for i in $(seq 1 10); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    log "train_manager ready (PID $MANAGER_PID)"
  else
    log "WARNING: train_manager may not have started"
  fi
fi

URL="http://localhost:$PORT"
log "Editor URL: $URL"
if [[ "$OPEN_BROWSER" == "1" ]]; then
  (xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || true) &
  disown 2>/dev/null || true
fi
