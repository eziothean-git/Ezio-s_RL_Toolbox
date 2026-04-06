#!/usr/bin/env bash
# start_editor.sh — 一键启动 myrl Editor WebUI
#
# 宿主机运行 train_manager，训练通过 docker exec 委派到容器。
# 自动杀掉旧进程，后台运行不占终端，日志写文件。
#
# 用法:
#   bash myrl/scripts/start_editor.sh [--port 7001] [--no-browser] [--container myrl-dev]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT=7001
CONTAINER="myrl-dev"
OPEN_BROWSER=1
LOG_FILE="$REPO_ROOT/myrl/work/editor.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    *) shift ;;
  esac
done

log() { printf "\033[36m[editor]\033[0m %s\n" "$*"; }

# ── 杀掉旧进程 ──
OLD_PIDS=$(pgrep -f "train_manager.py.*--port $PORT" 2>/dev/null || true)
if [[ -n "$OLD_PIDS" ]]; then
  log "Killing old train_manager (PIDs: $OLD_PIDS)..."
  echo "$OLD_PIDS" | xargs kill 2>/dev/null || true
  sleep 1
  # 确保死干净
  echo "$OLD_PIDS" | xargs kill -9 2>/dev/null || true
fi

# ── 确保日志目录 ──
mkdir -p "$(dirname "$LOG_FILE")"

# ── 选择 Python（优先 myrl-train env，有 torch 才能自动发现 reward schema） ──
MYRL_PYTHON="${MYRL_PYTHON:-}"
if [[ -z "$MYRL_PYTHON" ]]; then
  TRAIN_PYTHON="$HOME/myrl_work/.mamba/envs/myrl-train/bin/python3"
  if [[ -x "$TRAIN_PYTHON" ]]; then
    MYRL_PYTHON="$TRAIN_PYTHON"
  else
    MYRL_PYTHON="python3"
  fi
fi

# ── 启动 ──
log "Starting train_manager on :$PORT (container=$CONTAINER, python=$MYRL_PYTHON)..."
cd "$REPO_ROOT"

PYTHONPATH="myrl/src:${PYTHONPATH:-}" \
  nohup "$MYRL_PYTHON" myrl/scripts/train_manager.py \
    --port "$PORT" \
    --container "$CONTAINER" \
  > "$LOG_FILE" 2>&1 &

MANAGER_PID=$!
disown "$MANAGER_PID" 2>/dev/null || true

# ── 等待就绪 ──
for _ in $(seq 1 10); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  log "train_manager ready (PID $MANAGER_PID)"
  log "Log: $LOG_FILE"
else
  log "WARNING: train_manager may not have started — check $LOG_FILE"
fi

# ── 打开浏览器 ──
URL="http://localhost:$PORT"
log "Editor: $URL"
if [[ "$OPEN_BROWSER" == "1" ]]; then
  (xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || true) &
  disown 2>/dev/null || true
fi
