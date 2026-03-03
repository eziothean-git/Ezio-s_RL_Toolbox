#!/usr/bin/env bash
set -euo pipefail

log() { printf "[myrl] %s\n" "$*"; }

# -------------------------
# Basic info
# -------------------------
log "container: $(hostname)"
log "mode=${MYRL_MODE:-dev} headless=${HEADLESS:-1}"
log "repo=/workspace/myrl workdir=${MYRL_WORKDIR:-/workspace/myrl_work}"

REPO="/workspace/myrl"
WORK="${MYRL_WORKDIR:-/workspace/myrl_work}"
mkdir -p "$WORK" "$WORK/.cache" "$WORK/.logs" "$WORK/.deps" || true

# -------------------------
# Runtime server settings (IPC boundary)
# -------------------------
# You will later implement myrl/runtime/server.py inside this repo.
# For now, we only wire the plumbing and make it optional.
MYRL_RUNTIME_SERVER="${MYRL_RUNTIME_SERVER:-0}"   # 1 to auto-start runtime server
MYRL_RUNTIME_TRANSPORT="${MYRL_RUNTIME_TRANSPORT:-tcp}"  # tcp | unix (future)
MYRL_RUNTIME_HOST="${MYRL_RUNTIME_HOST:-0.0.0.0}"
MYRL_RUNTIME_PORT="${MYRL_RUNTIME_PORT:-52031}"
MYRL_RUNTIME_UNIX="${MYRL_RUNTIME_UNIX:-$WORK/myrl_runtime.sock}"
MYRL_RUNTIME_LOG="${MYRL_RUNTIME_LOG:-$WORK/.logs/runtime_server.log}"

export MYRL_WORKDIR="$WORK"
export MYRL_RUNTIME_TRANSPORT MYRL_RUNTIME_HOST MYRL_RUNTIME_PORT MYRL_RUNTIME_UNIX

# -------------------------
# Locate Isaac/Kit python launcher (do NOT install packages into it)
# -------------------------
pick_kit_python() {
  # Prefer official wrapper if present (it usually sets USD/Omni env vars correctly)
  if command -v python.sh >/dev/null 2>&1; then echo "python.sh"; return 0; fi
  if [ -x "/isaac-sim/python.sh" ]; then echo "/isaac-sim/python.sh"; return 0; fi
  if [ -x "/isaaclab/python.sh" ]; then echo "/isaaclab/python.sh"; return 0; fi

  # Fallback to embedded python if exposed
  if [ -x "/isaac-sim/kit/python/bin/python3" ]; then echo "/isaac-sim/kit/python/bin/python3"; return 0; fi
  if [ -x "/kit/python/bin/python3" ]; then echo "/kit/python/bin/python3"; return 0; fi

  # LAST resort: system python (not always correct for Isaac)
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return 0; fi
  if command -v python >/dev/null 2>&1; then echo "python"; return 0; fi
  return 1
}

KITPY=""
if KITPY="$(pick_kit_python 2>/dev/null)"; then
  log "kit python launcher: $KITPY"
else
  log "WARNING: no python launcher found. Runtime server cannot start."
fi

# -------------------------
# (Optional) Tools-in-container venv (DEBUG ONLY)
# Default OFF. Your tool env should live outside runtime container.
# -------------------------
MYRL_TOOLS_IN_CONTAINER="${MYRL_TOOLS_IN_CONTAINER:-0}"  # 1 enables venv + pip install
REQ="/opt/myrl/requirements-dev.txt"
VENV="$WORK/.venv/dev"
STAMP="$WORK/.deps/dev_venv.installed"
PIP_CACHE="$WORK/.cache/pip"

maybe_setup_tools_venv() {
  [ "$MYRL_TOOLS_IN_CONTAINER" = "1" ] || return 0

  log "MYRL_TOOLS_IN_CONTAINER=1: enabling tools venv inside runtime container (debug mode)."
  mkdir -p "$PIP_CACHE" "$WORK/.deps" || true

  # Create venv using kit python if possible; do NOT touch base site-packages
  if [ ! -d "$VENV" ]; then
    log "Creating venv: $VENV"
    "$KITPY" -m venv "$VENV" || {
      log "venv creation failed. This image may not ship venv module."
      log "Install python3-venv (if apt exists) OR use external tool env (recommended)."
      return 0
    }
  fi

  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  export PIP_CACHE_DIR="$PIP_CACHE"
  export PIP_DISABLE_PIP_VERSION_CHECK=1

  if [ -f "$REQ" ] && [ ! -f "$STAMP" ]; then
    log "Installing dev deps into venv (isolated)..."
    python -m pip install -U pip
    python -m pip install "setuptools<82" "wheel<0.46"
    python -m pip install "packaging<24" "docstring-parser==0.16" || true
    python -m pip install --upgrade-strategy only-if-needed -r "$REQ"
    date > "$STAMP"
    log "Dev deps installed (venv). Stamp: $STAMP"
  else
    log "Dev deps already installed (venv) or requirements missing."
  fi
}

# -------------------------
# Start runtime server (optional)
# -------------------------
start_runtime_server() {
  [ "$MYRL_RUNTIME_SERVER" = "1" ] || return 0
  [ -n "$KITPY" ] || { log "No kit python; cannot start runtime server."; return 0; }

  # Server entrypoint you will implement later.
  # Convention: myrl/runtime/server.py inside repo.
  local server_py="$REPO/myrl/runtime/server.py"
  if [ ! -f "$server_py" ]; then
    log "Runtime server requested but missing: $server_py"
    log "Skipping server start. (Create myrl/runtime/server.py later.)"
    return 0
  fi

  log "Starting myrl runtime server..."
  log "  transport=$MYRL_RUNTIME_TRANSPORT host=$MYRL_RUNTIME_HOST port=$MYRL_RUNTIME_PORT"
  log "  log=$MYRL_RUNTIME_LOG"

  # Use nohup so it survives if you open interactive shells
  # If you prefer foreground server, run with: MYRL_RUNTIME_SERVER_FG=1
  local fg="${MYRL_RUNTIME_SERVER_FG:-0}"
  if [ "$fg" = "1" ]; then
    "$KITPY" "$server_py" \
      --transport "$MYRL_RUNTIME_TRANSPORT" \
      --host "$MYRL_RUNTIME_HOST" \
      --port "$MYRL_RUNTIME_PORT" \
      --unix "$MYRL_RUNTIME_UNIX"
  else
    nohup "$KITPY" "$server_py" \
      --transport "$MYRL_RUNTIME_TRANSPORT" \
      --host "$MYRL_RUNTIME_HOST" \
      --port "$MYRL_RUNTIME_PORT" \
      --unix "$MYRL_RUNTIME_UNIX" \
      >"$MYRL_RUNTIME_LOG" 2>&1 &
    log "Runtime server PID=$!"
  fi
}

# -------------------------
# Main flow
# -------------------------
# IMPORTANT: keep IsaacLab base env untouched by default.
if [ "$MYRL_TOOLS_IN_CONTAINER" = "1" ]; then
  maybe_setup_tools_venv
else
  log "Tools venv in container: OFF (recommended). Use external tool env + IPC."
fi

start_runtime_server

log "Entering CMD: $*"
exec "$@"
