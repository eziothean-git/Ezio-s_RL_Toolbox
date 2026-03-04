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
# 初始化 Isaac Sim 环境（PYTHONPATH / LD_LIBRARY_PATH 等）并把 python3 加入 PATH
#
# 必须通过 source setup_python_env.sh 来设置 PYTHONPATH，直接加 bin 到 PATH 不够——
# /isaac-sim/kit/python/lib/python3.11/site-packages 及 python_packages 需要被加入才能
# import torch / toml 等。
# -------------------------
setup_isaac_env() {
  local isaac_dir="/isaac-sim"
  local setup_sh="$isaac_dir/setup_python_env.sh"

  # python.sh 在 source setup_python_env.sh 之前先 export 这三个变量；
  # 缺少任何一个都会导致 AppLauncher 报 KeyError。
  export CARB_APP_PATH="$isaac_dir/kit"
  export ISAAC_PATH="$isaac_dir"
  export EXP_PATH="$isaac_dir/apps"

  if [ -f "$setup_sh" ]; then
    # setup_python_env.sh 内部引用了多个可能未定义的变量（PYTHONPATH, LD_LIBRARY_PATH...）
    # 用 set +u 临时关掉 unbound variable 检查，source 结束后恢复
    set +u
    # shellcheck disable=SC1090
    source "$setup_sh"
    set -u
    log "Sourced Isaac env: $setup_sh (CARB/ISAAC/EXP paths set)"
  else
    log "WARNING: $setup_sh not found; python3/torch may not be importable."
  fi

  # 把 kit python bin 加入 PATH，使 `python3` 可直接调用
  local bin_dir="$isaac_dir/kit/python/bin"
  if [ -x "$bin_dir/python3" ] && [[ ":$PATH:" != *":$bin_dir:"* ]]; then
    export PATH="$bin_dir:$PATH"
    log "Added kit python to PATH: $bin_dir"
  fi
}
setup_isaac_env

# ── Source ROS2 Humble（如果已安装）──────────────────────────────────────
# Dockerfile 里装了 ros-humble-ros-base，source 后 rclpy 才可用。
# 必须在 Isaac env 之后 source，避免 ROS python path 覆盖 kit python。
setup_ros2_env() {
  local ros_setup="/opt/ros/humble/setup.bash"
  if [ -f "$ros_setup" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$ros_setup"
    set -u
    log "Sourced ROS2 env: $ros_setup"
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  else
    log "ROS2 not found (skipping); install ros-humble-ros-base if needed"
  fi
}
setup_ros2_env

# ─────────────────────────────────────────────────────────────────────────（不用 pip install，不触碰 kit python 依赖树）
#
# instinct_rl / instinctlab 以 rw 挂载，editable 修改即时生效；
# myrl 通过 /workspace/myrl 挂载。
# -------------------------
INSTINCT_RL_SRC="${INSTINCT_RL_DIR:-/workspace/myrl/third_party/instinct_rl}"
INSTINCTLAB_SRC="${INSTINCTLAB_DIR:-/workspace/myrl/third_party/instinctlab}"
MYRL_SRC="/workspace/myrl/src"

add_upstream_to_pythonpath() {
  local additions=""

  if [ -d "$INSTINCT_RL_SRC" ]; then
    additions="$INSTINCT_RL_SRC:$additions"
    log "PYTHONPATH += $INSTINCT_RL_SRC (instinct_rl)"
  else
    log "WARNING: instinct_rl not found at $INSTINCT_RL_SRC (check volume mount)"
  fi

  if [ -d "$INSTINCTLAB_SRC/source/instinctlab" ]; then
    additions="$INSTINCTLAB_SRC/source/instinctlab:$additions"
    log "PYTHONPATH += $INSTINCTLAB_SRC/source/instinctlab (instinctlab)"
  else
    log "WARNING: instinctlab not found at $INSTINCTLAB_SRC/source/instinctlab (check volume mount)"
  fi

  if [ -d "$MYRL_SRC" ]; then
    additions="$MYRL_SRC:$additions"
    log "PYTHONPATH += $MYRL_SRC (myrl)"
  else
    log "WARNING: myrl src not found at $MYRL_SRC"
  fi

  if [ -n "$additions" ]; then
    export PYTHONPATH="${additions}${PYTHONPATH:-}"
  fi
}
add_upstream_to_pythonpath

# -------------------------
# Git safe.directory（挂载 repo 与容器 root 所有者不同，git 拒绝操作）
# instinct_rl 的 OnPolicyRunner 在 learn() 里调用 store_code_state()，
# 需要读取 /opt/instinct_rl 的 git 状态，否则报 "dubious ownership"。
# -------------------------
git config --global --add safe.directory /workspace/myrl/third_party/instinct_rl 2>/dev/null || true
git config --global --add safe.directory /workspace/myrl/third_party/instinctlab 2>/dev/null || true
git config --global --add safe.directory /workspace/myrl 2>/dev/null || true


# -------------------------
# 安装 instinctlab 运行时缺失依赖（持久化到工作目录，不污染 kit python 环境）
#
# instinctlab 的 motion_reference 在顶层 import pytorch_kinematics，
# 即使不用 motion reference 功能也会触发。
# 策略：用 pip install --target 把包装到 $WORK/.site-packages（持久化 volume），
# 再把该目录加入 PYTHONPATH。kit python 本体不受影响。
# scipy / lxml / prettytable 已在 kit 环境中，无需重装。
# -------------------------
INSTINCTLAB_DEPS_SITE="$WORK/.site-packages"
INSTINCTLAB_DEPS_STAMP="$WORK/.deps/instinctlab_runtime_deps.installed"

install_instinctlab_runtime_deps() {
  # 只有 instinctlab 挂载时才需要；stamp 文件存在则已完成
  [ -d "$INSTINCTLAB_SRC/source/instinctlab" ] || return 0
  [ -f "$INSTINCTLAB_DEPS_STAMP" ] && return 0

  log "Installing instinctlab runtime deps to $INSTINCTLAB_DEPS_SITE (--no-deps, persistent)..."
  mkdir -p "$INSTINCTLAB_DEPS_SITE" "$WORK/.cache/pip" "$WORK/.deps"

  PIP_CACHE_DIR="$WORK/.cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 \
    python3 -m pip install -q --no-deps \
      --target "$INSTINCTLAB_DEPS_SITE" \
      pytorch_seed \
      arm_pytorch_utilities \
      pytorch_kinematics \
    && {
      date > "$INSTINCTLAB_DEPS_STAMP"
      log "instinctlab runtime deps installed. Stamp: $INSTINCTLAB_DEPS_STAMP"
    } || {
      log "WARNING: instinctlab runtime deps install failed (will retry next start)."
    }
}
install_instinctlab_runtime_deps

# -------------------------
# 安装 MuJoCo socket 服务端依赖（持久化到工作目录，不污染 kit python 环境）
#
# mujoco: sim2sim 物理仿真后端
# msgpack-numpy: TCP socket 协议的序列化层
# -------------------------
install_mujoco_server_deps() {
  local stamp="$WORK/.deps/mujoco_server_deps.installed"
  [ -f "$stamp" ] && return 0
  log "Installing mujoco server deps to $INSTINCTLAB_DEPS_SITE (persistent)..."
  mkdir -p "$INSTINCTLAB_DEPS_SITE" "$WORK/.cache/pip" "$WORK/.deps"
  PIP_CACHE_DIR="$WORK/.cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 \
    python3 -m pip install -q \
      --target "$INSTINCTLAB_DEPS_SITE" \
      mujoco \
      msgpack-numpy \
    && { date > "$stamp"; log "mujoco server deps installed."; } \
    || log "WARNING: mujoco server deps install failed (will retry next start)."
}
install_mujoco_server_deps

# 把持久化 site-packages 加入 PYTHONPATH（无论是否刚刚安装，都需要）
if [ -d "$INSTINCTLAB_DEPS_SITE" ]; then
  export PYTHONPATH="$INSTINCTLAB_DEPS_SITE:${PYTHONPATH:-}"
  log "PYTHONPATH += $INSTINCTLAB_DEPS_SITE (instinctlab runtime deps)"
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
