#!/usr/bin/env bash
set -euo pipefail

# =========================
# utils
# =========================
log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
die() { log "FATAL: $*"; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

# =========================
# paths
# =========================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"
SCRIPTS_DIR="$REPO_ROOT/scripts"
THIRD_PARTY_DIR="$REPO_ROOT/third_party"
ISAACLAB_DIR="$THIRD_PARTY_DIR/isaaclab"

COMPOSE_FILE="$DOCKER_DIR/compose.yaml"
ENV_FILE="$DOCKER_DIR/.env"
RUN_DEV_SH="$SCRIPTS_DIR/run_dev.sh"

# =========================
# config (override via env)
# =========================
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-release/2.3.0}"

# For CI/non-interactive EULA accept:
#   MYRL_ACCEPT_EULA=Y bash scripts/bootstrap_dev.sh
MYRL_ACCEPT_EULA="${MYRL_ACCEPT_EULA:-${ACCEPT_EULA:-}}"

# =========================
# checks
# =========================
ensure_repo_layout() {
  [ -d "$DOCKER_DIR" ] || die "Missing docker dir: $DOCKER_DIR"
  [ -d "$SCRIPTS_DIR" ] || die "Missing scripts dir: $SCRIPTS_DIR"
  [ -d "$THIRD_PARTY_DIR" ] || mkdir -p "$THIRD_PARTY_DIR"

  [ -f "$COMPOSE_FILE" ] || die "Missing compose file: $COMPOSE_FILE"
  [ -f "$RUN_DEV_SH" ] || die "Missing runner script: $RUN_DEV_SH"
}

# =========================
# docker access
# =========================
in_docker_group() { id -nG | tr ' ' '\n' | grep -qx docker; }

docker_diag() {
  log "Docker diagnostics:"
  (docker --version || true)
  (systemctl is-active docker || true)
  (ls -l /var/run/docker.sock || true)
  (id -nG || true)
}

ensure_docker_daemon() {
  need_cmd docker || die "docker not found. Install Docker first."
  if ! systemctl is-active docker >/dev/null 2>&1; then
    docker_diag
    die "docker service is not active. Run: sudo systemctl enable --now docker"
  fi
}

run_in_docker_group() {
  local cmd="$1"
  # Apply docker group without logout/login. Prefer `sg`.
  if need_cmd sg; then
    exec sg docker -c "$cmd"
  fi
  if need_cmd newgrp; then
    exec newgrp docker -c "$cmd"
  fi
  die "Neither 'sg' nor 'newgrp' available. Please log out/in once after adding docker group."
}

ensure_docker_access_or_fix() {
  ensure_docker_daemon

  if docker info >/dev/null 2>&1; then
    log "Docker daemon accessible."
    return 0
  fi

  docker_diag

  if ! in_docker_group; then
    need_cmd sudo || die "sudo not found; cannot auto-add docker group."
    log "You are NOT in 'docker' group. Auto-fixing now (sudo required)..."
    sudo usermod -aG docker "${USER:?USER not set}"
    log "Added '$USER' to docker group."
    log "Re-running bootstrap in docker group (you may see your shell banner again)."

    local self="$0"
    local rerun
    rerun=$(cat <<EOF
export ISAACLAB_REPO=$(printf %q "$ISAACLAB_REPO");
export ISAACLAB_REF=$(printf %q "$ISAACLAB_REF");
export MYRL_ACCEPT_EULA=$(printf %q "$MYRL_ACCEPT_EULA");
bash $(printf %q "$self")
EOF
)
    run_in_docker_group "$rerun"
  fi

  log "You appear to be in docker group, but docker is still not accessible."
  log "Try: sudo systemctl restart docker"
  die "Docker not accessible."
}

# =========================
# NVIDIA Container Toolkit (official stable/deb + nvidia-ctk)
# =========================
configure_nvidia_toolkit_repo() {
  need_cmd sudo || die "sudo not found; cannot install NVIDIA Container Toolkit."
  need_cmd curl || die "curl not found."
  need_cmd gpg  || die "gpg not found (install gnupg2)."

  log "Configuring NVIDIA Container Toolkit apt repo (official stable/deb)..."
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends curl gnupg2 ca-certificates

  local keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  sudo rm -f "$keyring"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o "$keyring"

  local list_url="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
  log "Fetching repo list: $list_url"

  # Use a pipe into sed+tee like NVIDIA docs; retry with IPv4 if TLS/network flaky.
  if ! curl -fsSL "$list_url" \
      | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null; then
    log "Repo list fetch failed; retrying with IPv4 (-4)..."
    curl -4 -fsSL "$list_url" \
      | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null \
      || die "Failed to configure NVIDIA repo list (network/TLS). Fix DNS/proxy/network then re-run."
  fi

  sudo apt-get update -y
}

install_nvidia_container_toolkit() {
  need_cmd sudo || die "sudo not found."
  log "Installing nvidia-container-toolkit..."
  sudo apt-get install -y nvidia-container-toolkit
}

configure_docker_runtime_for_nvidia() {
  need_cmd sudo || die "sudo not found."
  need_cmd nvidia-ctk || die "nvidia-ctk not found (toolkit install failed?)"

  log "Configuring Docker runtime for NVIDIA (nvidia-ctk)..."
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
}

check_docker_gpu_support() {
  log "Checking Docker GPU support (--gpus all)..."
  # Use a CUDA base image just to run nvidia-smi.
  if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    log "Docker GPU support: OK"
    return 0
  fi

  log "Docker GPU support: NOT working yet."
  log "Installing/configuring NVIDIA Container Toolkit (official method)..."

  configure_nvidia_toolkit_repo
  install_nvidia_container_toolkit
  configure_docker_runtime_for_nvidia

  log "Re-checking Docker GPU support..."
  if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    log "Docker GPU support: OK (after toolkit install)"
    return 0
  fi

  log "Diagnostics (run manually):"
  log "  docker info | sed -n '1,200p'"
  log "  sudo journalctl -u docker --no-pager | tail -n 200"
  die "Docker GPU support still not available."
}

# =========================
# Isaac Sim EULA -> docker/.env
# =========================
ensure_isaac_eula() {
  mkdir -p "$DOCKER_DIR"

  # already accepted?
  if [ -f "$ENV_FILE" ] && grep -qE '^ACCEPT_EULA=Y$' "$ENV_FILE"; then
    log "Isaac Sim EULA already accepted: $ENV_FILE"
    return 0
  fi

  # non-interactive accept
  if [ "${MYRL_ACCEPT_EULA}" = "Y" ] || [ "${MYRL_ACCEPT_EULA}" = "y" ]; then
    log "MYRL_ACCEPT_EULA=Y provided. Writing $ENV_FILE"
    [ -f "$ENV_FILE" ] && grep -vE '^ACCEPT_EULA=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
    [ -f "$ENV_FILE.tmp" ] && mv "$ENV_FILE.tmp" "$ENV_FILE"
    echo "ACCEPT_EULA=Y" >> "$ENV_FILE"
    return 0
  fi

  log "NVIDIA Isaac Sim Additional Software and Materials License must be accepted."
  log "URL:"
  log "  https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/"
  echo
  read -r -p "Do you accept the Isaac Sim EULA? [y/N]: " ans
  case "$ans" in
    y|Y)
      [ -f "$ENV_FILE" ] && grep -vE '^ACCEPT_EULA=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
      [ -f "$ENV_FILE.tmp" ] && mv "$ENV_FILE.tmp" "$ENV_FILE"
      echo "ACCEPT_EULA=Y" >> "$ENV_FILE"
      log "EULA accepted. Stored at: $ENV_FILE"
      ;;
    *)
      die "EULA not accepted. Tip: MYRL_ACCEPT_EULA=Y bash scripts/bootstrap_dev.sh"
      ;;
  esac
}

# =========================
# Fetch IsaacLab (host-side source)
# =========================
fetch_isaaclab() {
  if [ -d "$ISAACLAB_DIR/.git" ]; then
    log "IsaacLab already exists: $ISAACLAB_DIR"
    log "  repo: $ISAACLAB_REPO"
    log "  ref : $ISAACLAB_REF"
    return 0
  fi

  need_cmd git || die "git not found. Install git first."
  log "Cloning IsaacLab (for host-side editing)..."
  log "  repo: $ISAACLAB_REPO"
  log "  ref : $ISAACLAB_REF"
  git clone --depth 1 --branch "$ISAACLAB_REF" "$ISAACLAB_REPO" "$ISAACLAB_DIR"
}

# =========================
# Main
# =========================
log "Repo root  : $REPO_ROOT"
log "Docker dir : $DOCKER_DIR"
log "3rd party  : $THIRD_PARTY_DIR"

ensure_repo_layout
fetch_isaaclab
ensure_docker_access_or_fix
check_docker_gpu_support
ensure_isaac_eula

log "Bootstrap OK."
log "Now run:"
log "  bash \"$RUN_DEV_SH\""
