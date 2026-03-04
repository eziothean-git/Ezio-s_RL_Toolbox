#!/usr/bin/env bash
set -euo pipefail

log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
die() { log "FATAL: $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"
BOOTSTRAP="$REPO_ROOT/scripts/bootstrap_dev.sh"

in_docker_group() { id -nG | tr ' ' '\n' | grep -qx docker; }

log "Repo root: $REPO_ROOT"
log "Docker dir: $DOCKER_DIR"

# 1) System bootstrap
bash "$BOOTSTRAP"

# 2) Prepare a temp script that runs compose in the correct directory
TMP_DIR="${TMPDIR:-/tmp}"
TMP_SCRIPT="$TMP_DIR/myrl_run_compose_$$.sh"
cleanup() { rm -f "$TMP_SCRIPT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

cat > "$TMP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$DOCKER_DIR"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="\${COMPOSE_DOCKER_CLI_BUILD:-1}"

# 允许容器内 root 用户连接本机 X Server（GUI 显示）
xhost +SI:localuser:root 2>/dev/null || true

echo "[\$(date '+%H:%M:%S')] docker compose build..."
docker compose build

echo "[\$(date '+%H:%M:%S')] docker compose run --rm dev..."
docker compose run --rm dev
EOF
chmod +x "$TMP_SCRIPT"

# 3) If docker group isn't effective in this session, run the temp script via sg/newgrp
if ! in_docker_group; then
  log "Current session is NOT in docker group yet."
  log "Running compose via 'sg docker' for this session (no logout needed)."

  if command -v sg >/dev/null 2>&1; then
    exec sg docker -c "bash \"$TMP_SCRIPT\""
  fi

  if command -v newgrp >/dev/null 2>&1; then
    exec newgrp docker -c "bash \"$TMP_SCRIPT\""
  fi

  die "Neither 'sg' nor 'newgrp' found. Please log out/in once, then re-run."
fi

# 4) Otherwise run normally
bash "$TMP_SCRIPT"
