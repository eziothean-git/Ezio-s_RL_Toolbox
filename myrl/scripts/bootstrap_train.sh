#!/usr/bin/env bash
set -euo pipefail

# =========================
# MyRL Train-time Bootstrap (clean, non-interactive)
# Assumed repo layout (single-layer):
#   <REPO_ROOT>/
#     env/train.yml
#     scripts/bootstrap_train.sh
#     third_party/
#       rsl_rl/
#       legged_gym/
#       isaac_gym/IsaacGym_Preview_4_Package.tar.gz
#
# Goals:
# - No sudo required
# - No docker required
# - No "eval shell hook" / no activation; use micromamba run -p
# - Isolate HOME to avoid ~/.conda permission junk
# - Auto select PyTorch index by GPU compute capability (A100/4090/5090/Blackwell)
# - IsaacGym installed from tarball, injected via .pth
# - pip editable for third_party with --no-deps (avoid PyPI isaacgym)
# =========================

log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

# ---------- repo paths ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="$REPO_ROOT/env/train.yml"
SCRIPTS_DIR="$REPO_ROOT/scripts"
THIRD_PARTY_DIR="$REPO_ROOT/third_party"

# ---------- workdir ----------
# NOTE: micromamba can break when paths contain spaces or single quotes.
# Default to a safe path outside the repo.
WORKDIR="${WORKDIR:-$HOME/myrl_work}"
TOOLS_DIR="$WORKDIR/.tools"
MAMBA_ROOT="$WORKDIR/.mamba"
ENV_PREFIX="$MAMBA_ROOT/envs/myrl-train"

CACHE_DIR="$WORKDIR/.cache/myrl"
BOOT_CONDARC="$WORKDIR/condarc"

ISAACGYM_TAR="$THIRD_PARTY_DIR/isaac_gym/IsaacGym_Preview_4_Package.tar.gz"
ISAACGYM_DIR="$WORKDIR/third_party/isaacgym"

if [[ "$WORKDIR" == *"'"* || "$WORKDIR" == *" "* ]]; then
  log "FATAL: WORKDIR path contains space or single-quote: $WORKDIR"
  log "Please set WORKDIR to a safe path, e.g.:"
  log "  WORKDIR=\"$HOME/myrl_work\" bash scripts/bootstrap_train.sh"
  exit 1
fi

# ---------- isolate HOME to avoid ~/.conda permission junk ----------
# libmamba may try to write ~/.conda/environments.txt even without activation.
# Your real ~/.conda is likely broken by historical setup; isolate HOME under WORKDIR.
BOOT_HOME="$WORKDIR/home"
export HOME="$BOOT_HOME"
mkdir -p "$HOME/.conda"
: > "$HOME/.conda/environments.txt" || true

mkdir -p "$WORKDIR" "$TOOLS_DIR" "$MAMBA_ROOT" "$CACHE_DIR" "$WORKDIR/third_party"

# ---------- sanity checks ----------
if [ ! -f "$ENV_YML" ]; then
  log "FATAL: Missing env file: $ENV_YML"
  exit 1
fi

if [ ! -d "$THIRD_PARTY_DIR/rsl_rl" ] || [ ! -d "$THIRD_PARTY_DIR/legged_gym" ]; then
  log "FATAL: Missing third_party libs under: $THIRD_PARTY_DIR"
  log "Expect:"
  log "  $THIRD_PARTY_DIR/rsl_rl"
  log "  $THIRD_PARTY_DIR/legged_gym"
  exit 1
fi

if [ ! -f "$ISAACGYM_TAR" ]; then
  log "FATAL: Missing IsaacGym tarball at: $ISAACGYM_TAR"
  exit 1
fi

# ---------- isolate conda config (avoid defaults/anaconda channel junk) ----------
cat > "$BOOT_CONDARC" <<'YAML'
channels:
  - conda-forge
channel_priority: strict
show_channel_urls: true
always_yes: true
YAML
export CONDARC="$BOOT_CONDARC"

# ---------- micromamba ----------
MAMBA_EXE="$TOOLS_DIR/bin/micromamba"
if [ ! -x "$MAMBA_EXE" ]; then
  log "1/8 Download micromamba..."
  cd "$TOOLS_DIR"
  curl -L -o micromamba.tar.bz2 "https://micro.mamba.pm/api/micromamba/linux-64/latest"
  tar -xjf micromamba.tar.bz2
  chmod +x "$MAMBA_EXE"
fi

log "Micromamba: $("$MAMBA_EXE" --version || true)"
log "Repo root : $REPO_ROOT"
log "Workdir   : $WORKDIR"
log "Env prefix: $ENV_PREFIX"
log "ThirdParty: $THIRD_PARTY_DIR"
log "IsaacGym tar: $ISAACGYM_TAR"
log "IsaacGym out: $ISAACGYM_DIR"
log "HOME (isolated): $HOME"
log "CONDARC: $CONDARC"

# ---------- create/update env ----------
log "2/8 Create/update env..."
"$MAMBA_EXE" --root-prefix "$MAMBA_ROOT" create -y -p "$ENV_PREFIX" -f "$ENV_YML"

# helper to run commands inside env without activation
mmrun() {
  "$MAMBA_EXE" --root-prefix "$MAMBA_ROOT" run -p "$ENV_PREFIX" "$@"
}

log "3/8 Verify Python in env..."
mmrun python -V
mmrun python -m pip install -U pip

# ---------- pip packages ----------
log "4/8 Install pip packages..."
mmrun python -m pip install rich typer wandb tqdm

# ---------- torch (auto select) ----------
detect_compute_cap() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cc=""
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
    if [ -n "$cc" ]; then
      echo "$cc"
      return 0
    fi
    cc="$(nvidia-smi -q 2>/dev/null | grep -m1 -E "CUDA Capability" | awk '{print $NF}' | tr -d ' ')"
    if [ -n "$cc" ]; then
      echo "$cc"
      return 0
    fi
  fi
  echo ""
}

choose_torch_index() {
  local cc="$1"
  local pre="${TORCH_PRE:-0}"
  local forced="${TORCH_INDEX_URL:-}"

  if [ -n "$forced" ]; then
    echo "$forced"
    return 0
  fi

  if [ -z "$cc" ]; then
    echo "https://download.pytorch.org/whl/cpu"
    return 0
  fi

  local major="${cc%%.*}"
  local minor="${cc#*.}"
  minor="${minor:-0}"
  local cc_int=$((major*10 + minor))

  # cc >= 12.0 (sm_120) -> cu128
  if [ "$cc_int" -ge 120 ]; then
    if [ "$pre" = "1" ]; then
      echo "https://download.pytorch.org/whl/nightly/cu128"
    else
      echo "https://download.pytorch.org/whl/cu128"
    fi
    return 0
  fi

  # default for A100(8.0), 4090(8.9), etc -> cu124
  if [ "$pre" = "1" ]; then
    echo "https://download.pytorch.org/whl/nightly/cu124"
  else
    echo "https://download.pytorch.org/whl/cu124"
  fi
}

install_torch() {
  local cc index
  cc="$(detect_compute_cap)"
  index="$(choose_torch_index "$cc")"

  log "5/8 Install PyTorch (auto select)..."
  log "GPU compute capability: ${cc:-none}"
  log "Torch index-url: $index"
  log "Torch nightly: ${TORCH_PRE:-0} (set TORCH_PRE=1 to force nightly)"
  log "Torch override: ${TORCH_INDEX_URL:-<none>} (set TORCH_INDEX_URL to override)"

  mmrun python -m pip install -U pip
  mmrun python -m pip install torch torchvision torchaudio --index-url "$index"

  # Hard validation: real CUDA kernel launch (catches sm mismatch)
  log "Validate torch CUDA kernel..."
  mmrun python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(0)

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = (x @ y).mean()
torch.cuda.synchronize()
print("gpu:", torch.cuda.get_device_name(0))
print("ok: kernel launch")
PY
}

install_torch

# ---------- unpack isaacgym ----------
log "6/8 Unpack IsaacGym..."
rm -rf "$ISAACGYM_DIR"
mkdir -p "$ISAACGYM_DIR"
tar -xzf "$ISAACGYM_TAR" -C "$ISAACGYM_DIR" --strip-components=1

if [ ! -d "$ISAACGYM_DIR/python" ]; then
  log "FATAL: IsaacGym python dir not found: $ISAACGYM_DIR/python"
  log "Check tar layout:"
  log "  tar -tzf \"$ISAACGYM_TAR\" | head -n 40"
  exit 1
fi

log "Register IsaacGym on sys.path via .pth..."
mmrun python - <<PY
import site, pathlib
pth_dir = pathlib.Path(site.getsitepackages()[0])
pth = pth_dir / "isaacgym_local.pth"
pth.write_text(r"$ISAACGYM_DIR/python" + "\n")
print("Wrote", pth, "->", pth.read_text().strip())
PY

# ---------- install editable third_party (no-deps to avoid PyPI isaacgym) ----------
log "7/8 Install third_party (editable, --no-deps)..."
mmrun python -m pip install -e "$THIRD_PARTY_DIR/rsl_rl" --no-deps
mmrun python -m pip install -e "$THIRD_PARTY_DIR/legged_gym" --no-deps

# ---------- smoke test ----------
SMOKE="$SCRIPTS_DIR/smoke_env.py"
if [ ! -f "$SMOKE" ]; then
  log "8/8 smoke_env.py missing; creating minimal one..."
  cat > "$SMOKE" <<'PY'
import subprocess, sys

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

print("PY:", sys.executable)

print("\n== GPU ==")
try:
    print(run("nvidia-smi -L"))
except Exception as e:
    print("nvidia-smi not available:", e)

print("\n== IsaacGym ==")
import isaacgym
from isaacgym import gymapi
print("isaacgym:", isaacgym.__file__)
print("gymapi:", gymapi)
print("\n== Torch ==")

import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    y = torch.randn(256, 256, device="cuda")
    _ = (x @ y).mean()
    torch.cuda.synchronize()
    print("gpu:", torch.cuda.get_device_name(0))
    print("ok: kernel launch")

print("\n== Third Party ==")
import rsl_rl, legged_gym
print("rsl_rl:", rsl_rl.__file__)
print("legged_gym:", legged_gym.__file__)

print("\nSMOKE: OK")
PY
fi

log "8/8 Run smoke test..."
mmrun python "$SMOKE"

log "Bootstrap OK."
echo
echo "Examples:"
echo "  $MAMBA_EXE --root-prefix \"$MAMBA_ROOT\" run -p \"$ENV_PREFIX\" python -c \"import torch; print(torch.cuda.is_available())\""
echo
echo "Notes:"
echo "  - Uses WORKDIR=$WORKDIR (safe path, no spaces/quotes)."
echo "  - HOME is isolated under WORKDIR to avoid ~/.conda permission junk."
echo "  - IsaacGym injected via .pth; legged_gym installed with --no-deps."
