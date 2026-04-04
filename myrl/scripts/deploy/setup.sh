#!/usr/bin/env bash
set -euo pipefail
# ── setup.sh — 服务器端环境部署 ─────────────────────────────────────
#
# 在远程服务器上运行（需要国际互联网）。
# 检测环境 → 拉镜像 → 构建容器 → 冒烟验证。
#
# 用法:  cd ~/Ezios_RL_Toolbox && bash myrl/scripts/deploy/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_DIR="$REPO_ROOT/myrl/docker"

ts() { date "+%H:%M:%S"; }
info() { echo "[$(ts)] [setup] $*"; }
fail() { echo "[$(ts)] [setup] ✗ $*" >&2; exit 1; }
ok()   { echo "[$(ts)] [setup] ✓ $*"; }

info "仓库根目录: $REPO_ROOT"

# ── Step 1: 检测基本依赖 ────────────────────────────────────────────
info "Step 1: 环境检测"

command -v docker >/dev/null 2>&1 || fail "docker 未安装"
ok "docker: $(docker --version | head -1)"

docker info 2>/dev/null | grep -q "Runtimes.*nvidia" || {
    # 尝试检测 nvidia-container-toolkit
    if ! command -v nvidia-container-runtime >/dev/null 2>&1 && \
       ! dpkg -l nvidia-container-toolkit >/dev/null 2>&1; then
        fail "nvidia-container-toolkit 未安装（docker 无法访问 GPU）"
    fi
}
ok "nvidia-container-toolkit 可用"

nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi 不可用（检查驱动）"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

# ── Step 2: 拉取/构建 Docker 镜像 ───────────────────────────────────
info "Step 2: Docker 镜像"

IMAGE="myrl/isaaclab-dev:2.3.2"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "镜像已存在: $IMAGE"
else
    info "构建镜像（首次需要拉取基础镜像，约 20-30 分钟）..."
    cd "$DOCKER_DIR"
    docker compose build
    cd "$REPO_ROOT"
    ok "镜像构建完成: $IMAGE"
fi

# ── Step 3: 创建工作目录 ────────────────────────────────────────────
info "Step 3: 工作目录"

WORK_DIR="${MYRL_WORK_DIR:-$HOME/myrl_work}"
mkdir -p "$WORK_DIR"
ok "工作目录: $WORK_DIR"

# ── Step 4: 启动容器 + 冒烟验证 ─────────────────────────────────────
info "Step 4: 容器冒烟验证"

cd "$DOCKER_DIR"

# 停止已有容器
docker compose down 2>/dev/null || true

# 启动
docker compose up -d
ok "容器已启动"

# 等待 entrypoint 完成初始化（pip install 等）
info "等待 entrypoint 初始化（首次可能需要几分钟安装依赖）..."
for i in $(seq 1 120); do
    if docker exec myrl-dev python3 -c "import torch; print('torch OK')" 2>/dev/null; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        fail "容器初始化超时（120s）。查看日志: docker logs myrl-dev"
    fi
    sleep 1
done
ok "torch 可用"

# GPU 验证
GPU_OK=$(docker exec myrl-dev python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$GPU_OK" = "True" ]; then
    ok "容器内 CUDA 可用"
else
    fail "容器内 CUDA 不可用！检查 nvidia-container-toolkit 配置"
fi

# Isaac Lab 验证
if docker exec myrl-dev python3 -c "from isaaclab.app import AppLauncher; print('isaaclab OK')" 2>/dev/null; then
    ok "Isaac Lab 可导入"
else
    fail "Isaac Lab 导入失败"
fi

cd "$REPO_ROOT"

# ── Step 5: 训练冒烟测试（可选） ────────────────────────────────────
info "Step 5: 训练冒烟测试（5 iterations）"
info "（跳过请 Ctrl+C，不影响部署）"

docker exec myrl-dev python3 /workspace/myrl/scripts/ablation_probe.py \
    --task Instinct-Locomotion-Flat-G1-v0 --num_envs 32 --steps 50 \
    2>&1 | tail -20

ok "冒烟测试完成"

# ── 完成 ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  ✓ 部署完成"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 启动 train_manager（宿主机进程）:"
echo "     cd '$REPO_ROOT'"
echo "     nohup python3 myrl/scripts/train_manager.py --port 7001 --compose-file myrl/docker/compose.yaml > /tmp/train_manager.log 2>&1 &"
echo ""
echo "  2. 从本地建立 SSH 隧道:"
echo "     bash myrl/scripts/deploy/remote.sh user@this_server"
echo ""
echo "  3. 打开浏览器: http://localhost:7001"
echo ""
