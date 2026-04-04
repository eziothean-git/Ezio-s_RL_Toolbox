#!/usr/bin/env bash
set -euo pipefail
# ── remote.sh — 本地一键连接远程训练服务器 ──────────────────────────
#
# 用法:  bash myrl/scripts/deploy/remote.sh user@server
#
# 建立 SSH 隧道（7001=Editor, 7002=Oscilloscope, 7000=LogServer），
# 然后打开浏览器。

if [ -z "${1:-}" ]; then
    echo "用法: bash $0 user@server"
    echo ""
    echo "端口映射:"
    echo "  7001 → Editor WebUI"
    echo "  7002 → Oscilloscope"
    echo "  7000 → SSE Log Server"
    exit 1
fi

TARGET="$1"

# 检查端口是否已被占用（可能已有隧道）
for port in 7001 7002 7000; do
    if ss -tlnp 2>/dev/null | grep -q ":$port " || lsof -i ":$port" >/dev/null 2>&1; then
        echo "[remote] 端口 $port 已被占用，可能已有隧道。跳过建立新连接。"
        echo "[remote] 直接打开: http://localhost:7001"
        xdg-open "http://localhost:7001" 2>/dev/null || open "http://localhost:7001" 2>/dev/null || echo "请手动打开浏览器"
        exit 0
    fi
done

echo "[remote] 建立 SSH 隧道到 $TARGET ..."
echo "  本地 :7001 → Editor WebUI"
echo "  本地 :7002 → Oscilloscope"
echo "  本地 :7000 → Log Server"
echo ""

# 后台建立隧道
ssh -fN \
    -L 7001:127.0.0.1:7001 \
    -L 7002:127.0.0.1:7002 \
    -L 7000:127.0.0.1:7000 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "$TARGET"

echo "[remote] 隧道已建立。"

# 等端口就绪
for i in $(seq 1 10); do
    if curl -s http://localhost:7001/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -s http://localhost:7001/health >/dev/null 2>&1; then
    echo "[remote] ✓ train_manager 可达"
else
    echo "[remote] ⚠ train_manager 未响应（可能还未启动，手动在服务器启动后刷新浏览器）"
fi

# 打开浏览器
xdg-open "http://localhost:7001" 2>/dev/null || open "http://localhost:7001" 2>/dev/null || {
    echo "[remote] 请手动打开: http://localhost:7001"
}

echo ""
echo "[remote] 关闭隧道: kill \$(lsof -t -i:7001) 或关闭终端"
