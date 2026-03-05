# Tailscale 远程访问配置指南

## 概述

Tailscale 是基于 WireGuard 的轻量 VPN，让训练服务器和开发机通过私有 100.x.x.x 地址
互联，无需端口转发或公网暴露。`train_manager.py` 绑定 Tailscale 接口，仅对已入网设备
可见。

---

## 安装

### 训练服务器（需要一次性 root 权限）

```bash
# 方式 A：官方一键安装（需 sudo）
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# 方式 B：请 sysadmin 安装后，普通用户直接 up
tailscale up
```

安装后获取本机 Tailscale IP：
```bash
tailscale ip -4   # 输出示例: 100.64.1.23
```

### 开发机

同上步骤安装，两台机器加入同一 Tailscale 账号（或 Tailnet）后自动互通。

---

## 启动 TrainManager（服务器端）

```bash
# 绑定到 Tailscale 接口（仅对同 Tailnet 设备可见）
TAILSCALE_IP=$(tailscale ip -4)
python myrl/scripts/train_manager.py \
    --bind "$TAILSCALE_IP" \
    --port 7001 \
    --inner-port 7000

# 开发环境测试：绑定 0.0.0.0（仅局域网内使用）
python myrl/scripts/train_manager.py --port 7001
```

---

## 客户端连接（开发机）

```bash
SERVER_IP=100.64.1.23   # 替换为服务器的 Tailscale IP

# CLI 方式
export MYRL_HOST=$SERVER_IP
export MYRL_PORT=7001
python myrl/scripts/train_cli.py status

# TUI 方式
python myrl/scripts/train_tui.py --host $SERVER_IP --port 7001

# 或带默认任务（供 [S] 键直接启动）
python myrl/scripts/train_tui.py \
    --host $SERVER_IP --port 7001 \
    --task myrl/Locomotion-Flat-G1Smoke-v0 \
    --num_envs 4096
```

---

## 工作流示例

```bash
# 1. 服务器：后台启动管控服务
nohup python myrl/scripts/train_manager.py --bind 100.64.1.23 --port 7001 \
    > ~/train_manager.log 2>&1 &

# 2. 开发机：CLI 验证连接
python myrl/scripts/train_cli.py --host 100.64.1.23 status

# 3. 开发机：启动训练
python myrl/scripts/train_cli.py --host 100.64.1.23 start \
    --task myrl/Locomotion-Flat-G1Smoke-v0 --num_envs 4096

# 4. 开发机：TUI 实时监控
python myrl/scripts/train_tui.py --host 100.64.1.23 --port 7001

# 5. 开发机：保存 checkpoint 后继续（不中断训练）
python myrl/scripts/train_cli.py --host 100.64.1.23 checkpoint

# 6. 开发机：优雅停止
python myrl/scripts/train_cli.py --host 100.64.1.23 stop
```

---

## 信号协议（供参考）

| 操作 | 信号 | 行为 |
|------|------|------|
| `/halt` | SIGUSR1 | 当前迭代结束 → 保存 checkpoint → halt 等待 |
| `/resume` | SIGUSR2 | 清除 halt，恢复训练循环 |
| `/checkpoint` | SIGUSR1 + 10s + SIGUSR2 | 保存 checkpoint 后自动恢复 |
| `/stop` | SIGTERM | 当前迭代结束 → 保存 checkpoint → 退出 |
| `/kill` | SIGKILL | 立即终止（无 checkpoint） |

---

## 防火墙注意事项

Tailscale 使用 UDP 41641（或 HTTPS 443 降级），**无需开放任何 TCP 端口**到公网。
`train_manager.py` 的 7001 端口只在 Tailscale 虚拟网络内可达。

如果服务器有严格出站规则导致 Tailscale 无法建立直连，可使用 DERP 中继：
```bash
tailscale ping 100.64.1.23   # 查看是否为直连或中继
```

---

## Tailscale 版本要求

- Tailscale >= 1.20 支持用户态安装（`tailscale up --userspace-networking`）
- 无 root 运行方式适用于无法修改内核路由表的共享 HPC 集群
