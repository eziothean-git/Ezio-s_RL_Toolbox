# myrl — 机器人强化学习工具箱

从零构建的机器人 RL 框架，面向腿式机器人全身控制。算法层依托 **instinct_rl**（PPO / AMP / TPPO / VAE / MoE），任务层依托 **instinctlab**（InstinctRlEnv + gymnasium 任务），以 Isaac Sim 为主训练后端，MuJoCo + ROS2 支撑 Sim2Sim 与 Sim2Real 迁移。

```
算法层        instinct_rl  (PPO / AMP / TPPO / VAE / MoE)
─────────────────────────────────────────────────────────────
任务/环境层   instinctlab  (InstinctRlEnv + gymnasium 任务)
─────────────────────────────────────────────────────────────
适配层        myrl/core/compat/  (后端 + Views)             ← 核心抽象
─────────────────────────────────────────────────────────────
仿真后端      Isaac Lab（训练）│ MuJoCo（Sim2Sim）│ 真实机器人（ROS2）
```

---

## 环境要求

| 组件 | 版本 | 备注 |
|------|------|------|
| OS | Ubuntu 22.04 | ROS2 Humble 依赖 |
| GPU | NVIDIA，CUDA 12.x | A100 / RTX 30xx 及以上 |
| Docker | 24+ | 仅 Dev 模式 |
| NVIDIA Container Toolkit | 最新 | 仅 Dev 模式 |
| Python | 3.10（训练）/ 3.11（Dev 容器） | |
| Isaac Sim | 5.1.0 | 仅 Dev 容器 |
| Isaac Lab | 2.3.2 | 仅 Dev 容器 |

---

## 目录结构

```
Ezio's_RL_Toolbox/
├── myrl/
│   ├── scripts/                      # 入口脚本
│   │   ├── bootstrap_train.sh        # 服务器：训练环境一键部署
│   │   ├── bootstrap_dev.sh          # PC：开发 Docker 环境一键部署
│   │   ├── run_dev.sh                # PC：启动开发容器
│   │   ├── train.py                  # Isaac Lab 训练
│   │   ├── play.py                   # Isaac Lab 推理
│   │   ├── play_mujoco.py            # MuJoCo 推理（无需 Isaac Sim）
│   │   ├── start_mujoco_server.py    # 独立 MuJoCo 仿真服务器
│   │   └── start_ros2_bridge.py      # ROS2 ↔ TCP 桥接节点
│   ├── docker/
│   │   ├── Dockerfile.dev            # Isaac Lab 2.3.2 + ROS2 Humble
│   │   ├── compose.yaml
│   │   └── entrypoint.sh
│   └── src/myrl/
│       ├── tasks/                    # 任务注册（gym.register）
│       ├── assets/                   # 资产解析器
│       └── core/
│           ├── compat/
│           │   ├── backends/         # IsaacLabBackend, MuJoCoBackend
│           │   └── views/            # RobotHandle, JointView, BodyView, ContactView
│           ├── sim_server/           # TCP 协议、MuJoCoSimServer、Ros2Bridge
│           ├── obs/                  # ObsHistoryManager
│           └── task/                 # ObsBuilder, RewardBuilder（WIP）
```

---

## 部署指南

### 模式一 — 训练服务器（无 Docker）

适用于无 root、无 Docker 的 GPU 服务器，所有依赖安装在用户目录下。

```bash
# 一次性初始化（约 5 分钟，自动下载 micromamba + torch）
WORKDIR=~/myrl_work bash myrl/scripts/bootstrap_train.sh

# 开始训练
micromamba run -p ~/myrl_work/.mamba/envs/myrl-train \
    python myrl/scripts/train.py \
    --task myrl/Locomotion-Flat-G1Smoke-v0 \
    --num_envs 4096 \
    --headless
```

`bootstrap_train.sh` 会自动检测 GPU 计算能力并选择对应的 PyTorch 索引（`cu124` 或 `cu128`）。训练日志写入 `logs/myrl/<experiment>/`。

**恢复训练：**
```bash
python myrl/scripts/train.py \
    --task myrl/Locomotion-Flat-G1Smoke-v0 \
    --resume --load_run 20260304_120000
```

---

### 模式二 — 开发机（本地 PC，Docker）

在容器内提供完整的 Isaac Lab + ROS2 环境，支持 GPU 直通和 X11/Wayland 显示。

```bash
# 一次性初始化（安装 Docker + NVIDIA Container Toolkit，克隆 Isaac Lab）
bash myrl/scripts/bootstrap_dev.sh

# 启动开发容器（首次构建镜像约 10 分钟）
bash myrl/scripts/run_dev.sh
```

容器内 `python3` 即可使用，Isaac Sim、instinct_rl、instinctlab、myrl 均已通过 `PYTHONPATH` 注入，无需 `pip install`。

**使用自定义上游仓库路径：**
```bash
INSTINCT_RL_DIR=/path/to/instinct_rl \
INSTINCTLAB_DIR=/path/to/instinctlab \
bash myrl/scripts/run_dev.sh
```

**容器内训练：**
```bash
# 容器内执行
python3 myrl/scripts/train.py --task myrl/Locomotion-Flat-G1Smoke-v0 --headless
```

---

## Sim2Sim 迁移（Isaac Lab → MuJoCo）

MuJoCo 后端作为独立 TCP 服务运行，训练好的策略通过 `MuJoCoBackend`（与 `IsaacLabBackend` 实现相同的 `VecEnv` 接口）连接。推理侧完全不依赖 Isaac Sim。

**步骤一 — 启动 MuJoCo 仿真服务器**（独立终端）：

```bash
# 内置 DummyTask 冒烟测试（无需 MJCF）
python myrl/scripts/start_mujoco_server.py --task dummy --num_envs 4 --port 7777

# 真实任务（实现了 MuJoCoTask ABC 的自定义类）
python myrl/scripts/start_mujoco_server.py \
    --task myrl.tasks.mujoco.g1_walk:G1WalkTask \
    --mjcf myrl/assets/robots/humanoid_x/robot.xml \
    --num_envs 16 --port 7777
```

**步骤二 — 运行推理：**

```bash
# 冒烟测试（随机动作）
python myrl/scripts/play_mujoco.py --num_envs 4 --num_steps 100

# 加载真实 checkpoint
python myrl/scripts/play_mujoco.py \
    --load_run logs/myrl/g1_smoke/20260304_120000 \
    --checkpoint model_5000.pt \
    --host localhost --port 7777
```

**Sim2Sim 保真度的核心约束**：`MuJoCoTask.obs_format()` 的返回值必须与 Isaac Lab 训练环境产生的 `obs_format` **完全一致**，包括每个 term 的维度和顺序。

---

## Sim2Real 迁移（MuJoCo → 真实机器人，via ROS2）

ROS2 桥接将仿真服务器与真实机器人对接。观测数据经 ROS2 topic（传感器总线）流动，TCP 通道只传 rewards/dones/log 元数据。

```
真实机器人传感器 → ROS2 topics (/myrl/{task_id}/obs/{group})
                                         ↓
                                   Ros2Bridge
                                         ↓  TCP（只含 rewards/dones）
                                MuJoCoSimServer
                                         ↑
                            策略（via MuJoCoBackend）
```

**步骤一 — 以 ROS 模式启动仿真服务器：**

```bash
python myrl/scripts/start_mujoco_server.py \
    --task dummy --num_envs 4 --port 7777 \
    --ros --task_id robot1
```

**步骤二 — 启动 ROS2 桥接节点**（需先 source ROS2）：

```bash
source /opt/ros/humble/setup.bash

python myrl/scripts/start_ros2_bridge.py \
    --port 7777 --task_id robot1 --hz 50 \
    --history_cfg '{"policy": {"base_ang_vel": 8}}'
```

**步骤三 — 验证通路：**

```bash
# 查看 topic 列表
ros2 topic list | grep myrl

# 查看 obs 数据
ros2 topic echo /myrl/robot1/obs/policy --once

# 注入测试动作
ros2 topic pub /myrl/robot1/action std_msgs/msg/Float32MultiArray \
    "{data: [0.1, 0.0, -0.1]}" --once
```

`--history_cfg` 参数配置 `ObsHistoryManager` 的 term 级历史长度。机器人传感器只发布当前帧，历史展开由桥接侧完成。

---

## 编写自定义任务

### Isaac Lab 任务（训练阶段）

在 `myrl/src/myrl/tasks/<domain>/config/<task_name>/__init__.py` 中注册：

```python
import gymnasium as gym

gym.register(
    id="myrl/MyTask-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "myrl.tasks.<domain>.config.<task_name>:MyTaskEnvCfg",
        "instinct_rl_cfg_entry_point": "myrl.tasks.<domain>.config.<task_name>.agents:MyRunnerCfg",
    },
)
```

实现 `MyTaskEnvCfg`（继承 `InstinctLabRLEnvCfg`）和 `MyRunnerCfg`（继承 `InstinctRlOnPolicyRunnerCfg`）。

### MuJoCo 任务（Sim2Sim / Sim2Real）

继承 `myrl.core.sim_server.mujoco_task.MuJoCoTask`：

```python
from myrl.core.sim_server.mujoco_task import MuJoCoTask
import numpy as np

class G1WalkTask(MuJoCoTask):
    @property
    def num_actions(self) -> int:
        return 29  # 必须与 Isaac Lab 训练时完全一致

    @property
    def max_episode_length(self) -> int:
        return 1000

    def obs_format(self) -> dict:
        # 必须与 Isaac Lab 训练的 obs_format 完全一致
        return {"policy": {"base_ang_vel": (24,), "joint_pos": (87,)}}

    def compute_obs(self, model, datas) -> dict:
        ...

    def compute_reward(self, model, datas, actions) -> np.ndarray:
        ...

    def is_terminated(self, model, datas):
        ...

    def apply_action(self, model, data, action) -> None:
        ...

    def reset_env(self, model, data, env_id) -> None:
        ...
```

---

## 核心设计约束

1. **适配层边界** — 任务代码与算法代码不得直接 import 仿真后端。所有机器人状态读取必须通过 `RobotHandle` / `JointView` / `BodyView` / `ContactView`。

2. **obs_format 一致性** — Isaac Lab 训练、MuJoCo Sim2Sim、ROS2 桥接三者使用的 `obs_format` 必须完全一致（维度和 term 顺序），否则推理时会出现静默错误。

3. **奖励 shape** — 奖励始终为 `Tensor[num_envs, num_rewards]`（即使只有单路奖励也保留第二维），与 instinct_rl 多 critic 约定保持一致。

4. **历史归属模型层** — 仿真后端只发送当前帧观测，`ObsHistoryManager` 负责维护 ring buffer 并输出历史展开的观测。

---

## 当前进度

| 组件 | 状态 |
|------|------|
| 训练环境 bootstrap 脚本 | 完成 |
| 开发 Docker 环境 | 完成 |
| Isaac Lab → instinct_rl 训练链路 | 完成（Phase A，5 iter 验证） |
| IsaacLabBackend（Phase B） | 已实现，待 loss 对比验证 |
| MuJoCo TCP 仿真服务器 | 完成（23/23 测试） |
| ROS2 桥接节点 | 完成（32/32 测试） |
| MuJoCo 推理脚本 | 完成 |
| ObsHistoryManager | 完成（32/32 测试） |
| RobotHandle / Views | 部分（代理层，完整 View API 开发中） |
| ObsBuilder / RewardBuilder | 骨架 |
| humanoid_x 机器人资产 | 空（开发中） |
