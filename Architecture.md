# myrl 架构设计文档

> **版本**：v1.2 | **日期**：2026-04-04 | **作者**：Ezio
>
> 本文档是 myrl 框架的顶层架构蓝图，为后续每个模块的详细设计提供统一参照。

---

## 目录

- [1. 愿景与设计哲学](#1-愿景与设计哲学)
- [2. 总体架构](#2-总体架构)
- [3. Compat 层 — 仿真抽象](#3-compat-层--仿真抽象)
  - [3.1 Backend 子系统](#31-backend-子系统)
  - [3.2 View 子系统](#32-view-子系统)
  - [3.3 DataBus — 数据总线](#33-databus--数据总线)
- [4. Task 层 — 观测/奖励/终止](#4-task-层--观测奖励终止)
  - [4.1 ObsBuilder](#41-obsbuilder)
  - [4.2 ObsHistoryManager](#42-obshistorymanager)
  - [4.3 RewardBuilder](#43-rewardbuilder)
  - [4.4 RewardLibrary — 奖励资产化系统](#44-rewardlibrary--奖励资产化系统)
  - [4.5 Reward/Obs Pipeline — 管线即 YAML](#45-rewardobs-pipeline--管线即-yaml)
- [5. Algo 层 — 算法适配](#5-algo-层--算法适配)
- [6. 资产系统](#6-资产系统)
  - [6.1 AssetStore — 内容寻址资产库](#61-assetstore--内容寻址资产库)
  - [6.2 实验合成与打包](#62-实验合成与打包)
  - [6.3 资产 YAML 配置规范](#63-资产-yaml-配置规范)
- [7. 日志、可观测性与调试](#7-日志可观测性与调试)
  - [7.1 三层日志架构](#71-三层日志架构)
  - [7.2 Oscilloscope — 示波器](#72-oscilloscope--示波器)
- [8. 训练管控系统](#8-训练管控系统)
- [9. 仿真服务系统](#9-仿真服务系统)
- [10. 注册表与可复现性](#10-注册表与可复现性)
- [11. 部署架构](#11-部署架构)
- [12. 数据流全景](#12-数据流全景)
- [13. 模块状态与路线图](#13-模块状态与路线图)
- [附录 A. 完整 API 索引](#附录-a-完整-api-索引)
- [附录 B. 配置文件规范](#附录-b-配置文件规范)
- [附录 C. CLI/TUI 命令索引](#附录-c-clitui-命令索引)

---

## 1. 愿景与设计哲学

### 1.1 使命

myrl 是 **Ezio 的机器人强化学习工具箱**——一个面向长期演化的个人 Robotics 基建。它不是某个单一功能的框架，而是一个有机生长的体系：每一层的存在都为下一层创造了自然的需求。

### 1.2 自然生长链

框架的各部分不是独立设计出来的，而是沿着一条内在逻辑自然推演出来的：

```
Compat 层（多后端适配器）
  │  "不同仿真后端的差异应该被屏蔽"
  │  → View API 让 Task/Algo 层无需关心后端
  │
  ▼
资产系统
  │  "obs/reward 管线、传感器、执行器、URDF、场景、脚本……
  │   都应该作为 asset 来管理"
  │  → AssetStore 内容寻址 + 版本化
  │
  ▼
实验合成 + 打包
  │  "既然所有东西都是 asset，那复用它们组合实验就是自然的"
  │  → ExperimentComposer + .myrlpkg
  │
  ▼
分布式训练 + 管控
  │  "既然有了打包分发，那在远程服务器上管理多个 experiment 也是自然的"
  │  → train_manager + CLI + TUI + SSE
  │
  ▼
Evaluate + Deploy
  │  "有 train 就有 evaluate 和 deploy 的需求"
  │  → MuJoCo 后端 + ROS2 桥接 + 实机接口
  │  → 又绕回到 Compat 层——它就是针对不同后端的适配器
  │
  ▼
调试 + 可观测性
  │  "开发过程中需要像玩 Besiege 一样直接看到、摸到仿真中的一切"
  │  → DataBus（信号总线）+ Oscilloscope（示波器）
  │  → 可选挂件，不影响训练链路
  │
  ▼
奖励/观测管线
    "奖励项应该像资产一样组合，归一化应该可控可视"
    → @reward_fn 资产化 + YAML 声明式管线 + Transform 链
    → 与 DataBus 集成后可实时检视每条信号的幅度和分布
```

**闭环**：Compat 层既是起点（屏蔽后端差异）也是终点（适配部署目标），整个工具箱围绕它形成闭环。

### 1.3 设计支柱

| 支柱 | 含义 | 实践 |
|------|------|------|
| **跨层隔离** | Algo/Task 不直接 import 仿真后端，降低认知负担 | Compat View API + SimBackend ABC |
| **一切皆资产** | obs/reward/sensor/actuator/URDF/scene/script 统一管理 | AssetStore 10 种 AssetType + SHA256 |
| **可观测性** | 流经机器人的任何信号都可以被 tap、可视化、录制 | DataBus pub/sub + 示波器（可选挂件） |
| **YAML 即管线** | 奖励/观测管线声明式定义，运行后定格 | reward_pipeline + obs_pipeline YAML |
| **可复现** | 训练只信任 Manifest，不信任环境状态 | 内容寻址 + RunManifest |
| **最小部署** | Train-time 只需 bash + Python + GPU | bootstrap_train.sh 裸机自举 |

### 1.4 核心不变式（Invariants）

1. **跨层隔离**：Algo/Task 层永远不直接 import 仿真后端，只通过 Compat 层的 View / Backend API
2. **强类型 View**：所有机器人状态读取通过 `RobotHandle` → `JointView` / `BodyView` / `ContactView` / `SensorView`
3. **观测格式合约**：`obs_format: dict[str, dict[str, tuple]]` 是唯一的观测结构描述
4. **DataBus 是可选的**：无 DataBus 时直接返回数据，有 DataBus 时额外 publish。不能成为必经路径
5. **运行时定格**：一旦 `train.py` / `play.py` 启动，reward/obs 管线配置不可变
6. **资产不可变性**：注册到 AssetStore 的资产 SHA256 寻址，不可修改，只能发布新版本
7. **日志不可丢失**：JSONL sink always-on

---

## 2. 总体架构

### 2.1 分层总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Algo 层 (instinct_rl)                              │
│  PPO · TPPO · WasabiPPO · EstimatorPPO · OnPolicyRunner               │
├─────────────────────────────────────────────────────────────────────────┤
│                      Task 层 (myrl.core.task)                           │
│  ObsBuilder · RewardBuilder · RewardLibrary · TransformLibrary         │
│  ObsHistoryManager · Reward/Obs Pipeline (YAML)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                Compat 层 (myrl.core.compat) ← 闭环枢纽                  │
│  ┌──────────────────────────┐  ┌────────────────────────────────────┐  │
│  │ Views (状态读取)          │  │ Backends (后端适配)                │  │
│  │  RobotHandle             │  │  IsaacLabBackend (开发/调试)       │  │
│  │  ├── JointView           │  │  MuJoCoBackend (sim2sim/部署)     │  │
│  │  ├── BodyView            │  │  [MJXBackend] (大规模训练)         │  │
│  │  ├── ContactView         │  │                                    │  │
│  │  └── SensorView          │  │  DataBus (可选信号总线)            │  │
│  └──────────────────────────┘  └────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                      仿真后端 (Sim Backends)                            │
│  Isaac Gym · Isaac Lab · MuJoCo · [MJX] · [ROS2 → 实机]               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         横切系统                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 资产系统      │  │ 实验合成/打包 │  │ 日志/可观测   │  │ 训练管控    │ │
│  │ AssetStore   │  │ Composer     │  │ JSONL+wandb  │  │ Manager    │ │
│  │ 10种AssetType│  │ Packager     │  │ SSE Server   │  │ CLI+TUI    │ │
│  │ CLI + TUI    │  │ .myrlpkg     │  │ Oscilloscope │  │ ProcessCtrl│ │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤  ├────────────┤ │
│  │ 注册表        │  │ 仿真服务      │  │              │  │ 部署系统    │ │
│  │ RunManifest  │  │ MuJoCoServer │  │              │  │ Docker     │ │
│  │ Checksum     │  │ ROS2 Bridge  │  │              │  │ Bootstrap  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 双模式运行拓扑

```
┌──────────────────────────────────────┐
│   开发者本地 (CODE-TIME)              │
│   Docker 容器 · Isaac Lab · GUI      │
│                                      │
│  ┌─ play.py / train.py ────────────┐│
│  │ AppLauncher · IsaacLabBackend   ││
│  │ OnPolicyRunner                   ││
│  │ DataBus（可选，Code-time 挂载）   ││
│  │ └─ Oscilloscope（可选挂件）       ││
│  └──────────────────────────────────┘│
│                                      │
│  ┌─ 工具 ──────────────────────────┐│
│  │ asset_tui · reward_inspector    ││
│  │ log_viewer · registry_cli       ││
│  └──────────────────────────────────┘│
└──────────────────┬───────────────────┘
                   │ TailScale / SSH
┌──────────────────▼───────────────────┐
│   GPU 服务器 (TRAIN-TIME)             │
│   裸机 · 无 Docker · 无 GUI          │
│                                      │
│  ┌─ train_manager :7001 ───────────┐│
│  │ ProcessCtrl · SSEBroadcaster    ││
│  │ └─ train.py                     ││
│  │    :7000 SSE Log                ││
│  │    DataBus 无订阅者 → 零开销     ││
│  └──────────────────────────────────┘│
│  bootstrap_train.sh · micromamba     │
└──────────────────────────────────────┘
```

---

## 3. Compat 层 — 仿真抽象

> **定位**：整个工具箱的闭环枢纽。上层（Task/Algo）通过 View 和 Backend API 访问仿真，永远不直接 import 后端。同一套 Compat 接口既服务于训练（Isaac Gym/Lab），也服务于评估（MuJoCo），也服务于部署（ROS2/实机），降低开发者的认知负担。View 同时作为 DataBus 的自动 publisher（可选）。

```
myrl/src/myrl/core/compat/
├── backends/
│   ├── base.py              # SimBackend ABC
│   ├── isaaclab_backend.py  # IsaacLabBackend(SimBackend, VecEnv)
│   ├── mujoco_backend.py    # MuJoCoBackend(SimBackend, VecEnv)
│   └── [mjx_backend.py]     # MJXBackend(SimBackend, VecEnv) [待建]
├── views/
│   ├── robot.py             # RobotHandle — 聚合入口
│   ├── joints.py            # JointView
│   ├── bodies.py            # BodyView
│   ├── contacts.py          # ContactView
│   └── sensor.py            # SensorView ABC + ImuView
└── [databus/]               # DataBus (可选信号总线) [待建]
    ├── bus.py
    ├── channel.py
    └── tap.py
```

Compat 层包含三个子系统：**Backend**（多后端适配）、**View**（强类型状态读取）、**DataBus**（可选信号总线）。

Backend 和 View 的完整 API 详见后续各小节。此处先阐述 DataBus 的设计，因为它是可观测性体系的基础。

### 3.1 Backend 子系统

**SimBackend ABC 合约**（详见 `base.py`）：

```python
class SimBackend(ABC):
    num_envs: int;  num_actions: int;  num_rewards: int;  device: torch.device
    @abstractmethod
    def step(actions) -> (obs, rewards[N,R], dones[N], extras)
    @abstractmethod
    def reset() -> (obs, extras)
    @abstractmethod
    def get_observations() -> (obs, extras)
    @abstractmethod
    def close() -> None
```

所有 Backend 同时实现 `instinct_rl.env.VecEnv`，可直接传给 `OnPolicyRunner`。

**Backend 矩阵**：

| Backend | 运行模式 | 通信 | 状态 |
|---------|---------|------|------|
| IsaacLabBackend | 进程内 | 直接调用 | ✅ |
| MuJoCoBackend | 跨进程 | TCP SimProto | ✅ |
| MJXBackend | 进程内 | JAX→torch DLPack | ❌ 待建 |

### 3.2 View 子系统

**RobotHandle** 聚合所有 View，提供根体快捷属性和子 View 访问器：

| 访问器 | 返回 | 主要属性 |
|--------|------|---------|
| `robot.joints` | `JointView` | pos, vel, acc, torque, pos_rel, select(ids) |
| `robot.bodies` | `BodyView` | root_pos_w, root_quat_w, root_lin_vel_w/b, projected_gravity_b |
| `robot.contacts(name)` | `ContactView` | net_forces_w, force_magnitude, air_time, in_contact, first_contact(dt) |
| — | `SensorView` | ImuView: lin_acc_b, ang_vel_b |

**Bridge 函数**：`make_term(fn)` / `make_rew(fn)` 将 `(RobotHandle)->Tensor` 包装为 Isaac Lab `ObservationTermCfg.func` 兼容签名。

### 3.3 DataBus — 数据总线

> **定位**：Compat 层的可选组件。View / RewardBuilder / ObsBuilder 向它 publish，示波器和诊断工具从它 subscribe。**无订阅者时零开销**。去掉 DataBus 框架照常运行。

#### 3.3.1 核心概念

| 概念 | 说明 |
|------|------|
| **Channel** | 一条命名的数据流，如 `robot/joints/pos`、`reward/track_lin_vel_xy_exp` |
| **Publish** | View / Builder 在计算完成后将 tensor 推送到 channel |
| **Tap** | 订阅者获取的句柄，可配置：采样率、环形缓冲长度、幅度增益/偏移 |
| **零开销原则** | channel 无 Tap 时 publish 是 no-op（不分配内存、不拷贝 tensor） |

#### 3.3.2 Channel 路径约定

```
robot/                          # RobotHandle 发布
├── joints/
│   ├── pos                     # (N, J) 关节位置
│   ├── vel                     # (N, J) 关节速度
│   ├── torque                  # (N, J) 施加力矩
│   └── pos_rel                 # (N, J) 相对偏差
├── bodies/
│   ├── root_pos_w              # (N, 3) 根体位置
│   ├── root_quat_w             # (N, 4) 根体朝向
│   ├── root_lin_vel_b          # (N, 3) 体坐标线速度
│   └── projected_gravity_b     # (N, 3) 投影重力
├── contacts/
│   ├── net_forces_w            # (N, B, 3) 接触力
│   ├── air_time                # (N, B) 空中时长
│   └── in_contact              # (N, B) 是否接触
└── sensors/
    └── imu/
        ├── lin_acc_b            # (N, 3) 加速度
        └── ang_vel_b            # (N, 3) 角速度

obs/                            # ObsBuilder 发布
├── policy                      # (N, D_policy) 策略观测
└── critic                      # (N, D_critic) 特权观测

reward/                         # RewardBuilder 发布
├── total                       # (N,) 加权总奖励
├── track_lin_vel_xy_exp        # (N,) 单项 raw 值
├── penalize_joint_torque_l2    # (N,) 单项 raw 值
└── _weights                    # dict 当前权重快照

action/                         # VecEnv.step 发布
└── raw                         # (N, A) 原始动作

episode/                        # VecEnv 发布
├── dones                       # (N,) 终止 mask
└── time_outs                   # (N,) 超时 mask
```

#### 3.3.3 设计 API（草案）

```python
class DataBus:
    """全局数据总线。单例，通过 get_databus() 获取。"""

    def publish(channel: str, data: Tensor, *, env_ids: Tensor | None = None) -> None
        """发布数据到 channel。无 Tap 时为 no-op。"""

    def tap(channel: str, *, buffer_len: int = 256,
            downsample: int = 1, gain: float = 1.0, offset: float = 0.0
    ) -> Tap
        """订阅 channel，返回 Tap 句柄。"""

    def list_channels() -> list[str]
        """列出所有已注册的 channel 路径。"""

    def channel_info(channel: str) -> ChannelInfo
        """返回 channel 元数据（shape, dtype, publish 频率）。"""

class Tap:
    """订阅句柄。"""

    @property
    def latest(self) -> Tensor | None
        """最新一帧数据（经 gain/offset 处理后）。"""

    @property
    def buffer(self) -> Tensor
        """环形缓冲区内容 (buffer_len, *shape)。"""

    def set_gain(gain: float) -> None
    def set_offset(offset: float) -> None
    def set_downsample(rate: int) -> None
    def close() -> None
        """取消订阅。"""

def get_databus() -> DataBus | None:
    """获取全局 DataBus。Train-time 可返回 None。"""

def enable_databus() -> DataBus:
    """显式启用 DataBus（Code-time 调用）。"""
```

#### 3.3.4 View 集成方式（侵入性最小）

```python
# 改动前（现有 JointView）：
@property
def pos(self) -> Tensor:
    d = self._asset.data.joint_pos
    return d[:, self._ids] if self._ids is not None else d

# 改动后（添加 publish，零开销守恒）：
@property
def pos(self) -> Tensor:
    d = self._asset.data.joint_pos
    result = d[:, self._ids] if self._ids is not None else d
    _bus = get_databus()
    if _bus is not None:        # Train-time 时 _bus 为 None → 零开销
        _bus.publish("robot/joints/pos", result)
    return result
```

**设计约束**：
- `get_databus()` 返回 `None` 时所有 publish 路径被编译器短路（无函数调用开销）
- 可以考虑更激进的优化：模块级 `_BUS` 变量，import 时绑定一次

---

> **注**：Oscilloscope（示波器）作为 DataBus 的消费者插件，详见 [§7.2 Oscilloscope](#72-oscilloscope--示波器)。

---

## 4. Task 层 — 观测/奖励/终止

> **定位**：定义「任务是什么」——观测空间、奖励信号、终止条件、课程调度。与仿真后端解耦，只通过 View API 读取状态。

### 4.1 ObsBuilder

```
myrl/src/myrl/core/task/obs_builder.py
```

**职责**：管理多个观测分组（policy / critic / amp_policy ...），生成兼容 instinct_rl 的 `obs_format`。

```python
class ObsGroup:
    add(term_name, func, shape) -> ObsGroup   # 注册观测项
    remove(term_name) -> ObsGroup
    obs_format() -> dict[str, tuple]           # {term_name: shape}
    compute(env) -> Tensor                     # (N, flat_dim) 拼接

class ObsBuilder:
    __getattr__(group_name) -> ObsGroup        # 懒创建 builder.policy / builder.critic
    get_obs_format() -> dict[str, dict[str, tuple]]  # 完整 obs_format
    compute(env) -> dict[str, Tensor]                 # {group: flat_tensor}
```

**与 ObsHistoryManager 的关系**：

```
ObsBuilder.compute(env) → 当前帧 obs_pack
    ↓
ObsHistoryManager.push(obs_pack) → 历史展开后的 obs_pack
    ↓
VecEnv.extras["observations"] → 传给 OnPolicyRunner
```

### 4.2 ObsHistoryManager

```
myrl/src/myrl/core/obs/history_manager.py
```

**职责**：环境无关的纯 torch 环形缓冲区，per-term 粒度历史管理。

```python
class ObsHistoryManager:
    __init__(obs_format, history_cfg, num_envs, device)

    push(obs_pack: dict[str, Tensor]) -> dict[str, Tensor]
    # 输入当前帧，输出 oldest→newest concat 的历史展开

    reset(env_ids: list[int] | None) -> None
    # 在 done 处理后立即调用，清零指定环境的历史

    get_output_format() -> dict[str, dict[str, tuple]]
    # 返回历史展开后的 obs_format（shape 变为 original_dim * history_length）
```

**history_cfg 格式**：

```yaml
# 组级别（整个 policy 组 8 帧历史）
policy: 8
critic: 1

# term 级别（精细控制）
policy:
  base_ang_vel: 8
  joint_pos: 3
  depth_image: 1
```

### 4.3 RewardBuilder

```
myrl/src/myrl/core/task/reward_builder.py
```

**职责**：组合多个 reward term，支持动态权重、激活切换、transform 流水线、checkpoint 持久化。

```python
class RewardBuilder:
    # 注册
    add(term_name, func, weight, active=True) -> self
    add_from_lib(term_name, weight, *, lib_name, robot_name, **params) -> self
    remove(term_name) -> self

    # 动态调整
    set_weight(term_name, weight) -> self
    set_active(term_name, active) -> self

    # Transform 流水线
    add_transform(transform: RewardTransform) -> self
    add_transform_from_lib(name, **params) -> self

    # 计算
    compute(env, step=0, return_per_term=False)
        -> (total_reward: Tensor[N], per_term: dict[str, Tensor[N]])
    # 流水线：① 无条件计算全部 term → ② 顺序执行 transforms → ③ 加权求和（仅 active）

    # 检视 & 持久化
    list_terms() -> dict[str, dict]
    state_dict() -> dict
    load_state_dict(d) -> None
```

**compute 流水线详解**：

```
Step 1: 计算全部 term（含 inactive，供 transform 感知）
    per_term = {name: func(env) for name, func in terms}

Step 2: 顺序执行 Transform 链
    for transform in transforms:
        per_term, weights = transform.apply(per_term, weights, step)

Step 3: 加权求和（仅 active terms）
    total = sum(per_term[name] * weights[name] for name in active_terms)
```

### 4.4 RewardLibrary — 奖励资产化系统

```
myrl/src/myrl/core/task/reward_lib/
├── __init__.py    # @reward_fn / @transform_fn 装饰器
├── meta.py        # RewardTermMeta / TransformMeta 元数据
├── library.py     # RewardLibrary / TransformLibrary 单例注册表
├── transform.py   # RewardTransform ABC + 4 内置算子
└── adapters.py    # make_instinctlab_rew_func() 适配器
```

**注册 & 使用流程**：

```python
# ① 定义 reward term（任意 Python 文件）
@reward_fn(
    description="Exponential kernel tracking of horizontal velocity",
    tags=["locomotion", "command_tracking", "dense"],
    params=TrackLinVelXYExpParams,
    version="1.0.0",
)
def track_lin_vel_xy_exp(robot: RobotHandle, params: TrackLinVelXYExpParams) -> Tensor:
    error = robot.get_command("base_velocity")[:, :2] - robot.root_lin_vel_b[:, :2]
    return torch.exp(-error.square().sum(dim=-1) / params.std**2)

# ② 查询 & 构建（需 AppLauncher 之后）
lib = get_reward_library()
lib.list_names()                          # ['track_lin_vel_xy_exp', ...]
lib.list_by_tag("locomotion")             # 按标签过滤
meta = lib.get("track_lin_vel_xy_exp")
meta.params_json_schema()                 # 标准 JSON Schema
func = lib.build("track_lin_vel_xy_exp", robot_name="robot", std=0.3)
```

**RewardTermMeta 结构**：

```python
@dataclass
class RewardTermMeta:
    name: str                    # 全局唯一名
    module: str                  # 源模块路径
    source_file: str             # 源文件绝对路径
    source_line: int             # 源行号
    version: str                 # 语义版本
    description: str             # 单行描述
    long_description: str        # 详细说明（可选）
    tags: list[str]              # 分类标签
    params: type[BaseModel]      # Pydantic 参数模型
    output_description: str      # 输出语义
    author: str
    added_in: str
    _func: Callable              # 原始函数引用

    params_json_schema() -> dict # 标准 JSON Schema
    to_dict() -> dict            # YAML/JSON 序列化
```

**4 内置 Transform 算子**：

| 算子 | 修改目标 | 有状态 | 功能 |
|------|---------|--------|------|
| `RunningNormalize` | per_term | ✅ Welford | 在线标准差归一化 |
| `RelativeRebalance` | weights | ✅ EMA | 追踪贡献比例，自动调权 |
| `ClipReward` | per_term | ✅ 可选 EMA | 截断 ±threshold |
| `WeightSchedule` | weights | ❌ | 线性/余弦课程调度（依赖 step 参数） |

**Transform ABC 合约**：

```python
class RewardTransform(ABC):
    @abstractmethod
    def apply(per_term, weights, step) -> (per_term, weights)

    def state_dict() -> dict       # checkpoint 持久化
    def load_state_dict(d) -> None
```

### 4.5 内置 Reward Terms

```
myrl/src/myrl/tasks/locomotion/mdp/rewards/
├── locomotion.py       # 3 个 locomotion terms
└── regularization.py   # 3 个 regularization terms
```

| Term | 参数 | 输出语义 |
|------|------|---------|
| `track_lin_vel_xy_exp` | `std`, `command_name` | exp(-||v_cmd - v_actual||² / std²) |
| `track_ang_vel_z_exp` | `std`, `command_name` | exp(-|ω_cmd - ω_actual|² / std²) |
| `feet_air_time_biped` | `foot_body_ids`, `threshold`, `sensor_name` | 奖励双足交替悬空 |
| `penalize_joint_torque_l2` | `joint_ids` | -||τ||² |
| `penalize_lin_accel` | — | -||a_root||² |
| `penalize_orientation` | — | -(g_proj_x² + g_proj_y²) |

### 4.6 待建模块

| 模块 | 说明 | 优先级 |
|------|------|--------|
| `Termination` | 终止条件管理器（跌倒/超限/自碰撞） | P3 |
| `Curriculum` | 课程调度器（地形难度/扰动强度/指令范围） | P3 |
| `BaseTask` | myrl 原生任务基类（整合 ObsBuilder + RewardBuilder + Termination） | P3 |

---

## 5. Algo 层 — 算法适配

> **定位**：利用 instinct_rl 提供的成熟算法，不重复造轮子。myrl 的创新在 Compat 和 Task 层，Algo 层以**最小接口对接 instinct_rl**。

### 5.1 当前策略：直接使用 instinct_rl

```python
# train.py 中的使用方式
from instinct_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(env, train_cfg, log_dir, device)
runner.learn(num_learning_iterations)
```

instinct_rl 已内化至 `myrl/third_party/instinct_rl/`（`.git` 改名 `.git_upstream`），可直接修改。

**OnPolicyRunner 的 myrl 扩展**（最小改动）：
- `_log_sinks: list[LogSink]` — 外部日志 sink 列表
- `add_log_sink(sink)` — 注册日志 sink
- `_dispatch_log_sinks(locs)` — 在 `log()` 末尾分发事件

### 5.2 算法矩阵

| 算法 | 类名 | 用途 |
|------|------|------|
| 标准 PPO | `PPO` | 基线训练 |
| 教师蒸馏 PPO | `TPPO` | 特权信息→student |
| AMP PPO | `WasabiPPO` | 运动模仿（对抗奖励） |
| 状态估计 PPO | `EstimatorPPO` | 学习状态重建 |
| VAE 蒸馏 | `VAEDistillPPO` | 潜空间编码 |
| 全组合 | `WasabiEstimatorPPO` | AMP + 状态估计 |

### 5.3 待建：自定义网络扩展

```
myrl/src/myrl/core/algo/    # 当前为空
```

**规划**：
- Transformer encoder（替代 MLP/CNN，用于长序列观测）
- 不 fork instinct_rl，通过继承和 `modules.build_actor_critic()` 的工厂注册

---

## 6. 资产系统

### 6.1 AssetStore 概述

```
myrl/src/myrl/assets/
├── __init__.py        # 资产解析 API：has_asset / resolve_asset / require_asset
├── asset_store.py     # AssetStore — 内容寻址存储
├── composer.py        # ExperimentComposer — 实验合成器
└── packager.py        # PackageBuilder / PackageReader — .myrlpkg 打包
```

#### 6.1.1 AssetStore — 内容寻址资产库

**存储结构**：

```
myrl/asset_store/
├── blobs/
│   └── <sha256[:2]>/<sha256[2:]>   # 内容寻址 blob（文件或 .tar.gz）
└── assets/
    ├── robot_model/g1_29dof@1.0.0.yaml      # 元数据记录
    ├── reward_fn/locomotion_rewards@1.0.0.yaml
    └── ...
```

**AssetType 枚举（10 种）**：

| 类型 | 说明 | 典型内容 |
|------|------|---------|
| `ROBOT_MODEL` | 机器人模型 | URDF/USD + meshes 目录 |
| `ACTUATOR_CFG` | 执行器配置 | PD 增益、力矩限制 YAML |
| `SENSOR_CFG` | 传感器配置 | 接触传感器、IMU 参数 YAML |
| `TERRAIN` | 地形资产 | 平面/台阶/碎石 + 元数据 |
| `REWARD_FN` | 奖励函数 | Python 模块（含 @reward_fn） |
| `REWARD_PIPELINE` | 奖励管线 | term 列表 + 权重 + transform YAML |
| `OBS_PIPELINE` | 观测管线 | term 列表 + scale + history YAML |
| `ALGO_CFG` | 算法配置 | PPO 超参数 YAML |
| `ENV_SCRIPT` | 环境脚本 | gym.register + EnvCfg 定义 |
| `EXPERIMENT_CFG` | 实验配置 | 引用上述所有资产的顶层 YAML |

**核心 API**：

```python
class AssetStore:
    register(asset_type, name, version, source_path, **metadata) -> AssetRecord
    get(name, version, asset_type) -> AssetRecord
    list(asset_type=None) -> list[AssetRecord]
    export(record, dest_path) -> None
    verify(record) -> bool   # SHA256 校验
```

**AssetRecord**：

```python
@dataclass
class AssetRecord:
    name: str
    version: str
    asset_type: AssetType
    content_hash: str        # SHA256
    blob_path: str           # 内容寻址路径
    is_directory: bool
    description: str
    tags: list[str]
    created_at: str
    metadata: dict
    asset_id: str            # property: "{name}@{version}"
```

#### 6.1.2 资产解析 API

**文件**：`myrl/src/myrl/assets/__init__.py`

```python
MYRL_ASSETS_DIR = Path(__file__).parents[3] / "assets"  # myrl/assets/

has_asset(relative_path: str) -> bool
resolve_asset(relative_path: str) -> str | None       # 返回绝对路径或 None
resolve_asset_dir(relative_path: str) -> str | None    # 解析目录
require_asset(relative_path: str) -> str               # 不存在则抛异常
```

**优先级**：`MYRL_ASSETS_DIR` 环境变量 > 默认 `myrl/assets/` 目录。

#### 6.1.3 资产 YAML 配置规范

**实验配置** (`experiments/*.yaml`)：

```yaml
name: g1_locomotion_v1
version: "1.0.0"
description: "G1 平地行走基线实验"

assets:
  robot_model:     { name: g1_29dof,     version: "1.0.0", source: myrl/assets/robots/g1/ }
  actuator_cfg:    { name: g1_default,   version: "1.0.0", source: myrl/assets/actuator_cfgs/g1_default.yaml }
  sensor_cfg:      { name: g1_contact,   version: "1.0.0", source: myrl/assets/sensor_cfgs/g1_contact.yaml }
  terrain:         { name: flat_plane,   version: "1.0.0", source: myrl/assets/terrains/flat_plane/ }
  reward_fns:      # 列表，可引用多个 reward 模块
    - { name: locomotion_rewards, version: "1.0.0", source: myrl/src/myrl/tasks/.../rewards/ }
  reward_pipeline: { name: g1_loco_v1,   version: "1.0.0", source: myrl/assets/reward_pipelines/g1_loco_v1.yaml }
  obs_pipeline:    { name: g1_standard_obs, version: "1.0.0", source: myrl/assets/obs_pipelines/g1_standard_obs.yaml }
  algo_cfg:        { name: ppo_standard, version: "1.0.0", source: myrl/assets/algo_cfgs/ppo_standard.yaml }
  env_script:      { name: g1_flat_packaged, version: "1.0.0", source: .../env_script.py }
```

**奖励管线** (`reward_pipelines/*.yaml`)：

```yaml
name: g1_loco_v1
version: "1.0.0"
terms:
  track_lin_vel_xy_exp:  { weight: 1.5,    params: { std: 0.25 } }
  track_ang_vel_z_exp:   { weight: 0.75,   params: { std: 0.25 } }
  feet_air_time_biped:   { weight: 0.5,    params: { threshold: 0.35 } }
  penalize_joint_torque_l2: { weight: -0.001 }
  penalize_orientation:     { weight: -1.0 }
  penalize_lin_accel:       { weight: -0.01 }
transforms: []  # 可添加 RunningNormalize / WeightSchedule 等
```

**观测管线** (`obs_pipelines/*.yaml`)：

```yaml
name: g1_standard_obs
version: "1.0.0"
policy:
  base_ang_vel:     { func: mdp.base_ang_vel,     scale: 0.25, history_length: 1 }
  projected_gravity:{ func: mdp.projected_gravity, scale: 1.0,  history_length: 1 }
  velocity_commands:{ func: mdp.generated_commands,scale: 1.0,  history_length: 1 }
  joint_pos:        { func: mdp.joint_pos_rel,     scale: 1.0,  history_length: 1 }
  joint_vel:        { func: mdp.joint_vel_rel,     scale: 0.05, history_length: 1 }
  last_actions:     { func: mdp.last_action,       scale: 1.0,  history_length: 1 }
```

**算法配置** (`algo_cfgs/*.yaml`)：

```yaml
name: ppo_standard
version: "1.0.0"
runner:
  num_steps_per_env: 24
  max_iterations: 30000
  save_interval: 1000
policy:
  class_name: ActorCritic
  init_noise_std: 1.0
  actor_hidden_dims: [256, 256, 128]
  critic_hidden_dims: [256, 256, 128]
algorithm:
  class_name: PPO
  learning_rate: 0.001
  num_learning_epochs: 5
  num_mini_batches: 4
  gamma: 0.99
  lam: 0.95
  desired_kl: 0.01
  schedule: adaptive
  entropy_coef: 0.005
```

**执行器配置** (`actuator_cfgs/*.yaml`)：

```yaml
name: g1_default
version: "1.0.0"
type: position_pd
default_gains: { kp: 100.0, kd: 2.0 }
joint_overrides:
  left_knee_joint:  { kp: 150.0, kd: 3.0 }
  right_knee_joint: { kp: 150.0, kd: 3.0 }
torque_limits:    # 正则表达式匹配
  ".*hip.*":   88.0
  ".*knee.*":  139.0
  ".*ankle.*": 50.0
```

**传感器配置** (`sensor_cfgs/*.yaml`)：

```yaml
name: g1_contact
version: "1.0.0"
sensors:
  - type: contact
    name: contact_forces
    body_regex: "Robot/.*"
    history_length: 3
    track_air_time: true
  - type: imu
    name: base_imu
    prim_path: torso_link
```

---

### 6.2 实验合成与打包

#### 6.2.1 ExperimentComposer — 7 步流水线

```
myrl/src/myrl/assets/composer.py
```

```python
class ExperimentComposer:
    __init__(package_path: str)           # 接收 .myrlpkg 路径
    manifest: PackageManifest             # 包元数据
    compose(num_envs, device) -> (env, runner_cfg_dict)
```

**7 步流水线**：

```
① 读取 package.yaml + experiment.yaml
    ↓
② os.environ["MYRL_ASSETS_DIR"] = pkg.assets_dir
    ↓
③ sys.path.insert(0, pkg/reward_fns/) → import → @reward_fn 自动注册
    ↓
④ 运行 terrain/generators/*.py → 注册地形生成 term
    ↓
⑤ 加载 reward_pipeline YAML → 构建 RewardBuilder（deferred params 暂存）
    ↓
⑥ import env_script → 注入 _COMPOSER_REWARD_BUILDER
    ↓
⑦ gym.register + gym.make → 返回 (env, runner_cfg_dict)
```

**调用约束**：必须在 `AppLauncher` 之后调用 `compose()`（Isaac Lab 依赖）。

#### 6.2.2 Packager — .myrlpkg 格式

```python
class PackageBuilder:
    from_yaml_file(yaml_path, repo_root) -> PackageBuilder   # 从实验 YAML 构建
    from_experiment_cfg(name, version) -> PackageBuilder       # 从 AssetStore 构建
    build(output_dir) -> str                                   # 返回 .myrlpkg 路径

class PackageReader:
    __init__(package_path)
    manifest: PackageManifest
    list_reward_fn_dirs() -> list[str]
    get_terrain_generators_dir() -> str
    get_reward_pipeline_path() -> str
    get_env_script_path() -> str

class PackageManifest:
    package_version: str
    package_id: str            # UUID
    experiment_name: str
    created_at: str
    source_experiment_cfg: str
    asset_checksums: list[AssetChecksum]
```

**.myrlpkg 内部结构**：

```
experiment_name.myrlpkg (zip)
├── package.yaml          # PackageManifest
├── experiment.yaml       # 原始实验配置
├── assets/               # 收集的资产快照
│   ├── robots/g1/
│   ├── actuator_cfgs/
│   └── ...
├── reward_fns/           # Python 模块
│   └── locomotion.py
├── reward_pipeline.yaml  # 奖励管线
├── obs_pipeline.yaml     # 观测管线
├── algo_cfg.yaml         # 算法配置
└── env_script.py         # 环境脚本
```

---

## 7. 日志、可观测性与调试

```
myrl/src/myrl/logging/
├── __init__.py              # build_sinks() 工厂
├── sinks/
│   ├── base.py              # LogSink ABC + LogEvent
│   ├── jsonl_sink.py        # JSONLSink
│   └── wandb_sink.py        # WandbSink
└── server/
    ├── log_server.py        # SSELogServer
    └── log_client.py        # SSEClient
```

### 7.1 三层日志架构

```
OnPolicyRunner.log()
    ↓ _dispatch_log_sinks(locs)
    ↓
    ├── JSONLSink        →  {log_dir}/metrics.jsonl    (always-on, 逐行 flush)
    ├── WandbSink        →  wandb.run.summary          (可选，通过 --wandb 启用)
    └── SSELogServer     →  HTTP :7000                  (可选，通过 --log_server_port)
                              ├── /stream     (SSE 实时流)
                              ├── /history    (历史回看)
                              ├── /metrics    (最新快照)
                              └── /health     (健康检查)
```

**LogEvent 结构**：

```python
@dataclass
class LogEvent:
    iteration: int            # 训练迭代数
    timestamp: float          # time.time()
    metrics: dict[str, float] # 标量指标（Loss, reward, fps...）
    extras: dict[str, Any]    # 非标量（直方图、图片...）
```

**LogSink ABC**：

```python
class LogSink(ABC):
    @abstractmethod
    def write(event: LogEvent) -> None
    def close() -> None
```

**build_sinks 工厂**：

```python
def build_sinks(args_cli, log_dir, run_name) -> list[LogSink]:
    """根据 CLI 参数自动构建 sink 列表。"""
    sinks = []
    if not args_cli.no_jsonl:
        sinks.append(JSONLSink(log_dir))
    if args_cli.wandb:
        sinks.append(WandbSink())
    if args_cli.log_server_port:
        sinks.append(SSELogServer(args_cli.log_server_host, args_cli.log_server_port))
    return sinks
```

### 7.2 SSE 日志协议

**SSE 事件格式**：

```
event: log
data: {"iteration": 1000, "timestamp": 1712345678.9, "metrics": {"Loss/value": 0.5, ...}}

event: log
data: {"iteration": 1001, ...}
```

**SSEClient**：

```python
class SSEClient:
    stream() -> Iterator[dict]                    # 阻塞式 SSE 订阅
    fetch_history(n: int) -> list[dict]           # 历史回看
    fetch_metrics() -> dict[str, float]           # 最新快照
    health_check() -> bool                        # 服务器存活检查
```

**技术要点**：
- 必须用 `resp.readline()` 逐行读取 SSE，不能用 `read(4096)`（缓冲区不满会阻塞）
- SSE handler 必须在 `end_headers()` 后 `self.wfile.flush()`
- 每个 SSE 客户端获取独立 `queue.Queue`，多播不互相阻塞

### 7.3 Oscilloscope — 示波器

> **定位**：DataBus 的消费者插件。**Code-time only**，在 Docker 容器内运行。去掉它框架照常运行。

```
myrl/src/myrl/debug_tools/oscilloscope/   [待建]
├── scope.py           # 主类（管理 Tap 集合）
├── inspector.py       # Inspector 面板（选中体的状态详情）
├── signal_view.py     # 信号波形视图（类示波器 UI）
└── interaction.py     # 鼠标交互（拾取/施力/关节覆写）
```

**功能矩阵**：

| 功能 | 数据源 | 优先级 |
|------|--------|--------|
| Inspector 面板（选中体状态数值表） | DataBus tap | P1 |
| 信号波形（多通道实时叠加） | DataBus tap buffer | P1 |
| 3D 视口覆盖层（力箭头、关节轴） | Isaac Lab viewer | P1 |
| 鼠标拾取/施力 | Isaac Lab env | P2 |
| 暂停/单步/慢放 | AppLauncher | P2 |
| 关节滑块/指令覆写 | Isaac Lab env | P3 |

示波器**叠加在** Isaac Lab viewer 之上（`omni.ui` 或 `imgui` overlay），不替代它。

启用方式：`--oscilloscope` 参数或 `MYRL_OSCILLOSCOPE=1` 环境变量。

---

## 8. 训练管控系统

```
myrl/scripts/
├── train_manager.py     # HTTP 服务端（:7001），进程生命周期管理
├── train_cli.py         # stdlib-only CLI 客户端
└── train_tui.py         # Textual btop 风格 TUI
```

### 8.1 train_manager — 管控服务端

**架构**：

```
train_manager.py (:7001)
├── ProcessCtrl          # 训练进程生命周期
│   └── subprocess       # train.py (:7000 SSE Log)
├── SSEBroadcaster       # 多客户端 SSE 事件分发
├── GPUMetrics           # nvidia-smi + /proc/stat 采样
├── SSEProxy             # :7000 → SSEBroadcaster 转发
└── HTTP Handler
    ├── GET /health
    ├── GET /status
    ├── GET /stream?filter=<type>
    ├── GET /history?n=200
    ├── GET /console?n=200
    ├── POST /start
    ├── POST /stop
    ├── POST /kill
    ├── POST /halt
    ├── POST /resume
    └── POST /checkpoint
```

**进程生命周期状态机**：

```
                    POST /start
    stopped ──────────────────────► starting
       ▲                               │
       │ POST /stop                    │ subprocess ready
       │ (SIGTERM)                     ▼
    stopping ◄──────────────────── running
       ▲                               │
       │                     POST /halt│ (SIGUSR1)
       │                               ▼
       │                            halted
       │                               │
       │                     POST /resume (SIGUSR2)
       │                               │
       └───────────────────────────────┘
```

**SSE 事件类型**：

| 类型 | 来源 | 内容 |
|------|------|------|
| `system` | train_manager | 启动/停止/错误事件 |
| `train` | SSEProxy → :7000 | 训练指标（Loss, reward, fps） |
| `console` | stdout/stderr 捕获 | 控制台输出行 |
| `status` | 定期轮询 | GPU/CPU/内存使用率 |

### 8.2 train_cli — 命令行客户端

```
python scripts/train_cli.py <command> [options]
环境变量: MYRL_HOST (默认 localhost), MYRL_PORT (默认 7001)
```

| 命令 | 说明 |
|------|------|
| `status` | 进程状态 + GPU 指标 + 迭代进度 + ETA |
| `start --task <id> --num_envs N` | 启动训练 |
| `stop` | SIGTERM 优雅停止 |
| `kill` | SIGKILL 强制终止 |
| `halt` | SIGUSR1 在迭代边界暂停 |
| `resume` | SIGUSR2 恢复 |
| `checkpoint` | 保存检查点 + 恢复（halt → 10s → resume） |
| `stream [--filter type]` | tail -f 风格 SSE 流 |
| `console [--n N]` | 控制台日志 |

### 8.3 train_tui — btop 风格 TUI

```
python scripts/train_tui.py --host <ip> --port 7001
python scripts/train_tui.py --mock   # 模拟数据测试布局
```

**布局**：

```
┌─ myrl Training Dashboard ──────────────────────────────────┐
│                                                             │
│  ┌─ GPU / System ──────┐  ┌─ Console ─────────────────────┐│
│  │ GPU 0: ████░ 78%    │  │ [14:32:05] iter 1000/30000   ││
│  │ VRAM:  ██████ 12G   │  │ [14:32:05] mean_reward: 2.3  ││
│  │ CPU:   ██░░░ 45%    │  │ [14:32:06] Loss/value: 0.52  ││
│  │ RAM:   ████░ 16G    │  │ [14:32:06] fps: 12345        ││
│  ├──────────────────────┤  │ ...                           ││
│  │ State: running       │  │                               ││
│  │ Task: G1-Loco-v0     │  │                               ││
│  │ Iter: 1000 / 30000   │  │                               ││
│  ├──────────────────────┤  │                               ││
│  │ Metrics              │  │                               ││
│  │ reward    2.3  ↑0.1  │  │                               ││
│  │ Loss      0.52 ↓0.02 │  │                               ││
│  │ fps       12345      │  │                               ││
│  └──────────────────────┘  └───────────────────────────────┘│
│ [S]tart [T]erm [H]alt [R]esume [K]ill [C]heckpoint [Q]uit  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 仿真服务系统

```
myrl/src/myrl/core/sim_server/
├── protocol.py          # SimProto — TCP 帧协议
├── base_server.py       # SimServer ABC
├── mujoco_server.py     # MuJoCoSimServer — MuJoCo 向量化仿真
├── mujoco_task.py       # MuJoCoTask ABC + DummyTask
└── ros2_bridge.py       # Ros2Bridge — ROS2 ↔ TCP 双向桥接
```

### 9.1 SimProto — TCP 帧协议

```
帧格式: [4 字节 big-endian uint32 长度] [msgpack-numpy 编码的 payload]

消息类型 (MsgType):
  HANDSHAKE_REQ/RESP   # 握手：交换 num_envs, obs_format, num_actions 等
  STEP_REQ/RESP        # 物理步：发送 actions，返回 obs/rewards/dones
  RESET_REQ/RESP       # 重置：返回初始 obs
  GET_OBS_REQ/RESP     # 获取当前 obs
  CLOSE                # 关闭连接
  ERROR                # 错误报告
```

### 9.2 MuJoCoSimServer

```python
class MuJoCoSimServer(SimServer):
    __init__(task, mjcf_path, num_envs, sim_steps_per_ctrl, host, port, include_obs_in_response)

    register_obs_callback(fn: (dict) -> None)  # obs_callback 在 TCP resp 之前调用
    # ROS 模式下 include_obs_in_response=False，TCP 只传 rewards/dones
```

**向量化仿真**：N 个 `MjData` 实例并行 step，自动 reset on done。

### 9.3 MuJoCoTask ABC

```python
class MuJoCoTask(ABC):
    # 必须实现
    num_actions: int                                              # property
    max_episode_length: int                                       # property
    obs_format() -> dict[str, dict[str, tuple]]                   # 观测格式
    compute_obs(model, datas: list) -> dict[str, np.ndarray]      # 计算观测
    compute_reward(model, datas, actions) -> np.ndarray           # 计算奖励
    is_terminated(model, datas) -> np.ndarray                     # 终止判断
    apply_action(model, datas, actions) -> None                   # 施加动作
    reset_env(model, data, env_id) -> None                        # 单环境重置

    # 可选覆写
    num_rewards: int = 1                                          # 默认单奖励
```

### 9.4 ROS2 桥接架构（目标架构）

```
                    ROS2 Topics (传感器总线)
                ┌──────────────────────────────────┐
                │  /myrl/{task_id}/obs/policy       │ Float32MultiArray
                │  /myrl/{task_id}/obs/critic       │ (可选)
                └──────────┬───────────────────────┘
                           │ subscribe
                ┌──────────▼───────────────────────┐
                │       Ros2Bridge                  │
                │  ┌─ ObsHistoryManager ────────┐  │
                │  │  push(current_obs)          │  │
                │  │  → history-expanded obs     │  │
                │  └────────────────────────────┘  │
                └──────────┬───────────────────────┘
                           │ TCP (actions → rewards/dones)
                ┌──────────▼───────────────────────┐
                │     MuJoCoSimServer               │
                │  obs_callback (publish to ROS)    │
                │  step → rewards/dones (TCP only)  │
                └──────────────────────────────────┘
```

**同步保障**：
1. Server 先调用 `obs_callback`（ROS publish）
2. 再发送 TCP `STEP_RESP`（只含 rewards/dones）
3. Bridge 用 `Event.wait()` 等待 ROS obs 到达

---

## 10. 注册表与可复现性

```
myrl/src/myrl/registry/
├── registry.py    # RunRegistry — 运行清单管理
├── manifest.py    # RunManifest — 训练运行元数据
└── checksum.py    # SHA256 工具
```

### 10.1 RunManifest — 训练运行元数据

```python
@dataclass
class RunManifest:
    manifest_version: str
    run_id: str                  # UUID
    experiment_name: str
    created_at: str
    task_id: str                 # gymnasium task id
    env_cfg_sha256: str          # 环境配置哈希
    max_iterations: int
    num_envs: int
    seed: int
    device: str
    assets: list[AssetEntry]     # 资产清单 + SHA256
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_iteration: int
    myrl_commit: str             # git commit hash
    package_id: str | None       # .myrlpkg UUID
    metrics_snapshot: dict       # 最终指标快照

    @classmethod
    def from_train_run(log_dir, task_id, agent_cfg, runner) -> RunManifest
    # 训练结束时自动生成
```

### 10.2 RunRegistry

```python
class RunRegistry:
    __init__(registry_dir=None)   # 默认 myrl/registry/

    save(manifest) -> str         # 返回 run_id
    load(run_id, experiment_name=None) -> RunManifest
    list_runs(experiment_name=None) -> list[RunManifest]
    verify(manifest) -> dict[str, bool]  # 校验所有 SHA256
```

**存储结构**：

```
myrl/registry/
├── manifests/<sha8>.yaml                    # 内容寻址
└── runs/<experiment>/<run_id> → symlink     # 按实验索引
```

---

## 11. 部署架构

### 11.1 Train-time 部署（GPU 服务器）

```bash
# 零依赖自举（无 Docker、无 sudo、无 GUI）
WORKDIR=~/myrl_work bash myrl/scripts/bootstrap_train.sh

# 自举完成后的环境
~/myrl_work/
├── .mamba/envs/myrl-train/    # micromamba 环境
│   └── bin/python3            # Python 3.10 + torch + CUDA
├── .site-packages/            # 持久化额外依赖
└── logs/                      # 训练日志
```

**bootstrap_train.sh 做什么**：
1. 下载 micromamba（如果需要）
2. 从 `env/train.yml` 创建 conda 环境
3. 自动检测 GPU 算力 → 选择 PyTorch CUDA 索引（cu124/cu128）
4. 安装 PyTorch + 验证 CUDA kernel launch
5. 解压 IsaacGym tarball → `.pth` 注册（不用 PyPI 的 isaacgym）
6. 安装 third_party 依赖（--no-deps 避免冲突）
7. 运行 smoke_env.py 验证

### 11.2 Code-time 部署（开发机）

```bash
bash myrl/scripts/run_dev.sh
# → bootstrap_dev.sh（Docker + NVIDIA Container Toolkit + EULA）
# → docker compose build
# → docker compose run --rm dev
```

**容器环境**：

| 组件 | 说明 |
|------|------|
| 基础镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2` |
| 额外安装 | ROS2 Humble (ros-base + messages) |
| 网络 | `host` 模式（X11 + GPU 直通） |
| 共享内存 | 16GB（Isaac Sim 要求） |
| 挂载卷 | 仓库 `:rw`，工作目录 `:rw`，X11 socket |
| PYTHONPATH | instinct_rl, instinctlab, myrl/src（不 pip install） |

### 11.3 部署矩阵

| 场景 | 仿真后端 | 部署方式 | GPU |
|------|---------|---------|-----|
| 高速训练 | Isaac Gym | 裸机 bootstrap | ✅ 必须 |
| 开发调试 | Isaac Lab | Docker 容器 | ✅ 必须 |
| Sim2Sim | MuJoCo (TCP) | 裸机/容器皆可 | ❌ 可选 |
| 大规模微调 | MJX (JAX) | 裸机 | ✅ 必须 |
| 实机部署 | ROS2 bridge | ROS2 节点 | ❌ 可选 |

---

## 12. 数据流全景

### 12.1 训练循环数据流

```
┌─ OnPolicyRunner.learn() ──────────────────────────────────────────────┐
│                                                                        │
│  for iteration in range(max_iterations):                               │
│                                                                        │
│    ┌─ 采集阶段 ────────────────────────────────────────────────────┐   │
│    │  for step in range(num_steps_per_env):                         │   │
│    │                                                                │   │
│    │    obs ──► ActorCritic.act(obs, critic_obs)                    │   │
│    │                ↓                                               │   │
│    │           actions: Tensor[N, A]                                │   │
│    │                ↓                                               │   │
│    │    VecEnv.step(actions) ──► IsaacLabBackend / MuJoCoBackend    │   │
│    │                ↓                                               │   │
│    │    (obs, rewards[N,R], dones[N], extras)                       │   │
│    │                ↓                                               │   │
│    │    PPO.process_env_step(rewards, dones, extras)                │   │
│    │                                                                │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│    ┌─ 更新阶段 ────────────────────────────────────────────────────┐   │
│    │  PPO.compute_returns(last_critic_obs)                          │   │
│    │  PPO.update(iteration) → (loss_dict, stat_dict)                │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│    ┌─ 日志阶段 ────────────────────────────────────────────────────┐   │
│    │  runner.log(locs)                                              │   │
│    │    ├── TensorBoard writer                                      │   │
│    │    └── _dispatch_log_sinks(locs) → [JSONL, wandb, SSE]         │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 12.2 观测数据流

```
仿真后端 (Isaac Lab / MuJoCo)
    ↓ 原始传感器数据
ObservationManager / MuJoCoTask.compute_obs()
    ↓ {group: {term: Tensor}}
ObsBuilder.compute(env)   ← myrl Task 层
    ↓ {group: flat_Tensor}
ObsHistoryManager.push(obs_pack)   ← 可选
    ↓ {group: history_expanded_Tensor}
VecEnv.extras["observations"]
    ↓
OnPolicyRunner → ActorCritic.act(policy_obs, critic_obs)
```

### 12.3 奖励数据流

```
RobotHandle (View API)
    ↓ 状态查询
@reward_fn terms  (通过 RewardLibrary.build() 实例化)
    ↓ per_term: {name: Tensor[N]}
RewardTransform 链 (RunningNormalize → RelativeRebalance → ...)
    ↓ 修改后的 per_term + weights
RewardBuilder.compute()  加权求和
    ↓ total_reward: Tensor[N]
VecEnv.step() → rewards → PPO.process_env_step()
```

### 12.4 资产数据流

```
开发者定义实验
    ↓ experiments/g1_locomotion_v1.yaml
asset_cli.py register  → AssetStore (SHA256 寻址)
    ↓
asset_cli.py pack  → PackageBuilder → .myrlpkg
    ↓
asset_cli.py deploy  → rsync/scp 到 GPU 服务器
    ↓
train.py --package xxx.myrlpkg
    ↓
ExperimentComposer.compose()
    ↓ 7 步流水线
(env, runner_cfg_dict) → OnPolicyRunner.learn()
    ↓ 训练完成
RunManifest.from_train_run() → RunRegistry.save()
```

### 12.5 远程管控数据流

```
GPU 服务器 (Train)                    开发机 (DEV)
┌────────────────────┐              ┌────────────────────┐
│ train_manager :7001│◄─── HTTP ───►│ train_cli.py       │
│  └─ train.py :7000 │              │ train_tui.py       │
│     ├─ JSONL (disk) │              │ log_viewer.py      │
│     ├─ wandb (cloud)│              │ reward_inspector.py│
│     └─ SSE → :7001  │              └────────────────────┘
│        SSEProxy      │                     ▲
│        SSEBroadcaster│──── SSE stream ─────┘
│        GPUMetrics    │
└────────────────────┘
        ▲
        │ TailScale / SSH tunnel
        │
    任意网络位置
```

---

## 13. 模块状态与路线图

### 13.1 当前状态（2026-04-04）

```
████████████████████████░░░  ~82% 完成

代码量统计：
  核心库 (src/myrl/):  ~4,600 行
  脚本 (scripts/):     ~3,500 行
  总计:                ~8,000+ 行 Python（不含 third_party 和资产文件）
```

### 13.2 模块状态总览

| 模块 | 状态 | 验证 | 行数 |
|------|------|------|------|
| **Compat / IsaacLabBackend** | ✅ 完整 | Phase A 端到端 | 249 |
| **Compat / MuJoCoBackend** | ✅ 完整 | 23/23 测试 | 184 |
| **Compat / MJXBackend** | ❌ 待建 | — | 0 |
| **Views (全部)** | ✅ 完整 | 奖励 term 已依赖 | ~330 |
| **ObsBuilder** | ✅ 完整 | — | 54 |
| **ObsHistoryManager** | ✅ 完整 | 32/32 测试 | 140 |
| **RewardBuilder** | ✅ 完整 | Phase E 验证 | 155 |
| **RewardLibrary** | ✅ 完整 | Phase E 验证 | ~850 |
| **内置 Reward Terms** | ✅ 完整 | — | ~240 |
| **SimServer / Protocol** | ✅ 完整 | 23/23 测试 | ~560 |
| **MuJoCoTask ABC** | ✅ 完整 | DummyTask 验证 | 197 |
| **ROS2 Bridge** | ⚠️ ~70% | 核心结构完整 | 394 |
| **AssetStore** | ✅ 完整 | 运行时数据已填充 | 274 |
| **ExperimentComposer** | ✅ 完整 | — | 256 |
| **Packager** | ✅ 完整 | — | 474 |
| **RunRegistry** | ✅ 完整 | — | ~320 |
| **日志系统** | ✅ 完整 | SSE 验证通过 | ~540 |
| **train_manager** | ✅ 完整 | — | 578 |
| **train_cli** | ✅ 完整 | — | 208 |
| **train_tui** | ✅ 完整 | — | 362 |
| **asset_cli** | ✅ 完整 | — | 323 |
| **asset_tui** | ✅ 完整 | — | 445 |
| **Docker / Bootstrap** | ✅ 完整 | 端到端验证 | ~670 |
| **DataBus** | ❌ 待建 | — | 0 |
| **Oscilloscope 示波器** | ❌ 待建 | — | 0 |
| **core/algo 扩展** | ❌ 待建 | — | 0 |

### 13.3 路线图（愿景重定义后）

```
2026 Q2                        Q3                          Q4
──┬────────────────────────┬──────────────────────────┬────────────
  │                        │                          │
  ▼                        ▼                          ▼
┌────────────────────────┐┌──────────────────────────┐┌──────────────────┐
│ ★ DataBus 核心          ││ Oscilloscope v1           ││ Oscilloscope v2  │
│ channel pub/sub         ││ Inspector 面板            ││ 鼠标拾取/施力    │
│ View 自动 publish       ││ 信号波形（多通道叠加）     ││ 关节滑块覆写     │
│ 零订阅者零开销验证       ││ 3D overlay（力箭头等）    ││ 暂停/单步/慢放   │
│                        ││                          ││                  │
│ Phase B 验证            ││ myrl 原生任务             ││ MJX 后端         │
│ IsaacLabBackend 对比    ││ humanoid_x 行走 v0       ││ JAX 大规模训练   │
│                        ││                          ││                  │
│ reward YAML 管线验证    ││ obs_pipeline YAML 集成    ││ 实机部署 v1      │
│ (已有基础，端到端串通)   ││ Composer 端到端           ││                  │
└────────────────────────┘└──────────────────────────┘└──────────────────┘
      ★ P0 核心创新               P1-P3                     P4+
```

### 13.4 各阶段里程碑

| 阶段 | 状态 | 关键交付 |
|------|------|---------|
| A — instinctlab wrapper 训练链路 | ✅ | 5 iter, EXIT_CODE:0 |
| B — IsaacLabBackend 自有 compat 层 | ⚠️ | 切换后 loss < 1e-4 |
| C — 资产解析 + myrl 原生任务注册 | ✅ | G1Smoke 端到端 |
| D — MuJoCo TCP 后端 + ROS2 桥接 | ✅ | 23/23 测试 |
| D+ — ROS=传感器总线 + ObsHistoryManager | ✅ | 32/32 测试 |
| E — 奖励函数资产化系统 | ✅ | 全验证通过 |
| F — 三层日志体系 + 上游库内化 | ✅ | SSE 验证通过 |
| G — 远程训练管控系统 | ✅ | 语法检查通过 |
| H — Asset/Experiment 管理系统 | ✅ | AssetStore + Composer + Packager |
| **★ I — DataBus 数据总线** | ❌ | **新 P0，核心创新** |
| **★ J — Oscilloscope 示波器 v1** | ❌ | **Inspector + 信号波形** |
| P-MJX — MJX 大规模训练后端 | ❌ | 待建 |

---

## 附录 A. 完整 API 索引

### 核心包 `myrl`

| 模块路径 | 主要导出 |
|---------|---------|
| `myrl.core.databus` | `DataBus`, `Tap`, `ChannelInfo`, `get_databus`, `enable_databus` [待建] |
| `myrl.debug_tools.oscilloscope` | `Oscilloscope`, `Inspector`, `SignalView` [待建] |
| `myrl.assets` | `has_asset`, `resolve_asset`, `require_asset`, `AssetStore`, `AssetType`, `AssetRecord`, `PackageBuilder`, `PackageReader` |
| `myrl.assets.composer` | `ExperimentComposer` |
| `myrl.core.compat.backends.base` | `SimBackend` |
| `myrl.core.compat.backends.isaaclab_backend` | `IsaacLabBackend` |
| `myrl.core.compat.backends.mujoco_backend` | `MuJoCoBackend` |
| `myrl.core.compat.views` | `RobotHandle`, `JointView`, `BodyView`, `ContactView`, `SensorView`, `ImuView`, `make_term`, `make_rew` |
| `myrl.core.task.obs_builder` | `ObsBuilder`, `ObsGroup` |
| `myrl.core.task.reward_builder` | `RewardBuilder` |
| `myrl.core.task.reward_lib` | `reward_fn`, `transform_fn`, `RewardLibrary`, `TransformLibrary`, `RewardTermMeta`, `TransformMeta`, `RewardTransform` |
| `myrl.core.obs.history_manager` | `ObsHistoryManager` |
| `myrl.core.sim_server` | `SimProto`, `MsgType`, `SimServer`, `MuJoCoSimServer`, `MuJoCoTask`, `DummyTask`, `Ros2Bridge` |
| `myrl.logging` | `LogSink`, `LogEvent`, `JSONLSink`, `WandbSink`, `SSELogServer`, `SSEClient`, `build_sinks` |
| `myrl.registry` | `RunRegistry`, `RunManifest`, `AssetEntry`, `sha256_file`, `sha256_bytes`, `sha256_dict` |

---

## 附录 B. 配置文件规范

### 文件位置约定

| 类型 | 路径 | 格式 |
|------|------|------|
| 实验配置 | `myrl/assets/experiments/*.yaml` | [见 §5.1.3] |
| 奖励管线 | `myrl/assets/reward_pipelines/*.yaml` | [见 §5.1.3] |
| 观测管线 | `myrl/assets/obs_pipelines/*.yaml` | [见 §5.1.3] |
| 算法配置 | `myrl/assets/algo_cfgs/*.yaml` | [见 §5.1.3] |
| 执行器配置 | `myrl/assets/actuator_cfgs/*.yaml` | [见 §5.1.3] |
| 传感器配置 | `myrl/assets/sensor_cfgs/*.yaml` | [见 §5.1.3] |
| 地形元数据 | `myrl/assets/terrains/*/terrain_meta.yaml` | 地形参数 |
| 机器人模型 | `myrl/assets/robots/*/` | URDF/USD + meshes |

### 版本规范

所有配置文件使用语义版本 `MAJOR.MINOR.PATCH`：
- MAJOR：不兼容的接口变更
- MINOR：向后兼容的功能新增
- PATCH：向后兼容的 bug 修复

---

## 附录 C. CLI/TUI 命令索引

### 训练 & 推理

| 命令 | 说明 |
|------|------|
| `python scripts/train.py --task <id>` | 本地训练（Isaac Lab） |
| `python scripts/train.py --package <pkg>` | 从 .myrlpkg 训练 |
| `python scripts/play.py --task <id> --load_run <dir>` | 策略推理/可视化 |
| `python scripts/play_mujoco.py --host <ip>` | MuJoCo 远程推理 |

### 训练管控

| 命令 | 说明 |
|------|------|
| `python scripts/train_manager.py` | 启动管控服务端 (:7001) |
| `python scripts/train_cli.py status` | 查看训练状态 |
| `python scripts/train_cli.py start --task <id>` | 远程启动训练 |
| `python scripts/train_cli.py halt / resume` | 暂停/恢复训练 |
| `python scripts/train_tui.py --host <ip>` | btop 风格 TUI |

### 资产管理

| 命令 | 说明 |
|------|------|
| `python scripts/asset_cli.py register <type> <name:ver> --source <path>` | 注册资产 |
| `python scripts/asset_cli.py list [--type <type>]` | 列出所有资产 |
| `python scripts/asset_cli.py pack <yaml>` | 打包 .myrlpkg |
| `python scripts/asset_cli.py deploy <pkg> <dest>` | 部署到远程 |
| `python scripts/asset_tui.py` | 资产管理 TUI |

### 日志 & 诊断

| 命令 | 说明 |
|------|------|
| `python scripts/log_viewer.py --host <ip>` | SSE 日志流查看 |
| `python scripts/reward_inspector.py --host <ip>` | 实时 reward 分解 |
| `python scripts/registry_cli.py list` | 查看运行记录 |
| `python scripts/registry_cli.py verify <run_id>` | 校验资产完整性 |

### 仿真服务

| 命令 | 说明 |
|------|------|
| `python scripts/start_mujoco_server.py --task dummy` | MuJoCo 仿真服务 |
| `python scripts/start_ros2_bridge.py --task_id <id>` | ROS2 桥接服务 |

### 部署

| 命令 | 说明 |
|------|------|
| `bash scripts/bootstrap_train.sh` | GPU 服务器自举 |
| `bash scripts/bootstrap_dev.sh` | 开发环境自举 |
| `bash scripts/run_dev.sh` | Docker 开发环境启动 |
