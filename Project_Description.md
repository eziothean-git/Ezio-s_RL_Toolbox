# 框架需求文档

> 本文档用于定义一个从零开始设计的机器人强化学习（RL）训练与开发框架需求说明。该框架以 **rslrl + leggedgym** 为核心训练算法基础，但在架构、工具链、部署方式与开发体验上进行系统性重构。
>
> 设计目标不是“更快做完一个项目”，而是 **长期可维护、可迁移、可调试、可复现**

---

## 1. 总体目标与设计原则

### 1.1 总体目标

* 从零构建一套 **可长期演化的机器人 RL 框架**
* 支持 **Gym / Isaac Lab / MuJoCo** 等多仿真后端的平滑切换
* 显著降低 sim2sim、env2env、robot2robot 的迁移成本
* 提供工程级调试、可视化与观测能力，而非“只能看 reward 曲线”
* 支持 **一键部署到租用服务器**（无 sudo / 无 Docker / kernel 权限不全）
* 训练与开发环境严格解耦，保证稳定性与可复现性

### 1.2 核心设计原则

1. **明确边界**：

   * 算法、任务、仿真、资产、工具链必须解耦
   * 允许脏，但脏必须集中在边界层（compat layer）

2. **强约束 API**：

   * 上层逻辑禁止直接操作底层 tensor / root state
   * 通过少量、稳定、语义明确的接口完成交互

3. **可观测优先**：

   * 一切 reward / obs / 接触 / 状态变化必须可被检查与记录

4. **可复现优先于便利**：

   * 训练只认 manifest
   * 资产、权重、配置必须内容可寻址

5. **最小假设部署环境**：

   * train-time 不依赖 Docker、不依赖 GUI、不依赖 sudo
   * code-time 才使用 Docker 与重型工具链

---

## 2. 总体架构分层

框架采用明确的分层设计，自下而上为：

```
Algo Layer        （算法层 / rslrl）
----------------
Task / Env Layer （任务与环境定义）
----------------
Compat Layer     （模型-环境兼容层 / Runtime API）
----------------
Sim Backend      （IsaacGym / IsaacLab / MuJoCo）
```

### 2.1 Sim Backend（仿真后端）

职责：

* 提供最底层的 step / reset / state 读写
* 执行物理仿真、接触求解、力/扭矩施加
* 提供渲染、debug draw、交互能力（若支持）

约束：

* Backend **不允许**直接被任务或算法访问
* 所有差异（坐标系、单位、索引、数据布局）必须在 compat layer 内消化

支持目标：

* Isaac Gym（训练主后端）
* Isaac Lab（开发 / 强交互调试）
* MuJoCo（快速开发 / sim2sim 对照）

### 2.2 Compat Layer（模型-环境兼容层 / Runtime API）

这是整个框架的**核心层**。

职责：

* 在模型（算法/任务）与仿真后端之间提供统一接口
* 封装所有 backend 差异
* 对上层暴露 **强类型 View + 原子操作 API**

设计目标：

* 初期实现可以很简陋
* 但 **必须存在，且必须被强制使用**

#### 2.2.1 统一的 View 抽象

Compat layer 至少应提供以下 View：

* `RobotHandle`：机器人实例（支持多环境）
* `JointView`：

  * position / velocity / torque
  * limits / gear ratio / actuator model
* `BodyView`：

  * pose / velocity / external wrench
* `ContactView`：

  * 接触点、法向、摩擦、接触对、力/impulse
* `SensorView`：

  * IMU、力传感器、触觉、视觉（可选）
* `DebugDraw`：

  * 点 / 线 / 箭头 / 文本
* `Interactor`：

  * 鼠标拾取、施加外力、拖拽（主要用于 Lab）

View 内部可以持有 tensor，但对上层：

* 禁止直接访问底层仿真 tensor
* 只允许通过受控方法读写

#### 2.2.2 原子操作 API

Compat layer 对上层仅暴露少量原子操作，例如：

* `reset(env_ids)`
* `step(actions)`
* `set_joint_targets(pos / vel / torque)`
* `apply_body_wrench(body_id, force, torque, env_ids)`
* `get_contacts(filter=...)`
* `get_time() / get_dt()`
* `set_domain_randomization(params)`

> 上层逻辑 **禁止** 调用 backend 的任何“后门接口”。

### 2.3 Task / Env Layer（任务与环境）

职责：

* 定义观测、动作、奖励、终止条件、reset 策略
* 定义 curriculum / domain randomization

约束：

* 只能通过 compat layer 与仿真交互
* 不允许直接读写 backend tensor

推荐组件：

* `ObsBuilder`
* `RewardBuilder`
* `Termination`
* `Curriculum`

### 2.4 Algo Layer（算法层）

职责：

* 提供 RL 算法实现（PPO / AMP / teacher-student 等）
* 通过 adapter 与 Task/Runtime 对接

约束：

* 算法层 **完全不知道** 使用的是哪种仿真后端
* 只接收标准化的 `obs, action, reward, done, info`

---

## 3. Debug 与交互工具系统

### 3.1 模块化 Debug 插件机制

Debug 工具以 **插件（Resource）** 形式存在，可按需挂载。

每个插件应支持以下生命周期钩子：

* `on_reset(sim, env_ids)`
* `on_pre_step(sim)`
* `on_post_step(sim, obs, info)`
* `on_render(sim)`（若 backend 支持）
* `on_keyboard(sim, key)` / `on_mouse(sim, event)`

### 3.2 典型 Debug 插件需求

* 接触点 / 接触力可视化
* 关节状态-时间绘图器（Joint Scope）
* Reward 分解与实时检查
* 鼠标/热键施加外力扰动
* NaN / 数值异常 / 梯度爆炸检测器

### 3.3 Backend 能力降级

* Isaac Lab：完整 3D 可视化 + 交互
* Isaac Gym：简化 viewer / 仅日志
* MuJoCo：按能力提供

插件 API 不变，backend 能力不足时自动降级。

---

## 4. 资产系统（Assets as Game Content）

资产以“游戏资源”的方式管理，与算法与环境逻辑解耦。

### 4.1 资产内容

* 机器人模型：URDF / USD / Mesh
* 执行器定义（actuators.yaml）
* 传感器定义（sensors.yaml）
* 运动学结构（kinematics.yaml）
* 默认姿态（default_pose.yaml）
* 地形资源（heightfield / mesh / usd）

### 4.2 资产设计原则

* 换机器人 = 换 assets 目录
* 资产版本必须可追踪、可校验
* 所有 ID / feet / mirror map 在 kinematics.yaml 中统一定义

---

## 5. Reward / Observation 构建规范

### 5.1 ObsBuilder

* 每个观测项需显式注册
* 支持 normalize / clip / noise / history
* 自动生成维度与统计信息

### 5.2 RewardBuilder

`RewardBuilder` 管理多个 reward term，计算加权和并记录 breakdown。

```python
builder = RewardBuilder()
builder.add(“vel_track”, func, weight=1.5)          # 直接添加函数
builder.add_from_lib(“vel_track”, weight=1.5,       # 从 RewardLibrary 添加
                     lib_name=”track_lin_vel_xy_exp”,
                     std=0.3)
builder.add_transform(RunningNormalize(...))         # 追加后处理算子
total, per_term = builder.compute(env, step=1000)   # step 用于课程调度
```

**compute 流水线**（按顺序）：
1. 所有 term 无条件求值（含 inactive，供 transform 感知）
2. 顺序执行 `_transforms` 列表（每个算子接收并返回 `per_term, weights`）
3. 对 active term 加权求和

**Checkpoint 支持**：`builder.state_dict()` / `load_state_dict()` 持久化有状态算子

### 5.3 RewardLibrary — 奖励函数资产化

奖励函数以”可发现、可验证、可导出 Schema”的形式管理。

#### 注册约定

每个 term = **Pydantic Params 类 + `@reward_fn` 装饰的函数**：

```python
class TrackLinVelXYExpParams(BaseModel):
    std: float = Field(0.25, ge=0.01, le=5.0,
                       json_schema_extra={“unit”: “m/s”})
    command_name: str = Field(“base_velocity”)

@reward_fn(
    description=”指数核跟踪水平线速度命令”,
    tags=[“locomotion”, “command_tracking”, “dense”],
    params=TrackLinVelXYExpParams,
    version=”1.0.0”,
    author=”ezio”,
    added_in=”2026-03-04”,
)
def track_lin_vel_xy_exp(robot: RobotHandle, params: TrackLinVelXYExpParams) -> Tensor:
    ...
```

函数签名约定：`(robot: RobotHandle, params: ParamsType) -> Tensor[num_envs]`

`RobotHandle` 仅用于 `TYPE_CHECKING`，注册期间不触发 isaaclab 导入。

#### 查询与实例化

```python
from myrl.core.task.reward_lib import get_reward_library
lib = get_reward_library()

lib.list_names()                             # 所有已注册 term 名
lib.list_by_tag(“locomotion”)               # 按 tag 筛选
meta = lib.get(“track_lin_vel_xy_exp”)
meta.params_json_schema()                   # 标准 JSON Schema（供前端消费）
meta.to_dict()                              # 完整元数据字典

func = lib.build(“track_lin_vel_xy_exp”, std=0.3)  # Pydantic 验证 + make_rew 包装
lib.export_yaml(“myrl/assets/reward_schemas/rewards_latest.yaml”)
```

#### 内置 term（`tasks/locomotion/mdp/rewards/`）

| term | 文件 | 说明 |
|------|------|------|
| `track_lin_vel_xy_exp` | locomotion.py | 指数核水平速度跟踪 |
| `track_ang_vel_z_exp` | locomotion.py | 指数核偏航角速度跟踪 |
| `feet_air_time_biped` | locomotion.py | 双足最小空中时间 |
| `penalize_joint_torque_l2` | regularization.py | 关节力矩 L2 惩罚 |
| `penalize_lin_accel` | regularization.py | 根体线加速度惩罚 |
| `penalize_orientation` | regularization.py | Roll/Pitch 姿态偏差惩罚 |

### 5.4 TransformLibrary — 奖励后处理算子资产化

算子以相同资产化机制管理，`@transform_fn` 装饰 `RewardTransform` 子类。

#### 内置算子

| 算子 | 注册名 | 说明 |
|------|--------|------|
| `RunningNormalize` | `running_normalize` | 按 term 运行期标准差归一化（Welford，有状态） |
| `RelativeRebalance` | `relative_rebalance` | EMA 追踪贡献占比，按目标比例调整权重 |
| `ClipReward` | `clip_reward` | 裁剪 ± EMA 平滑 |
| `WeightSchedule` | `weight_schedule` | 线性/余弦插值课程调度 |

#### `RewardTransform` ABC 接口

```python
class RewardTransform(ABC):
    def apply(self, per_term, weights, step) -> (per_term, weights): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d): ...
```

算子分两类：只改 `per_term`（归一化），或只改 `weights`（调度/重平衡）。

目标：防止 reward 尺度失控，防止”随手加一项 reward 就把 PPO 炸掉”。

---

## 6. 训练信息展示与远程控制

### 6.1 日志系统

三层日志体系：

1. wandb（曲线、视频、artifact）
2. 本地结构化日志（jsonl / parquet）
3. 远程 TUI（实时状态）

### 6.2 远程 TUI 控制需求

* pause / resume
* save checkpoint
* 开关 debug 插件
* 动态调整 reward 权重（热更新）
* 导出当前 env snapshot

实现方式：

* 训练进程内 RPC 服务（zmq / grpc / websocket）
* TUI 客户端独立运行

---

## 7. Train-time / Code-time 双模式设计

### 7.1 Train-time（训练发行版）

目标：**最大稳定性，最小依赖假设**

特点：

* 不依赖 Docker
* 不依赖 sudo
* 不依赖 GUI
* 使用 bash + Python 环境管理

部署方式：

* `bootstrap_train.sh`
* micromamba / venv
* 固定依赖版本

仅包含：

* rslrl
* compat layer
* IsaacGym backend
* 最小 debug + wandb

### 7.2 Code-time（开发发行版）

目标：**最大开发效率与交互能力**

特点：

* 使用 Docker
* 集成 Isaac Lab + MuJoCo
* 完整 debug / UI / 交互工具

用途：

* 本地开发
* 调奖励、调接触、调资产

---

## 8. 一键部署与资源中转（Registry）

### 8.1 Registry 职责

* 统一管理所有训练所需资源：

  * 资产
  * terrain
  * pretrained 权重
  * 数据集
  * 配置快照

### 8.2 Registry 特性

* 内容寻址（name + version + sha256）
* 本地 cache（可指定路径）
* 支持镜像源切换
* 支持断点续传

### 8.3 部署流程（Train-time）

1. 启动 bootstrap 脚本
2. 创建/激活 Python 环境
3. 拉取 artifacts 到 cache
4. 生成 Experiment Manifest
5. 启动训练

---

## 9. Experiment Manifest（可复现核心）

训练必须基于 manifest 运行。

Manifest 至少包含：

* 代码版本（git commit）
* 完整展开后的配置
* 所有 artifacts 的版本与 sha256
* 依赖版本（torch / cuda / backend）
* 随机种子

Resume / Reproduce 时必须校验 manifest。

---

## 10. 非目标（当前阶段不强制）

* 自动多卡调度
* 云原生编排
* GUI 编辑器
* 自动超参搜索

---

> 本文档定义的是 **框架级需求与结构边界**。

数据结构：
myrl/
  README.md
  LICENSE
  .gitignore
  pyproject.toml
  uv.lock                     # 若你用 uv（可选）
  requirements/
    train.txt                  # 训练子集（无lab/mujoco）
    code.txt                   # 开发全集（含lab/mujoco）
    dev.txt                    # lint/format/test
  env/
    train.yml                  # micromamba/conda 环境（训练端）
    code.yml                   # 本机conda
  docker/
    Dockerfile.code            # code-time：IsaacLab + Mujoco + tools
    docker-compose.yml         # code-time：挂载workspace/cache
  scripts/
    bootstrap_train.sh         # 服务器一键部署（无sudo/无docker）
    bootstrap_code.sh          # 可选：本地裸机开发部署（不建议含lab）
    myrl                       # 小CLI封装（bash调用python）
  configs/
    experiments/
      humanoid_walk.yaml       # “配方”(exp)，启动时生成manifest
    tasks/
      humanoid_walk.yaml
    sim/
      isaacgym.yaml
      isaaclab.yaml
      mujoco.yaml
    debug/
      default.yaml
  assets/
    robots/
      humanoid_x/
        robot.urdf
        meshes/
        actuators.yaml
        sensors.yaml
        kinematics.yaml
        default_pose.yaml
    terrains/
      plane/
        plane.usd | plane.obj | heightfield.npy
    reward_schemas/                  # 自动生成，不手写
      rewards_latest.yaml            # RewardLibrary JSON Schema 导出
      transforms_latest.yaml         # TransformLibrary JSON Schema 导出
  registry/
    manifests/
      humanoid_x_assets@0.1.0.json
      plane_terrain@0.1.0.json
    checksums/
      humanoid_x_assets@0.1.0.sha256
  runs/                        # 默认输出（可在config中改到大盘）
  src/
    myrl/
      __init__.py
      cli/
        __init__.py
        main.py                # python -m myrl ...
      core/
        compat/
          __init__.py
          runtime.py           # create_runtime(profile, backend, ...)
          views/
            __init__.py
            joints.py
            bodies.py
            contacts.py
            sensors.py
          backends/
            __init__.py
            base.py
            isaacgym_backend.py
            isaaclab_backend.py
            mujoco_backend.py
        task/
          __init__.py
          base_task.py
          obs_builder.py
          reward_builder.py          # add/add_from_lib/add_transform/compute(step)
          termination.py
          curriculum.py
          reward_lib/
            __init__.py              # @reward_fn / @transform_fn / get_*_library()
            meta.py                  # RewardTermMeta + TransformMeta
            library.py               # RewardLibrary + TransformLibrary（单例）
            transform.py             # RewardTransform ABC + 4 内置算子
            adapters.py              # instinctlab RewTerm 适配器
        algo/
          __init__.py
          rslrl_adapter.py     # 适配rslrl的VecEnv接口
      debug_tools/
        __init__.py
        plugins/
          __init__.py
          contact_viz.py
          joint_scope.py
          disturbance_gun.py
          reward_inspector.py
          sanity_checker.py
      tools/
        __init__.py
        pull.py                # registry拉取与cache管理
        manifest.py            # 生成/校验experiment manifest
        snapshot.py            # 导出env快照（可选）
      entrypoints/
        train.py
        dev.py
        play.py
     isaac_gym/
        IsaacGym_Preview_4_Package.tar.gz
