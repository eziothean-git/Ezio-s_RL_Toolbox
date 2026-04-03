# myrl MVP 进度追踪

> **更新日期**：2026-04-04
>
> 按自然生长链的顺序追踪每一层的 MVP 完成度。
> MVP 定义：**能在 Isaac Lab 容器内端到端跑通 G1 行走训练，过程中可通过示波器实时检视任意信号，奖励管线通过 YAML 声明式定义。**

---

## 总览

```
██████████████████████░░  91% MVP 完成

已完成  : 39 / 43 项
未开始  :  4 / 43 项
```

**最新验证**：2026-04-04 G1Smoke 全要素端到端测试通过（Isaac Lab 2.3.2 容器，RTX 5060，5 iter / 4 envs / 480 timesteps，EXIT_CODE:0）

---

## 1. Compat 层（多后端适配器）

> 屏蔽仿真后端差异，让 Task/Algo 层无需关心后端。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 1.1 | SimBackend ABC | `core/compat/backends/base.py` | ✅ 完成 | 57L，合约定义 |
| 1.2 | IsaacLabBackend | `core/compat/backends/isaaclab_backend.py` | ✅ 完成 | 250L |
| 1.3 | MuJoCoBackend | `core/compat/backends/mujoco_backend.py` | ✅ 完成 | 185L，TCP 客户端 |
| 1.4 | Phase B 验证 | — | ✅ 完成 | G1Smoke 5 iter 端到端通过（2026-04-04） |
| 1.5 | RobotHandle | `core/compat/views/robot.py` | ✅ 完成 | 116L，含 make_term/make_rew |
| 1.6 | JointView | `core/compat/views/joints.py` | ✅ 完成 | 55L + DataBus 集成 |
| 1.7 | BodyView | `core/compat/views/bodies.py` | ✅ 完成 | 58L + DataBus 集成 |
| 1.8 | ContactView | `core/compat/views/contacts.py` | ✅ 完成 | 61L + DataBus 集成 |
| 1.9 | SensorView + ImuView | `core/compat/views/sensor.py` | ✅ 完成 | 36L + DataBus 集成 |

**本层完成度：9/9 (100%)**

---

## 2. 资产系统

> obs/reward 管线、传感器、执行器、URDF、脚本……都作为 asset 管理。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 2.1 | AssetStore | `assets/asset_store.py` | ✅ 完成 | 274L，10 种 AssetType |
| 2.2 | 资产解析 API | `assets/__init__.py` | ✅ 完成 | has/resolve/require_asset |
| 2.3 | G1 机器人资产 | `assets/robots/g1/` | ✅ 完成 | URDF + STL + MuJoCo XML |
| 2.4 | 执行器配置 | `assets/actuator_cfgs/g1_default.yaml` | ✅ 完成 | PD 增益 + 力矩限制 |
| 2.5 | 传感器配置 | `assets/sensor_cfgs/g1_contact.yaml` | ✅ 完成 | 接触 + IMU |
| 2.6 | 算法配置 | `assets/algo_cfgs/ppo_standard.yaml` | ✅ 完成 | PPO 全参数 |
| 2.7 | 奖励管线 | `assets/reward_pipelines/g1_loco_v1.yaml` | ✅ 完成 | 6 term + weights |
| 2.8 | 观测管线 | `assets/obs_pipelines/g1_standard_obs.yaml` | ✅ 完成 | 6 obs term |
| 2.9 | asset_cli | `scripts/asset_cli.py` | ✅ 完成 | 323L |
| 2.10 | asset_tui | `scripts/asset_tui.py` | ✅ 完成 | 447L |

**本层完成度：10/10 (100%)**

---

## 3. 实验合成 + 打包

> 复用 asset 组合实验，打包分发。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 3.1 | ExperimentComposer | `assets/composer.py` | ✅ 完成 | 280L，7 步流水线（含 obs pipeline） |
| 3.2 | Packager | `assets/packager.py` | ✅ 完成 | 480L，含 get_obs_pipeline_path |
| 3.3 | 实验配置示例 | `assets/experiments/g1_locomotion_v1.yaml` | ✅ 完成 | 引用 8 种资产 |
| 3.4 | **Composer 端到端验证** | — | ❌ 未完成 | `--package xxx.myrlpkg` 全流程（需容器内测试） |

**本层完成度：3/4 (75%)**

---

## 4. 分布式训练 + 管控

> 远程服务器上管理多个 experiment。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 4.1 | train.py | `scripts/train.py` | ✅ 完成 | 354L，Phase A/B 切换 |
| 4.2 | play.py | `scripts/play.py` | ✅ 完成 | 175L |
| 4.3 | train_manager | `scripts/train_manager.py` | ✅ 完成 | 578L，HTTP + SSE |
| 4.4 | train_cli | `scripts/train_cli.py` | ✅ 完成 | 208L |
| 4.5 | train_tui | `scripts/train_tui.py` | ✅ 完成 | 362L |
| 4.6 | Docker 环境 | `docker/*` | ✅ 完成 | Dockerfile + compose + entrypoint |
| 4.7 | bootstrap_train | `scripts/bootstrap_train.sh` | ✅ 完成 | 裸机自举 |
| 4.8 | bootstrap_dev | `scripts/bootstrap_dev.sh` + `run_dev.sh` | ✅ 完成 | Docker 自举 |

**本层完成度：8/8 (100%)**

---

## 5. Evaluate + Deploy（MuJoCo / ROS2 / 实机）

> 训练完成后的评估和部署通路。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 5.1 | SimProto | `core/sim_server/protocol.py` | ✅ 完成 | TCP 帧协议 |
| 5.2 | SimServer ABC | `core/sim_server/base_server.py` | ✅ 完成 | 135L |
| 5.3 | MuJoCoTask ABC | `core/sim_server/mujoco_task.py` | ✅ 完成 | ABC + DummyTask |
| 5.4 | MuJoCoSimServer | `core/sim_server/mujoco_server.py` | ✅ 完成 | 向量化仿真 |
| 5.5 | ROS2 Bridge | `core/sim_server/ros2_bridge.py` | ✅ 完成 | 395L，TCP↔ROS |
| 5.6 | play_mujoco | `scripts/play_mujoco.py` | ✅ 完成 | 117L |
| 5.7 | start_mujoco_server | `scripts/start_mujoco_server.py` | ✅ 完成 | 174L |
| 5.8 | start_ros2_bridge | `scripts/start_ros2_bridge.py` | ✅ 完成 | — |

**本层完成度：8/8 (100%)**

---

## 6. 调试 + 可观测性（DataBus + 示波器）

> 像玩 Besiege 一样直接看到、摸到仿真中的一切。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 6.1 | DataBus 核心 | `core/databus/bus.py` | ✅ 完成 | 130L，19/19 测试通过 |
| 6.2 | Channel + Tap | `core/databus/channel.py` + `tap.py` | ✅ 完成 | 70L + 115L，支持 bool/float/int/uint8/image |
| 6.3 | View→DataBus 集成 | 修改 `views/*.py` | ✅ 完成 | joints/bodies/contacts/sensor/robot 全部集成 |
| 6.4 | RewardBuilder/ObsBuilder/Backend→DataBus | 修改 3 个文件 | ✅ 完成 | reward per-term + obs per-group + action/dones |
| 6.5 | **Oscilloscope 主类** | `debug_tools/oscilloscope/scope.py` | ❌ 未开始 | 管理 Tap 集合 |
| 6.6 | **Inspector 面板** | `debug_tools/oscilloscope/inspector.py` | ❌ 未开始 | 选中体状态数值表 |
| 6.7 | **信号波形** | `debug_tools/oscilloscope/signal_view.py` | ❌ 未开始 | 实时多通道叠加 |
| 6.8 | **3D overlay** | `debug_tools/oscilloscope/...` | ❌ 未开始 | 力箭头 + 关节轴标注（P3） |

**本层完成度：4/8 (50%)**

---

## 7. 奖励/观测管线（YAML 声明式）

> 奖励项像资产一样组合，归一化可控可视。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 7.1 | RewardBuilder | `core/task/reward_builder.py` | ✅ 完成 | 156L，含 transform 流水线 + DataBus |
| 7.2 | RewardLibrary | `core/task/reward_lib/library.py` | ✅ 完成 | 146L，单例注册表 |
| 7.3 | @reward_fn 装饰器 | `core/task/reward_lib/__init__.py` | ✅ 完成 | 元数据注册 |
| 7.4 | 4 内置 Transform | `core/task/reward_lib/transform.py` | ✅ 完成 | 339L |
| 7.5 | 内置 reward terms | `tasks/locomotion/mdp/rewards/` | ✅ 完成 | 6 个 term |
| 7.6 | ObsBuilder | `core/task/obs_builder.py` | ✅ 完成 | 54L + DataBus |
| 7.7 | ObsHistoryManager | `core/obs/history_manager.py` | ✅ 完成 | 160L |
| 7.8 | obs_pipeline YAML→Isaac Lab ObsCfg | `composer.py` + `env_script.py` | ✅ 完成 | YAML→ObservationsCfg 动态生成 |
| 7.9 | reward_pipeline YAML→RewardBuilder | `composer.py` | ✅ 完成 | Composer step 5a |

**本层完成度：9/9 (100%)**

---

## 8. 日志 + 注册表

> 训练过程可观测 + 实验可复现。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 8.1 | LogSink ABC | `logging/sinks/base.py` | ✅ 完成 | 40L |
| 8.2 | JSONLSink | `logging/sinks/jsonl_sink.py` | ✅ 完成 | 59L |
| 8.3 | WandbSink | `logging/sinks/wandb_sink.py` | ✅ 完成 | 46L |
| 8.4 | SSELogServer | `logging/server/log_server.py` | ✅ 完成 | 176L |
| 8.5 | SSEClient | `logging/server/log_client.py` | ✅ 完成 | 218L |
| 8.6 | RunRegistry | `registry/registry.py` | ✅ 完成 | 113L |
| 8.7 | RunManifest | `registry/manifest.py` | ✅ 完成 | 180L |
| 8.8 | log_viewer | `scripts/log_viewer.py` | ✅ 完成 | 118L |
| 8.9 | reward_inspector | `scripts/reward_inspector.py` | ✅ 完成 | 141L |
| 8.10 | registry_cli | `scripts/registry_cli.py` | ✅ 完成 | 131L |

**本层完成度：10/10 (100%)**

---

## 9. 任务定义

> 至少一个可端到端运行的 myrl 原生任务。

| # | 模块 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| 9.1 | G1 Smoke 任务 | `tasks/locomotion/config/g1_smoke/` | ✅ 完成 | 端到端验证通过（2026-04-04） |
| 9.2 | G1 Native 任务 | `tasks/locomotion/config/g1_native/` | ✅ 完成 | env_script: reward+obs+actuator+sensor 全部集成 |
| 9.3 | **Packaged 任务端到端** | — | ❌ 未完成 | g1_locomotion_v1.yaml → .myrlpkg → train（需容器内测试） |

**本层完成度：2/3 (67%)**

---

## 验证记录

| 日期 | 测试 | 环境 | 结果 |
|------|------|------|------|
| 2026-03-04 | Phase A G1Smoke 5 iter | Isaac Lab 容器 | ✅ EXIT_CODE:0 |
| 2026-03-04 | MuJoCo TCP 协议 | 宿主机 Python | ✅ 23/23 |
| 2026-03-04 | ObsHistoryManager | 宿主机 Python | ✅ 32/32 |
| 2026-04-04 | DataBus 核心 + 多类型 | 宿主机 Python | ✅ 19/19 |
| **2026-04-04** | **G1Smoke 全要素 e2e** | **Isaac Lab 2.3.2, RTX 5060** | **✅ 5 iter, 480 ts, 0.67s/iter** |
| 2026-04-04 | SignalServer HTTP/SSE | 宿主机 Python | ✅ 4/4 endpoints |
| **2026-04-04** | **GUI viewport 渲染** | **Isaac Lab 2.3.2, RTX 5060, Wayland** | **✅ OpenGL 后端，viewport 正常** |
| 2026-04-04 | Git push 28eeae2 | GitHub | ✅ master pushed |

---

## MVP 剩余项

| 优先级 | 项目 | 来源 | 工作量 | 说明 |
|--------|------|------|--------|------|
| **P1** | Oscilloscope v1 | 6.5-6.7 | ~500L | Inspector 面板 + 信号波形（DataBus 已就绪） |
| **P2** | Composer 端到端 | 3.4 | 容器内测试 | pack → --package → train |
| **P2** | Packaged 任务端到端 | 9.3 | 容器内测试 | 依赖 3.4 |
| **P3** | 3D overlay | 6.8 | ~400L | 力箭头/关节轴（omni.ui，MVP 后） |

**关键路径**：Oscilloscope v1（6.5-6.7）是 MVP 最后的核心功能。Composer 端到端（3.4+9.3）是容器内集成测试，不需要新代码。

---

## MVP 不需要的（明确排除）

| 模块 | 原因 |
|------|------|
| MJX Backend | 大规模训练后端，MVP 用 Isaac Lab 够了 |
| core/algo/ 自定义网络 | 算法定型前不做 |
| cli/ 统一入口 | scripts/ 散装足够 |
| entrypoints/ | pyproject.toml entry_points，非必要 |
| 鼠标拾取/施力/关节滑块 | 示波器 v2 功能 |
| TUI 管线编辑器 | YAML 编辑足够 |
| humanoid_x 新机器人 | G1 够验证 MVP |

---

## QOL 并行小项目（MVP 后持续改进）

> 这些不阻塞 MVP，但显著提升日常开发体验。可与 MVP 剩余项并行推进。

### Q1. 一键启动开发体验

> 目标：新开发者 clone 后一条命令进入完整开发环境。

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| Q1.1 | `make dev` 入口 | ❌ | Makefile 统一 bootstrap + compose + shell |
| Q1.2 | 首次启动向导 | ❌ | 检测缺失依赖，交互式引导（EULA、GPU 检查） |
| Q1.3 | devcontainer.json | ❌ | VS Code Remote Containers 一键打开 |
| Q1.4 | 容器内热重载 | ❌ | 代码变更后无需重建容器（当前已通过 volume mount 实现，需文档化） |

### Q2. 自动化 CI/CD 管线

> 目标：PR 合入前自动验证，训练产物自动归档。

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| Q2.1 | GitHub Actions 语法检查 | ❌ | py_compile + ruff lint 对所有 .py |
| Q2.2 | 单元测试 CI | ❌ | DataBus/ObsHistory/RewardBuilder 测试（无需 GPU） |
| Q2.3 | Docker 镜像缓存 | ❌ | ghcr.io 镜像 push，避免每次 build |
| Q2.4 | 容器内冒烟测试 | ❌ | GPU runner 或 self-hosted：5 iter G1Smoke |
| Q2.5 | .myrlpkg 产物归档 | ❌ | 训练完成后 artifact upload 到 release/S3 |

### Q3. Agent / Vibe 开发友好化

> 目标：让 AI Agent（Claude Code / Cursor / Copilot）和快速迭代开发更高效。

| # | 项目 | 状态 | 说明 |
|---|------|------|------|
| Q3.1 | CLAUDE.md 精简 & 聚焦 | ❌ | 当前 600+ 行，需拆分为核心 + 详细参考 |
| Q3.2 | 模块级 README | ❌ | 每个核心模块目录放简短 README（Agent 上下文友好） |
| Q3.3 | 类型标注补全 | ❌ | 公开 API 补全 type hints（Agent 推理更准确） |
| Q3.4 | 示例脚本集 | ❌ | `examples/` 目录：最小训练/打包/信号查看示例 |
| Q3.5 | `.claude/` hooks | ❌ | pre-commit lint、测试自动运行 |
| Q3.6 | 错误消息改进 | ❌ | 常见错误（EULA、GPU、import）给出明确修复建议 |
