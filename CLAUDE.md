# CLAUDE.md - Ezio's RL Toolbox

## Project Overview

A from-scratch robot reinforcement learning framework (`myrl`) built on top of
**instinct_rl** (algorithm layer) and **instinctlab** (environment/task layer),
targeting legged robot whole-body control with Isaac Sim as the primary sim backend.

**Language**: Chinese for docs/comments, English for code identifiers and commit messages.

## Architecture (4-Layer Design)

```
Algo Layer        instinct_rl (PPO / AMP / TPPO / VAE / MoE)
─────────────────────────────────────────────────────────────
Task / Env Layer  instinctlab (InstinctRlEnv + gymnasium tasks)
─────────────────────────────────────────────────────────────
Compat Layer      myrl/core/compat/ (Views + Runtime API)    ← myrl core innovation
─────────────────────────────────────────────────────────────
Sim Backend       Isaac Gym (train) | Isaac Lab (dev) | MuJoCo (sim2sim)
```

### Core Design Principles
1. **Strict boundary enforcement**: algo/task/sim must be decoupled; cross-layer access goes through compat layer only
2. **Strong-typed View API**: RobotHandle, JointView, BodyView, ContactView, SensorView - no direct tensor access
3. **Observability first**: all reward/obs/contact/state changes must be inspectable and recordable
4. **Reproducibility over convenience**: training runs on manifests; assets/weights/configs are content-addressable
5. **Minimal deployment assumptions**: train-time needs no Docker, no sudo, no GUI

---

## Upstream Frameworks — Detailed API Reference

### instinct_rl

Local clone: `/home/eziothean/instinct_rl/`

#### Observation Format System

The key abstraction — an `OrderedDict`-based 3-level hierarchy:

```python
# obs_format: describes the *structure* of observations
obs_format: dict[str, dict[str, tuple]] = {
    "policy": {
        "base_ang_vel": (24,),   # 8 timesteps * 3 dims
        "joint_pos":    (87,),   # 29 joints * 3
        "depth_image":  (2304,), # 64*36 flattened
    },
    "critic": {                  # optional privileged obs
        "motion_reference": (50,),
    },
    # any custom group for AMP, estimator, etc.
    "amp_policy":    {"state_vec": (128,)},
    "amp_reference": {"state_vec": (128,)},
}

# obs_pack: actual tensors, keyed same as obs_format
obs_pack: dict[str, Tensor] = {
    "policy": Tensor[num_envs, sum_of_policy_dims],
    "critic": Tensor[num_envs, sum_of_critic_dims],
}
```

Key utility functions (`instinct_rl.utils`):
```python
get_subobs_size(obs_segments: OrderedDict) -> int
get_obs_slice(segments, component_name) -> (slice, shape)
get_subobs_by_components(observations, component_names, obs_segments, cat=True)
```

#### VecEnv Abstract Interface (`instinct_rl.env.vec_env.VecEnv`)

```python
# Properties
env.num_envs: int
env.num_actions: int
env.num_rewards: int          # 1 for standard, >1 for multi-critic
env.max_episode_length: int | Tensor
env.device: torch.device

# Methods
env.get_obs_format() -> dict[str, dict[str, tuple]]
env.get_observations() -> tuple[Tensor, dict]
    # returns: (policy_obs[num_envs, obs_size], extras)
env.step(actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]
    # returns: (obs, rewards[num_envs, num_rewards], dones[num_envs], extras)
env.reset() -> tuple[Tensor, dict]
```

`extras` dict contract:
```python
extras = {
    "observations": {           # all obs groups as flat tensors
        "policy":   Tensor[num_envs, policy_size],
        "critic":   Tensor[num_envs, critic_size],  # optional
        # custom groups for AMP/estimator...
    },
    "time_outs": Tensor[num_envs],  # truncation flags (for GAE bootstrapping)
    "log": dict[str, float],         # scalar metrics for logging
}
```

#### PPO Algorithm (`instinct_rl.algorithms.PPO`)

```python
ppo = PPO(
    actor_critic,
    num_learning_epochs=5,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.998,
    lam=0.95,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    entropy_coef=0.0,
    use_clipped_value_loss=True,
    schedule="fixed",           # or "adaptive" (KL-based LR)
    desired_kl=0.01,
    device="cpu",
)

# Core loop methods:
ppo.init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1)
ppo.act(obs, critic_obs) -> Tensor[num_envs, num_actions]
ppo.process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
ppo.compute_returns(last_critic_obs)
ppo.update(current_learning_iteration) -> (loss_dict, stat_dict)
```

#### Algorithm Variants

| Class | Inherits | Adds |
|-------|----------|------|
| `PPO` | - | Vanilla PPO |
| `TPPO` | PPO | Teacher-student distillation (loads teacher checkpoint) |
| `EstimatorPPO` | PPO | State estimation reconstruction loss |
| `WasabiAlgoMixin` | - | AMP discriminator + reward |
| `WasabiPPO` | WasabiMixin, PPO | AMP + PPO combined |
| `WasabiEstimatorPPO` | WasabiMixin, EstimatorPPO | All combined |
| `VAEDistillPPO` | TPPO | + VAE latent encoding |

WASABI/AMP config keys:
```python
alg_cfg = {
    "class_name": "WasabiPPO",
    "actor_state_key": "amp_policy",       # obs group with policy state
    "reference_state_key": "amp_reference", # obs group with expert state
    "discriminator_reward_coef": 0.25,
    "discriminator_reward_type": "quad",   # or "log"
    "discriminator_loss_func": "MSELoss",  # or "BCEWithLogitsLoss"
    "discriminator_gradient_penalty_coef": 5.0,
}
```

#### Actor-Critic Networks (`instinct_rl.modules`)

```python
# Factory (supports "ClassName" or "module.path:ClassName")
actor_critic = modules.build_actor_critic(
    policy_class_name,   # e.g., "ActorCritic", "EncoderMoEActorCritic"
    policy_cfg,          # dict of constructor kwargs
    obs_format,
    num_actions,
    num_rewards,
)
```

Available classes:
- `ActorCritic` — standard MLP
- `ActorCriticRecurrent` — GRU/LSTM
- `MoEActorCritic` — Mixture of Experts
- `EncoderActorCritic` — with CNN/custom encoder
- `EncoderMoEActorCritic` — encoder + MoE (used in parkour tasks)
- `VaeActorCritic` — VAE latent space
- `EstimatorActorCritic` — learns state representations

#### OnPolicyRunner (`instinct_rl.runners.OnPolicyRunner`)

```python
runner = OnPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
runner.learn(num_learning_iterations, init_at_random_ep_len=False)
runner.save(path)
runner.load(path)
```

`train_cfg` dict structure:
```python
train_cfg = {
    "algorithm": {
        "class_name": "WasabiPPO",
        "learning_rate": 1e-3,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        # ... algorithm-specific kwargs
    },
    "policy": {
        "class_name": "EncoderMoEActorCritic",
        "actor_hidden_dims": [256, 128, 64],
        "num_moe_experts": 4,
        "encoder_configs": {...},
    },
    "normalizers": {             # optional per-group normalizers
        "policy": {"class_name": "EmpiricalNormalization"},
        "critic": {"class_name": "EmpiricalNormalization"},
    },
    "num_steps_per_env": 24,
    "max_iterations": 30000,
    "save_interval": 5000,
    "experiment_name": "my_task",
}
```

---

### instinctlab

Local clone: `/home/eziothean/instinctlab/`
Source: `source/instinctlab/instinctlab/`

#### Task Registration Pattern

```python
# file: tasks/mytask/config/__init__.py
import gymnasium as gym

gym.register(
    id="Instinct-MyTask-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        # Isaac Lab environment config
        "env_cfg_entry_point": "my_package.tasks.mytask:MyTaskEnvCfg",
        # instinct_rl runner config
        "instinct_rl_cfg_entry_point": "my_package.tasks.mytask.agents:MyRunnerCfg",
    },
)
# Play variant (simpler terrain, no disturbances)
gym.register(
    id="Instinct-MyTask-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "my_package.tasks.mytask:MyTaskEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": "my_package.tasks.mytask.agents:MyRunnerCfg",
    },
)
```

#### InstinctRlEnv

Extends `ManagerBasedRLEnv` (Isaac Lab) with:
- `MultiRewardManager` — multi-critic reward support
- `MonitorManager` — detailed statistics tracking
- `num_rewards` property

```python
# env returned by gym.make(task_id)
env.num_envs: int
env.num_rewards: int   # 1 or number of reward groups
env.step(action) -> VecEnvStepReturn
```

#### Environment Config Structure

```python
@configclass
class MyTaskEnvCfg(InstinctLabRLEnvCfg):
    # Scene
    scene: SceneCfg = SceneCfg()    # robot, contact sensors, cameras

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()          # or MultiRewardCfg
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    monitors: MonitorsCfg = None                # instinctlab addition

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            history_length=8,
            flatten_history_dim=True,
            noise=GaussianNoiseCfg(std=0.01),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        # For AMP: include these two groups
        amp_policy    = ObsTerm(func=mdp.amp_policy_state)
        amp_reference = ObsTerm(func=mdp.amp_reference_state)

    @configclass
    class CriticCfg(ObsGroup):  # optional privileged obs
        motion_reference = ObsTerm(func=mdp.motion_reference_state)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()   # set to None if not using privileged obs
```

#### Runner Config (instinctlab-side)

```python
# file: tasks/mytask/agents/instinct_rl_cfg.py
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlActorCriticCfg,
    InstinctRlPpoAlgorithmCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlConv2dHeadCfg,
)

@configclass
class MyRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "my_task"

    policy = InstinctRlActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
    )
    algorithm = InstinctRlPpoAlgorithmCfg(
        learning_rate=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
```

#### VecEnv Wrapper (`instinctlab.utils.wrappers.instinct_rl.InstinctRlVecEnvWrapper`)

Adapts `ManagerBasedRLEnv` → `instinct_rl.env.VecEnv`:
- Builds `obs_format` from Isaac Lab `ObservationManager`
- Flattens multi-dim obs groups into flat tensors
- Stacks multi-reward dict → `Tensor[num_envs, num_rewards]`
- Exposes `time_outs` for GAE bootstrapping

#### Multi-Reward Manager

```python
@configclass
class MultiRewardCfg:
    # Each nested @configclass becomes a reward group
    rewards = StandardRewardsCfg()    # group 1
    # aux_rewards = AuxRewardsCfg()  # group 2 (optional)

# Returns Tensor[num_envs, num_groups] instead of [num_envs]
```

#### Training Script (instinctlab reference)

```bash
# From instinctlab repo:
python scripts/instinct_rl/train.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --num_envs=4096 \
    --headless

# Play / eval:
python scripts/instinct_rl/play.py \
    --task=Instinct-Parkour-Target-Amp-G1-Play-v0 \
    --num_envs=16
```

---

## Repository Structure

```
Ezio's_RL_Toolbox/
├── CLAUDE.md                          # This file
├── Project_Description.md             # Full framework requirements (Chinese)
├── Ezio's_RL_Toolbox.code-workspace   # VSCode workspace
└── myrl/
    ├── assets/                        # Robot URDF/USD, terrain meshes (empty, WIP)
    │   ├── robots/humanoid_x/
    │   └── terrains/
    ├── configs/                       # YAML configs (empty, WIP)
    │   ├── debug/
    │   ├── experiments/
    │   ├── sim/
    │   └── tasks/
    ├── docker/                        # Code-time dev environment (DONE)
    │   ├── Dockerfile.dev             # Based on nvcr.io/nvidia/isaac-lab:2.3.0
    │   ├── compose.yaml
    │   ├── entrypoint.sh              # Runtime server plumbing, venv management
    │   └── requirements-dev.txt
    ├── env/
    │   └── train.yml                  # Conda env spec (Python 3.10)
    ├── registry/                      # Content-addressable artifact registry (empty, WIP)
    │   ├── checksums/
    │   └── manifests/
    ├── scripts/
    │   ├── bootstrap_train.sh         # Train-time: micromamba + torch + IsaacGym (DONE)
    │   ├── bootstrap_dev.sh           # Dev-time: Docker + NVIDIA toolkit + Isaac Lab (DONE)
    │   ├── run_dev.sh                 # Compose build + run wrapper (DONE)
    │   └── smoke_env.py               # Validates isaacgym + torch + rsl_rl + legged_gym (DONE)
    ├── src/myrl/                      # Application source code (empty, WIP)
    │   ├── cli/                       # Entry points: train, play, debug
    │   ├── core/
    │   │   ├── compat/                # ← THE CORE: Views + Backends
    │   │   │   ├── backends/          # isaacgym_backend.py, isaaclab_backend.py, mujoco_backend.py
    │   │   │   └── views/             # JointView, BodyView, ContactView, SensorView
    │   │   ├── task/                  # ObsBuilder, RewardBuilder, Termination, Curriculum
    │   │   └── algo/                  # instinct_rl adapter
    │   ├── debug_tools/plugins/       # contact_viz, joint_scope, reward_inspector, ...
    │   ├── tools/                     # registry, manifest, checksum
    │   └── entrypoints/               # train.py, play.py
    ├── third_party/                   # Vendored dependencies (gitignored)
    │   ├── isaac_gym/                 # IsaacGym Preview 4 tarball
    │   ├── isaaclab/                  # Isaac Lab 2.3.0 (git clone)
    │   ├── legged_gym/               # legacy, replaced by instinctlab
    │   └── rsl_rl/                    # legacy, replaced by instinct_rl
    └── work/                          # Runtime workdir (gitignored)
```

Reference repos (analyzed, not vendored):
- `/home/eziothean/instinct_rl/` — upstream instinct_rl
- `/home/eziothean/instinctlab/` — upstream instinctlab

---

## Current Status

### DONE

**基础设施**
- Framework requirements document (Project_Description.md)
- Train-time bootstrap (micromamba, auto GPU detection, PyTorch auto-index, IsaacGym .pth injection)
- Dev-time bootstrap (Docker, NVIDIA Container Toolkit, Isaac Lab clone, EULA management)
- Docker dev environment (Isaac Lab 2.3.2 container, runtime server plumbing)
- Docker GUI (X11/Wayland) support (`network_mode: host`, DISPLAY forwarding)
- Smoke test (validates GPU, torch stack)

**Phase A — 直接复用 instinctlab wrapper，串通训练链路（2026-03-03）**
- `src/myrl/__init__.py` — 包初始化
- `src/myrl/tasks/__init__.py` — `import_packages` 自动触发子任务 `gym.register()`
- `scripts/train.py` — 训练入口，自包含 CLI（含 resume / video / headless / debug 支持）
- `scripts/play.py` — 推理入口（`--load_run` / `--no_resume`）
- `src/pyproject.toml` — 使 `pip install -e` 生效，build-backend 用 `setuptools.build_meta`

**Phase B — myrl 自己的 compat 层（2026-03-03）**
- `src/myrl/core/compat/backends/base.py` — `SimBackend` ABC（IsaacGym/MuJoCo 扩展骨架）
- `src/myrl/core/compat/backends/isaaclab_backend.py` — 完整实现 `VecEnv` ABC，内化 `InstinctRlVecEnvWrapper` 逻辑
- Phase A → B 切换：train.py / play.py 各改一行 import，无其他改动

**Docker 容器环境修复 + 升级（2026-03-03）**
- `Dockerfile.dev`: 升级基础镜像 `isaac-lab:2.3.0` → `2.3.2`（解决 MultiMeshRayCaster / ray_cast_utils 版本不匹配）
- `entrypoint.sh`: `source /isaac-sim/setup_python_env.sh`（必须 source 才能使 torch/toml 可见，不能只加 PATH）
- `entrypoint.sh`: PYTHONPATH 追加方式挂载上游包（不用 `pip install -e`，避免破坏 kit 依赖树）
- `entrypoint.sh`: `git config --global safe.directory` 避免挂载 repo 的 dubious ownership 报错
- `entrypoint.sh`: `install_instinctlab_runtime_deps()` — 把 `pytorch_kinematics` + deps 装到 `$WORK/.site-packages`（持久化 volume，不污染 kit 环境）
- `compose.yaml`: instinct_rl / instinctlab 以 `:rw` 挂载到 `/opt/instinct_rl` / `/opt/instinctlab`

**Phase C — 资产解析器 + myrl 自有任务注册（2026-03-04）**
- `src/myrl/assets/__init__.py` — 资产解析器：`has_asset` / `resolve_asset` / `require_asset`，`MYRL_ASSETS_DIR = parents[3]/assets`
- `src/myrl/tasks/locomotion/` — 任务目录骨架（`__init__.py` + `mdp/` + `config/`）
- `src/myrl/tasks/locomotion/config/g1_smoke/` — G1 冒烟任务：`gym.register` × 2，资产优先级逻辑，继承 `G1FlatEnvCfg`
- `src/myrl/tasks/locomotion/config/g1_smoke/agents/ppo_cfg.py` — `G1SmokePPORunnerCfg`（全字段显式赋值）
- `scripts/train.py` + `play.py` — 追加 `import myrl.tasks`（AppLauncher 后，与 instinctlab.tasks 并列）

**端到端验证状态（2026-03-04）✅ 全部通过**
- Phase A + Phase C（`myrl/Locomotion-Flat-G1Smoke-v0`）：5 iterations，~0.5s/iter，EXIT_CODE:0
- 调试中发现并修复 2 个 bug：
  1. `import_packages` 参数名 `blacklist` → `blacklist_pkgs`
  2. `@configclass` 子类未覆盖父类 `MISSING` 字段 → `to_dict()` 序列化为 `{}` → `dict * Tensor` 崩溃

**Phase D — MuJoCo Socket 后端 + ROS2 双向桥接（2026-03-04）✅ 全部通过**
- `src/myrl/core/sim_server/protocol.py` — `MsgType` + `SimProto`（4字节帧头 + msgpack-numpy）
- `src/myrl/core/sim_server/base_server.py` — `SimServer` ABC（TCP 监听 + 消息分发）
- `src/myrl/core/sim_server/mujoco_task.py` — `MuJoCoTask` ABC + `DummyTask`（无需 MJCF 冒烟测试）
- `src/myrl/core/sim_server/mujoco_server.py` — `MuJoCoSimServer`：N×MjData 向量化仿真 + auto-reset
- `src/myrl/core/sim_server/ros2_bridge.py` — `Ros2Bridge`：rclpy 节点，双向桥接 ROS2 ↔ TCP
- `src/myrl/core/compat/backends/mujoco_backend.py` — `MuJoCoBackend(SimBackend, VecEnv)`：TCP 客户端
- `scripts/start_mujoco_server.py` — 独立启动服务端（`--task dummy` 或自定义任务）
- `scripts/start_ros2_bridge.py` — 启动 ROS2 桥接节点
- `scripts/play_mujoco.py` — 策略推理（支持 `--load_run` 加载 checkpoint）
- Docker 更新：Dockerfile 添加 `ros-humble-ros-base`；entrypoint 追加 `setup_ros2_env()`；
  compose.yaml 新增 `MUJOCO_GL: egl`
- 端到端验证：23/23 测试通过（host Python；MuJoCoBackend VecEnv 需容器内 torch 验证）

**Phase D 架构对齐 + ObsHistoryManager（2026-03-04）**
- `src/myrl/core/obs/history_manager.py` — `ObsHistoryManager`：环境无关，纯 torch ring buffer，term 粒度 history
- `src/myrl/core/sim_server/mujoco_server.py` — 新增 `register_obs_callback()` + `include_obs_in_response` 参数；obs_callback 在 TCP resp 之前调用（ROS 同步保障）
- `src/myrl/core/sim_server/ros2_bridge.py` — 架构反转：ROS topics 为传感器总线（承载 obs），TCP 只承载 rewards/dones；集成 ObsHistoryManager；topic 命名空间改为 `/myrl/{task_id}/...`
- `scripts/start_mujoco_server.py` — 新增 `--ros`, `--task_id`, `--no_obs_in_resp` 参数
- `scripts/start_ros2_bridge.py` — 新增 `--task_id`, `--history_cfg` 参数，移除旧 topic 参数


### WIP / TODO (in priority order)

1. **[P0] Phase B 验证** — 切换到 IsaacLabBackend，对比 5 iter loss（误差 < 1e-4）
2. **[P1] `core/compat/views/`** — JointView / BodyView / ContactView / SensorView + RobotHandle
   - 这是框架核心层，所有自定义任务必须通过 View 读机器人状态
3. **[P2] `core/task/`** — ObsBuilder / RewardBuilder / Termination / BaseTask（依赖 P1）
4. **[P3] 第一个 myrl 原生任务** — humanoid_x 行走 v0（依赖 P1 + P2 + P4）
5. **[P4] Robot assets** — humanoid_x URDF + actuators/sensors/kinematics YAML（可并行）
6. **[P5] `core/algo/`** — 自定义网络（Transformer encoder）/ 扩展 PPO（继承，不 fork）
7. **[P6] Debug tools** — contact_viz / joint_scope / reward_inspector / disturbance_gun（依赖 P1）
8. **[P7] 工程基础** — Experiment Manifest / Registry / CLI / Wandb / Remote TUI
9. **[P-MJX] MuJoCo JAX (MJX) 大规模训练后端** — 未来可扩展模块
   - 目标：替代 IsaacLabBackend 用于 MuJoCo 上的微调/对齐（1024+ envs）
   - 关键约束：同一 MuJoCoTask ABC + ObsHistoryManager，policy 接口不变；不经过 TCP/ROS，直接 JAX→torch tensor
   - 入口文件（待建）：`myrl/src/myrl/core/compat/backends/mjx_backend.py`
   - 依赖：`jax>=0.4`, `jaxlib[cuda12]`, `mujoco>=3.0`（mjx 内置）
   - 性能目标：1024 envs + 64×36 depth @ 50Hz，单 A100
   - 详细 API 笔记见 `memory/mjx_future.md`

---

## Development Modes

### Train-time (GPU server, headless)
```bash
# No Docker, no sudo required
WORKDIR=~/myrl_work bash myrl/scripts/bootstrap_train.sh
# Then use micromamba run -p ... to execute training
```

### Code-time (local, Docker)
```bash
bash myrl/scripts/run_dev.sh
# Opens interactive shell in Isaac Lab container with GPU support
```

容器内注意事项：
- **Python**: `python3` 可直接使用（entrypoint 已 `source /isaac-sim/setup_python_env.sh` 并加 PATH）
- **上游包可用性**: entrypoint 通过 PYTHONPATH 追加使 `instinct_rl` / `instinctlab` / `myrl` 可导入（不用 pip，不污染依赖树）
- **Isaac Sim 运行时**: `instinctlab` 和 `myrl.core.compat.backends.isaaclab_backend` 依赖 `omni.physics`，只有 `AppLauncher` 初始化后才能 import——train.py / play.py 已正确处理
- **上游 repo 路径覆盖**:
  ```bash
  INSTINCT_RL_DIR=/path/to/instinct_rl INSTINCTLAB_DIR=/path/to/instinctlab bash scripts/run_dev.sh
  ```

---

## Coding Conventions

- Python 3.10+ (train-time) / 3.11 (Isaac Lab code-time)
- Type hints required for public APIs
- Use `OrderedDict` for observation formats (instinct_rl convention)
- `@configclass` dataclasses for all configs (Isaac Lab convention)
- All configs in YAML, not Python hardcodes
- Shell scripts: `set -euo pipefail`, log with timestamp
- Git: conventional commits in English, descriptive messages
- Comments/docs in Chinese is fine; identifiers in English

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Isaac Sim | 5.1.0 | Physics simulation (code-time) |
| Isaac Lab | 2.3.2 (NGC tag) / pip `0.47.1` | Environment framework (code-time) |
| Isaac Gym | Preview 4 | Physics simulation (train-time, legacy) |
| instinct_rl | latest | RL algorithms (PPO, AMP, TPPO, VAE) |
| instinctlab | latest | Environment/task definitions |
| PyTorch | latest (cu124/cu128) | Deep learning |
| wandb | latest | Experiment tracking |

---

## Important Notes

1. **instinct_rl replaces rsl_rl** — `third_party/rsl_rl` is legacy; new code uses instinct_rl API
2. **instinctlab replaces legged_gym** — `third_party/legged_gym` is legacy; new tasks use `InstinctRlEnv`
3. **Compat layer is the core innovation** — it abstracts sim backend differences so algo/task code is backend-agnostic
4. Never import from sim backend directly in task/algo code — always go through compat layer
5. Docker `.env` contains EULA acceptance — do not commit (gitignored)
6. `work/` is ephemeral runtime state — always gitignored
7. Bootstrap scripts are designed for **hostile environments** (broken conda, no sudo, no Docker)
8. Path has apostrophe — always quote in bash: `"/home/eziothean/Ezio's_RL_Toolbox"`
9. `myrl/env/` dir gets caught by `env/` gitignore — use specific patterns if needed
