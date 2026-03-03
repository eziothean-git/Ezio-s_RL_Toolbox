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
Compat Layer      myrl/core/compat/ (Views + Runtime API)
─────────────────────────────────────────────────────────────
Sim Backend       Isaac Gym (train) | Isaac Lab (dev) | MuJoCo (sim2sim)
```

### Core Design Principles
1. **Strict boundary enforcement**: algo/task/sim must be decoupled; cross-layer access goes through compat layer only
2. **Strong-typed View API**: RobotHandle, JointView, BodyView, ContactView, SensorView - no direct tensor access
3. **Observability first**: all reward/obs/contact/state changes must be inspectable and recordable
4. **Reproducibility over convenience**: training runs on manifests; assets/weights/configs are content-addressable
5. **Minimal deployment assumptions**: train-time needs no Docker, no sudo, no GUI

## Upstream Frameworks

### instinct_rl (github.com/project-instinct/instinct_rl)
- RL algorithm library extending rsl_rl
- Key concepts: `obs_format` (OrderedDict), `obs_segment`, `obs_pack`, `obs_component`
- Factory pattern: `modules.build_actor_critic(class_name, config, obs_format, ...)`
- Algorithms: PPO, State Estimator, AMP/WASABI, TPPO (Teacher PPO), VAE Distillation
- Networks: ActorCritic, MoE ActorCritic, VAE ActorCritic, Encoder variants, VQ-VAE
- Extensible via `module_name:class_name` import syntax

### instinctlab (github.com/project-instinct/instinctlab)
- Environment layer built on Isaac Lab 2.3.2 / Isaac Sim 5.1.0
- Entry point: `instinctlab.envs:InstinctRlEnv`
- Task registration via gymnasium: `gym.register(id=..., entry_point="instinctlab.envs:InstinctRlEnv", kwargs={...})`
- Tasks live in `source/instinctlab/instinctlab/tasks/`
- Training: `python scripts/instinct_rl/train.py --task=<TASK_NAME> --headless`
- Export: `--exportonnx` flag for ONNX policy export (sim2real via instinct_onboard)

## Repository Structure

```
Ezio's_RL_Toolbox/
├── CLAUDE.md                          # This file
├── Project_Description.md             # Full framework requirements (Chinese)
├── Ezio's_RL_Toolbox.code-workspace   # VSCode workspace
└── myrl/
    ├── assets/                        # Robot URDF/USD, terrain meshes (empty, WIP)
    │   ├── robots/
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
    │   └── (planned: cli/, core/compat/, core/task/, core/algo/, debug_tools/, tools/, entrypoints/)
    ├── third_party/                   # Vendored dependencies (gitignored)
    │   ├── isaac_gym/                 # IsaacGym Preview 4 tarball
    │   ├── isaaclab/                  # Isaac Lab 2.3.0 (git clone)
    │   ├── legged_gym/               # legged_gym (legacy, to be replaced by instinctlab)
    │   └── rsl_rl/                    # rsl_rl (legacy, to be replaced by instinct_rl)
    └── work/                          # Runtime workdir (gitignored)
```

## Current Status

### DONE
- Framework requirements document (Project_Description.md)
- Train-time bootstrap (micromamba, auto GPU detection, PyTorch auto-index, IsaacGym .pth injection)
- Dev-time bootstrap (Docker, NVIDIA Container Toolkit, Isaac Lab clone, EULA management)
- Docker dev environment (Isaac Lab 2.3.0 container, runtime server plumbing)
- Smoke test (validates full stack: GPU, IsaacGym, torch, rsl_rl, legged_gym)

### WIP / TODO
- **Migrate from rsl_rl + legged_gym to instinct_rl + instinctlab**
- Implement `src/myrl/core/compat/` (Views, backends, Runtime API)
- Implement `src/myrl/core/task/` (ObsBuilder, RewardBuilder, Termination, Curriculum)
- Implement `src/myrl/core/algo/` (instinct_rl adapter instead of rslrl_adapter)
- Implement debug tools (contact_viz, joint_scope, disturbance_gun, reward_inspector, sanity_checker)
- Create robot assets (URDF + actuators/sensors/kinematics YAML)
- Create task configs (experiments, sim backends, debug profiles)
- Registry system (content-addressed artifact management)
- Experiment manifest system
- Remote TUI for training control

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

## Coding Conventions

- Python 3.10+ (train-time) / 3.11 (Isaac Lab code-time)
- Type hints required for public APIs
- Use `OrderedDict` for observation formats (instinct_rl convention)
- Reward functions registered by name with weight and expected scale
- All configs in YAML, not Python classes
- Shell scripts: `set -euo pipefail`, log with timestamp
- Git: conventional commits in English, descriptive messages

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Isaac Sim | 5.1.0 | Physics simulation (code-time) |
| Isaac Lab | 2.3.0-2.3.2 | Environment framework (code-time) |
| Isaac Gym | Preview 4 | Physics simulation (train-time, legacy) |
| instinct_rl | latest | RL algorithms (PPO, AMP, TPPO, VAE) |
| instinctlab | latest | Environment/task definitions |
| PyTorch | latest (cu124/cu128) | Deep learning |
| wandb | latest | Experiment tracking |

## Important Notes for Development

1. **instinct_rl replaces rsl_rl** - the third_party/rsl_rl is legacy; new code should use instinct_rl's API
2. **instinctlab replaces legged_gym** - the third_party/legged_gym is legacy; new tasks should use InstinctRlEnv
3. The compat layer is the **core innovation** of this project - it abstracts away sim backend differences
4. Never import from sim backend directly in task/algo code - always go through compat layer
5. Docker `.env` file contains EULA acceptance - do not commit it (gitignored)
6. The `work/` directory is ephemeral runtime state - always gitignored
7. Bootstrap scripts are designed for **hostile environments** (broken conda, no sudo, no Docker)
