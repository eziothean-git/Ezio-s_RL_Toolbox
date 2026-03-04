# myrl — Robot Reinforcement Learning Toolbox

A from-scratch robot RL framework targeting legged robot whole-body control. Built on top of
**instinct_rl** (algorithm layer) and **instinctlab** (environment/task layer), with Isaac Sim
as the primary training backend and MuJoCo + ROS2 for sim-to-sim and sim-to-real transfer.

```
Algo Layer        instinct_rl  (PPO / AMP / TPPO / VAE / MoE)
─────────────────────────────────────────────────────────────
Task / Env Layer  instinctlab  (InstinctRlEnv + gymnasium tasks)
─────────────────────────────────────────────────────────────
Compat Layer      myrl/core/compat/  (backends + Views)      ← core abstraction
─────────────────────────────────────────────────────────────
Sim Backend       Isaac Lab (train) │ MuJoCo (sim2sim) │ Real Robot via ROS2
```

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Ubuntu 22.04 | Required for ROS2 Humble |
| GPU | NVIDIA, CUDA 12.x | A100 / RTX 30xx or newer recommended |
| Docker | 24+ | Dev mode only |
| NVIDIA Container Toolkit | latest | Dev mode only |
| Python | 3.10 (train) / 3.11 (dev container) | |
| Isaac Sim | 5.1.0 | Dev container only |
| Isaac Lab | 2.3.2 | Dev container only |

---

## Repository Layout

```
Ezio's_RL_Toolbox/
├── myrl/
│   ├── scripts/               # Entry points
│   │   ├── bootstrap_train.sh # Server: one-shot train-time setup
│   │   ├── bootstrap_dev.sh   # PC: one-shot dev-time Docker setup
│   │   ├── run_dev.sh         # PC: start dev container
│   │   ├── train.py           # Isaac Lab training
│   │   ├── play.py            # Isaac Lab inference
│   │   ├── play_mujoco.py     # MuJoCo inference (no Isaac Sim)
│   │   ├── start_mujoco_server.py  # Standalone MuJoCo sim server
│   │   └── start_ros2_bridge.py    # ROS2 ↔ TCP bridge
│   ├── docker/
│   │   ├── Dockerfile.dev     # Isaac Lab 2.3.2 + ROS2 Humble
│   │   ├── compose.yaml
│   │   └── entrypoint.sh
│   └── src/myrl/
│       ├── tasks/             # Task registration (gym.register)
│       ├── assets/            # Asset resolver
│       └── core/
│           ├── compat/
│           │   ├── backends/  # IsaacLabBackend, MuJoCoBackend
│           │   └── views/     # RobotHandle, JointView, BodyView, ContactView
│           ├── sim_server/    # TCP protocol, MuJoCoSimServer, Ros2Bridge
│           ├── obs/           # ObsHistoryManager
│           └── task/          # ObsBuilder, RewardBuilder (WIP)
```

---

## Deployment

### Mode 1 — Train (GPU server, no Docker)

Designed for headless GPU servers. Requires no root, no Docker, no pre-installed conda.

```bash
# One-time setup (~5 min, downloads micromamba + torch)
WORKDIR=~/myrl_work bash myrl/scripts/bootstrap_train.sh

# Train
micromamba run -p ~/myrl_work/.mamba/envs/myrl-train \
    python myrl/scripts/train.py \
    --task myrl/Locomotion-Flat-G1Smoke-v0 \
    --num_envs 4096 \
    --headless
```

The bootstrap script auto-detects GPU compute capability and selects the correct
PyTorch index (`cu124` vs `cu128`). Logs are written to `logs/myrl/<experiment>/`.

**Resume training:**
```bash
python myrl/scripts/train.py \
    --task myrl/Locomotion-Flat-G1Smoke-v0 \
    --resume --load_run 20260304_120000
```

---

### Mode 2 — Dev (local PC, Docker)

Provides a full Isaac Lab + ROS2 environment in a container with GPU and X11/Wayland support.

```bash
# One-time setup (installs Docker + NVIDIA Container Toolkit, clones Isaac Lab)
bash myrl/scripts/bootstrap_dev.sh

# Start dev container (builds image on first run, ~10 min)
bash myrl/scripts/run_dev.sh
```

Inside the container, `python3` is immediately available with Isaac Sim, instinct_rl,
instinctlab, and myrl all on `PYTHONPATH`. No `pip install` required.

**Override upstream repo paths:**
```bash
INSTINCT_RL_DIR=/path/to/instinct_rl \
INSTINCTLAB_DIR=/path/to/instinctlab \
bash myrl/scripts/run_dev.sh
```

**Train inside container:**
```bash
# (inside container)
python3 myrl/scripts/train.py --task myrl/Locomotion-Flat-G1Smoke-v0 --headless
```

---

## Sim-to-Sim Transfer (Isaac Lab → MuJoCo)

The MuJoCo backend runs as a standalone TCP server. A trained policy connects to it
via `MuJoCoBackend`, which implements the same `VecEnv` interface as `IsaacLabBackend`.
No Isaac Sim dependency on the inference side.

**Step 1 — Start the MuJoCo simulation server** (separate terminal):

```bash
# Smoke test with built-in DummyTask (no MJCF needed)
python myrl/scripts/start_mujoco_server.py --task dummy --num_envs 4 --port 7777

# Real task (implements MuJoCoTask ABC)
python myrl/scripts/start_mujoco_server.py \
    --task myrl.tasks.mujoco.g1_walk:G1WalkTask \
    --mjcf myrl/assets/robots/humanoid_x/robot.xml \
    --num_envs 16 --port 7777
```

**Step 2 — Run inference:**

```bash
# Smoke (random actions)
python myrl/scripts/play_mujoco.py --num_envs 4 --num_steps 100

# With checkpoint
python myrl/scripts/play_mujoco.py \
    --load_run logs/myrl/g1_smoke/20260304_120000 \
    --checkpoint model_5000.pt \
    --host localhost --port 7777
```

The key constraint for sim2sim fidelity: the `MuJoCoTask.obs_format()` return value
**must be identical** to the `obs_format` produced by the Isaac Lab training environment.

---

## Sim-to-Real Transfer (MuJoCo → Real Robot via ROS2)

The ROS2 bridge connects the MuJoCo sim server to a real robot. Observations flow over
ROS2 topics (sensor bus), while the TCP channel carries only rewards/dones/log metadata.

```
Real Robot Sensors → ROS2 topics (/myrl/{task_id}/obs/{group})
                                  ↓
                            Ros2Bridge
                                  ↓  TCP (rewards/dones only)
                         MuJoCoSimServer
                                  ↑
                     Policy (via MuJoCoBackend)
```

**Step 1 — Start the sim server in ROS mode:**

```bash
python myrl/scripts/start_mujoco_server.py \
    --task dummy --num_envs 4 --port 7777 \
    --ros --task_id robot1
```

**Step 2 — Start the ROS2 bridge** (source ROS2 first):

```bash
source /opt/ros/humble/setup.bash

python myrl/scripts/start_ros2_bridge.py \
    --port 7777 --task_id robot1 --hz 50 \
    --history_cfg '{"policy": {"base_ang_vel": 8}}'
```

**Step 3 — Verify topics:**

```bash
ros2 topic list | grep myrl
ros2 topic echo /myrl/robot1/obs/policy --once

# Inject a test action
ros2 topic pub /myrl/robot1/action std_msgs/msg/Float32MultiArray \
    "{data: [0.1, 0.0, -0.1]}" --once
```

The `--history_cfg` argument configures per-term observation history managed by
`ObsHistoryManager` on the bridge side — the robot's sensors only publish current-frame data.

---

## Writing a Custom Task

### Isaac Lab task (training)

1. Create `myrl/src/myrl/tasks/<domain>/config/<task_name>/__init__.py` and register:

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

2. Implement `MyTaskEnvCfg` (inherits `InstinctLabRLEnvCfg`) and `MyRunnerCfg`
   (inherits `InstinctRlOnPolicyRunnerCfg`).

### MuJoCo task (sim2sim / sim2real)

Subclass `MuJoCoTask` from `myrl.core.sim_server.mujoco_task`:

```python
from myrl.core.sim_server.mujoco_task import MuJoCoTask
import numpy as np

class G1WalkTask(MuJoCoTask):
    @property
    def num_actions(self) -> int:
        return 29  # must match Isaac Lab training

    @property
    def max_episode_length(self) -> int:
        return 1000

    def obs_format(self) -> dict:
        # MUST be identical to Isaac Lab obs_format
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

## Key Design Constraints

1. **Compat layer boundary** — task and algo code never imports directly from a sim backend.
   All robot state access goes through `RobotHandle` / `JointView` / `BodyView` / `ContactView`.

2. **obs_format consistency** — the `obs_format` dict used by the policy network must be
   identical across Isaac Lab training, MuJoCo sim2sim, and the ROS2 bridge. Shape mismatches
   cause silent failures at inference time.

3. **Reward shape** — rewards are always `Tensor[num_envs, num_rewards]` (second dim retained
   even for single-reward tasks). This matches instinct_rl's multi-critic convention.

4. **History ownership** — sim backends emit current-frame observations only.
   `ObsHistoryManager` (model layer) owns the ring buffer and produces history-expanded obs.

---

## Status

| Component | Status |
|-----------|--------|
| Train-time bootstrap | Done |
| Dev Docker environment | Done |
| Isaac Lab → instinct_rl training pipeline | Done (Phase A, verified 5 iter) |
| IsaacLabBackend (Phase B) | Implemented, pending loss-parity validation |
| MuJoCo TCP sim server | Done (23/23 tests) |
| ROS2 bridge | Done (32/32 tests) |
| MuJoCo play / inference script | Done |
| ObsHistoryManager | Done (32/32 tests) |
| RobotHandle / Views | Partial (proxy layer, full View API WIP) |
| ObsBuilder / RewardBuilder | Skeleton |
| humanoid_x assets | Empty (WIP) |
