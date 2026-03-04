# myrl API & Framework Conventions

This document describes the contracts, data formats, and interface conventions that all
components in the myrl framework must follow. Deviating from these conventions breaks
cross-layer compatibility.

---

## Table of Contents

1. [obs_format Convention](#1-obs_format-convention)
2. [VecEnv Interface](#2-vecenv-interface)
3. [SimBackend ABC](#3-simbackend-abc)
4. [extras Dict Contract](#4-extras-dict-contract)
5. [IsaacLabBackend](#5-isaaclab-backend)
6. [MuJoCoBackend](#6-mujoco-backend)
7. [MuJoCoTask ABC](#7-mujocotask-abc)
8. [TCP Protocol (SimProto)](#8-tcp-protocol-simproto)
9. [MuJoCoSimServer](#9-mujocosimserver)
10. [Ros2Bridge](#10-ros2bridge)
11. [ObsHistoryManager](#11-obshistorymanager)
12. [Task Registration Pattern](#12-task-registration-pattern)

---

## 1. obs_format Convention

`obs_format` is the central schema shared by all components — network construction,
VecEnv adapters, MuJoCoTask, and ObsHistoryManager all rely on it.

```python
obs_format: dict[str, dict[str, tuple]] = {
    "policy": {                   # required: actor input
        "base_ang_vel": (24,),    # 8 timesteps × 3 dims (history already embedded)
        "joint_pos":    (87,),    # 29 joints × 3
        "depth_image":  (2304,),  # 64×36 flattened
    },
    "critic": {                   # optional: privileged obs for value function
        "motion_reference": (50,),
    },
    # arbitrary extra groups for AMP, estimator, etc.
    "amp_policy":    {"state_vec": (128,)},
    "amp_reference": {"state_vec": (128,)},
}
```

**Rules:**

- All shapes are flat 1-D tuples after history expansion. The raw per-term shape from
  Isaac Lab's `ObsTerm` may be multi-dimensional; it is flattened before being stored.
- When `ObsHistoryManager` is in use, shapes in `obs_format` represent the **current-frame**
  dimension. `get_output_format()` returns the history-expanded version.
- `obs_format` passed to `ppo.init_storage()` and `actor_critic` must be the
  **history-expanded** format (i.e. the output of `ObsHistoryManager.get_output_format()`
  when history is used, or the raw `obs_format` when it is not).
- Group and term names are strings; ordering within a group must be stable (use
  `OrderedDict` or rely on Python 3.7+ dict ordering).

---

## 2. VecEnv Interface

All backends implement `instinct_rl.env.VecEnv`. The interface below is what
`OnPolicyRunner` expects:

```python
class VecEnv(ABC):
    # Required attributes
    num_envs:      int
    num_actions:   int
    num_rewards:   int           # 1 for standard tasks, >1 for multi-critic
    max_episode_length: int | Tensor
    device:        torch.device

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """Return the obs_format dict (all groups, not only policy)."""

    def get_observations(self) -> tuple[Tensor, dict]:
        """Return current observations without stepping.

        Returns:
            obs:    Tensor[num_envs, policy_obs_size]
            extras: see extras dict contract
        """

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Step all environments by one control frame.

        Args:
            actions: Tensor[num_envs, num_actions]

        Returns:
            obs:     Tensor[num_envs, policy_obs_size]
            rewards: Tensor[num_envs, num_rewards]
            dones:   Tensor[num_envs]  dtype=long  (0 or 1)
            extras:  see extras dict contract
        """

    def reset(self) -> tuple[Tensor, dict]:
        """Reset all environments.

        Returns:
            obs:    Tensor[num_envs, policy_obs_size]
            extras: see extras dict contract
        """
```

**Reward shape invariant:** rewards are always `[num_envs, num_rewards]`. Single-reward
tasks must unsqueeze the last dim. This matches instinct_rl's multi-critic storage layout.

---

## 3. SimBackend ABC

`myrl.core.compat.backends.base.SimBackend` is a lightweight marker class. Concrete
backends inherit both `SimBackend` and `VecEnv`:

```python
class SimBackend(ABC):
    num_envs:    int
    num_actions: int
    num_rewards: int
    device:      torch.device

    @abstractmethod
    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]: ...

    @abstractmethod
    def reset(self) -> tuple[Tensor, dict]: ...

    @abstractmethod
    def get_observations(self) -> tuple[Tensor, dict]: ...

    @abstractmethod
    def close(self) -> None: ...
```

Backends must **never** be imported by task or algo code. All cross-layer access goes
through the compat layer.

---

## 4. extras Dict Contract

The `extras` dict returned by `step()` / `reset()` / `get_observations()` must follow
this schema:

```python
extras = {
    # All observation groups as flat tensors. Always present.
    "observations": {
        "policy":  Tensor[num_envs, policy_obs_size],
        "critic":  Tensor[num_envs, critic_obs_size],  # if privileged obs
        # ... other groups (amp_policy, amp_reference, etc.)
    },

    # Truncation flags for infinite-horizon GAE bootstrapping.
    # Present only when env.cfg.is_finite_horizon == False.
    "time_outs": Tensor[num_envs],   # bool

    # Scalar metrics for TensorBoard / wandb.
    # Optional; may be absent or empty.
    "log": dict[str, float],
}
```

`time_outs` distinguishes truncation (episode length limit) from termination (task failure).
It is used by `PPO.process_env_step()` for value bootstrapping. Omit it for finite-horizon
tasks; instinct_rl will treat all episode ends as true terminations.

---

## 5. IsaacLabBackend

`myrl.core.compat.backends.isaaclab_backend.IsaacLabBackend`

Wraps an `ManagerBasedRLEnv` (or `DirectRLEnv`) as a `VecEnv`. Intended as a drop-in
replacement for `instinctlab`'s `InstinctRlVecEnvWrapper`.

```python
from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend

env_raw = gym.make("myrl/MyTask-v0", cfg=env_cfg)
env = IsaacLabBackend(env_raw)

# env now satisfies VecEnv + SimBackend
runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device="cuda")
```

**Behaviour:**
- Calls `env.reset()` in `__init__` (OnPolicyRunner does not call reset itself).
- Builds `obs_format` from Isaac Lab's `observation_manager.active_terms`.
- Flattens multi-dimensional obs terms to 1-D before concatenation.
- Stacks multi-reward dict → `Tensor[num_envs, num_rewards]`.
- Sets `time_outs` only for infinite-horizon configs.

**Phase A → B switch** (one-line change in `train.py` / `play.py`):
```python
# Phase A (current):
from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper as EnvWrapper
# Phase B (target):
from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend as EnvWrapper
```

---

## 6. MuJoCoBackend

`myrl.core.compat.backends.mujoco_backend.MuJoCoBackend`

TCP client that connects to a running `MuJoCoSimServer`. Implements the same `VecEnv`
interface as `IsaacLabBackend` — the same policy code works unchanged.

```python
from myrl.core.compat.backends.mujoco_backend import MuJoCoBackend

env = MuJoCoBackend(host="localhost", port=7777, device="cpu")
# env.num_envs, env.num_actions, env.obs_format() all populated from server handshake

obs, extras = env.reset()
for _ in range(steps):
    actions = policy(obs)
    obs, rewards, dones, extras = env.step(actions)
env.close()
```

**Handshake:** On construction `MuJoCoBackend` sends `HANDSHAKE_REQ` and receives
`HANDSHAKE_RESP` containing `num_envs`, `num_actions`, `obs_format`, and `max_episode_length`.
Then it immediately sends `RESET_REQ`.

**Tensor conversion:** All numpy arrays received from the server are converted to `torch.Tensor`
on the requested `device`. Actions are converted back to numpy before transmission.

---

## 7. MuJoCoTask ABC

`myrl.core.sim_server.mujoco_task.MuJoCoTask`

User-implemented physics logic for a MuJoCo model. `MuJoCoSimServer` calls these methods
on every step/reset without knowing the concrete task type.

```python
class MuJoCoTask(ABC):

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Action space dimension. Must match Isaac Lab training num_actions."""

    @property
    def num_rewards(self) -> int:
        """Number of reward channels. Default 1."""
        return 1

    @property
    @abstractmethod
    def max_episode_length(self) -> int:
        """Steps before truncation. Server handles time-out bookkeeping."""

    @abstractmethod
    def obs_format(self) -> dict[str, dict[str, tuple]]:
        """Observation format. MUST be identical to Isaac Lab training obs_format."""

    @abstractmethod
    def compute_obs(self, model, datas: list) -> dict[str, np.ndarray]:
        """
        Args:
            model:  mujoco.MjModel (shared, read-only)
            datas:  list[mujoco.MjData], one per parallel env

        Returns:
            {"policy": ndarray[N, policy_dim], "critic": ndarray[N, critic_dim], ...}
            All arrays float32, already flattened per group.
        """

    @abstractmethod
    def compute_reward(self, model, datas: list, actions: np.ndarray) -> np.ndarray:
        """
        Returns:
            ndarray[N, num_rewards], dtype float32
        """

    @abstractmethod
    def is_terminated(self, model, datas: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            (terminated[N], truncated[N])  dtype bool
            terminated: task failure (e.g. fall)
            truncated:  time limit (server handles this; return all-False here)
        """

    @abstractmethod
    def apply_action(self, model, data, action: np.ndarray) -> None:
        """Write action to data.ctrl / qfrc_applied (single env, in-place)."""

    @abstractmethod
    def reset_env(self, model, data, env_id: int) -> None:
        """Randomize qpos/qvel for a single env (in-place)."""
```

**Sim2Sim fidelity rules:**

1. `obs_format()` must return the same group names, term names, and flat shapes as the
   Isaac Lab training environment.
2. `apply_action()` semantics must match the Isaac Lab `ActionCfg` (position / torque /
   velocity control mode must be consistent).
3. `reset_env()` randomisation range should match the Isaac Lab `EventCfg` domain
   randomisation used during training.

**Built-in DummyTask** is available for protocol smoke-testing (no MJCF required):
```bash
python myrl/scripts/start_mujoco_server.py --task dummy --num_envs 4
```

---

## 8. TCP Protocol (SimProto)

`myrl.core.sim_server.protocol`

All communication between `MuJoCoBackend` (client) and `MuJoCoSimServer` (server) uses
a framed msgpack-numpy protocol over a single persistent TCP connection.

### Frame format

```
[4 bytes: uint32 big-endian body_len] + [msgpack-numpy body]
```

### Message types

```python
class MsgType(IntEnum):
    HANDSHAKE_REQ  = 0   # client → server: open connection
    HANDSHAKE_RESP = 1   # server → client: metadata (num_envs, obs_format, ...)
    STEP_REQ       = 2   # client → server: {"actions": ndarray[N, A]}
    STEP_RESP      = 3   # server → client: obs / rewards / dones / extras
    RESET_REQ      = 4   # client → server: {}
    RESET_RESP     = 5   # server → client: initial obs
    GET_OBS_REQ    = 6   # client → server: {} (no physics step)
    GET_OBS_RESP   = 7   # server → client: current obs
    CLOSE          = 8   # client → server: graceful disconnect
    ERROR          = 9   # server → client: error string
```

### Payload schemas

**HANDSHAKE_RESP**
```python
{
    "type": MsgType.HANDSHAKE_RESP,
    "num_envs": int,
    "num_actions": int,
    "obs_format": dict,           # same structure as obs_format convention
    "max_episode_length": int,
}
```

**STEP_REQ**
```python
{"type": MsgType.STEP_REQ, "actions": ndarray[N, A]}
```

**STEP_RESP** (when `include_obs_in_response=True`)
```python
{
    "type": MsgType.STEP_RESP,
    "obs_all": {"policy": ndarray[N, D], ...},   # all groups
    "rewards": ndarray[N, num_rewards],
    "dones":   ndarray[N],                        # bool
    "extras":  {"time_outs": ndarray[N], "log": {}},
}
```

**STEP_RESP** (when `include_obs_in_response=False`, used with `--ros` mode)
```python
{
    "type": MsgType.STEP_RESP,
    "rewards": ndarray[N, num_rewards],
    "dones":   ndarray[N],
    "extras":  {"time_outs": ndarray[N], "log": {}},
}
```

### SimProto usage

```python
from myrl.core.sim_server.protocol import SimProto, MsgType
import socket

sock = socket.create_connection(("localhost", 7777))
SimProto.send(sock, {"type": MsgType.HANDSHAKE_REQ})
resp = SimProto.recv(sock)   # -> dict
```

---

## 9. MuJoCoSimServer

`myrl.core.sim_server.mujoco_server.MuJoCoSimServer`

Vectorised MuJoCo physics server. Accepts one persistent TCP connection and drives
N parallel `MjData` instances per control frame.

```python
from myrl.core.sim_server.mujoco_server import MuJoCoSimServer

server = MuJoCoSimServer(
    task=my_task,                   # MuJoCoTask instance
    mjcf_path="robot.xml",          # None for DummyTask
    num_envs=16,
    sim_steps_per_ctrl=4,           # physics substeps per control frame
    host="0.0.0.0",
    port=7777,
    include_obs_in_response=True,   # set False when using ROS mode
)

# Optional: register an obs callback (called BEFORE TCP response is sent)
def on_obs(obs_all: dict[str, np.ndarray]) -> None:
    # publish to ROS topics, write to shared memory, etc.
    ...

server.register_obs_callback(on_obs)
server.serve_forever()   # blocks; Ctrl-C to stop
```

**obs_callback ordering guarantee:** `obs_callback` is invoked synchronously before the
TCP response is sent. This means a downstream consumer (e.g. `Ros2Bridge`) that has
received the TCP response can safely read the latest obs from the ROS topic.

---

## 10. Ros2Bridge

`myrl.core.sim_server.ros2_bridge.Ros2Bridge`

Bidirectional bridge between a `MuJoCoSimServer` and a real robot (or any ROS2 system).

**Data flow:**

```
Sensors / state estimator → ROS topic /myrl/{task_id}/obs/{group}  (Float32MultiArray)
                                            ↓
                                      Ros2Bridge
                              (ObsHistoryManager, control loop)
                                            ↓  TCP  STEP_REQ
                                   MuJoCoSimServer
                                            ↓  TCP  STEP_RESP (rewards/dones only)
                                      Ros2Bridge
                                            ↓
                       Action publisher  /myrl/{task_id}/action  (Float32MultiArray)
```

```python
from myrl.core.sim_server.ros2_bridge import Ros2Bridge

bridge = Ros2Bridge(
    host="localhost",
    port=7777,
    task_id="robot1",               # must match server --task_id
    control_hz=50.0,
    history_cfg={"policy": {"base_ang_vel": 8}},  # or None
    node_name="myrl_ros2_bridge",
    obs_wait_timeout=0.1,           # seconds to wait for ROS obs after TCP resp
    device="cpu",
)
bridge.spin()   # blocks on rclpy.spin; control loop runs in daemon thread
```

**Topic naming convention:**
- Obs input:    `/myrl/{task_id}/obs/{group_name}`  — `Float32MultiArray`
- Action output: `/myrl/{task_id}/action`           — `Float32MultiArray`
- Reward output: `/myrl/{task_id}/reward`           — `Float32MultiArray`

**External action injection** (bypassing ROS, for testing):
```python
bridge.set_actions(np.zeros((num_envs, num_actions), dtype=np.float32))
```

---

## 11. ObsHistoryManager

`myrl.core.obs.history_manager.ObsHistoryManager`

Sim-agnostic ring-buffer for per-term observation history. Belongs to the model layer —
sim backends emit current-frame obs only; history is maintained here.

```python
from myrl.core.obs.history_manager import ObsHistoryManager

# obs_format: current-frame shapes (no history yet)
obs_format = {"policy": {"base_ang_vel": (3,), "joint_pos": (29,)}}

# history_cfg: two supported formats
history_cfg_group = {"policy": 8}                          # all terms in group: 8 frames
history_cfg_term  = {"policy": {"base_ang_vel": 8,         # per-term precision
                                "joint_pos":    1}}

mgr = ObsHistoryManager(
    obs_format=obs_format,
    history_cfg=history_cfg_group,
    num_envs=512,
    device="cuda",
)

# Push current frame, receive history-expanded obs_pack
obs_pack_current = {"policy": torch.randn(512, 32)}        # [N, 3+29]
obs_pack_history = mgr.push(obs_pack_current)
# obs_pack_history["policy"].shape == (512, 8*3 + 1*29) == (512, 53)

# Reset done environments (call after done-env detection, before next push)
done_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
mgr.reset(done_ids)

# Query history-expanded obs_format for network construction
expanded_format = mgr.get_output_format()
# {"policy": {"base_ang_vel": (24,), "joint_pos": (29,)}}
```

**Initialisation:** all ring-buffer slots are zero-filled at construction (cold-start safe).

**history_cfg resolution order:**
1. If `group` key maps to `int` → all terms in that group use the same length.
2. If `group` key maps to `dict` → per-term lookup; missing terms default to `1`.
3. Groups absent from `history_cfg` entirely default to `1` for all their terms.

---

## 12. Task Registration Pattern

All myrl tasks are registered through gymnasium and discovered automatically when
`import myrl.tasks` is executed (after `AppLauncher` in train/play scripts).

### Registration (Isaac Lab task)

```python
# file: myrl/src/myrl/tasks/<domain>/config/<task_name>/__init__.py
import gymnasium as gym

gym.register(
    id="myrl/MyTask-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":        "<pkg>.tasks.<domain>.<task>:MyTaskEnvCfg",
        "instinct_rl_cfg_entry_point": "<pkg>.tasks.<domain>.<task>.agents:MyRunnerCfg",
    },
)
# Play variant (simpler terrain, eval only)
gym.register(
    id="myrl/MyTask-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":        "<pkg>.tasks.<domain>.<task>:MyTaskEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": "<pkg>.tasks.<domain>.<task>.agents:MyRunnerCfg",
    },
)
```

### Task ID namespace

- `myrl/*` — tasks defined in this repository
- `Instinct-*` — tasks from upstream instinctlab (available after `import instinctlab.tasks`)

Both namespaces are loaded in `train.py` / `play.py` and are selectable via `--task`.

### Runner config

```python
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlActorCriticCfg,
    InstinctRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass

@configclass
class MyRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations    = 30000
    save_interval     = 5000
    experiment_name   = "my_task"
    seed              = 42

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

All fields must be explicitly set — do not rely on `MISSING` defaults from parent classes,
as unresolved `MISSING` values cause silent serialisation failures in `to_dict()`.

---

## Coding Conventions

| Rule | Detail |
|------|--------|
| Language | Code identifiers in English; comments and docs in Chinese or English |
| Type hints | Required for all public API signatures |
| Configs | `@configclass` dataclasses (Isaac Lab convention); no Python hardcodes |
| Observation dicts | `OrderedDict` or stable-insertion-order `dict` (Python 3.7+) |
| Shell scripts | `set -euo pipefail`; log with timestamps |
| Git commits | Conventional commits in English, descriptive body |
| Import discipline | Never import sim backends in task/algo code; always go through compat layer |
| Reward shape | Always `[num_envs, num_rewards]`, even for single-reward tasks |
