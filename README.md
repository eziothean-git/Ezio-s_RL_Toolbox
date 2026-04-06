# myrl -- Robot Reinforcement Learning Toolbox

A from-scratch robot RL framework for legged robot whole-body control, with a
browser-based **visual editor** for experiment configuration, sensor placement,
and real-time training monitoring.

```
 Editor WebUI        WebGPU 3D viewer + obs graph + reward editor + fleet mgmt
 ─────────────────────────────────────────────────────────────────────────────
 Algo Layer          instinct_rl  (PPO / AMP / TPPO / VAE / MoE)
 ─────────────────────────────────────────────────────────────────────────────
 Task / Env Layer    instinctlab  (InstinctRlEnv + gymnasium tasks)
 ─────────────────────────────────────────────────────────────────────────────
 Compat Layer        myrl/core/compat/  (Views + Sensor Drivers)
 ─────────────────────────────────────────────────────────────────────────────
 Sim Backend         Isaac Lab (train) │ MuJoCo (sim2sim) │ Real Robot (ROS2)
```

---

## Quick Start

### 1. Launch the Editor

The editor runs on your host machine and manages Docker containers for training.

```bash
# One-time: bootstrap the dev environment
bash myrl/scripts/bootstrap_dev.sh

# Start editor + train manager (opens browser automatically)
bash myrl/scripts/start_editor.sh
```

Open `http://localhost:7001` -- select an experiment from the sidebar to begin.

### 2. Train

From the **Run** tab in the editor, select an experiment and click **Start Training**.
The editor delegates training to the Docker container via `docker exec`.

Or from the command line:

```bash
bash myrl/scripts/run_dev.sh   # start container
# inside container:
python3 myrl/scripts/train.py --task myrl/Locomotion-Flat-G1Native-v0 --headless
```

### 3. Remote Training (GPU Servers)

```bash
# On a headless GPU server (no Docker needed)
WORKDIR=~/myrl_work bash myrl/scripts/bootstrap_train.sh
micromamba run -p ~/myrl_work/.mamba/envs/myrl-train \
    python myrl/scripts/train.py \
    --task myrl/Locomotion-Flat-G1Native-v0 \
    --num_envs 4096 --headless
```

Add remote servers through the **Servers** tab in the editor for fleet management.

---

## Editor WebUI

The browser-based editor (`http://localhost:7001`) provides:

### Experiment Editor
- **Project > Task** tree sidebar -- experiments contain train/play task variants
- **Robot & Sensors** -- WebGPU 3D model viewer + link tree + sensor attachment
  - Click robot links to attach sensors (depth camera, LiDAR, force, IMU)
  - Sensor manifest auto-saved to YAML, referenced by obs pipeline
- **Reward Pipeline** -- visual ratio editor with schema-driven parameter forms
  - `@reward_fn` decorated functions auto-discovered at startup
- **Obs Pipeline v2** -- canvas-based block graph editor (DAG)
  - Drag-to-connect wiring, zoom/pan, floating add-block menu
  - Block types: obs (MDP/sensor) > modifier (scale/noise/history) > encoder (CNN/MLP) > group (policy/critic)
- **Algorithm Config** -- form editor for PPO/AMP/TPPO parameters

### Training Control
- Launch training from browser with configurable num_envs, device, headless mode
- Real-time console output via SSE streaming
- GPU utilization, iteration counter, ETA display
- Stop / Halt / Resume / Checkpoint controls

### Fleet Management
- Add remote GPU servers (SSH tunnel or direct Tailscale)
- Sync code, deploy packages, start/stop remote train managers
- Train on any server from the same UI

### Tech Stack
- **Zero external dependencies** -- vanilla ES modules, no build tools, no CDN
- **WebGPU** for 3D robot rendering (graceful fallback when unavailable)
- **Python stdlib HTTP server** -- train_manager.py uses no pip packages except optional PyYAML

---

## Architecture

### Core Abstractions

**Views** -- all task/reward code accesses robot state through typed views, never
directly from the sim backend:

```python
robot = RobotHandle.from_env(env)
robot.joints.pos          # JointView: (num_envs, num_joints)
robot.bodies.root_pos_w   # BodyView: (num_envs, 3)
robot.contacts("feet")    # ContactView: forces, air_time, in_contact
robot.depth_camera()      # DepthCameraView: depth, depth_flat, history
robot.height_scan()       # HeightScanView: heights_w, heights_rel
robot.force_sensor()      # ForceSensorView: forces, torques, magnitude
```

**Sensor Driver Model** -- sensors use a Protocol-based driver architecture:

```
SensorView (View layer)     DepthCameraView / HeightScanView / ForceSensorView
       ↓ consumes
Protocol (contract)          DepthCameraProto / HeightScanProto / ForceSensorProto
       ↑ implements
Driver (backend adapter)     IsaacLabDepthCamera / IsaacLabHeightScanner / ...
       ↑ reads
Sim Backend                  Isaac Lab RayCasterCamera / RayCaster / ContactSensor
```

**Sensor Manifest** -- per-robot YAML declaring available sensors:

```yaml
# myrl/assets/sensor_cfgs/g1_29dof_sensors.yaml
sensors:
  - name: front_depth_camera
    type: depth_camera
    mount_link: head_link
    config: { width: 64, height: 36, fov_deg: 87, max_range: 5 }
  - name: left_foot_height_scan
    type: height_scanner
    mount_link: left_ankle_roll_link
    config: { size: [0.3, 0.2], resolution: 0.05 }
```

Created visually in the editor's WebGPU 3D viewer, consumed by the obs pipeline.

**Obs Pipeline v2** -- block graph DAG compiled to instinct_rl-compatible config:

```yaml
blocks:
  - { id: depth_cam, type: obs, kind: sensor, sensor_name: front_depth_camera, outputs: [encoder] }
  - { id: encoder, type: encoder, kind: conv2d, output_size: 128, outputs: [policy_group] }
  - { id: policy_group, type: group, kind: policy }
```

**Reward Library** -- `@reward_fn` decorator auto-registers terms with Pydantic schemas:

```python
@reward_fn(description="Track velocity", tags=["locomotion"], params=MyParams)
def track_lin_vel_xy_exp(robot: RobotHandle, params: MyParams) -> Tensor:
    ...
```

The editor auto-discovers all registered terms at startup -- no manual schema files.

---

## Repository Structure

```
myrl/
├── scripts/
│   ├── editor/                # WebUI (16 ES module files)
│   │   ├── index.html
│   │   ├── css/               # theme, layout, obs-graph, robot-viewer, servers, modal
│   │   └── js/                # app, state, api, sidebar, reward, obs-graph, algo,
│   │       │                  #   training, sse, fleet, robot-viewer
│   │       └── webgpu/        # stl-parser, gpu-pipeline, orbit-camera,
│   │                          #   urdf-transform, link-picker, sensor-gizmos
│   ├── train_manager.py       # HTTP API server (stdlib, ~1100 lines)
│   ├── fleet_manager.py       # Remote server management
│   ├── start_editor.sh        # One-command launcher
│   ├── train.py / play.py     # Isaac Lab entry points
│   └── deploy/                # Remote deployment scripts
├── src/myrl/
│   ├── core/
│   │   ├── compat/
│   │   │   ├── views/         # RobotHandle, JointView, BodyView, ContactView,
│   │   │   │                  #   DepthCameraView, HeightScanView, ForceSensorView
│   │   │   ├── sensors/       # Protocols + IsaacLab drivers
│   │   │   └── backends/      # IsaacLabBackend, MuJoCoBackend
│   │   ├── robot/             # URDF parser (stdlib xml.etree)
│   │   ├── task/              # ObsBuilder, RewardBuilder, RewardLibrary,
│   │   │                      #   ObsPipelineV2, PipelineCompiler
│   │   ├── obs/               # ObsHistoryManager
│   │   ├── sim_server/        # MuJoCo TCP server, ROS2 bridge
│   │   └── databus/           # Pub/sub signal bus for debugging
│   ├── tasks/                 # Task registration (gym.register)
│   ├── assets/                # Asset resolver
│   ├── registry/              # Content-addressable run manifests
│   └── logging/               # JSONL + wandb + SSE sinks
├── assets/
│   ├── robots/g1/             # URDF + STL meshes (8 variants, 84 meshes)
│   ├── experiments/           # Experiment YAML definitions
│   ├── sensor_cfgs/           # Sensor manifests per robot
│   ├── obs_pipelines/         # Obs pipeline v2 YAML (block graph)
│   ├── reward_pipelines/      # Reward term + weight YAML
│   └── algo_cfgs/             # Algorithm parameter YAML
├── docker/                    # Isaac Lab 2.3.2 container
│   ├── Dockerfile.dev
│   ├── compose.yaml
│   └── entrypoint.sh
└── third_party/               # Internalized upstream (instinct_rl, instinctlab, isaaclab)
```

---

## Logging & Monitoring

| Sink | Activation | Output |
|------|-----------|--------|
| JSONL | Always on | `{log_dir}/metrics.jsonl` |
| Weights & Biases | `--wandb` | `wandb.init(sync_tensorboard=True)` |
| SSE Log Server | `--log_server_port 7000` | HTTP stream at `/stream` |

The editor's **Run** tab consumes the SSE stream for real-time console and metrics.

---

## Sim-to-Sim / Sim-to-Real

```
Isaac Lab (train) ──checkpoint──> MuJoCo (sim2sim) ──ROS2──> Real Robot
                                     ↑ TCP                    ↑ ROS2 topics
                              MuJoCoSimServer             Ros2Bridge
```

The MuJoCo backend implements the same `VecEnv` interface -- a trained policy loads
without code changes. The ROS2 bridge routes observations over topics and
rewards/dones over TCP, with `ObsHistoryManager` handling frame buffering.

---

## Key Design Principles

1. **View boundary** -- task code never imports sim backend types directly
2. **Sensor driver model** -- Protocol-based, swappable per backend
3. **Declarative configuration** -- experiments, sensors, obs, rewards all in YAML
4. **Visual-first editing** -- browser editor is the primary config interface
5. **Reproducibility** -- content-addressable manifests, experiment packaging (.myrlpkg)
6. **Zero-dep browser tools** -- no npm, no CDN, no build step
7. **Train anywhere** -- same experiment runs on local Docker, remote GPU, or fleet

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| A | instinctlab wrapper training pipeline | Done |
| B | IsaacLabBackend (own compat layer) | Implemented, pending validation |
| C | Asset resolver + myrl task registration | Done |
| D | MuJoCo TCP backend + ROS2 bridge | Done (23/23 + 32/32 tests) |
| E | Reward library + transform system | Done |
| F | Three-layer logging (JSONL/wandb/SSE) | Done |
| G | Remote training management | Done |
| H | Asset/experiment management system | Done |
| I | Editor WebUI + Oscilloscope | Done |
| J | Oscilloscope v2 + probe refactor | Done |
| K | Remote deploy + ablation | Done |
| L | Fleet manager | Done |
| M | Visual config editors (reward/algo/obs) | Done |
| N | Obs pipeline v2 schema + compiler | Done |
| O | Editor modularization + obs graph enhancement | Done |
| P | Reward schema auto-discovery + tree sidebar | Done |
| Q | Sensor views + URDF parser + WebGPU viewer | Done |

---

## License

Private repository. All rights reserved.
