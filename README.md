# RAMMP-CuRobo

GPU motion planning (NVIDIA cuRobo) + safe execution for the RAMMP Kinova
Gen3 7-DoF, as a standalone repo any RAMMP module can adopt with a few
lines. Planning runs identically against the MuJoCo simulation and the real
arm's `ros2_kortex` bringup — same controller names, same action, same code.

```
core/                     Layer 1 — pip package `rammp-curobo`: pure-Python
                          cuRobo wrapper, NO ROS imports (configs baked in)
rammp_curobo_interfaces/  ROS 2 action/srv definitions (dependency-free)
rammp_curobo_ros/         Layer 2 — planner node + safety-gated executor
examples/                 plan_only.py (no ROS) / plan_and_execute.py
scripts/                  config baking, live sim checks
docs/HARDWARE_BRINGUP.md  the real-arm runbook — READ BEFORE TOUCHING HARDWARE
```

> **Hardware safety, non-negotiable:** a human holds the physical e-stop
> during ALL hardware runs. Execution is opt-in at three separate layers
> (node `execute:=true`, example `--execute`, typed `yes`), starts at 25%
> speed or less, and every plan is re-validated against limits and the
> arm's live state before anything reaches the controller.

## Install (Jetson AGX Orin)

Verified on: JetPack 6.2.2 (L4T R36.5.0), CUDA 12.6.11, Ubuntu 22.04,
ROS 2 Humble, Python 3.10. cuRobo's install is the long pole — pin
everything; do not "upgrade" any of it.

### 1. PyTorch (Jetson CUDA wheels)

```bash
python3 -m pip install --no-cache torch torchvision \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Known-good: **torch 2.10.0 / torchvision 0.25.0** (this wheel's cuSOLVER
even runs cuRobo's joint-space trajopt, which older jp6 wheels could not —
the planner auto-falls-back if yours can't).

### 2. cuRobo — PINNED v0.7.8, built from source

v0.8.0 is an API rewrite. Never upgrade past v0.7.8 here.

```bash
sudo apt-get install -y git-lfs && git lfs install
git clone https://github.com/NVlabs/curobo.git && cd curobo
git checkout tags/v0.7.8
export TORCH_CUDA_ARCH_LIST="8.7+PTX"     # Orin = sm_87
export MAX_JOBS=4
python3 -m pip install -U "packaging>=24.1" "setuptools>=70,<80"
python3 -m pip install -e . --no-build-isolation   # 20-40 min on the Orin
python3 -m pip install "warp-lang==1.5.1"  # v0.7.8 needs warp 1.5.x
```

(On this lab's Jetson a v0.7.8 checkout already lives at
`~/RAMMP-Kinova/ros2_ws/curobo` — installed and working; skip this step.)

### 3. This repo

```bash
git clone <this-repo> ~/RAMMP-CuRobo && cd ~/RAMMP-CuRobo
python3 -m pip install --user --no-build-isolation -e ./core
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
    --packages-select rammp_curobo_interfaces rammp_curobo_ros
```

### 4. Arm driver workspace (for execution)

Execution goes through ros2_control. Sim and real both come from the
RAMMP-Kinova workspace (`~/RAMMP-Kinova/ros2_ws`), set up once via its
`scripts/setup_ros2_kortex.sh` (clones Kinovarobotics/ros2_kortex@humble +
pinned deps, builds `kortex_bringup` and `mujoco_sim`). Follow that repo's
README if starting fresh.

Performance: `sudo nvpmodel -m 0 && sudo jetson_clocks` before demos.

## Smoke test (no hardware, no ROS)

```bash
python3 -m pytest core/tests -q
```

10 offline tests run anywhere; the 7 GPU tests plan real collision-free
trajectories in the sim-kitchen world (first run compiles kernels — takes
a minute or two; ~20 s warm). Or plan one motion directly:

```bash
python3 examples/plan_only.py --joints 0.3 0.262 3.142 -2.269 0.0 0.960 1.571
```

## Run against the simulation

```bash
# terminal 1 — RAMMP-Kinova's sim (physics + ros2_control + controllers)
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/RAMMP-Kinova/ros2_ws/install/setup.bash
ros2 launch mujoco_sim mujoco_bringup.launch.py

# terminal 2 — planner node (execute enabled: it's a sim)
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash && source ~/RAMMP-Kinova/ros2_ws/install/setup.bash
source ~/RAMMP-CuRobo/install/setup.bash
ros2 launch rammp_curobo_ros planner.launch.py execute:=true use_sim_time:=true

# terminal 3 — plan, preview, confirm, execute at 25% speed
source ... (as terminal 2)
python3 examples/plan_and_execute.py \
    --joints 0.2 0.262 3.142 -2.269 0.0 0.960 1.571 --execute --speed-scale 0.25
```

Without `--execute` the same command is a pure dry-run (prints the
trajectory, nothing moves). `scripts/sim_execution_checks.py` additionally
verifies the refusal gates and mid-motion cancel against the live sim.

## Run on the real arm

Follow **docs/HARDWARE_BRINGUP.md** step by step — coordination (three
stacks can claim this arm; only one may run), measuring
`world_real_bench.yaml`, the kortex bringup command with its gotchas, the
dry-run gate, the first small joint move at 15% speed, and the required
abort drill. Human on the e-stop throughout.

## Safety model (execution gates)

Every `ExecuteTrajectory` goal must pass, in order: node `execute`
parameter true → speed scale in (0, 1] (default 0.25, exact time dilation
— plans are never sped up) → joint names match → finite, within position
AND velocity limits → monotonic timing and step-continuity (rejects the
stale-buffer/discontinuity failure mode) → arm's live `/joint_states`
within 0.05 rad of the trajectory start (stale plans refused) → controller
accepts. Cancel at any time stops the controller and holds position; after
completion the executor verifies arrival within 0.08 rad.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `pip install -e` fails (`build_editable` hook / permission denied) | old pip + isolated build env; use `python3 -m pip install --user --no-build-isolation -e ./core` |
| pytest crashes collecting (`No module named '_pytest.scope'`) | user-site anyio plugin vs system pytest; repo `pytest.ini` disables it (`-p no:anyio`) — run pytest from the repo root |
| first plan takes minutes | one-time CUDA kernel compile (warmup); subsequent runs ~20 s init, ~0.2-2 s per plan |
| `plan_single_js` fails / `DT_EXCEPTION` on older Jetson wheels | known torch-wheel cuSOLVER gap; `joint_space_method: auto` falls back to FK-pose planning automatically. Keep `enable_graph: false` on Jetson always |
| plan succeeds but joints differ from a joint goal | FK-pose fallback reached the POSE via another joint family — check `goal_mismatch_rad`; the example refuses >0.5 rad without `--allow-mismatch` |
| execution refused: "arm is X rad from the trajectory start" | plan is stale (arm moved since planning) — re-plan; this gate is intentional |
| node warns about SIM world without sim time | you're (probably) on the real arm with the kitchen world — relaunch with `world:=world_real_bench.yaml` (measured!) |
| nodes can't see each other's topics | `ROS_LOCALHOST_ONLY=1` must be exported in EVERY shell (non-interactive shells skip `~/.zshrc` — export explicitly, the sim launcher does) |
| both bringups fight / controllers flap | MuJoCo sim and kortex bringup both claim `/controller_manager` — run exactly one |
| `update_world` seems ignored / obstacles missing | cuRobo v0.7.8: cylinders/spheres in a WorldConfig are silently dropped (cuboids only), and an empty world silently keeps the previous one — the library guards both, custom worlds go in as boxes |
| `AttributeError: wp.torch` in mesh collision | newer warp needs explicit `import warp.torch` — the library does this; if embedding cuRobo yourself, copy that |
| `ros2 topic echo` prints "A message was lost!!!" | benign QoS depth artifact of echo on a 500 Hz topic |
| arm won't move, controller error mentions tolerances | check speed scale isn't absurdly low (goal-time), and that `arm_driver`'s persisted 25 deg/s soft limit isn't what you're seeing |

## Integrating from another RAMMP module

See **INTEGRATION.md** — 5 lines for pure-Python planning, ~15 for
plan+execute over ROS actions.
