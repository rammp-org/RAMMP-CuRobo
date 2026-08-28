# RAMMP-CuRobo

GPU motion planning (NVIDIA cuRobo) for the RAMMP Kinova Gen3 7-DoF, as a
standalone planning service any RAMMP module can adopt: **in** goes an end
start configuration and an end position (tool pose or joint goal), **out**
comes the collision-free, time-parameterized joint trajectory for the arm
to execute. The planner never touches the arm: it holds no driver, no
controller client and no `/joint_states` subscription, and execution is
entirely the caller's.

```
core/                     Layer 1 — pip package `rammp-curobo`: pure-Python
                          cuRobo wrapper, NO ROS imports (configs baked in)
rammp_curobo_interfaces/  ROS 2 action/srv definitions (dependency-free)
rammp_curobo_ros/         Layer 2 — plan-only planner node + tour_demo
                          (the showcase: 4 random points chain-planned
                          into one merged trajectory; nothing moves)
examples/                 plan_only.py (no ROS)
scripts/                  config baking
docker/                   the planning service as a container (Jetson/JP6)
docs/HARDWARE_BRINGUP.md  the real-arm runbook — READ BEFORE TOUCHING HARDWARE
```

> **Nothing in this repo can move the arm.** There is no executor, no
> controller client and no execute flag — a plan is data until whoever
> owns the robot chooses to run it, under their gates (on this bench,
> `kinova_arm_ros2`). A human still holds the physical e-stop during ALL
> hardware runs. See **issue #6** for why the boundary is drawn here.

## Install (Jetson AGX Orin)

Verified on: JetPack 6.2.2 (L4T R36.5.0), CUDA 12.6.11, Ubuntu 22.04,
ROS 2 Humble, Python 3.10. cuRobo's install is the long pole — pin
everything; do not "upgrade" any of it. (Prefer the container? See
**docker/** — same pins, one build.)

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
source /opt/ros/humble/setup.zsh
colcon build --symlink-install \
    --packages-select rammp_curobo_interfaces rammp_curobo_ros
```

### 4. Arm driver workspace (only for executing on this bench)

Execution goes through ros2_control. Sim and real both come from the
RAMMP-Kinova workspace (`~/RAMMP-Kinova/ros2_ws`), set up once via its
`scripts/setup_ros2_kortex.sh` (clones Kinovarobotics/ros2_kortex@humble +
pinned deps, builds `kortex_bringup` and `mujoco_sim`). This repo
deliberately does not include or launch any arm driver — one
`/controller_manager` per arm, owned elsewhere.

Performance: `sudo nvpmodel -m 0 && sudo jetson_clocks` before demos.

## Smoke test (no hardware, no ROS)

```bash
python3 -m pytest core/tests -q
```

The offline tests run anywhere; the GPU tests plan real collision-free
trajectories in the sim-kitchen world (first run compiles kernels — takes
a minute or two; ~20 s warm). Or plan one motion directly:

```bash
python3 examples/plan_only.py --joints 0.3 0.262 3.142 -2.269 0.0 0.960 1.571
```

> Shell note: the Jetson's default shell is **zsh** — source the `.zsh`
> setup files as shown. From bash, use the `.bash` variants instead
> (sourcing `setup.bash` from zsh fails with "no such file or directory:
> .../setup.sh").

## Run the planning service

```bash
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh && source ~/RAMMP-CuRobo/install/setup.zsh
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
```

That's the whole service: `/rammp_curobo/plan_to_pose` and
`/rammp_curobo/plan_to_joints` take a goal carrying the configuration to
plan from (`start_joints`, **required** — the planner has no view of any
arm) and return the trajectory. See **INTEGRATION.md** for client code,
and **docker/** to run the same thing as a container.

## The showcase

```bash
ros2 run rammp_curobo_ros tour_demo            # 4 random points
ros2 run rammp_curobo_ros tour_demo --points 6 --seed 3
```

`tour_demo` samples reachable targets in the frontal cone and chain-plans
`start → P1 → … → Pn → start`, each segment starting from the previous
segment's planned endpoint, then merges the lot into one continuous
trajectory and prints it. No arm need exist — the start configuration is
given (`--start`, defaulting to the planner's retract pose), so it doubles
as the repo's end-to-end smoke test: if it prints a tour, the node, the
world and chained planning all work.

The demo deliberately has no notion of "home". Where the arm belongs is
the arm layer's call, not the planner's.

## Execution is not here

This repo plans. It does not execute, and it cannot: there is no
`ExecuteTrajectory` action, no `FollowJointTrajectory` client, no
`/joint_states` subscription, no controller-manager client, no gripper
client, no `execute` parameter.

Executing a plan means handing the returned `trajectory_msgs/JointTrajectory`
to whoever owns the robot, with their gates and their e-stop discipline.
On this bench that is **`kinova_arm_ros2`**, whose `GoToEEPose` /
`GoToJointConfig` actions call this planner, supply the arm's measured `q`
as `start_joints`, validate what comes back, and run it through the
supervisor and driver with a path tolerance.

Two safety authorities with different rules is worse than either alone —
that, plus plans made from a joint state up to 2 s stale, is why the
executor that used to live here was removed (**issue #6**).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `pip install -e` fails (`build_editable` hook / permission denied) | old pip + isolated build env; use `python3 -m pip install --user --no-build-isolation -e ./core` |
| pytest crashes collecting (`No module named '_pytest.scope'`) | user-site anyio plugin vs system pytest; repo `pytest.ini` disables it (`-p no:anyio`) — run pytest from the repo root |
| first plan takes minutes | one-time CUDA kernel compile (warmup); subsequent runs ~20 s init, ~0.2-2 s per plan |
| `plan_single_js` fails / `DT_EXCEPTION` on older Jetson wheels | known torch-wheel cuSOLVER gap; `joint_space_method: auto` falls back to FK-pose planning automatically. Keep `enable_graph: false` on Jetson always |
| plan succeeds but joints differ from a joint goal | FK-pose fallback reached the POSE via another joint family — check `goal_mismatch_rad` before acting on it |
| goal aborted: "start_joints is required" | `start_joints` is not optional — the planner has no view of any arm, so the caller must send the configuration to plan from |
| node warns about SIM world without sim time | you're (probably) on the real arm with the kitchen world — relaunch with `world:=world_real_bench.yaml` (measured!) |
| nodes can't see each other's topics | `ROS_LOCALHOST_ONLY=1` must be exported in EVERY shell (non-interactive shells skip `~/.zshrc` — export explicitly; RAMMP-Kinova's `tools/launch_stack.zsh` does) |
| both bringups fight / controllers flap | MuJoCo sim and kortex bringup both claim `/controller_manager` — run exactly one |
| `update_world` seems ignored / obstacles missing | cuRobo v0.7.8: cylinders/spheres in a WorldConfig are silently dropped (cuboids only), and an empty world silently keeps the previous one — the library guards both, custom worlds go in as boxes |
| `AttributeError: wp.torch` in mesh collision | newer warp needs explicit `import warp.torch` — the library does this; if embedding cuRobo yourself, copy that |
| `ros2 topic echo` prints "A message was lost!!!" | benign QoS depth artifact of echo on a 500 Hz topic |

## Integrating from another RAMMP module

See **INTEGRATION.md** — 5 lines for pure-Python planning, ~15 for
plan-over-ROS-actions, and the Docker service.
