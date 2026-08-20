# RAMMP-CuRobo

GPU motion planning (NVIDIA cuRobo) for the RAMMP Kinova Gen3 7-DoF, as a
standalone planning service any RAMMP module can adopt: **in** goes an end
position (tool pose or joint goal), **out** comes the collision-free,
time-parameterized joint trajectory for the arm to execute. The planner
never owns the arm — execution is the caller's (or the optional
safety-gated executor's, pointed at an existing ros2_control bringup).

```
core/                     Layer 1 — pip package `rammp-curobo`: pure-Python
                          cuRobo wrapper, NO ROS imports (configs baked in)
rammp_curobo_interfaces/  ROS 2 action/srv definitions (dependency-free)
rammp_curobo_ros/         Layer 2 — planner node + safety-gated executor
                          + tour_demo (the showcase: 4 random points, one
                          merged full-speed trajectory)
examples/                 plan_only.py (no ROS) / plan_and_execute.py
scripts/                  config baking, live sim checks
docker/                   the planning service as a container (Jetson/JP6)
docs/HARDWARE_BRINGUP.md  the real-arm runbook — READ BEFORE TOUCHING HARDWARE
```

> **Hardware safety, non-negotiable:** a human holds the physical e-stop
> during ALL hardware runs. Execution is opt-in at three separate layers
> (node `execute:=true`, example/demo `--execute`, typed confirmation —
> except the `seeker`, which by owner decision moves autonomously once
> given a target), defaults to 25% speed
> (`tour_demo` alone runs full-speed, behind its own all-caps warning
> and typed 'go'), and every plan is re-validated against limits and
> the arm's live state before anything reaches the controller.

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
`/rammp_curobo/plan_to_joints` take a goal (optionally with explicit
`start_joints` — no `/joint_states` needed) and return the trajectory.
See **INTEGRATION.md** for client code, and **docker/** to run the same
thing as a container.

## Run with execution (this bench)

Start the arm side first — this repo never launches it:

```bash
# sim (terminal 1):
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh && source ~/RAMMP-Kinova/ros2_ws/install/setup.zsh
ros2 launch mujoco_sim mujoco_bringup.launch.py
# real arm instead: the kortex bringup per docs/HARDWARE_BRINGUP.md —
# human on the e-stop
```

Then arm the planner and run the demo (terminals sourced the same way,
plus this repo's `install/setup.zsh`):

```bash
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml execute:=true
ros2 run rammp_curobo_ros tour_demo --execute   # 4 random points, one
                                                # merged trajectory, lap time
```

Without `--execute`, `tour_demo` pre-plans and prints the tour dry.
`scripts/sim_execution_checks.py` additionally verifies the refusal gates
and mid-motion cancel against the live sim.

## Easter egg: dance

`ros2 run rammp_curobo_ros dance_demo` choreographs randomized rounds of
bobs, sways, circles, wrist twists and shimmies (dry-run by default;
`--execute` + typed `dance` to move, default 40% speed). Same skeleton
and gates as the tour: safe-box waypoints, chained pre-planning, ONE
merged trajectory per round, Ctrl+C = hold. First run:
`--rounds 1 --moves 4 --speed 0.25`, workspace clear, hand on the e-stop.

## Seeker ("go to the bottle")

`ros2 run rammp_curobo_ros seeker --ros-args -p target:="go to the bottle"`
— a continuous perceive-decide-act controller, not a script. One loop:
YOLO on the wrist D405 finds the named object (80 COCO classes +
synonyms; weights `~/yolo11s-seg.pt`, never downloaded), 3 agreeing
frames localize it, and cuRobo plans straight to it through the live
perceived world — pre-grasp, then grasp. There is no creep-closer
phase: `tool_frame` sits 12 cm ahead of the flange, so poses part-way
to the object are unreachable while poses AT it are fine. Move the
object and it re-plans; hide it and it returns to a survey pose; lift
it and it holds (never chases a hand); kill the camera and it stops.
Retarget any time via the `/seeker/set_target` service ("" idles);
status on `/seeker/status`; live detection view on `:8767`. Autonomous
once targeted (owner decision 2026-08-19) — the planner's `execute`
param, every executor gate, ≤0.25 speed, and the human on the e-stop
are the safety layers. The D405 driver needs `align_depth.enable:=true`.

**Grasping.** Instead of guessing a pose, the seeker asks
**GraspGenX** (NVlabs, Apache-2.0, runs on this Jetson — ~1.3 s and
~660 MiB per object) for real 6-DoF grasps on the masked depth, then
plans pre-grasp → grasp with cuRobo and closes the gripper. Start its
server once, out-of-tree:

```bash
cd ~/GraspGenX && ~/graspgen_venv/bin/python client-server/graspgenx_server.py \
    --config ext/graspgenx_checkpoints/release --assets_dir assets \
    --port 5556 --default_gripper robotiq_2f_85
```

We speak its ZMQ protocol directly (`grasps.py`) — no GraspGenX import,
so the socket being down just means `grasp:=false` behaviour. Its grasp
frame coincides with our `end_effector_link`, so the handoff is one
`tool_offset` push along the grasp's approach axis; that offset
(0.120 our `tool_frame` vs 0.136 GraspGenX's fingertip) is a parameter.

## Perceived world (cameras)

The `cameras` node gives the planner live spatial awareness from the
**wrist-mounted D405**: depth is deprojected to base_link using TF *at
each frame's timestamp* (no extrinsic calibration — the camera rides the
arm; frames captured mid-motion are dropped), the arm erases itself via a
capsule self-filter, a voxel accumulator with hysteresis kills flicker,
and surviving clusters become named cuboid obstacles pushed to the
planner at ~2 Hz — merged on top of the static `world_real_bench.yaml`
baseline. Decay is **frustum-scoped**: a voxel is only forgotten when the
camera provably sees through its location, so the world the wrist has
mapped survives when it looks away. Planning-only: the node never
commands motion.

```bash
# terminal A — the camera driver
ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 camera_name:=d405
# terminal B — perception (defaults to the wrist config)
ros2 run rammp_curobo_ros cameras
```

No monitor needed: the node serves a **live browser view** at
`http://<jetson>:8766/` (camera feed with the perceived boxes drawn on
it, plus gating/ignore status; `view:=false` disables). Or watch
`/cameras/world_markers` in RViz — a box placed in front of the
gripper appears within ~2 s (±3 cm validates the bracket mount), survives
the wrist looking away, and fades ~3 s after the camera sees its spot
empty. The wrist only maps where it has looked: plans through never-seen
space rely on the baseline world. The camera YAML also declares a
`sensor_params` contract the node pushes to the driver at startup —
for the D405 (passive stereo, no projector) that's the High Accuracy
preset, which makes textureless surfaces return holes (unknown, safe)
instead of hallucinated depth (phantom obstacles — field-bitten). `scripts/cameras_checks.py` verifies
the service loop live; `/cameras/set_ignore_region` masks the object
you're about to grasp (see INTEGRATION.md). A fixed bench camera can be
added via the `cameras` param after running
`scripts/calibrate_camera_extrinsics.py` (browser-click, no fiducials).

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
| nodes can't see each other's topics | `ROS_LOCALHOST_ONLY=1` must be exported in EVERY shell (non-interactive shells skip `~/.zshrc` — export explicitly; RAMMP-Kinova's `tools/launch_stack.zsh` does) |
| both bringups fight / controllers flap | MuJoCo sim and kortex bringup both claim `/controller_manager` — run exactly one |
| `update_world` seems ignored / obstacles missing | cuRobo v0.7.8: cylinders/spheres in a WorldConfig are silently dropped (cuboids only), and an empty world silently keeps the previous one — the library guards both, custom worlds go in as boxes |
| `AttributeError: wp.torch` in mesh collision | newer warp needs explicit `import warp.torch` — the library does this; if embedding cuRobo yourself, copy that |
| `ros2 topic echo` prints "A message was lost!!!" | benign QoS depth artifact of echo on a 500 Hz topic |
| arm won't move, controller error mentions tolerances | check speed scale isn't absurdly low (goal-time), and that `arm_driver`'s persisted 25 deg/s soft limit isn't what you're seeing |

## Integrating from another RAMMP module

See **INTEGRATION.md** — 5 lines for pure-Python planning, ~15 for
plan-over-ROS-actions, and the Docker service.
