# RAMMP-CuRobo

GPU motion planning (NVIDIA cuRobo) for the RAMMP Kinova Gen3 7-DoF, as a
standalone planning service any RAMMP module can adopt: **in** goes an end
position (tool pose or joint goal), **out** comes the collision-free,
time-parameterized joint trajectory for the arm to execute. The planner
never owns the arm — execution is the caller's, or the optional
safety-gated executor's, which drives the kinova_arm_ros2 driver's
`/execute_joint_trajectory` action (sim and real through the same action).

```
core/                     Layer 1 — pip package `rammp-curobo`: pure-Python
                          cuRobo wrapper, NO ROS imports (configs baked in)
rammp_curobo_interfaces/  ROS 2 action/srv definitions (dependency-free)
rammp_curobo_ros/         Layer 2 — planner node + safety-gated executor
                          + tour_demo (the showcase: 4 random points, one
                          merged trajectory — 0.32 speed until the
                          driver's reference cap rises)
examples/                 plan_only.py (no ROS) / plan_and_execute.py
scripts/                  config baking, live sim checks
docker/                   the planning service as a container (Jetson/JP6)
docs/HARDWARE_BRINGUP.md  the real-arm runbook — READ BEFORE TOUCHING HARDWARE
```

> **Hardware safety, non-negotiable:** a human holds the physical e-stop
> during ALL hardware runs. Execution is opt-in at three separate layers
> (node `execute:=true`, example/demo `--execute`, typed confirmation),
> defaults to 25% speed (32% for the tour demo), and every plan is
> re-validated against limits and the arm's live state before anything
> reaches the driver.

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

Execution goes through **rammp-org/kinova_arm_ros2** (`kinova_arm_node`,
the thin ROS 2 shell over the kinova-gen3-driver 1 kHz RT core — it
replaced the ros2_kortex stack here 2026-08-14). On this Jetson it is
built at `/tmp/kinova-ros2-ws` by the driver author's rsync dev loop
(abra has no GitHub key; note /tmp does not survive reboots). Source its
`install/setup.zsh` under ours when executing; building
`rammp_curobo_ros` itself does NOT need it, and neither does the
planning-only Docker image (the driver interface is a lazy, runtime-only
dependency).

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
# sim (terminal 1) — NOTE: the driver's --sim is a STATIC transport stub
# (measured q never moves): right for exercising gates/comms, no motion.
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh && source /tmp/kinova-ros2-ws/install/setup.zsh
cd /tmp/kinova-ros2-ws/src/kinova-gen3-driver
ros2 run kinova_arm_ros2 kinova_arm_node --sim --urdf models/gen3_7dof_2f85.urdf
# real arm instead: same node with --ip 192.168.1.10, ATTENDED ONLY, per
# docs/HARDWARE_BRINGUP.md — human on the e-stop
```

Then arm the planner and run the demo (terminals sourced the same way,
plus this repo's `install/setup.zsh`):

```bash
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml execute:=true
ros2 run rammp_curobo_ros tour_demo --execute   # 4 random points, one
                                                # merged trajectory, lap time
```

Without `--execute`, `tour_demo` pre-plans and prints the tour dry. Its
default `--speed` is 0.32 — the highest scale that keeps the driver's
divergence guard armed under its 0.5 rad/s reference cap; faster scales
drop the guard and then lag their timestamps (raise it when the driver's
`max_ref_speed` goes up).
`scripts/sim_execution_checks.py` verifies the refusal gates, cancel-hold,
and the no-motion arrival catch against the sim stub.

## Safety model (execution gates)

Every `ExecuteTrajectory` goal must pass, in order: node `execute`
parameter true → speed scale in (0, 1] (default 0.25, exact time dilation
— plans are never sped up) → joint names match → finite, within position
AND velocity limits → monotonic timing and step-continuity (rejects the
stale-buffer/discontinuity failure mode) → arm's live `/joint_states`
within 0.05 rad of the trajectory start (stale plans refused) → driver
accepts. Cancel at any time cancels the driver goal — the arm stops and
holds; after
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
| nodes can't see each other's topics | `ROS_LOCALHOST_ONLY=1` must be exported in EVERY shell (non-interactive shells skip `~/.zshrc` — export it explicitly) |
| `/joint_states` looks silent / planner says "no fresh joint state" | kinova_arm_node publishes BEST-EFFORT — CLI needs `ros2 topic echo --qos-reliability best_effort /joint_states`; our nodes subscribe sensor-data QoS already |
| execution succeeds per driver but our node reports TRACKING FAILURE | the driver completes on its timer with no goal check; the arm lagged (speed_scale too high for its 0.5 rad/s reference cap) or never moved — see docs/HARDWARE_BRINGUP.md faults section |
| `update_world` seems ignored / obstacles missing | cuRobo v0.7.8: cylinders/spheres in a WorldConfig are silently dropped (cuboids only), and an empty world silently keeps the previous one — the library guards both, custom worlds go in as boxes |
| `AttributeError: wp.torch` in mesh collision | newer warp needs explicit `import warp.torch` — the library does this; if embedding cuRobo yourself, copy that |
| `ros2 topic echo` prints "A message was lost!!!" | benign QoS depth artifact of echo on a high-rate topic |

## Integrating from another RAMMP module

See **INTEGRATION.md** — 5 lines for pure-Python planning, ~15 for
plan-over-ROS-actions, and the Docker service.
