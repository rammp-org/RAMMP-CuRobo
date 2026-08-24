# RAMMP-CuRobo — agent notes

Standalone cuRobo planning for the RAMMP Kinova Gen3 7-DoF (+ Robotiq
2F-85): a planning SERVICE — end position in, collision-free joint
trajectory out — plus showcases (`tour_demo`, the `dance_demo`
easter egg, `tag_follow`, `sweep_demo`, and the `seeker`: "go to the
bottle" via YOLO on the wrist camera — a continuous perceive/decide/act
loop (blocking cuRobo hops, survey pose when lost; helpers in
seek_core.py; never chases a lifted target)), a safety-gated
executor for this bench, and live spatial awareness (the `cameras` node:
wrist-D405 depth → cuboid obstacles → planner world at ~2 Hz; TF
extrinsics, no calibration; plus the calibrated fixed Orbbec). Parts:
`core/` (pip `rammp-curobo`, pure Python, NO ROS imports — keep it that
way; perception.py is the pure pipeline), `rammp_curobo_interfaces/`
(rosidl, dependency-free by policy), `rammp_curobo_ros/` (ament_python:
planner_node + cameras + the demos), `docker/` (the service
containerized for other RAMMP codebases). This repo deliberately
contains and launches NO arm driver — bringup is the RAMMP-Kinova
workspace's, execution ownership is the caller's. The 2026-08-14 cleanup removed the scan/palm demos; the
scan pipeline was deliberately REVIVED 2026-08-17 as `perception.py` +
`cameras` (the palm demo remains history-only).

## Environment (this lab's Jetson AGX Orin, 192.168.1.11)

- JetPack 6.2.2 (L4T R36.5.0), CUDA 12.6.11, Ubuntu 22.04, ROS 2 Humble.
- torch 2.10.0 / torchvision 0.25.0 from https://pypi.jetson-ai-lab.io/jp6/cu126.
- cuRobo **PINNED v0.7.8** (v0.8 is an API rewrite — never upgrade),
  pip-editable from `~/RAMMP-Kinova/ros2_ws/curobo`; warp-lang 1.5.1.
- Arm: Kinova Gen3 7-DoF at **192.168.1.10**. ros2_control stack (sim
  MuJoCo + real ros2_kortex) lives in `~/RAMMP-Kinova/ros2_ws` — source it
  before ours for execution.
- `ROS_LOCALHOST_ONLY=1` in EVERY shell that runs ROS here. Non-interactive
  shells skip ~/.zshrc — export it explicitly or nodes won't discover each
  other (field-verified split-DDS failure).
- THREE mutually exclusive arm stacks exist on this machine: this repo via
  ros2_kortex, Demo-Software's `arm_driver` (Kortex API direct), and the
  custom C++ driver in `~/atdev/kinova-gen3-driver`. One at a time, ever.

## Invariants (hard-won in RAMMP-Kinova; verified still true here)

- cuRobo v0.7.8: WorldConfig cylinders/spheres silently DROPPED (cuboids
  only); `update_world` with zero cuboids silently keeps the old world;
  more cuboids than `collision_cache_obb` raises. `world.py` guards all.
- Collision re-checking: `validate_goal_msg` checks names/limits/velocity/
  timing/continuity/start-match and NEVER collision, so a trajectory
  planned before an obstacle appeared passes every gate. `~/check_trajectory`
  (read-only) is the only thing that catches it — a reactive client must
  poll it. Its verdict flips at `world_padding` (MEASURED 0.020 m), NOT
  `collision_activation_distance` (0.03): that knob feeds the IK/trajopt
  COST terms and never reaches `check_constraints`.
- `planner_node` OWNS SIGINT (`SignalHandlerOptions.NO`). rclpy's default
  handler kills the context synchronously, and the cancel path is
  delivered by polling `is_cancel_requested` from the executor thread —
  so with the default handler Ctrl+C during motion did not stop the arm,
  it stopped watching it. `scripts/abort_checks.py` is the tripwire; never
  hand SIGINT back to rclpy in a node that can command motion.
- The arm's SELF-MODEL for perception is cuRobo's own collision spheres
  (`self_model_gen3_2f85.yaml`, baked — never hand-edit), placed by TF.
  Arm-link frames are identical between cuRobo's URDF and ros2_kortex's
  (verified, all 8 joints); the gripper is expressed in end_effector_link
  (the URDFs attach it with different yaw and names). `self_radius` is a
  MARGIN over the sphere radii, not a capsule radius.
- Camera-pose error is solved OFF THE ARM (SelfRegistrar: coarse grid +
  point-to-plane ICP on the camera-facing surface of the spheres, raw
  frames BEFORE masking). Never estimate it from perceived boxes: the
  self-filter leaves only the fringe of a displaced arm, and a
  box-centroid estimate measures the fringe (over-corrected 5 cm on the
  bench). Local ICP alone is not enough either — errors beyond one arm
  radius push the drawn arm into its model and the far surface is a
  perfectly good local minimum.
- `colcon build --symlink-install` symlinks `config/*.yaml` but COPIES the
  Python sources into `build/`. pytest reads the source tree, `ros2 run`
  reads the copy — always rebuild before running a node, or you will test
  code that is not running.
- Publish/consume ONLY the trimmed interpolated plan — result buffers are
  padded; a stale tail caused violent motion in the field. `validate.py`'s
  continuity check is the tripwire; never bypass it.
- `enable_graph` stays FALSE on Jetson (`enable_graph_attempt=None` matters:
  cuRobo silently auto-enables the graph after 3 failed attempts). The
  graph planner needs torch.svd/cuSOLVER the Jetson wheels historically
  lack; linalg is routed to magma at init.
- `plan_single_js` WORKS on torch 2.10.0 (verified 2026-08-11, mismatch
  5e-7) — on older wheels it died; `joint_space_method: auto` handles both.
  The FK-pose fallback reaches the POSE but may flip joint families —
  `goal_mismatch_rad` is the guard, the example refuses >0.5 rad.
- cuRobo `velocity_scale` stays 1.0. Slow execution = time dilation in
  `retime.py` / the executor (t/s, v*s, a*s²), never plan-time scaling.
- Quaternions: cuRobo wxyz, ROS xyzw. Library API takes xyzw by default.
- Robot config = `configs/robot_gen3_2f85.yaml`, GENERATED by
  `scripts/bake_robot_config.py` (audit-tuned spheres, gripper shell,
  retract=home). Regenerate, don't hand-edit spheres.
- retract_config must stay in the HOME elbow family (joint_3 ≈ π) — the
  stock retract seeds the opposite family and every crossing plan winds.
- `ee_link` is `tool_frame`: 0.120 m beyond the wrist flange (≈ fingertip
  midpoint). Tool corrections (spin 90°, tip 21 mm) are sim-measured and
  ship DISABLED in gen3.yaml until re-measured on the real arm.
- Sim starts at q=0 (not home). Both sim and real expose the SAME
  controller names; never run both bringups (shared /controller_manager).
- Perceived world: UpdateWorldBoxes REPLACES the perceived set every call
  (never accumulates), always merged on a sticky baseline world — so
  updates are never empty and the table survives. Ignore region lives in
  the cameras node, not the planner. Wrist-camera decay is
  FRUSTUM-SCOPED — a voxel is forgotten only when the camera provably
  sees through it; out-of-view voxels are remembered. Don't "fix" the
  accumulator back to global decay (a narrow-FOV wrist camera would
  evaporate the world every time it looks away). Depth frames pair with
  TF at THEIR stamp and are dropped while the camera moves — never
  "latest" TF (field-verified time-skew class). Primary camera = wrist
  D405 (TF extrinsics, no calibration; mount YAML is photo-estimated,
  validated by the §6 acceptance). A fixed camera is calibrated by
  scripts/calibrate_orbbec.py (tag-on-gripper eye-to-hand; make_tag.py
  prints the 60 mm tag) — what produced camera_orbbec_bench.yaml;
  scripts/calibrate_camera_extrinsics.py (fingertip-click Kabsch, no
  fiducials) is the fallback. Startup self-registration (SelfRegistrar
  above) absorbs residual translation error. Never hand-edit mounts.

## Safety (do not weaken)

Execution gates live in `rammp_curobo_ros/planner_node.py` (`_execute_cb`:
execute param, speed clamp) + `executor.py` (the rest) and are all
verified live: execute param (default false) → speed clamp (0,1] → name /
limit / continuity / monotonic-time checks → live start-state match →
cancel = controller stop+hold → arrival check. Typed gates survive only
on the interactive trio: tour_demo 'go' (default speed 1.0!), dance_demo
'dance' (0.4), the example's 'yes'. Everything else that moves — seeker,
tag_follow, sweep_demo execute:=true, calibrate_orbbec / prove_avoidance
/ go_home --execute — is autonomous once started BY OWNER DECISION
2026-08-19 (no typed gates, no countdowns; Ctrl+C stops and holds;
node/executor gates unchanged). Hardware runs follow
docs/HARDWARE_BRINGUP.md with a human on the physical e-stop; never
drive the real arm autonomously from an agent session.

## Build / test

```bash
python3 -m pip install --user --no-build-isolation -e ./core   # pip quirks: see README
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rammp_curobo_interfaces rammp_curobo_ros
python3 -m pytest core/tests -q          # offline + GPU smoke
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio
python3 scripts/sim_execution_checks.py  # live gates+abort, needs sim running
```

pytest needs the repo-root `pytest.ini` (`-p no:anyio` — user-site anyio
plugin clashes with system pytest). Style: Ruff v0.3.0 defaults
(pre-commit), PEP 8, mdformat/gfm for Markdown.
