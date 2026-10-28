# RAMMP-CuRobo — agent notes

Standalone cuRobo planning for the RAMMP Kinova Gen3 7-DoF (+ Robotiq
2F-85): a planning SERVICE — start config + end position in, collision-free
joint trajectory out — plus one plan-only showcase (`tour_demo`). Parts:
`core/` (pip `rammp-curobo`, pure Python, NO ROS imports — keep it that
way), `rammp_curobo_interfaces/` (rosidl, dependency-free by policy),
`rammp_curobo_ros/` (ament_python node), `docker/` (the service
containerized for other RAMMP codebases), `sim/mebot_sim/` (the MEBot
wheelchair sim stack, ROS 2 + MuJoCo; not part of the package, and
COLCON_IGNOREd like `core/`).

**The robot model is the arm's own.** No URDF ships here: the node waits
(RELIABLE + TRANSIENT_LOCAL, `robot_description_timeout_s`, default 30 s)
for the URDF `arm_state_publisher` latches on `/arm/robot_description`,
writes it to a temp file and hands `core` the path
(`CuRoboPlanner.from_config(..., urdf_path=path)`; `core` stays file-based,
and loading the baked config without one is refused). No model in time =
a FATAL naming `arm_state_publisher` and exit 1. The model is read ONCE, at
startup: a later, different model on the topic is never used (the node logs
an ERROR each time one arrives), so restart the planner whenever the robot's
mount changes. Planning is in the model's root,
`arm_mount` (level with the chair, not with gravity), with `ee_link`
`arm_tool_frame` and joints `arm_joint_1..7`; every returned trajectory is
stamped `header.frame_id: arm_mount`, joint plans included
(kinova-gen3-ros2's `GoToEEPose` refuses any pose plan not stamped
`arm_mount` (#47), on every mount).

**This repo NEVER touches the arm** (issue #6, 2026-08-24): outside `sim/`
there is no executor, no `/joint_states` subscription, no controller or
controller-manager client, no gripper client, no `execute` parameter, and
no arm-package dependency in a `package.xml`. `start_joints` is REQUIRED on
both plan actions — the caller owns the robot and its state. The consumer
is `kinova_arm_ros2`, which plans here and executes under its own gates;
the dependency arrow points only that way. Do not add an edge back. The
sim, `sim/mebot_sim`, is a test kit outside the package: it depends on
`rammp_arm_interfaces`, its simulated arm serves the driver's names, and
its client subscribes `/joint_states` and sends `/execute_joint_trajectory`
goals, to the sim arm only and never on the bench's domain, 0 (see
Safety). None of it may leave `sim/`.

Removed deliberately, git history has it — don't reintroduce casually: the
camera-scanning and palm-demo code (2026-08-14), and `executor.py`,
`ExecuteTrajectory.action`, `plan_and_execute.py`, `sim_execution_checks.py`
plus the gripper services (2026-08-24, issue #6).

## Environment (this lab's Jetson AGX Orin, 192.168.1.11)

- JetPack 6.2.2 (L4T R36.5.0), CUDA 12.6.11, Ubuntu 22.04, ROS 2 Humble.
- torch 2.10.0 / torchvision 0.25.0 from https://pypi.jetson-ai-lab.io/jp6/cu126.
- cuRobo **PINNED v0.7.8** (v0.8 is an API rewrite — never upgrade),
  pip-editable from `~/RAMMP-Kinova/ros2_ws/curobo`; warp-lang 1.5.1.
- Arm: Kinova Gen3 7-DoF at **192.168.1.10**. Execution lives elsewhere
  (`kinova_arm_ros2`; the ros2_control stack in `~/RAMMP-Kinova/ros2_ws`).
  Nothing here needs it sourced — the planner runs standalone.
- `ROS_LOCALHOST_ONLY=1` in EVERY shell that runs ROS here. Non-interactive
  shells skip ~/.zshrc — export it explicitly or nodes won't discover each
  other (field-verified split-DDS failure).
- Mutually exclusive arm stacks exist on this machine: `kinova_arm_ros2` +
  the custom C++ driver in `~/atdev/kinova-gen3-driver`, RAMMP-Kinova's
  ros2_kortex bringup, and Demo-Software's `arm_driver` (Kortex API
  direct). One at a time, ever. This repo is in none of them — it plans
  alongside whichever is up, and claims nothing.

## Invariants (hard-won in RAMMP-Kinova; verified still true here)

- cuRobo v0.7.8: WorldConfig cylinders/spheres silently DROPPED (cuboids
  only); `update_world` with zero cuboids silently keeps the old world;
  more cuboids than `collision_cache_obb` raises; two boxes of one name
  collapse into one; a NaN position, zero quaternion or negative dim loads
  as a box that never collides. `world.py` guards all, in pure Python,
  before cuRobo is imported.
- Obstacle layers (issue #19): a planner config's `layers:` are composed
  into EVERY world, set_world pushes included; a world file's `layers:`
  last only until the next push. Safety geometry (the chair, the person)
  belongs in the planner config. Only an `excludable: true` layer may be
  dropped, by `ignore=`, from Python only (no IDL change).
- The chair is NOT package data (issue #19, Swapnil's review of #28). The
  three layers, `planner.yaml` and `world.yaml` describe ONE chair
  and where the arm sits on it, so they ship with that robot:
  rammp-deployments `december_2026/config/robots/flatbot/curobo/`, bind-mounted
  at `/config/robot` and selected with
  `config:=/config/robot/curobo/planner.yaml` (profile `flatbot.yaml`, which
  MUST also override `world` to `""` — a non-empty `world` param replaces
  whatever the config names). `core` needed no change: `resolve_config` takes
  absolute paths, and a layer resolves next to the config that lists it. Do
  NOT restore a packaged copy "as a fallback": Docker makes `/config/robot`
  an empty directory when it starts before `set_robot.py`, and a fallback
  would let a bare layer name resolve against the image and plan the chair
  against stale geometry. `test_the_chair_is_not_packaged_and_resolves_to_no_stale_copy`
  pins that.
- `core/tests/data/mebot/` is this checkout's FIXTURE copy of those five
  files — for the tests and the sim, so verifying the chair needs no
  deployment checkout. The layers in it are GENERATED by
  `scripts/generate_layers.py` from `scripts/mebot_layers.yaml` and the
  pinned MEBot URDF: edit the inputs, run `make layers` (on unchanged inputs
  it must leave no diff) and commit what it writes. Never hand-edit them.
  `make layers OUTDIR=<robots>/mebot/curobo` writes the robot's copy from the
  same pinned inputs. The bytes match because the generator is
  deterministic, not because one run writes both: `make layers OUTDIR=` runs
  it twice, and only the in-repo copy is diff-asserted here (the deployment
  repo composes its own copy in CI). `outputs:` in
  `mebot_layers.yaml` names the fixture copy, and `stack.layer_files()`
  follows it — which is why the sim's digest pin in
  `sim/mebot_sim/config/scenarios.yaml` survived the move untouched.
- A layer states the padding it was generated for
  (`meta.padding_assumed_m`); composing it with LESS is REFUSED, naming the
  layer and the shortfall. NOT because the boxes are undersized — they
  circumscribe the chair, and each layer carries the containment certificate
  that proves it (`under_m3: 0.000000`). Padding is margin, not size. What is
  padding-dependent is what the generator DECIDED at that padding: the person
  layer is carved back around the arm's base column by
  `padding_assumed_m + keep_out_margin_m`, so a smaller padding leaves part of
  the occupant uncovered beside the arm; and a platform merge was only made
  where the padded box still cleared those spheres. So the set is certified
  for one padding and no other. MORE is allowed — it only grows the boxes, and
  growth into the arm's spheres is caught by the keep-out guard, by the box it
  grew. This is the only part of `meta` anything reads; before #19 only a test
  held the two in step, which cannot reach a file that ships from another repo.
- Nothing padded may touch the arm's joint-independent sphere column (the
  arm_base_link spheres and the arm_shoulder_link ones on the arm_joint_1 axis: z
  -0.015 .. 0.383 m, radius up to 0.080 m with the buffer), or every state
  is INVALID_START. The planner refuses a layer box that does; the
  generator writes no layer within 4 mm of it.
- Worktree tests and `make layers` need `PYTHONPATH=<worktree>/core`: this
  Jetson's editable install is `~/RAMMP-CuRobo/core`, and
  `core/tests/conftest.py` and the generator refuse another checkout's
  `rammp_curobo`. The sim targets set it themselves.
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
- cuRobo `velocity_scale` stays 1.0. Slow execution = time dilation
  (t/s, v*s, a*s²) applied by whoever executes, never plan-time scaling.
  `retime.py` is the library-side implementation.
- Quaternions: cuRobo wxyz, ROS xyzw. Library API takes xyzw by default.
- Robot config = `configs/robot_gen3_2f85.yaml`, GENERATED by
  `scripts/bake_robot_config.py` (audit-tuned spheres, gripper shell,
  retract_config from that script's `RETRACT_POSE` — the ONLY place that
  pose is defined). Regenerate, don't hand-edit spheres. `make bake` runs
  it inside a released image (it needs cuRobo's content tree and yourdfpy)
  and expects no diff. The spheres were audited on NVIDIA's model and are
  MOVED onto the upstream links through FK of both models, never re-tuned;
  `test_model_parity.py` pins every sphere and `arm_tool_frame` to NVIDIA's
  within 1 mm (and 0.1°).
- The URDF is sanitised before cuRobo reads it (`urdf_model.prepare_urdf`):
  `<visual>`, `<collision>` and `<ros2_control>` are stripped (the mesh
  paths point at packages this image lacks; the spheres are the collision
  model), and a model that is not ours (root not `arm_mount`, no
  `arm_tool_frame`, unprefixed names) is refused by name. So is a model whose
  joints below `arm_base_link` drift from `configs/arm_kinematics.json` (what
  the spheres were baked on: 1 mm / 0.1°, structure, joint set; the mount
  joint may vary, limit differences are only logged).
- The gripper is locked open by its ACTUATED joint only
  (`lock_joints: {arm_robotiq_85_left_knuckle_joint: 0.0}`; cuRobo v0.7.8
  refuses mimic joints in `lock_joints` and folds the five mimics in itself).
  The 2F-85's mimic joints carry no `<limit>`, so `prepare_urdf` fills them
  from the joint each follows (kinova-gen3-ros2#48 may add them upstream).
- Joint limits are the URDF's — what the driver enforces: j2/j4/j6
  ±2.24/±2.57/±2.09 (the continuous joints parse as ±6.28).
  `limit_clamp_rad` stays 0.05, so a start from the Gen3 park pose
  (`arm_joint_4` ≈ -2.674) is REFUSED, naming the joint
  (kinova-gen3-ros2#49). Don't widen the clamp or the limit to hide it.
- `joint_limit_margin_rad` (0.1) is cuRobo's `position_limit_clip`: every
  plan stays that far inside the URDF limits. `joint_limits()` still
  returns the URDF's (widened back); the constraint re-check enforces the
  margin. A start inside it is REFUSED by name, never shifted — shifting
  would put the plan's first point off the arm's measured q.
- A world file must say `base_frame: arm_mount`; any other frame, or none,
  is refused (`load_scene`), since the boxes are applied in `arm_mount`.
- `world_empty.yaml` (no obstacles) is accepted at CONSTRUCTION only: the
  node applies its `world` parameter to the config before building the
  planner, and `update_world` still refuses an empty world. The startup log
  says `Collision world: N boxes (<file>)`.
- Test fixtures `core/tests/data/arm_level.urdf` / `arm_tilt30.urdf` are
  rendered by kinova-gen3-ros2's `kinova_gen3_render`; how is in
  `core/tests/data/README.md`. Re-render, don't hand-edit.
- retract_config must stay in the bent-elbow family (arm_joint_3 ≈ π) — the
  stock retract seeds the opposite family and every crossing plan winds.
  It is the IK SEED, read back as `planner.retract_pose`. The planner has
  no home/start pose of its own: `start` is required on every plan call
  and there is no default to fall back on (issue #6).
- `ee_link` is `arm_tool_frame`: 0.120 m beyond the wrist flange (≈ fingertip
  midpoint). Tool corrections (spin 90°, tip 21 mm) are sim-measured and
  ship DISABLED in gen3.yaml until re-measured on the real arm.
- Sim starts at q=0 (not home). Both sim and real expose the SAME
  controller names; never run both bringups (shared /controller_manager).

## Safety (do not weaken)

Nothing here can move the arm, and that IS the safety model — one execution
authority, not two with different rules. The gates that used to live in
`executor.py` are `kinova_arm_ros2`'s now: it refuses goals without a
measured state, plans from the arm's measured `q` (so a plan's first point
is where the arm is), validates the returned trajectory, and executes with
a path tolerance under the driver's rate limiter.

What survives here is planning-side integrity: `validate.py`'s continuity
check, the trimmed-plan rule, `check_start_joints` in `planner_node.py`, and
the sim-world-without-sim-time tripwire. Hardware sessions still follow
docs/HARDWARE_BRINGUP.md with a human on the physical e-stop; never drive
the real arm autonomously from an agent session.

The MEBot sim stack (`sim/mebot_sim`, `make sim`) serves the real arm
driver's names (`/joint_states`, `/execute_joint_trajectory`), so it never
runs on the bench's ROS domain, 0: a `ROS_DOMAIN_ID` that is not a plain
domain from 1 to 232 becomes 88, `/mebot_sim` refuses to start beside
anything already serving those names, and its client flies nothing unless
the sim alone serves them. Keep all three.

## Build / test

```bash
python3 -m pip install --user --no-build-isolation -e ./core   # pip quirks: see README
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rammp_curobo_interfaces rammp_curobo_ros
RAMMP_CUROBO_GPU=1 python3 -m pytest core/tests -q   # offline + GPU smoke
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio
ros2 run rammp_curobo_ros tour_demo      # live end-to-end plan check
make layers                              # regenerate the MEBot layers: no diff
make sim-test                            # the sim package's tests (Jetson)
make sim                                 # the MEBot sim suite (Jetson; GPU + ROS)
```

`RAMMP_CUROBO_GPU=1` turns a skip of the GPU tests (no CUDA, no torch or
cuRobo) into a failure, so on the Jetson the core suite cannot pass on its
offline tests alone; `make verify` sets it, CI does not.

pytest needs the repo-root `pytest.ini` (`-p no:anyio` — user-site anyio
plugin clashes with system pytest; `-p no:launch_testing -p no:launch_ros`
— pytest 9 refuses their hooks with ROS on PYTHONPATH). Style: Ruff v0.3.0
defaults (pre-commit), PEP 8, mdformat/gfm for Markdown.
