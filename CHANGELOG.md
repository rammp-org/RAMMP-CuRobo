# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project follows [semantic versioning](CONTRIBUTING.md#versioning) from 1.0.0
on. Downstream repos pin an exact tag, so **this file is how you decide whether
to move a pin** — a tag on its own does not tell you that.

Add entries under `Unreleased` as work merges into `dev`; the release PR renames
that heading to the new version and bumps the five places the number lives.

## [Unreleased]

The planner plans against the arm's own model, in `arm_mount`, for tilted arm
mounts (rammp-deployments `docs/superpowers/specs/2026-10-15-tilted-arm-mount-design.md`).
Breaking: move a pin only together with kinova-gen3-ros2's `arm_` names and
`arm_mount` (#47).

### Changed (breaking)

- **A required topic.** The node builds its robot model from the URDF
  `arm_state_publisher` latches on `/arm/robot_description` (RELIABLE +
  TRANSIENT_LOCAL), waiting `robot_description_timeout_s` (default 30 s); no
  model in time is a FATAL naming `arm_state_publisher` and exit 1. No URDF
  ships here any more. The model is read once: a later, different model is
  ignored, so restart the planner when the robot's mount changes. In `core`,
  `CuRoboPlanner(...)` / `from_config(...)` take the model as a file,
  `urdf_path=` (keyword); the baked config refuses to load without one.
- **Joint names** `joint_1..7` → `arm_joint_1..7`, on goals, in
  `gen3.yaml` and on every returned trajectory.
- **Frame.** Goals and plans are in `arm_mount`, the URDF's root (level with
  the chair), with the tool at `arm_tool_frame` (was `base_link` /
  `tool_frame`). On a level mount the numbers are unchanged; on a tilted one
  they are the arm's tilt apart, as intended.
- **Joint limits** are the URDF's, the ones the driver enforces: j2/j4/j6
  ±2.41/±2.66/±2.23 → **±2.24/±2.57/±2.09** rad (continuous j1/3/5/7 parse as
  ±6.28). `limit_clamp_rad` stays 0.05, so a plan from the Gen3's park pose
  (`arm_joint_4` ≈ -2.674) is now **refused**, naming `arm_joint_4`
  (kinova-gen3-ros2#49).
- **The container's CMD** execs the planner node directly (PID 1) instead of
  `ros2 launch`, so the container's exit code is the node's: a missing model
  is a non-zero exit, and `docker stop` exits 0 promptly.

### Added

- **Every plan states its frame:** `trajectory.header.frame_id` is
  `arm_mount` on pose and joint plans alike (core: `Trajectory.frame_id`).
  kinova-gen3-ros2's `GoToEEPose` refuses an unstamped pose plan on a tilted
  mount (#47).
- **`world_empty.yaml`**: plan against the arm alone (flatbot). Accepted at
  construction only (the node's `world` parameter is applied before the
  planner is built); `update_world` / `set_world` still refuse an empty
  world. The startup log reports `Collision world: N boxes (<file>)`, and a
  world file with an unknown top-level key is refused.

### Unchanged on purpose

- **The collision spheres** are moved onto the upstream links through FK of
  both models, not re-tuned: same count and radii, every sphere and
  `arm_tool_frame` within 1 mm (and 0.1°) of NVIDIA's model, which stays the
  sphere source. `make bake` now runs inside the image.
- **The gripper** is still locked open, now by its actuated joint only
  (`arm_robotiq_85_left_knuckle_joint`); cuRobo holds the five mimics with it.
  The URDF gives the mimic joints no limits, so they are filled in for cuRobo
  from the joint each follows (kinova-gen3-ros2#48 may add them upstream).
- **The retract seed's values**, `[0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]`.
- **`HOLD_LEVEL`'s maths.** Its documented limitation is now the real one:
  it holds level relative to `arm_mount`, which is level with the chair, not
  with gravity — on a ramp it is off by the ramp angle.

## [1.1.0] — 2026-10-02

### Added

- Plan while holding the tool's orientation (carry an object without tipping
  it, #16) and approach a goal through one blended via point the arm does not
  stop at (#17). New core types `PoseConstraint` and `ViaPoint`; `PlanToPose`
  gains `hold` (`HOLD_NONE`, `HOLD_LEVEL` or `HOLD_FIXED`, constants on the
  goal) and `approach_via` (new message `ApproachVia`). Both are appended at
  the end and default to off, so a client that sets neither plans exactly as
  before.
- A held plan is checked, not trusted: refused before planning
  (`CONSTRAINT_UNSATISFIABLE`) if the start disagrees with the goal on the
  held components, and after planning (`CONSTRAINT_VIOLATED`) if any waypoint
  breaks the hold or the plan does not end at the goal's full orientation —
  yaw included, since `HOLD_LEVEL` frees yaw only in transit. One tolerance,
  `constraint_tolerance_deg` (default 2 deg), for all three.
- A hold and a via point together are refused (`BAD_CONSTRAINT`): cuRobo's
  approach metric applies no per-step pose cost before `at_fraction`, so the
  hold would silently lapse for most of the path.
- Verified on the Jetson against real cuRobo (`ghcr.io/rammp-org/rammp-curobo:1.0.0`):
  on a goal the unconstrained plan reaches with ~20 deg of tilt, `HOLD_LEVEL`
  stays under 2 deg at every waypoint and ends on the goal's 1.2 rad of yaw.

### Fixed

- A plan that raised inside cuRobo left its hold installed, silently
  constraining the next, unconstrained plan (cuRobo resets the metric only on
  a normal return). The planner now resets it on the exception path.

## [1.0.0] — 2026-09-08

First tagged release, and the first published container. The planner has been in
bench use for weeks; this release does not add planning behaviour, it declares
the interface stable and gives `kinova_arm_ros2` something to pin other than a
moving branch.

### The promise

The public surface — the `rammp_curobo_interfaces` action and service
definitions, the `rammp_curobo` core API, and the container's ROS surface — is
now covered by semantic versioning. See
[Versioning](CONTRIBUTING.md#versioning), including the rule that in ROS 2
**adding a field to an existing message is a MAJOR change**, not an additive one.

### Added

- Container publishing. A `v*` tag builds and pushes
  `ghcr.io/rammp-org/rammp-curobo` (semver tags plus `latest`), so consumers pull
  a tag instead of building the ~1 h JetPack image themselves.
- CI. Three workflows split by cost: pre-commit over all files, core tests plus
  colcon on amd64 **and** arm64 on every PR, and the JetPack container build on
  the promotion PR into `main`, on merges, and on tags. A failed build on a merge
  or tag files one deduplicated `ci-failure` issue and closes it on recovery.
- The `main`/`dev`/`feature` branching model, with both branches protected and
  the gates above required.
- `CONTRIBUTING.md`, this changelog, a `Makefile`, and the sibling repos' hook
  set — hadolint over the Dockerfile, actionlint over the workflows, shellcheck
  over `entrypoint.sh`, plus Ruff, mdformat and general hygiene.

### Changed

- Version is `1.0.0` across `core/pyproject.toml`, `core/rammp_curobo/__init__.py`,
  both `package.xml` files, and `rammp_curobo_ros/setup.py`.

### The state this release declares stable

Carried in from earlier work, unchanged here and listed because a 1.0.0 is a
claim about all of it:

- **This repo never touches the arm** (issue #6). No executor, no `/joint_states`
  subscription, no controller client, no `execute` parameter. `start_joints` is
  required on both plan actions — the caller owns the robot and its state. That
  is the safety model: one execution authority, not two with different rules.
- Plan-only ROS surface: `/rammp_curobo/plan_to_pose`,
  `/rammp_curobo/plan_to_joints`, `/rammp_curobo/set_world`.
- cuRobo pinned to **v0.7.8** by tag *and* commit hash; torch 2.10.0 /
  torchvision 0.25.0 from the jp6/cu126 index; warp 1.5.1; numpy 1.26.4.
- Cyclone DDS, matching the rest of the fleet.

### Known limits

- The container is **arm64/JetPack only** — `nvcr.io/nvidia/l4t-jetpack` has no
  x86 variant. An x86-64 CUDA image is issue #8.
- CI's container job is a **build-and-link gate, not a functional one**: hosted
  runners have no GPU, so `core/tests/test_smoke.py` skips there and the CUDA
  kernel JIT never runs. The functional gate is the Jetson, via `.hil.yml`.

[1.0.0]: https://github.com/rammp-org/RAMMP-CuRobo/releases/tag/v1.0.0
[1.1.0]: https://github.com/rammp-org/RAMMP-CuRobo/compare/v1.0.0...v1.1.0
[unreleased]: https://github.com/rammp-org/RAMMP-CuRobo/compare/v1.1.0...HEAD
