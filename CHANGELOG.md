# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project follows [semantic versioning](CONTRIBUTING.md#versioning) from 1.0.0
on. Downstream repos pin an exact tag, so **this file is how you decide whether
to move a pin** — a tag on its own does not tell you that.

Add entries under `Unreleased` as work merges into `dev`; the release PR renames
that heading to the new version and bumps the five places the number lives.

## [Unreleased]

## [2.0.0] — 2026-10-30

The planner now plans against the arm's own model, read at start-up from
`/arm/robot_description`, in `arm_mount`, so a tilted arm mount plans correctly
(rammp-deployments `docs/superpowers/specs/2026-10-15-tilted-arm-mount-design.md`,
#38). Also in this release: obstacle layers, the first of them the MEBot
wheelchair and the person in it (#19, #28); `/rammp_curobo/blend`, one
trajectory through several plans without stopping between them (#48); and
planner fixes for accelerations, joint limits and continuous-joint winding
(#35, #45, #36).

MAJOR because what the planner expects and returns changed without a field
changing: joint names gain `arm_`, goals and plans move to `arm_mount`, a new
topic is required, the joint limits tighten, and world files are checked
strictly. `PlanToPose`, `PlanToJoints` and `SetWorld` carry the same fields as
in 1.1.0; their comments now name the new joints and frame. `Blend.action` is
new.

### Upgrading from 1.1.0

1. **Move the pin together with kinova-gen3-ros2 2.0.0, on every robot, dojo
   included.** Each refuses the other's old version: the new kinova-gen3-ros2
   refuses every plan whose joints are not `arm_joint_1`…`arm_joint_7` and every
   pose plan not stamped `arm_mount`, and this planner exits without the new
   kinova-gen3-ros2's `arm_state_publisher`.

1. **Something must publish `/arm/robot_description`** (kinova-gen3-ros2's
   `arm_state_publisher`). With no model within `robot_description_timeout_s`
   (default 30 s) the planner logs a FATAL naming `arm_state_publisher` and
   exits 1. A slow first `arm_state_publisher` start can do this, and sheppy does
   not restart the planner: fix `arm_state_publisher`, then
   `sheppy restart planner`. When the mount changes, restart the planner too: it
   reads the model once.

1. **Clients send and read `arm_joint_1..7`, poses in `arm_mount`, the tool at
   `arm_tool_frame`** (were `joint_1..7`, `base_link`, `tool_frame`). On a level
   mount the numbers do not change. On a tilted mount they differ by the tilt,
   as intended.

1. **Choose a world per robot.** The image's default CMD loads `gen3_real.yaml`,
   whose world is `world_real_bench.yaml`: dojo's bench. On flatbot, either move
   the pin together with the per-robot worlds, or give flatbot alone
   `world_empty.yaml` in the sheppy manifest (rammp-deployments
   `december_2026/sheppy-manifest.yaml`):

   ```yaml
   - id: curobo_planner
     kind: docker
     ros_node_name: rammp_curobo          # key for the params file sheppy writes
     container:
       image: ghcr.io/rammp-org/rammp-curobo:2.0.0
       # ... (network, runtime, volumes as today)
       # sheppy appends --ros-args --params-file only after a command; without
       # one it replaces the image CMD and the container exits 2.
       command:
         - /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node
         - --ros-args
         - -p
         - config:=gen3_real.yaml
     params:
       world: world_real_bench.yaml       # dojo's bench; flatbot overrides
   ```

   and in flatbot's profile `overrides: {planner: {world: world_empty.yaml}}`.
   sheppy applies overrides only to params the alternative declares, so the
   `params:` entry is required. A plain `-p world:=world_empty.yaml` does not
   work: `params:` without a `command:` makes sheppy replace the image CMD with
   `--ros-args --params-file ...` alone, which `/entrypoint.sh` cannot exec, so
   the container exits 2. And hard-coding the world in the shared manifest's
   `command:` drops dojo's bench, so dojo plans through real objects.

1. **On the chair, name the config by path.** The MEBot config, world and layers
   no longer ship in the image; they live in rammp-deployments
   `december_2026/config/robots/<robot>/curobo/`, bind-mounted read-only at
   `/config/robot`. Use `config:=/config/robot/curobo/planner.yaml`, and if the
   manifest declares a `world` parameter, override it to `""` (a non-empty
   `world` replaces the one the config names, so a bench world left in place
   plans the chair against the wrong room). Profile `flatbot.yaml` does both.
   `gen3_mebot.yaml` is now `planner.yaml` and `world_mebot.yaml` is
   `world.yaml`. A bare `planner.yaml` fails to resolve, naming every path tried.

1. **Expect new refusals.** The j2/j4/j6 limits tighten to the URDF's, and
   starts and joint goals within 0.1 rad of a limit are refused: both the Gen3's
   park pose (`arm_joint_4` ≈ -2.674) and Kinova's factory Retract
   (`arm_joint_4` -2.548) are now refused. World files must say
   `base_frame: arm_mount`, have an `obstacles` key and no unknown or repeated
   keys.

1. **Delete `execution:` from planner configs.** `speed_scale` /
   `max_speed_scale` were never read. A config that keeps the block still loads,
   with a warning.

1. **Core API callers** pass the model as `urdf_path=` (keyword) to
   `CuRoboPlanner(...)` / `from_config(...)`, and `load_scene` defaults to
   `frame="arm_mount"`.

### Changed (breaking)

#### The arm's own model, in `arm_mount` (#38)

- **A required topic.** The node builds its robot model from the URDF
  `arm_state_publisher` latches on `/arm/robot_description` (RELIABLE +
  TRANSIENT_LOCAL), waiting `robot_description_timeout_s` (default 30 s); no
  model in time is a FATAL naming `arm_state_publisher` and exit 1. No URDF
  ships here any more. The model is read once: a later, different model is
  never used, and the node logs an ERROR each time one arrives ("a DIFFERENT
  model was published on /arm/robot_description; ... restart the planner").
  In `core`, `CuRoboPlanner(...)` / `from_config(...)` take the model as a
  file, `urdf_path=` (keyword); the baked config refuses to load without one.
- **Joint names** `joint_1..7` → `arm_joint_1..7`, on goals, in `gen3.yaml` and
  on every returned trajectory.
- **Frame.** Goals and plans are in `arm_mount`, the URDF's root (level with the
  chair), with the tool at `arm_tool_frame` (was `base_link` / `tool_frame`).
- **Joint limits** are the URDF's, the ones the driver enforces: j2/j4/j6
  ±2.41/±2.66/±2.23 → **±2.24/±2.57/±2.09** rad (continuous j1/3/5/7 parse as
  ±6.28). `limit_clamp_rad` stays 0.05, so a plan from the Gen3's park pose
  (`arm_joint_4` ≈ -2.674) is refused, naming `arm_joint_4`
  (kinova-gen3-ros2#49).
- **A drifted model is refused.** The collision spheres sit on the link frames
  of `arm_level.urdf`; `make bake` records every joint below `arm_base_link` in
  `configs/arm_kinematics.json`. A model whose joints differ from it (an origin
  by more than 1 mm or 0.1°, an axis by more than 0.1°, a parent, child, type or
  mimic, or a joint missing or extra) is a FATAL naming the joint, and exit 1.
  The mount (`arm_base_joint`) may vary. Limit differences are logged as a
  WARNING and the model's are used. An upstream description change that moves
  a link means a re-bake and a release.
- **The container's CMD** execs the planner node directly (PID 1) instead of
  `ros2 launch`, so the container's exit code is the node's: a missing model is
  a non-zero exit, and `docker stop` exits 0 promptly.

#### World files (#38)

- **Strict.** A world file with an unknown top-level key is refused, and so is
  one without an `obstacles` key (`obstacles: null` and a missing key are both
  refused; an empty list is `obstacles: []`). Both used to load. `obstacles`,
  `objects` and `targets` must each be a list.
- **`base_frame: arm_mount` is required.** Boxes are applied in the planning
  frame whatever the file says, so a world in any other frame is refused, naming
  the file and the value, including one with no `base_frame`, which used to
  default to `base_link`. The packaged worlds already say `arm_mount`. In
  `core`, `load_scene(path, frame="arm_mount")`, and `update_world` refuses a
  `Scene` in another frame.

#### Joint-limit margin (#45)

- **Starts and joint goals near a limit are refused.** One within
  `joint_limit_margin_rad` (new, default 0.1 rad) of a URDF position limit fails
  as `START_IN_LIMIT_MARGIN` / `GOAL_IN_LIMIT_MARGIN`, naming the joint, where it
  used to plan. Kinova's factory Retract (`arm_joint_4` -2.548) is one. The
  0.05 rad clamp now matters only with the margin set to 0. A pose goal
  reachable only near a limit now fails as `IK_FAIL`.

#### The MEBot chair (#28)

- **The chair ships with the robot, not in the image.** `planner.yaml`,
  `world.yaml` and the three `layer_mebot_*.yaml` describe ONE chair's geometry
  and where the arm sits on it, so they are per-robot data in rammp-deployments
  `december_2026/config/robots/<robot>/curobo/`, bind-mounted read-only at
  `/config/robot` (#19). Re-measuring the chair, and the Gen1.2 → Gen1.5 repin,
  are now a commit there rather than a release here. There is deliberately NO
  packaged fallback: Docker makes `/config/robot` an empty directory when a
  container starts before `set_robot.py` has run, and a fallback would plan the
  chair against stale geometry in the image with no warning. `core` needed no
  change: `resolve_config` already took absolute paths, and a layer resolves
  next to the config that lists it. `core/tests/data/mebot/` is this checkout's
  fixture copy, so verifying the chair needs no deployment checkout.
- **Fixed file names.** `gen3_mebot.yaml` is `planner.yaml` and
  `world_mebot.yaml` is `world.yaml`, the same in every robot's `curobo/`
  directory.
- **The chair's arm is on its left, tilted 30° on its mount.** The layers were
  generated for the RAMMS model's arm, level on the chair's right rail; flatbot's
  is on the left, with the joystick on the right, and leans 30° forward on the
  `new_adapter_plate_v3` (its `mount.yaml`). The mount's y is mirrored to
  0.265 m, and the generator takes the tilt as `mount.arm_base_joint.rpy_deg`:
  it places the arm's fixed spheres, which the person is carved around and the
  platform's merges clear, while the boxes stay square to the chair in
  `arm_mount`. The person is 9 boxes, so the chair's start-up world is 26.
  Neither layer models the joystick. The tests' chair planners and the sim's arm
  use the tilted model (`arm_tilt30.urdf`).

### Added

- **`/rammp_curobo/blend`** (`Blend.action`; core `CuRoboPlanner.blend_legs`,
  `blend.py`, #48): one trajectory from plans this planner returned (pose legs
  with their own hold, joint legs), each at its own speed scale, passing the goals
  between still moving. Consecutive legs overlap in time; each overlap keeps the
  joint velocity and acceleration limits, the held legs' holds (each against its
  own goal's `hold_orientation`), the tool's closest pass within the junction's
  `blend_radius` of its goal and every state within it of the planned path, and
  is collision-checked state by state. Legs not on the planner's dt are
  resampled onto it (cubic Hermite). On abra, in position mode at full speed,
  blending cut motion time by 21–43 %. kinova-gen3-ros2's `GoToEEPoses` /
  `GoToJointConfigs` are built on it.
- **Every plan states its frame** (#38): `trajectory.header.frame_id` is
  `arm_mount` on pose and joint plans alike (core `Trajectory.frame_id`); a
  failed plan's trajectory is empty, with an empty frame.
- **`world_empty.yaml`** (#38): plan against the arm alone (flatbot). Accepted at
  construction only (the node's `world` parameter is applied before the planner
  is built); `update_world` / `set_world` still refuse an empty world. The
  start-up log reports `Collision world: N boxes (<file>)`.
- **Obstacle layers** (#28): named groups of boxes, each in a file of its own.
  - The planner config's new `layers:` key lists layers composed into every
    world, `set_world` pushes included; a world file's new `layers:` key lists
    layers that belong to that scene and are replaced with it. Layer boxes are
    named `<layer>/<box>`, are in `arm_mount` and are padded like every other
    box. `Scene(layers=)` carries a scene's layers.
  - `ignore={"person"}` on `update_world` leaves an `excludable: true` layer out
    of that one world (Python only); ignoring a layer that is not excludable is
    refused.
  - A layer box that reaches into the arm's joint-independent spheres is
    refused: every start state would be in collision.
  - A layer states the `world_padding` it was generated for
    (`meta.padding_assumed_m`), and a world composed with LESS than that is
    refused, naming the layer, its file and the shortfall. The boxes still
    circumscribe the chair; what depends on padding is the person layer's
    carve-out around the arm's base column and where platform boxes were merged.
    More padding is allowed. A layer that states no padding (built in code, or
    hand-written) composes at any padding. This is the only part of `meta`
    anything reads.
  - The MEBot layers, shipped with the robot (see above): `platform` (16 boxes)
    and `floor` (1 box, its top at the URDF's lowest point) from the drive base's
    URDF in rammp-org/Demo-Software, pinned by commit and sha256; `person`
    (9 boxes, excludable) from the ANSUR II sizes and the seat in
    `scripts/mebot_layers.yaml`: any occupant from the 5th-percentile woman to
    the 95th-percentile man, plus 0.05 m.
  - `scripts/generate_layers.py`, its inputs `scripts/mebot_layers.yaml`, and
    `make layers`; `--outdir` / `make layers OUTDIR=<dir>` writes a robot's copy
    from the same pinned inputs. CI regenerates the committed layers on amd64 and
    arm64 and fails on any difference (the generator is deterministic; only the
    in-repo copy is diff-asserted here).
  - A consumer on `gen3.yaml` or `gen3_real.yaml`, which list no layers, gets the
    collision worlds it got before, box for box, except where *Fixed* says
    otherwise.
- **Which box:** a failed plan's error and `check_state_valid` name the world
  boxes the start or goal reach into, with depths, and a failed plan's error ends
  with the world's summary. New `CuRoboPlanner.world_boxes` and
  `CuRoboPlanner.world_summary()`.
- `joint_limits()` reports `acceleration` too (#35).
- The ROS 2 sim stack `sim/mebot_sim`: a MuJoCo Gen3 serving the arm driver's
  names beside the exact MEBot and a seated body, two planner nodes, and a
  paired suite; `make sim` and `make sim-test` (Jetson).
- `RAMMP_CUROBO_GPU=1` makes a GPU test that would skip fail the core suite
  instead, so it cannot pass on its offline tests alone; `make verify` sets it.

### Changed

- With config layers, cuRobo warms up in the scene's own world and the layers
  load after: warm-up plans from the retract, which on the chair is inside the
  person layer. The planner warns at start-up when the retract is not a valid
  state in its world.
- The `SetWorld` success message is `world set to <world> (<summary>)`, and the
  node logs the summary at start-up and in its ready line.
- A failed plan's error is cuRobo's reason followed by the boxes and the summary
  above. `_finish` keeps its v1.0.0 signature, so a subclass that overrides it
  keeps working: the boxes and the summary are added after it returns.
- `make_world_config`'s default `cache_obb` is 60, the planner's (was 40).
- `make bake` runs inside the image.
- The core depends on `scipy` (already in the Jetson image via cuRobo).

### Removed

- `execution.speed_scale` / `execution.max_speed_scale` from `gen3.yaml` and the
  config defaults, and `CuRoboPlanner.execution` (#22, #34). Nothing ever read
  them, so the "quarter speed" bring-up default they declared was never applied:
  the arm ran every plan at full speed. Speed scaling is the executor's job
  (`kinova-gen3-ros2#37`). A config that still carries an `execution:` block
  loads as before, with a warning that the block is ignored. Library-side
  retiming (`Trajectory.scaled()`, `retime.py`) is unchanged. Dropping the
  exported attribute would be MAJOR on its own; it is waived for this one
  attribute only, because no consumer references it.

### Unchanged on purpose

- **The collision spheres** are moved onto the upstream links through FK of both
  models, not re-tuned: same count and radii, every sphere and `arm_tool_frame`
  within 1 mm (and 0.1°) of NVIDIA's model, which stays the sphere source.
- **The gripper** is still locked open, now by its actuated joint only
  (`arm_robotiq_85_left_knuckle_joint`); cuRobo holds the five mimics with it.
  The URDF gives the mimic joints no limits, so they are filled in for cuRobo
  from the joint each follows (kinova-gen3-ros2#48 may add them upstream).
- **The retract seed**, `[0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]`.
- **`HOLD_LEVEL`'s maths.** Its documented limitation is now the real one: it
  holds level relative to `arm_mount`, which is level with the chair, not with
  gravity. On a ramp it is off by the ramp angle.

### Fixed

#### Trajectories

- **Planned accelerations reach the message** (#15, #35). `trajectory_to_msg`
  copies them into each `JointTrajectoryPoint`; they were computed and then
  dropped. A consumer that picks its interpolation order from what the points
  carry (`joint_trajectory_controller`, the kinova driver) moves from cubic to
  quintic on our plans with no change on its side.
- **Positions, velocities and accelerations agree** (#15, #35). cuRobo's
  `LINEAR_CUDA` resample lerps each one independently between its optimized
  knots, so on any plan optimized coarser than `interpolation_dt` the positions
  kinked at every knot while the accelerations claimed a smooth ramp (measured
  95–368 % apart). The planner now samples all three from one quintic spline
  through the same knots, landing on the goal at rest, before validation and the
  constraint re-check, so both see exactly what ships. On the Jetson over 22
  plans: a consumer's quintic peak jerk drops from a median 7397 to 227 rad/s³,
  its peak acceleration from 4.1× to 1.02× the planned one, at 1.6 ms per plan;
  positions move by at most 6.1 mrad. If the knots do not reproduce cuRobo's
  resample, the plan ships as cuRobo returned it but without accelerations, so a
  consumer stays on cubic.
- **Shipped accelerations are validated** against the robot's acceleration
  limit, with the same 1 % slack as velocities.
- **Plans no longer reach a joint limit** (#45). cuRobo's IK put pose goals ON a
  limit (`arm_joint_4` = 2.57 for a `GoToEEPose` on the real arm on 2026-10-19);
  the quintic resample passed it by up to 0.7 mrad and the constraint re-check
  rejected the plan ("constraint re-check flagged N/M states infeasible").
  cuRobo now plans inside `joint_limit_margin_rad` of every URDF limit, and when
  the spline still leaves the narrowed limits the planner ships cuRobo's own
  plan, without accelerations, after the same re-check and validation. On a
  Jetson Orin, 159 plans per model to that day's two goals: 34 rejects on the 30°
  model and 2 on the level model before, none after, and every plan kept at
  least 0.1 rad from every limit (before: down to 0.1 mrad).
- **No full revolution on a continuous joint** (#14, #36). `plan_to_pose` could
  take joint 1/3/5/7 the long way round when IK landed it on a 2π branch away
  from the start: a full revolution for a goal at the arm's current pose with
  `arm_joint_3` parked at -π. A plan that ends off the start's branch is now
  replanned in joint space to its final configuration moved onto that branch,
  the correction `plan_to_joints` already applied. If that replan fails the
  original plan is returned with a warning in `PlanResult.error`, which the node
  appends to the result `message`. Plans with a hold or via point are not
  replanned (a joint-space replan would drop them) but are logged as a warning.
- A slowed plan's `time_from_start` never carries `nanosec` 1e9.

#### Worlds and configs

- `update_world(world, ignore="mug")` leaves out the prop `mug`: a bare string is
  one name. 1.0.0 read it as the set of its letters, so it kept `mug` and left
  out any prop named `m`, `u` or `g`.
- Two boxes of one name are refused, naming both, instead of collapsing into one.
- A box with a non-finite position or rpy, a negative or non-finite dim, or a
  dim of 0 that no padding grows (a `no_pad_names` box, or any box at
  `world_padding: 0`) is refused: cuRobo loaded it as a box that never collides.
  A padded 0 is a slab, as in 1.0.0. A `world_padding` below 0 or not finite is
  refused too.
- A key written twice in a planner config or a world file is refused, naming the
  file, the key and its line, instead of the last one silently winning (a second
  `layers:` would have dropped the first's layers). A top-level `layer:` or a
  `layers:` under `planner:` is refused too: either would have left the chair
  out.
- A malformed world entry is a `ValueError` naming the file and the entry
  (`obstacles[3] (table): missing 'dims'`), not a bare `KeyError`.
- World files, planner configs and the robot config are read as UTF-8 whatever
  the locale.
- The world guards (empty world, cache overflow and the above) run before cuRobo
  is imported, so they hold without it.
- A failed plan whose world summary cannot be made is still reported, with
  `[world summary unavailable: ...]` in its error.

#### Sim

- The sim arm watches the ROS graph for 5 s, not one look after 1 s, before it
  decides no driver already serves its names (an action server, by its status
  publisher: a client lists the action's name too). `make sim` and
  `make sim-test` export `ROS_LOCALHOST_ONLY=1`, but leave it to a
  `CYCLONEDDS_URI` when one is set.

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
[2.0.0]: https://github.com/rammp-org/RAMMP-CuRobo/compare/v1.1.0...v2.0.0
[unreleased]: https://github.com/rammp-org/RAMMP-CuRobo/compare/v2.0.0...HEAD
