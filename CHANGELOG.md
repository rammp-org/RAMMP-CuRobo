# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project follows [semantic versioning](CONTRIBUTING.md#versioning) from 1.0.0
on. Downstream repos pin an exact tag, so **this file is how you decide whether
to move a pin** — a tag on its own does not tell you that.

Add entries under `Unreleased` as work merges into `dev`; the release PR renames
that heading to the new version and bumps the five places the number lives.

## [Unreleased]

Obstacle layers, and the first of them: the MEBot wheelchair and the person in
it (issue #19). No interface change: the `rammp_curobo_interfaces` definitions
are untouched, and a consumer on `gen3.yaml` or `gen3_real.yaml`, which list no
layers, gets the collision worlds it got before, box for box, except where
Fixed says otherwise: a world 1.0.0 built wrong is refused, and a bare-string
`ignore=` leaves out the prop it names. The release is MINOR.

### Added

- Obstacle layers: named groups of boxes, each in a file of its own. The
  planner config's new `layers:` key lists layers composed into every world,
  `set_world` pushes included; a world file's new `layers:` key lists layers
  that belong to that scene and are replaced with it. Layer boxes are named
  `<layer>/<box>` and padded like every other box. `Scene(layers=)` carries a
  scene's layers.
- `ignore={"person"}` on `update_world` leaves an `excludable: true` layer out
  of that one world (Python only); ignoring a layer that is not excludable is
  refused.
- The MEBot layers, in `configs/layer_mebot_*.yaml`. `platform` (16 boxes) and
  `floor` (1 box, its top at the URDF's lowest point) come from the drive
  base's URDF in rammp-org/Demo-Software, pinned by commit and sha256;
  `person` (8 boxes, excludable) from the ANSUR II sizes and the seat in
  `scripts/mebot_layers.yaml`: any occupant from the 5th-percentile woman to
  the 95th-percentile man, plus 0.05 m.
- The chair config `gen3_mebot.yaml` and its start-up world `world_mebot.yaml`.
- `scripts/generate_layers.py` with its inputs `scripts/mebot_layers.yaml`, and
  `make layers`. CI regenerates the committed layers from those inputs and the
  pinned URDF on amd64 and arm64 and fails on any difference.
- `CuRoboPlanner.world_boxes` and `CuRoboPlanner.world_summary()`.
- Which layer, which box: a failed plan's error and `check_state_valid` name
  the world boxes the start or goal reach into, with depths, and a failed
  plan's error ends with the world's summary.
- A layer box that reaches into the arm's joint-independent spheres is refused:
  every start state would be in collision.
- The ROS 2 sim stack `sim/mebot_sim`: a MuJoCo Gen3 serving the arm driver's
  names beside the exact MEBot and a seated body, two planner nodes, and a
  paired suite; `make sim` and `make sim-test` (Jetson).

### Changed

- With config layers, cuRobo warms up in the scene's own world and the layers
  load after: warm-up plans from the retract, which on the chair is inside the
  person layer. The planner warns at start-up when the retract is not a valid
  state in its world.
- The SetWorld success message is `world set to <world> (<summary>)`, and the
  node logs the summary at start-up and in its ready line.
- A failed plan's error is cuRobo's reason followed by the boxes and the
  summary above. `_finish` keeps its v1.0.0 signature, so a subclass that
  overrides it keeps working: the boxes and the summary are added after it
  returns.
- `make_world_config`'s default `cache_obb` is 60, the planner's (was 40).

### Fixed

- `update_world(world, ignore="mug")` leaves out the prop `mug`: a bare string
  is one name. 1.0.0 read it as the set of its letters, so it kept `mug` and
  left out any prop named `m`, `u` or `g`.
- Two boxes of one name are refused, naming both, instead of collapsing into
  one.
- A box with a non-finite position or rpy, or dims that are not finite and
  above 0, is refused: cuRobo loaded it as a box that never collides. So is a
  `world_padding` below 0 or not finite.
- The world guards (empty world, cache overflow and the above) run before
  cuRobo is imported, so they hold without it.
- A malformed world entry is a `ValueError` naming the file and the entry
  (`obstacles[3] (table): missing 'dims'`), not a bare `KeyError`.
- World files, planner configs and the robot config are read as UTF-8 whatever
  the locale.
- A key written twice in a planner config or a world file is refused, naming the
  file, the key and its line, instead of the last one silently winning: a second
  `layers:` would have dropped the layers of the first. A top-level `layer:` or
  a `layers:` under `planner:` is refused too: either would have left the chair
  out.

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
[unreleased]: https://github.com/rammp-org/RAMMP-CuRobo/compare/v1.0.0...HEAD
