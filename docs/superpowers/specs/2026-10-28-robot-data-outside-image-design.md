# Robot data outside the image — design

2026-10-28. Repos: RAMMP-CuRobo (the planner, `ghcr.io/rammp-org/rammp-curobo`)
and rammp-deployments (`december_2026/`).

## Intent

One cuRobo image must serve every arm, every MEBot base and every deployment
environment. We may switch arms, we do switch MEBot bases, and we regularly
change environments; none of that should need a new image or a commit to
RAMMP-CuRobo. Everything the planner needs to describe a robot and its world
comes from rammp-deployments. The image holds code, plus examples.

The planner defines what a robot must supply, and tells you, loudly and all at
once, what is missing or wrong. It still spins up standalone, with nothing
else running, for development and demos.

## Goals

- No robot- or site-specific value is read from the image unless the caller
  asked for the shipped example by name.
- The contract — which files a robot supplies, and which fields — is defined
  by the planner: written down in one doc, and enforced at startup by one
  check that lists every problem it finds, not just the first.
- `docker run ghcr.io/rammp-org/rammp-curobo:<tag> example` plans, with no
  `arm_state_publisher`, no deployment checkout and no robot. (Changed
  2026-10-28: a bare `docker run` plans `/config/robot/curobo/planner.yaml`,
  and stops naming it when nothing is mounted.)
- The startup log names where each piece came from (example in the image, or
  a path under `/config/robot`), so running the example on a robot is visible
  in its first lines.
- Swapping an arm, a base or a site is a rammp-deployments commit.

## Non-goals

- Generating collision spheres for a new arm. The shipped example's spheres
  (Gen3 + Robotiq 2F-85, derived from cuRobo's bundled `kinova_gen3.yml` and
  audited) are what exists; a new arm brings its own robot file. Deferred.
- More than one arm in one running planner.
- Changing the ROS action interfaces. `PlanToPose` / `PlanToJoints` already
  take variable-length joint arrays; only their comments name the Gen3, and
  those comments get generalised.
- Upgrading cuRobo (pinned v0.7.8).

## Anti-goals (a failure even if it works)

- A quiet fallback: a robot config that names a file it does not have must
  never resolve that file from the image. Docker can start the planner with an
  empty `/config/robot` (before `set_robot.py` runs); a fallback would then
  plan the chair against stale geometry. This is the issue #19 lesson, and
  `test_the_chair_is_not_packaged_and_resolves_to_no_stale_copy` already pins
  one instance of it.
- Robot data partially defaulted: a robot file or a world is whole or refused,
  never merged field by field with the example's.
- Anything here gaining the ability to move the arm (issue #6 stands).

## Constraints

- cuRobo v0.7.8 only; `core/` stays free of ROS imports; the interfaces
  package stays dependency-free.
- Existing safety invariants in CLAUDE.md stay true: the URDF-vs-spheres
  check, the fixed-sphere guard, padding certification of layers, trimmed
  plans, the start-joint checks.
- rammp-deployments' layout: the active robot's directory is bind-mounted
  read-only at `/config/robot`; per-robot planner files live in
  `config/robots/<robot>/curobo/`.

## The contract (decided)

A robot supplies a **robot directory**, conventionally
`config/robots/<robot>/curobo/`, mounted at `/config/robot/curobo/`:

| File                                     | Says                                                                                                                                                                                                                  | Required       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `planner.yaml`                           | names the robot file and the start-up world; lists the always-on layers; optional planner settings                                                                                                                    | yes            |
| robot file (any name, e.g. `robot.yaml`) | cuRobo's robot config: root link, tool link, planned joints, collision spheres, self-collision pairs and buffers, locked joints, IK seed (`retract_config`), plus the kinematics reference the spheres were fitted on | yes            |
| start-up world (e.g. `world.yaml`)       | `base_frame` equal to the robot file's root link; the scene the planner starts with; replaced by the first `set_world` push                                                                                           | yes            |
| layer files                              | always-on obstacles (chair, person), with `meta.padding_assumed_m`                                                                                                                                                    | only if listed |

The arm's URDF still arrives on `/arm/robot_description` from
`arm_state_publisher` (rendered from the robot's `mount.yaml`) — except in
example mode, below.

**Resolution rule.** Every file a named `planner.yaml` references resolves
next to that `planner.yaml` (or as an absolute path). Never from the image.
The only way to use a shipped file from a robot config is an explicit
`example:` reference (e.g. `robot: example:gen3_2f85`), and the log then says
`(example, image)` for it. Robots in rammp-deployments carry their own copy of
the robot file (preferred: a new image can then never change a robot's
spheres or IK seed); `example:` is allowed, and visible in the log.

The same rule covers `set_world` / `update_world` pushes: a pushed world
named by a relative path resolves next to `planner.yaml`, or is absolute, or
says `example:`; never the image by bare name. (My call, for consistency;
callers pushing `world_real_bench.yaml` by bare name break, part of the MAJOR
change below.)

**Planner settings** (padding, attempts, timeouts, limit margins, tool
corrections) keep defaulting field by field from `PLANNER_DEFAULTS`, as today.
They are tuning, not robot data.

**The check.** Before cuRobo is imported, the planner validates the robot
directory and the URDF together and raises ONE error listing every problem:
a missing file, a missing required field, the world's `base_frame` not
matching the robot file's root, a URDF missing a link or joint the robot file
names, a URDF whose kinematics below the arm's base drift from the robot
file's reference, a layer generated for more padding than configured. The
node logs it as FATAL and exits 1, as it does today for a missing model.

**No Gen3 names in Python.** The names `core/rammp_curobo/urdf_model.py`
hard-codes (`ROOT_LINK`, `EE_LINK`, `ARM_JOINTS`, `GRIPPER_JOINTS`,
`BASE_LINK`, `KINEMATICS_REFERENCE`) and `config.PLANNER_DEFAULTS`'
`joint_names` come from the robot file instead (`base_link`, `ee_link`,
`cspace.joint_names`, `lock_joints` plus the mimic joints read from the URDF,
and two new fields for the arm's base link and the kinematics reference). The
check keeps its protection — a URDF that is not the arm the spheres were
fitted on is refused by name — for any arm.

## Example mode (decided)

With no `config` given, the planner runs the shipped example: the Gen3 +
2F-85 robot file, the default planner settings, the example world, and — my
call, from "the planner should work standalone" — a shipped example Gen3 URDF
(level mount), so it plans without `arm_state_publisher`. Once a `config` is
named, the example URDF is never used; the URDF comes only from the topic.

Using the right config is the user's job; example mode does not refuse to
start. But it must be impossible to miss that no config made it in:

- a startup banner saying no config was given and the example robot from the
  image is running;
- if a model is (or later becomes) latched on `/arm/robot_description`, a
  WARN saying a robot is being published and this planner is ignoring it
  because no config was given;
- every plan result's `message` and the `SetWorld` reply carry an
  `(example robot)` marker, so a caller's own logs show it too.

## Moves (decided)

RAMMP-CuRobo keeps, as examples and test data: the Gen3 + 2F-85 robot file
and its kinematics reference, the example world(s), `world_empty.yaml`, the
MEBot fixture set in `core/tests/data/mebot/` and its example generator input
`scripts/mebot_layers.yaml`, and the tools (`bake_robot_config.py`,
`generate_layers.py`). The Docker image copies code and the example set only —
not the tests, the sim or the scripts (`COPY .` goes); `make verify`
bind-mounts the checkout's tests.

rammp-deployments gains, per robot in `config/robots/<robot>/curobo/`:

- **flatbot**: `planner.yaml`, its robot file, `world.yaml`, the generated
  layers, and the generator input — flatbot's own chair measurements (arm
  mount offset, seat, person margins). `make layers INPUT=… OUTDIR=…` in
  RAMMP-CuRobo writes from there. It starts from PR #46's values: platform,
  seat AND the fitted person, arm at y 0.36 — all unmeasured best guesses.
  Known consequence: flatbot's rest pose sits 25 mm inside `person/torso`, so
  plans starting from rest are refused until the rest pose, the measurements
  or per-part ignore (#19 PR 2) change that.
- **dojo**: what `gen3_real.yaml` + `world_real_bench.yaml` say today
  (pedestal, table), as `planner.yaml`, robot file and `world.yaml`, in the
  same deployments PR (my call).
- **gen_1_5** gets nothing: no profile runs a planner there; starting one
  without files is the intended loud failure (my call).
- `sheppy-manifest.yaml`: the planner nodes stop defaulting to
  `gen3_real.yaml` / `world_real_bench.yaml`; every profile that runs a
  planner points `config` at `/config/robot/curobo/planner.yaml`.

## Left to the builder

- Order of PRs and how to keep each robot running between them (the nightly
  image and the deployments manifest must not disagree for a night).
- The exact field names of the two new robot-file fields, the `example:`
  syntax, and the file names in the example set.
- Whether the `world` node parameter survives (today a non-empty one replaces
  the config's world, which every robot profile must override to `""`).
- Which tests move to fixtures and how `PACKAGED_CONFIG_DIR` lookups in tests
  are rewritten.
- The written contract's home (README section or its own doc) and the
  versioning: removing bare-name resolution of packaged files is a MAJOR
  change for anyone launching `config:=gen3_real.yaml`.

## Relationship to open work

- PR #46 (seat, fitted person, mount 0.36) stays as is: the generator change
  and the example/fixture set. flatbot's real measurements land in
  rammp-deployments under this design, not in RAMMP-CuRobo.
- The per-part ignore / per-plan `ignore_layers` work (PR 2 of #19) is
  independent and can go before or after.
