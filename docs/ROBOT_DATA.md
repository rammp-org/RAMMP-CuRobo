# Robot data: what a robot supplies

One planner image serves any arm, any MEBot base and any site. The image holds
code and a worked **example**; everything that describes a particular robot
or its world comes from outside it, from the robot's own directory. This page
is the contract: what that directory holds, how the planner finds it, and what
it checks at startup. Design:
`docs/superpowers/specs/2026-10-28-robot-data-outside-image-design.md`.

## The robot directory

Conventionally `config/robots/<robot>/curobo/` in rammp-deployments, mounted
read-only at `/config/robot/curobo/` in the planner container and selected with
`config:=/config/robot/curobo/planner.yaml`.

| File                                     | Says                                                                                                                                                                 | Required       |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `planner.yaml`                           | names the robot file and the start-up world; lists the always-on layers; optional planner settings                                                                   | yes            |
| robot file (any name, e.g. `robot.yaml`) | cuRobo's robot config: root link, tool link, planned joints, collision spheres, self-collision, locked joints, IK seed (`retract_config`), and a `model_check` block | yes            |
| kinematics reference (JSON)              | the arm's joints below its base, as the spheres were fitted on them (`model_check.kinematics_reference`)                                                             | yes            |
| start-up world (e.g. `world.yaml`)       | `base_frame` equal to the robot file's root link; the scene the planner starts with; replaced by the first `set_world` push                                          | yes            |
| layer files                              | always-on obstacles (the chair, the person), each with `meta.padding_assumed_m`                                                                                      | only if listed |

The arm's URDF is **not** in the directory. It arrives on
`/arm/robot_description` from `arm_state_publisher` (rendered from the robot's
`mount.yaml`), except in example mode, below.

## Resolution rule

Every file a named `planner.yaml` references resolves **next to that
`planner.yaml`**, or as an absolute path. Never from the image by bare name.
That covers `robot`, `world`, each entry of `layers`, a robot file's
`model_check.kinematics_reference`, a world file's own `layers`, and the
generator's `keep_out.robot_config`. The same holds for
`update_world` and `SetWorld` pushes: a world named by a relative path
resolves next to the planner config, an absolute path is taken as is, and a
bare packaged name is refused.

The one way to reach a shipped file is an explicit `example:<name>`, e.g.
`robot: example:robot_gen3_2f85.yaml`. The log then says `(example, image)`
for that piece. `example:` with an unknown name lists the examples. A
reference that fails to resolve but shares an example's name says so and names
the `example:` spelling to use. Nothing falls back silently: Docker can start
the planner with an empty `/config/robot`, and a fallback would plan against
the image's stale geometry.

Robots should carry their own copy of the robot file (a new image then can
never change their spheres or IK seed); `example:` is allowed and visible.
A copied robot file does not follow a re-bake of the example: when a new image
re-bakes the example's spheres or seed, re-copy the file deliberately.

## `planner.yaml`

| Key           | Required | Notes                                                                                                                                                                               |
| ------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `robot`       | yes      | the robot file                                                                                                                                                                      |
| `world`       | yes      | the start-up world; an empty one (`obstacles: []`) is accepted here only                                                                                                            |
| `layers`      | no       | obstacle layer files composed into every world, pushes included; default `[]`                                                                                                       |
| `planner`     | no       | tuning knobs; each defaults field by field from `config.PLANNER_DEFAULTS`                                                                                                           |
| `joint_names` | no       | the same joints as the robot file's `cspace.joint_names`, in the order plans are returned; default the robot file's order. Any other set is refused ("not the robot file's joints") |
| `tool`        | no       | tool corrections (`spin_deg`, `tip_offset_m`); default 0.0                                                                                                                          |

Robot data (robot file, world) is whole or refused: it is never merged field
by field with the example's. Only the planner settings default. The planned
joints are always the robot file's `cspace.joint_names`; `joint_names` here
only reorders them.

## The robot file

cuRobo's robot config (`robot_cfg: kinematics: ...`) plus one block beside it.
The planner reads these fields; the example's values are given for orientation
(`core/rammp_curobo/configs/robot_gen3_2f85.yaml`, generated by
`scripts/bake_robot_config.py`).

| Field                                                                                | Meaning                                                                                                                 | Example                                   |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `robot_cfg.kinematics.base_link`                                                     | the root link: the frame plans, worlds and layers are in                                                                | `arm_mount`                               |
| `robot_cfg.kinematics.ee_link`                                                       | the tool link goals are given for                                                                                       | `arm_tool_frame`                          |
| `robot_cfg.kinematics.cspace.joint_names`                                            | the planned joints, in order (also the order of every returned plan)                                                    | `arm_joint_1` .. `arm_joint_7`            |
| `robot_cfg.kinematics.cspace.retract_config`                                         | the IK seed, not a home pose                                                                                            | bent-elbow family (`arm_joint_3` near pi) |
| `robot_cfg.kinematics.lock_joints`                                                   | joints held fixed, by their actuated joint only (cuRobo folds in mimics)                                                | `arm_robotiq_85_left_knuckle_joint: 0.0`  |
| `robot_cfg.kinematics.urdf_path`                                                     | the sentinel `from_robot_description`: the model comes from the topic                                                   | `from_robot_description`                  |
| `robot_cfg.kinematics.collision_spheres`, `collision_link_names`, `self_collision_*` | the collision model                                                                                                     | audited Gen3 + 2F-85 spheres              |
| `model_check.arm_base_link`                                                          | where the kinematics reference starts; the joint above it (the mount) may vary                                          | `arm_base_link`                           |
| `model_check.kinematics_reference`                                                   | JSON of every joint below `arm_base_link` the spheres were fitted on, resolved next to the robot file or via `example:` | `arm_kinematics.json`                     |

`load_robot_spec` refuses a file missing `base_link`, `ee_link`,
`cspace.joint_names`, `model_check.arm_base_link` or
`model_check.kinematics_reference`, naming the file and each one. The rest of
`robot_cfg` is cuRobo's.

## The startup check

Before cuRobo is imported, `check_robot_dir` validates the whole directory and
the URDF together and raises **one** `RobotDataError` (a `ValueError`) whose
`.problems` lists every problem it found, so a bad deployment is fixed in one
pass: a missing file, a missing robot-file field, a world whose `base_frame`
is not the robot's root link, a layer generated for more padding than
configured (or an invalid box, a duplicate name, a world over the box cache),
a URDF missing the root or tool link, a joint the robot file names or a link
it puts collision spheres on, a URDF
whose joints below `arm_base_link` drift from the kinematics reference (1 mm,
0.1 deg, structure, joint set), a moving joint with no `<limit>` that follows
no other. The node logs it FATAL ("robot data or the model on /arm/robot_description:
robot data in <dir>: N problems ...") and exits 1.

The check needs the URDF, so it runs once the model has arrived. Only a
missing or unparseable `planner.yaml` fails at once: the planner config is read
**before** the node waits for a model, so an unmounted
`/config/robot/curobo/planner.yaml` exits 1 immediately (the node no longer
waits out a late bind-mount). Every other problem is reported after the model
arrives; without `arm_state_publisher` the "no model" FATAL comes first. To
check a directory offline, run
`check_robot_dir(cfg, cfg_dir, urdf_xml)` (`rammp_curobo.contract`) on the
planner config and a URDF file's text.

## Example mode

With no `config` (the node's default, `""`), the planner runs the shipped example: the Gen3 + 2F-85 robot file
(`example:robot_gen3_2f85.yaml`), the default planner settings, the kitchen
world and a shipped level-mount Gen3 URDF (`example:arm_level.urdf`), so it
plans standalone with no `arm_state_publisher`, deployment checkout or robot.
Once a `config` is named the example URDF is never used.

The image's default command names `config:=/config/robot/curobo/planner.yaml`,
so a bare `docker run <image>` with no robot directory exits 1 naming that
file. Example mode in the image is one explicit word: `docker run <image> example` (further arguments go to the node). `ros2 run` and `ros2 launch`
from a checkout are example mode with no arguments.

It cannot be mistaken for the real thing. Three signals:

1. a startup banner, logged at WARN: no config given, the example robot from the image is
   running;
1. a WARN if a model is (or later becomes) latched on `/arm/robot_description`:
   a robot is being published and this planner is ignoring it;
1. an `(example robot)` marker on every plan result's `message` and on the
   `SetWorld` reply, so a caller's own logs show it.

## Swapping an arm, a base or a site

A rammp-deployments commit; the image does not change.

- **A site** (new bench, new room): edit `world.yaml` in the robot directory.
- **A base** (MEBot chair geometry, where the arm sits): edit the generator
  input and regenerate the layers into the robot directory:
  `make layers DATA=<robot>/<inputs>.yaml OUTDIR=<robot>/curobo`
  (`DATA=` without `OUTDIR=` is refused, so a checkout's fixtures are not
  overwritten). The generator takes the robot file from next to its input. Commit what it writes.
- **An arm**: bring a robot file with its own spheres, `retract_config`, joint
  names, locked joints and `model_check`, and the kinematics reference JSON
  its spheres were fitted on. Generating spheres for a new arm is not
  automated; start from the example's robot file and
  `scripts/bake_robot_config.py`.
- In all cases: `planner.yaml` names the files with relative paths, the
  profile that runs the planner passes `config:=/config/robot/curobo/planner.yaml`
  and overrides the `world` parameter to `""` (a non-empty `world` replaces
  the config's), and the planner is restarted when the robot's mount
  changes.
- Check before deploying: run `check_robot_dir` on the directory and the
  arm's URDF offline, or start the planner with `arm_state_publisher` up; a
  bad directory exits 1 with every problem in the one FATAL.
