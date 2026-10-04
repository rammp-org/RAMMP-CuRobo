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
scripts/                  config baking; generate_layers.py (the MEBot
                          layers: the chair and floor from its URDF, the
                          person from ANSUR II sizes)
sim/                      mebot_sim: the planner on the MEBot in a MuJoCo
                          sim over ROS 2 (`make sim`, Jetson)
docker/                   the planning service as a container (Jetson/JP6)
docs/HARDWARE_BRINGUP.md  the real-arm runbook — READ BEFORE TOUCHING HARDWARE
```

> **Nothing in this repo can move the arm.** There is no executor, no
> controller client and no execute flag — a plan is data until whoever
> owns the robot chooses to run it, under their gates (on this bench,
> `kinova_arm_ros2`). The one exception is a test kit: the MEBot sim
> (`sim/`) flies plans through the arm driver's interface, but only on its
> simulated arm, never on the bench's ROS domain (0) and never beside a
> real driver. A human still holds the physical e-stop during ALL hardware
> runs. See **issue #6** for why the boundary is drawn here.

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
trajectories in the sim-kitchen world and on the MEBot wheelchair (first
run compiles kernels — takes a minute or two; warm, they take ~30 s and
the whole suite ~45 s). Without CUDA, torch or cuRobo the GPU tests skip,
and the suite passes on the offline tests alone; where they must run, as
on the Jetson, set `RAMMP_CUROBO_GPU=1` and a skip fails instead.
`make verify` sets it. Or plan one motion directly:

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
and **docker/** to run the same thing as a container. With the arm on the
MEBot wheelchair, run `config:=gen3_mebot.yaml` instead (see **Obstacle
layers** below).

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
client, no `execute` parameter. Only the MEBot sim's test client reads
`/joint_states` and sends trajectories, and only to its simulated arm (see
**How it is tested**).

Executing a plan means handing the returned `trajectory_msgs/JointTrajectory`
to whoever owns the robot, with their gates and their e-stop discipline.
On this bench that is **`kinova_arm_ros2`**, whose `GoToEEPose` /
`GoToJointConfig` actions call this planner, supply the arm's measured `q`
as `start_joints`, validate what comes back, and run it through the
supervisor and driver with a path tolerance.

Two safety authorities with different rules is worse than either alone —
that, plus plans made from a joint state up to 2 s stale, is why the
executor that used to live here was removed (**issue #6**).

## Obstacle layers: the wheelchair and the person

On the MEBot wheelchair the arm works beside the chair it is mounted on and
the person sitting in it. The chair's drive base and the person reach the
planner as **layers** (issue #19): named groups of boxes, each in a file of
its own, composed into the collision world beside the scene's own
obstacles.

- **Config layers** are listed under `layers:` in the planner config and
  are composed into **every** world the planner builds, `set_world` pushes
  included. A push replaces the scene, never these, so a consumer that
  pushes a whole world before every plan cannot drop the chair by leaving
  it out.
- **Scene layers** are listed under `layers:` in a world file and belong to
  that scene: the next push replaces them with its own. Geometry that must
  never go away belongs in the planner config.

A layer file listed by name is found next to the file that lists it or
among the packaged configs, never in the current directory. Its boxes are
named `<layer>/<box>` (`person/lap`), padded by `world_padding` like every
other box, and count against the same 60-box cache; a world that overflows
it is refused, naming every source, and nothing is truncated.
`planner.world_summary()` says where a world's boxes come from and
`planner.world_boxes` lists them. A failed plan's error, and
`check_state_valid`, name the boxes a state reaches into, deepest first.

### The chair config

```bash
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_mebot.yaml
```

`gen3_mebot.yaml` is the Gen3 on the MEBot, its base at the RAMMS model
pose (0.505, -0.265, 0.376) m in the MEBot's frame, square to the chair —
unverified on the real chair. It starts in a world of 25 boxes:

| Layer      | File                        | Boxes | In the world as                     |
| ---------- | --------------------------- | ----- | ----------------------------------- |
| `platform` | `layer_mebot_platform.yaml` | 16    | a config layer, not excludable      |
| `person`   | `layer_mebot_person.yaml`   | 8     | a config layer, excludable          |
| `floor`    | `layer_mebot_floor.yaml`    | 1     | a scene layer of `world_mebot.yaml` |

The platform is the drive base alone, with every joint at zero, and its
coverage numbers below are for that pose. Nothing models the rest of
the chair: its seat, backrest, armrests and footplate, or the plate the arm
is mounted on. Over the seat and at the backrest the person layer is the
only geometry.

The person layer leaves a gap of its own: the column carved around the
arm's base, ±0.105 m about the arm's axis in x and y and from z -0.040 to
0.409 m (`base_link`), which no person box enters, since a padded box there
would put every start state in collision. The arm's base and shoulder fill
the middle of it; nearly two thirds of what the carve takes out of the
person lies beside them, where the arm can move. So a part of the occupant
beside the mount, the right hip, elbow or forearm, is not kept from the arm.

The node logs the world's summary at start-up and in its ready line, and
returns it with every world SetWorld takes — the visible proof that the
chair config is active:

```text
Collision world: 25 boxes: scene 0 | config layers: platform 16, person 8 | scene layers: floor 1
```

The floor belongs to the start-up scene, so the first push replaces it: a
pushed scene brings its own floor. The two config layers take 24 of the
cache's 60 boxes, which leaves 36 for a pushed scene.

**The bench retract is not a valid start on the chair.** The robot
config's `retract_config`, which is also cuRobo's IK seed, is 154.4 mm
inside the padded person layer (`person/lap`), and the planner warns about
it at start-up. Plan from the arm's measured state, as every caller must
anyway. A start inside a layer fails `INVALID_START`, naming the boxes:

```text
... [start collides with person/lap (154 mm), person/torso_head_3 (134 mm), person/torso_head_1 (89 mm)] [world: 25 boxes: scene 0 | config layers: platform 16, person 8 | scene layers: floor 1]
```

### Leaving the person out

A layer marked `excludable: true` may be left out of one world. The person
is; the platform is not:

```python
planner.update_world("world_mebot.yaml", ignore={"person"})  # platform untouched
planner.update_world("world_mebot.yaml")                     # the person is back
```

`ignore={"platform"}` is refused (`layer 'platform' is not excludable`),
and so is a name that is both a layer and a prop. This is Python-only:
SetWorld carries a world name and nothing else, and a field on it would be
a MAJOR change. It is the hook a later approach to the person's mouth will
use.

A world without the person has nothing over the seat or at the backrest
either, so a plan made in it is free to pass through them. If yours go near
them, carry boxes for them in the scene you push.

### Generated, with coverage numbers: `make layers`

The three layer files are GENERATED by `scripts/generate_layers.py`. Every
number in them comes from `scripts/mebot_layers.yaml`, which gives each
input its source (the mount, the budgets, the anthropometry, the seat, the
floor), or from the URDF and the robot config it names. The platform is
the MEBot drive base's URDF in rammp-org/Demo-Software, fetched by commit
and checked by sha256 — from the cache, a local clone or GitHub raw; no
copy is committed. The generator runs on the CPU (numpy, PyYAML) and
writes the same bytes on amd64 and arm64.

```bash
make layers    # regenerate the three files; fails on any diff
```

Never hand-edit a layer file: change the inputs, run `make layers` and
commit what it writes. CI regenerates the files from the pinned URDF and
fails on any difference.

| Layer      | Boxes | What it holds, and how well                                                                                                                                                                                                                     |
| ---------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `platform` | 16    | The MEBot drive base: its URDF's 46 collision primitives (40 boxes, 6 cylinders), every joint at zero. Under-coverage 0, by certificate; over-coverage 10.9 % of the primitives' bounding boxes (budget 15 %), about 15 % of their true shapes. |
| `person`   | 8     | Any seated occupant from the ANSUR II 5th-percentile woman to the 95th-percentile man, plus 0.05 m on every face, in three segments (torso and head, lap, lower legs and feet) carved around the arm's base.                                    |
| `floor`    | 1     | A 3 × 3 × 0.1 m slab whose top, z -0.4593 m, is the URDF's lowest point.                                                                                                                                                                        |

The platform's under- and over-coverage are in its file's `meta`, which the
regeneration freezes with the boxes; the share of the true shapes is
measured on a 1 mm lattice. Every padded box clears the arm's
joint-independent collision spheres (its base, and the shoulder on the
joint_1 axis) by at least 4 mm, or the generator writes nothing: the
platform by 53.4 mm, the person by 5.1 mm.

**What the person costs.** The arm's base is 0.265 m from the occupant's
centreline, and the person layer is over-sized on purpose, so it confines
the arm. Of 1,000,000 arm configurations drawn at random (each joint
uniform over its range, the four continuous ones over two full turns), the
share whose collision spheres clear the padded boxes:

| World                   | All configurations | Tool high and in front (z > 0.3 m, x > 0.2 m) |
| ----------------------- | ------------------ | --------------------------------------------- |
| floor                   | 98.1 %             | 100.0 %                                       |
| floor, platform         | 88.2 %             | 99.7 %                                        |
| floor, platform, person | 2.3 %              | 1.3 %                                         |

Plans near the person, and anything handed to them, need the person layer
excluded, which leaves the seat and the backrest uncovered too (see
**Leaving the person out**). The sizes are data: tuning them is one edit to
`scripts/mebot_layers.yaml` and `make layers`.

### How it is tested

Every kind of box the planner emits has a pair of tests that fails if
cuRobo ignores it: with the box on the path the plan detours or fails, and
without it the same plan goes straight. `core/tests/test_smoke.py` (GPU)
holds the pairs for plain, rotated and layer boxes and for the generated
platform, reads cuRobo's collision cache back against the composed boxes,
and checks cuRobo's spheres against every padded box along the chair's
plans. `test_layers.py` and `test_layergen.py` run offline, in CI too.
Among them, every occupant of the person's range — each of its twelve
sizes at either end of its range, 4096 occupants, with the backrest at
either end of its own — lies inside the committed person layer or the
column carved around the arm's base.

```bash
make sim        # the MEBot sim suite (Jetson): exit 0 only if every gate passes
make sim-test   # the sim package's own tests (Jetson; no ROS graph, no GPU)
```

`make sim` flies the planner's plans on a simulated Gen3 beside the exact
MEBot, from its URDF, and a seated body, over ROS on a domain of its own
(88 by default, never 0). Planned without a layer, a request hits the
shape that layer covers; planned with it, the request is refused or flown
clear. The seated body, at the sitting heights of the 5th-percentile
woman, the 50th-percentile man and the 95th-percentile man and in three
postures, lies inside the padded person layer or that column. On the
Jetson (2026-10-03) it ran 45 cases in about three minutes, every gate
passed, and the chair planner flew 7 of its 8 ordinary reaches touching
nothing. See `sim/mebot_sim/README.md`.

### Meshes: not attempted

The issue's stretch goal, mesh layers, was not attempted. cuRobo v0.7.8's
MESH checker has three silent traps (in `geom/sdf/world_mesh.py`):

- an update with no meshes keeps the old ones (lines 101-117);
- a mesh whose name is reused keeps its old geometry (lines 167-173);
- a mesh-cache overflow reallocates the cache even under
  `fix_cache_reference` (lines 103-105), which by its docstring breaks
  recorded CUDA graphs (lines 96-99).

cuRobo's own documentation also puts its cuboid checker at 4x the speed of
its mesh checker (<https://curobo.org/get_started/2c_world_collision.html>).
So layers are boxes only, and the paired box tests above are the evidence
that cuRobo honours them.

## Troubleshooting

| symptom                                                            | cause / fix                                                                                                                                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pip install -e` fails (`build_editable` hook / permission denied) | old pip + isolated build env; use `python3 -m pip install --user --no-build-isolation -e ./core`                                                                                                  |
| pytest crashes collecting (`No module named '_pytest.scope'`)      | user-site anyio plugin vs system pytest; repo `pytest.ini` disables it (`-p no:anyio`) — run pytest from the repo root                                                                            |
| first plan takes minutes                                           | one-time CUDA kernel compile (warmup); subsequent runs ~20 s init, ~0.2-2 s per plan                                                                                                              |
| `plan_single_js` fails / `DT_EXCEPTION` on older Jetson wheels     | known torch-wheel cuSOLVER gap; `joint_space_method: auto` falls back to FK-pose planning automatically. Keep `enable_graph: false` on Jetson always                                              |
| plan succeeds but joints differ from a joint goal                  | FK-pose fallback reached the POSE via another joint family — check `goal_mismatch_rad` before acting on it                                                                                        |
| goal aborted: "start_joints is required"                           | `start_joints` is not optional — the planner has no view of any arm, so the caller must send the configuration to plan from                                                                       |
| node warns about SIM world without sim time                        | you're (probably) on the real arm with the kitchen world — relaunch with `world:=world_real_bench.yaml` (measured!)                                                                               |
| nodes can't see each other's topics                                | `ROS_LOCALHOST_ONLY=1` must be exported in EVERY shell (non-interactive shells skip `~/.zshrc` — export explicitly; RAMMP-Kinova's `tools/launch_stack.zsh` does)                                 |
| both bringups fight / controllers flap                             | MuJoCo sim and kortex bringup both claim `/controller_manager` — run exactly one                                                                                                                  |
| `update_world` seems ignored / obstacles missing                   | cuRobo v0.7.8: cylinders/spheres in a WorldConfig are silently dropped (cuboids only), and an empty world silently keeps the previous one — the library guards both, custom worlds go in as boxes |
| `INVALID_START` on `gen3_mebot.yaml`                               | the start is inside a layer; the error names the boxes and how deep. The bench retract is inside the person layer — plan from the arm's measured state                                            |
| set_world refused: "empty apart from the configured layers"        | the pushed scene has no obstacles, objects or layers of its own (a world file cut short?); the planner config's layers alone are never taken as a world                                           |
| `AttributeError: wp.torch` in mesh collision                       | newer warp needs explicit `import warp.torch` — the library does this; if embedding cuRobo yourself, copy that                                                                                    |
| `ros2 topic echo` prints "A message was lost!!!"                   | benign QoS depth artifact of echo on a 500 Hz topic                                                                                                                                               |

## Integrating from another RAMMP module

See **INTEGRATION.md** — 5 lines for pure-Python planning, ~15 for
plan-over-ROS-actions, and the Docker service.
