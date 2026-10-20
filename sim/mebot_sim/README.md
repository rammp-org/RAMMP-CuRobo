# mebot_sim: the MEBot sim stack

The planner's obstacle layers for the MEBot wheelchair (issue #19), tested
the way a consumer uses the planner: over ROS. A MuJoCo Gen3 stands in for
the arm driver, with the driver's names, beside two of this repo's planner
nodes, on a ROS domain of its own:

| Node                    | What it is                                                                                                                                                                                                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/mebot_sim`            | the menagerie Gen3 + 2F-85 with physics, the exact MEBot (its pinned URDF through MuJoCo's own importer), a seated stand-in and the floor; `/joint_states`, `/execute_joint_trajectory`, the arm's model latched on `/arm/robot_description` (`core/tests/data/arm_level.urdf`, a level mount), a live view |
| `/rammp_curobo`         | the planner with `config:=gen3_mebot.yaml`: the platform and person layers in every world, what a consumer on the chair gets                                                                                                                                                                                |
| `/rammp_curobo_control` | the planner with `config:=gen3.yaml`, which composes no layers: the suite pushes it scenes that list only some, so that the same request is planned without a layer                                                                                                                                         |

Contacts are MuJoCo's, against the true shapes: any arm or gripper contact
with the MEBot, the stand-in or the floor pressing with more than 1 N. A
flight is judged on the sim's first report sampled after it ended, once the
arm has had 0.5 s to come to rest: one sampled during the flight would miss
what it touched at its end, and with no later report within 5 s the run
ends early. The layers the planner was told about are drawn as translucent
overlays (blue platform, orange person, grey floor) and collide with
nothing.

## Safety

`/mebot_sim` serves the real driver's names, so the stack never runs on the
bench's ROS domain, 0. When `ROS_DOMAIN_ID` is unset, 0, or anything but a
plain domain from 1 to 232, the launch files and every node move to 88 and
say so; another domain from 1 to 232 is kept. `/mebot_sim` also refuses to
start beside anything that already serves `/joint_states`,
`/arm/robot_description` or `/execute_joint_trajectory`, and the client flies nothing unless the sim
alone does. No node here can reach an arm.

## Run it

On the Jetson, from the checkout:

```sh
make sim                                  # build .sim_ws/, run the suite: exit 0 only if every gate passes
ROS_DOMAIN_ID=93 make sim                 # on another domain (88 is the default)
make sim SIM_ARGS=port:=8766              # launch arguments, e.g. another live-view port
make sim-test                             # the package's own tests: no ROS graph, no GPU
```

`make sim` builds `rammp_curobo_interfaces`, `rammp_curobo_ros` and this
package into the git-ignored `.sim_ws/`, with `SIM_OVERLAY` (default
`~/rammp_deps_ws/install/setup.bash`) providing `rammp_arm_interfaces`, and
launches `suite.launch.py`: the stack, with `run_suite` beside it. The
launch ends when `run_suite` does, and exits with its verdict. A run takes
about three minutes, the build and the planners' warm-up included.

By hand, in shells that have each sourced ROS, the overlay and
`.sim_ws/install/setup.bash`:

```sh
ros2 launch mebot_sim sim.launch.py [posture:=none] [host:=127.0.0.1,<address>] [port:=8765]
ros2 run mebot_sim try_goal --pos 0.45 0.20 0.25 --control    # one goal, on both planners
ros2 run mebot_sim run_suite [--only S2] [--speed 0.5]         # the suite, against the running stack
ros2 run mebot_sim check_layers                               # S0 alone: CPU, no ROS graph
ros2 run mebot_sim find_pairs                                 # rewrite config/scenarios.yaml (GPU)
```

The live view, `http://127.0.0.1:8765` by default, streams the sim while
the stack is up: orbit, pan and zoom, send a tool goal to either planner,
push the joints by hand, switch the stand-in's posture. No test depends on
it.

Prerequisites, all on this Jetson: ROS 2 Humble with `rmw_cyclonedds_cpp`;
the overlay; Python `mujoco` (3.10) and Pillow, rendering on EGL;
`~/mujoco_menagerie` (`kinova_gen3`, `robotiq_2f85`) and the DeepMind
humanoid of ROS's `mujoco_vendor` (the launch's `menagerie:=` and
`humanoid:=`); the pinned MEBot URDF, from the generator's cache, a
Demo-Software clone or GitHub; cuRobo v0.7.8 on the GPU for the planners.

## The scenarios

`run_suite` flies the requests that `find_pairs` wrote to
`config/scenarios.yaml`, each from its start, the sim arm put there first.

| Scenario | What runs                                                                                                                                                                                                                       | Gates  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| S0       | `check_layers`, no planning: MuJoCo's parse of the URDF against the committed platform boxes, and the stand-in at three sizes, in three postures, its back at each end of the backrest's range, against the padded person layer | G0, G1 |
| S1       | 3 platform pairs, the chair empty. With the floor and the platform, the control planner refuses each or flies it clear; with the floor alone it plans it, and the flight hits the MEBot                                         | G2, G3 |
| S2       | 3 person pairs, the stand-in seated: the chair planner refuses each or flies it clear; the control planner, with the floor and the platform, plans it, and the flight hits the stand-in                                         | G2, G3 |
| S3       | 8 joint goals from one start on the chair planner: every flight clear, and at least `min_chair_reaches` of them planned and flown                                                                                               | G2, G7 |
| S4       | a consumer's push to the chair planner, `config/consumer_world.yaml` (8 boxes, no layers); then the S2 requests again                                                                                                           | G4, G2 |
| S5       | a plan from the bench retract on the chair planner, nothing flown                                                                                                                                                               | G5     |

A pair's straight joint path takes MuJoCo's arm, posed along it, into the
shape its layer covers (`crossing_mm` in `scenarios.yaml`), and cuRobo finds
that path blocked with the layer and clear without it: the layer is what
stands between the request and the shape. Every flight is flown at half
speed (`--speed`), time dilated by the client as consumers do.

## The gates

`run_suite` exits 1 if any gate fails, or if the run ends early. A run of
every scenario (no `--only`) also fails a gate that none of its cases bore
on, and G6 when either planner flew nothing for it to judge: nothing is
skipped.

- **G0, registration.** Every surface sample of MuJoCo's own parse of the
  MEBot URDF lies inside a box of the committed platform layer, unpadded,
  and the same boxes shrunk by 1 mm leave samples outside. The samples
  cover only what MuJoCo parsed, so its parse must also hold every
  collision primitive of the URDF as the generator reads it (46: 40 boxes
  and 6 cylinders), link by link; the line names any it lacks. A failure
  means the platform no longer covers the chair the sim simulates, or the
  check could not have failed.
- **G1, stand-in coverage.** The stand-in is seated at each size of
  `standin.sizes` in `scenarios.yaml`, the sitting heights of the
  5th-percentile woman, the 50th-percentile man and the 95th-percentile man
  (ANSUR II row 8), in each posture, with its back at either end of the
  backrest face's range: 18 bodies. Every point of each, sampled through its
  volume on a 6 mm grid, lies in the person layer as the chair planner pads
  it, or in the arm's carved column, which holds the arm's base and the
  space around it that no box may fill. The arm can move in that space, so a
  part of the body there is not kept from it (the main README's **The chair
  config**). The column comes from the generator, as the layer does, and a
  column carved too wide, or a body too small, would leave no gap to fail
  on. So two bounds of `thresholds` in `scenarios.yaml` hold as well: at
  least `covered_pct` percent of the body (99) lies in the layer itself, and
  the body is the whole humanoid at the scale that seats it at its size, its
  volume within `body_volume_pct` percent (2) of the humanoid's own times
  the scale cubed. The sizes are this suite's data, not the occupant
  `scripts/mebot_layers.yaml` sizes the layer for, so a layer that holds
  fewer bodies than they span fails: with the legs sized by the
  95th-percentile thigh alone, the shins and feet of the two smaller bodies
  hang behind them, and 6 of the 18 fail. With no size listed, S0 is an
  error: G1 would judge no body.
- **G2, no contact with the layers on.** No flight planned with the layers
  that cover the scene touches the MEBot, the stand-in or the floor. A
  refusal passes: the planner may decline a request it cannot do safely.
  A refusal is cuRobo finding no plan (`IK_FAIL`, `GRAPH_FAIL`,
  `TRAJOPT_FAIL`, `FINETUNE_TRAJOPT_FAIL`), or the planner's re-check
  flagging states of the plan it found (`LIBRARY_VALIDATION_FAILED`). Any
  other failure judged nothing and fails: a busy planner, an exception, a
  re-check that could not run, an answer with no message.
- **G3, the positive controls.** Every request planned without the layer
  that covers a shape is planned, flown, and hits that shape. This is the
  sim showing that it sees the chair and the body, and that leaving the
  layer out is dangerous; without it, G2 could pass on a sim that sees
  nothing.
- **G4, a push keeps the layers.** After a consumer's push of a scene that
  lists no layers, the chair planner's reply names the platform and person
  layers with their boxes, and the S2 requests are still refused or flown
  clear.
- **G5, diagnostics.** A plan from the bench retract fails, and its error
  names the person box the start is in and how deep.
- **G6, the sim flew the plans.** Each planner flew a plan that touched
  nothing, and every flight that touched nothing came to rest within
  `end_error_rad` (0.05 rad) of its plan's last point. Each planner is
  judged on its own flights: the control planner's show nothing of how the
  sim flies the chair planner's plans. A run of one scenario judges the
  flights it made.
- **G7, the chair planner plans.** It planned and flew at least
  `min_chair_reaches` (6, in `thresholds` in `scenarios.yaml`) of the 8 S3
  reaches, touching nothing; the line `S3 reaches` gives the count. G2 and
  G4 pass a refusal, so a chair planner that refused every request would
  pass them: G7 fails it, and so does G6 in a run of every scenario. A
  prototype of the suite flew 7 of the 8, and so does the suite: the first reach is refused
  (`FINETUNE_TRAJOPT_FAIL`).

Reported, not gated: how many of the requests asked of the chair planner
it flew, plan times, the closest approach to each shape in the flights with
the layers on, and how far the arm was from a plan's last point as a flight
that touched nothing ended. MuJoCo's arm meshes and cuRobo's spheres differ
by design, by up to a few millimetres, so the closest approach is a number
to read, not a gate.

## Reading the result

One line per case: the case, the planner and the world it planned in (or
what was checked against what), what came of it, PASS or FAIL. The world is
as the planner summed it up in its reply to the push, its boxes by the
layers they come from:

```text
/rammp_curobo (25 boxes: scene 0 | config layers: platform 16, person 8 | scene layers: floor 1)
```

Then each gate's verdict, the reported numbers, and the result file:
`${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/sim/<UTC>-<git sha>.json`, the
same cases with their numbers, outside the repo.

**`premise broken; re-run find_pairs`** means the requests no longer test
what they were found to test: `scenarios.yaml` was found with other layer
files than the committed ones (their sha256 is written beside the requests),
or holds fewer requests than `find_pairs` finds (3 platform pairs, 3 person
pairs, 8 reaches), or a positive control did not hit its shape, or a start
or goal found valid was refused. The planners plan to joints by `auto`,
whose fallback plans to the goal's pose, so a failed plan does not say that
its joint goal is in a layer. Each request planned with its layers that the
planner refused or flew is therefore followed by a plan from its goal to
itself, in the same world: the planner refuses that one `INVALID_START`,
naming the boxes, when the goal is no longer valid. After `make layers`
changes a layer, run `ros2 run mebot_sim find_pairs` and commit the
`scenarios.yaml` it writes. It searches with fixed seeds and keeps the
hand-written part of the file (the stand-in's postures and the thresholds)
as it is.

The suite holds the planners to the layer files, not to what a planner
config lists. A chair config that drops the person layer is no stale
request but a regression, and the gates catch it: tried on a scratch copy,
the person-pair and reach flights hit the stand-in (G2), the push's reply
lacks `person 8` (G4), and the retract plans (G5).

## Files

| Path                                 | What it is                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `mebot_sim/stack.py`                 | what every process shares: the domain rule, this checkout's `rammp_curobo`, the planner clients |
| `mebot_sim/scene.py`                 | the MuJoCo scene, built with MjSpec                                                             |
| `mebot_sim/sim_arm.py`, `view.py`    | `/mebot_sim`: physics, the driver's names, contacts, the live view                              |
| `mebot_sim/client.py`, `try_goal.py` | the test client, and one goal through the stack                                                 |
| `mebot_sim/check_layers.py`          | S0                                                                                              |
| `mebot_sim/find_pairs.py`            | the seeded searches that write `config/scenarios.yaml`, with the planner in-process on the GPU  |
| `mebot_sim/run_suite.py`             | S0 to S5 and the gates                                                                          |
| `launch/sim.launch.py`               | the stack                                                                                       |
| `launch/suite.launch.py`             | the stack and `run_suite`, ending with its verdict                                              |
| `config/scenarios.yaml`              | the stand-in's postures and the thresholds (by hand); the start and the requests (`find_pairs`) |
| `config/control_*.yaml`              | the control planner's scenes: the floor, and the floor and the platform                         |
| `config/consumer_world.yaml`         | S4's push                                                                                       |
| `test/`                              | `make sim-test`                                                                                 |
