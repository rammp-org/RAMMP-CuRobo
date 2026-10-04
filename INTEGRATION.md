# Integrating rammp-curobo from another RAMMP module

Three ways in, same planner underneath: import the pure-Python core, call
the ROS actions of a planner node you launch yourself, or run the
**Docker image** (see `docker/README.md`) and call the same ROS actions
with zero GPU-stack setup on your side. In every case the contract is:
end position in → collision-free time-parameterized joint trajectory out;
executing it is your side's job.

## Pure Python (no ROS) — the 5 lines

```python
from rammp_curobo import CuRoboPlanner

planner = CuRoboPlanner.from_config("gen3.yaml")   # ~20 s GPU init, keep it alive
res = planner.plan_to_joints(q_goal, q_now)        # start is REQUIRED, not optional
if res.success:                                     # res is a PlanResult
    hand_off(res.joint_traj)   # numpy: joint_names, positions (N,7), velocities, dt
```

Also available: `planner.plan_to_pose(pos, quat, start)` (quat is **xyzw**
by default — pass `quat_order='wxyz'` for cuRobo order),
`planner.update_world(obstacles)` (list of dicts, a Scene, or a world YAML),
`planner.fk(q)`, `planner.check_state_valid(q)`. Slow a plan down for
execution with `res.joint_traj.scaled(0.25)` — exact time dilation, never
faster than planned.

`start` is a required positional argument on both plan calls. The planner
does not know where the arm is, holds no default pose to fall back on, and
will not guess — pass the arm's measured joints. `planner.retract_pose` is
the IK seed read from the robot config; it is a planner parameter, not a
home, and it is not a substitute for a measurement.

Config is one YAML (`core/rammp_curobo/configs/gen3.yaml` — robot file,
world file, obstacle layers, planner knobs). Copy it next to your code and
point `from_config` at your copy to customize; bare names resolve to the
packaged defaults. No cuRobo types cross the API.

Requirements: the pinned GPU stack from the README (torch + cuRobo v0.7.8)
and `pip install --user --no-build-isolation -e <repo>/core`.

## Over ROS 2 (planning)

Depend on `rammp_curobo_interfaces` (dependency-free rosidl package). With
the planner node running (`ros2 launch rammp_curobo_ros planner.launch.py`):

```python
from rammp_curobo_interfaces.action import PlanToJoints
from rclpy.action import ActionClient

plan_client = ActionClient(node, PlanToJoints, '/rammp_curobo/plan_to_joints')

goal = PlanToJoints.Goal(
    target_joints=[0.2, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571],
    start_joints=measured_q,     # REQUIRED — you own the arm's state
)
plan = plan_client.send_goal(goal).result          # sync form; async works too
if plan.success:
    execute(plan.trajectory)     # your controller, your gates, your e-stop
```

`start_joints` is **not optional**. This planner holds no `/joint_states`
subscription and no view of any robot, so a goal without it is aborted with
`"start_joints is required"`. Sending the arm's freshly measured `q` is also
what makes the plan safe to run: its first point *is* where the arm is, so
there is no stale-plan catch-up sweep to guard against.

`/rammp_curobo/plan_to_pose` takes a `geometry_msgs/Pose`;
`/rammp_curobo/set_world` swaps the collision world's scene. That is the
entire public surface — there is nothing here that can move an arm.

## Planning on the MEBot (≥ 1.1.0)

With the arm on the MEBot wheelchair, run the planner with the chair
config: `config:=gen3_mebot.yaml` for the node, or
`CuRoboPlanner.from_config("gen3_mebot.yaml")`. The chair's drive base,
modelled with every joint at zero (layer `platform`, 16 boxes), and the
person in it (layer `person`, 8 boxes) are then in every collision world the
planner builds. Nothing models the rest of the chair: its seat, backrest,
armrests and footplate, or the plate the arm is mounted on. Nor does the
person layer cover the column carved around the arm's base (±0.105 m about
its axis in x and y, z -0.040 to 0.409 m in `base_link`): most of it is open
space beside the shoulder that the arm can move through, so the occupant's
right hip, elbow or forearm beside the mount is not kept from the arm there.
See the README's **Obstacle layers** for how the layers are made and what
they cost. For a caller:

- **A pushed scene replaces only your task geometry.** `set_world` and
  `update_world` replace the scene, as before; the layers the config lists
  stay in every world. The start-up scene's floor is task geometry too: the
  first push replaces it, so carry your own floor in the scenes you push.
  The two layers take 24 of the 60-box cache, which leaves 36 for your
  scene. A scene with no obstacles, objects or layers of its own is
  refused, and so is one that overflows the cache; the previous world
  stays.

- **The SetWorld reply lists the layers.** On success it reads
  `world set to <world> (<summary>)`, e.g. for an 8-box scene:

  ```text
  world set to /config/my_world.yaml (32 boxes: scene 8 | config layers: platform 16, person 8 | scene layers: none)
  ```

  `platform 16, person 8` there is your proof that the chair is in the
  world. A refused push replies with the reason, as before.

- **The bench retract is not a valid start on the chair.** The robot
  config's retract (`planner.retract_pose`, also cuRobo's IK seed) is
  154.4 mm inside the padded person layer, and the planner warns about it
  at start-up. Start from the arm's measured state, which `start_joints`
  requires anyway. A failed plan's error now ends with the boxes its start
  or goal reach into and where the world's boxes come from, e.g.
  `[start collides with person/lap (154 mm), ...] [world: 25 boxes: ...]`;
  over ROS it arrives in the result's `message` as `<status>: <error>`.

- **Excluding the person is Python-only.**
  `planner.update_world(world, ignore={"person"})` leaves the person out of
  that one world, the platform untouched; the next push without `ignore`
  brings the person back. SetWorld has no field for it — adding one would
  be a MAJOR change — and `ignore={"platform"}` is refused. Over the seat
  and at the backrest the person layer is the only geometry, so that world
  has none there: if your plans go near them, carry boxes for them in the
  scene you push.

- **A subclass that overrides `CuRoboPlanner._finish` keeps working.**
  `_finish(result, t0, q_goal=None)` keeps its v1.0.0 signature and code;
  the note on which boxes a failure involves is added after it returns. A
  failure cuRobo reports carries it whether your override passes it on,
  relabels it or leaves its error empty. A plan cuRobo found and your
  override rejects comes back exactly as your override made it, with no
  note.

- **A 1.0.0 image refuses the chair config.** `gen3_mebot.yaml` and its
  layer files ship from 1.1.0 on, and a 1.0.0 image answers
  `config 'gen3_mebot.yaml' not found`. Mounting the files into one does not
  help: 1.0.0 ignores `layers:`, finds no obstacles in `world_mebot.yaml`
  and refuses it as an empty world at start-up. It never plans without the
  chair; pin ≥ 1.1.0.

## Adopting into a RAMMP module

- Clone this repo into the workspace `src/` (or add as a submodule under
  `third_party/`); colcon picks up the two ament packages, the pip core
  installs once per machine (add to setup.sh alongside the other pip deps,
  rosdep-stub pattern like `python3-kortex-api`).
- **The dependency arrow points one way.** Your package depends on
  `rammp_curobo_interfaces`; this repo depends on nothing of yours. If you
  find yourself adding your driver's IDL to `rammp_curobo_ros/package.xml`,
  the design has gone wrong — see issue #6.
- **No mutual-exclusion problem to solve.** This planner never claims the
  arm, a `/controller_manager`, or a gripper, so it can run alongside any
  driver. Exactly one thing should execute; that thing is not this.
- On this bench the caller is **`kinova_arm_ros2`**: its `GoToEEPose` and
  `GoToJointConfig` actions plan here, hand the trajectory to the supervisor
  with a path tolerance, and own the arm end to end. Its `CuroboPlanClient`
  is a worked reference for a well-behaved caller.
