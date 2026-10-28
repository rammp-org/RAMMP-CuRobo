# Integrating rammp-curobo from another RAMMP module

Three ways in, same planner underneath: import the pure-Python core, call
the ROS actions of a planner node you launch yourself, or run the
**Docker image** (see `docker/README.md`) and call the same ROS actions
with zero GPU-stack setup on your side. In every case the contract is:
end position in → collision-free time-parameterized joint trajectory out;
executing it is your side's job.

**Frames and names.** Send Cartesian goals in **`arm_mount`** (the root of
the arm's URDF: Z up when the chair is level, X forward, origin at the arm's
base) for the tool frame `arm_tool_frame`; joints are `arm_joint_1..7`.
Every returned trajectory carries `header.frame_id: "arm_mount"`, joint plans
too. Read it: kinova-gen3-ros2's `GoToEEPose` refuses any pose plan not
stamped `arm_mount` (#47), on every mount. On a level mount (dojo) `arm_mount`
and the arm's base coincide, so the numbers are unchanged.

## Pure Python (no ROS) — the 5 lines

```python
from rammp_curobo import CuRoboPlanner

planner = CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf)  # ~20 s GPU init, keep it alive
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

`urdf_path` is REQUIRED: the arm's model as a file, i.e. what
`/arm/robot_description` carries, written to disk (the ROS node does exactly
that; `core/tests/data/arm_level.urdf` is a rendered example). Without it the
baked robot config refuses to load. `pos`/`quat` are `arm_tool_frame` in
`arm_mount`, and `res.joint_traj.frame_id` is `"arm_mount"`.

Config is one YAML (`core/rammp_curobo/configs/gen3.yaml` — robot file,
world file, obstacle layers, planner knobs). Copy it next to your code and
point `from_config` at your copy to customize; bare names resolve to the
packaged defaults. No cuRobo types cross the API.

Requirements: the pinned GPU stack from the README (torch + cuRobo v0.7.8)
and `pip install --user --no-build-isolation -e <repo>/core`.

## Over ROS 2 (planning)

Depend on `rammp_curobo_interfaces` (dependency-free rosidl package). With
the planner node running (`ros2 launch rammp_curobo_ros planner.launch.py`;
it needs `arm_state_publisher` latching `/arm/robot_description` in the same
domain, and exits 1 without it — see the README):

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
    assert plan.trajectory.header.frame_id == 'arm_mount'
    execute(plan.trajectory)     # your controller, your gates, your e-stop
```

`start_joints` is **not optional**. This planner holds no `/joint_states`
subscription and no view of any robot, so a goal without it is aborted with
`"start_joints is required"`. Sending the arm's freshly measured `q` is also
what makes the plan safe to run: its first point *is* where the arm is, so
there is no stale-plan catch-up sweep to guard against.

`/rammp_curobo/plan_to_pose` takes a `geometry_msgs/Pose` (`arm_tool_frame`
in `arm_mount`; the goal has no frame field, so convert before sending), plus two optional
trailing fields that default to off: `hold` (keep the tool's orientation:
`HOLD_NONE`, `HOLD_LEVEL` or `HOLD_FIXED`) and `approach_via` (one blended via
point). They cannot be combined. `HOLD_LEVEL` holds level relative to
`arm_mount`, which is level with the chair, not with gravity: on a ramp it is
off by the ramp angle. A held plan is checked, not trusted; when a
check fails the result has `success = false` and `message` starts with the
status:

| status                     | when                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `CONSTRAINT_UNSATISFIABLE` | before planning: the start disagrees with the goal on the held components                           |
| `CONSTRAINT_VIOLATED`      | after planning: a waypoint breaks the hold, or the plan does not end at the goal's full orientation |
| `BAD_CONSTRAINT`           | a malformed via point, or a hold sent together with a via                                           |

`PlanToPose.action` has the full contract. `/rammp_curobo/set_world` swaps the
collision world's scene. There is nothing here that can move an arm.

### Through several goals without stopping: `/rammp_curobo/blend`

Plan each leg with `plan_to_pose` / `plan_to_joints`, each from the last point
of the leg before it, then hand the trajectories to `blend` for ONE
trajectory that passes the goals between still moving. Per leg: its
`speed_scale` (`(0, 1]` of its planned speed; empty: all 1), the `hold` it
was planned with (empty: none), and for a held leg the `hold_orientation`, its
goal's orientation as sent to `plan_to_pose` (empty: the leg's planned end,
which is within the planner's tolerance of the goal, so an overlap can then
reach up to twice that from the goal). Per junction, one fewer than legs:
`blend_radius` (m), how far the tool may pass from that junction's goal to
cut the corner; `0` stops on it. The result carries the speeds, so execute it
at speed scale 1.

```python
from geometry_msgs.msg import Quaternion
from rammp_curobo_interfaces.action import Blend, PlanToPose

level = pose_client.send_goal(level_goal).result.trajectory   # hold HOLD_LEVEL
free = joints_client.send_goal(free_goal).result.trajectory   # from level's last point
plan = blend_client.send_goal(
    Blend.Goal(
        legs=[level, free],
        blend_radius=[0.03],
        speed_scale=[0.8, 0.4],
        hold=[PlanToPose.Goal.HOLD_LEVEL, PlanToPose.Goal.HOLD_NONE],
        hold_orientation=[level_goal.target.orientation, Quaternion()],
    )
).result
# plan.corner_m: per junction, how close the tool came to that goal
```

How: consecutive legs are overlapped in time, so the arm moves by the sum of
both while they overlap (`core/rammp_curobo/blend.py`). Each overlap is the
longest whose states:

- keep the joint velocity and acceleration limits;
- bring the tool within the radius of the junction's goal at its closest
  pass, and keep it within the radius of the planned path;
- keep each held leg's hold against its hold orientation, measured as
  `plan_to_pose` measures it;
- pass cuRobo's constraint check (collision, self-collision, limits), one by one.

The whole trajectory is then validated like any plan. A radius is a bound, not
a promise: where the limits, a hold or the world allow no cut, the arm stops on
the goal and `corner_m` says 0. A leg must start and end at rest and start
within 1e-5 rad of where the one before it ends (plan it from that point), its points (at least 2)
evenly spaced and finite, with positions and velocities, or the goal fails naming the leg
(`invalid legs: leg 2/3: ...`, or `BAD_LEG: leg 2/3: ...`).

## Planning on the MEBot (≥ 1.1.0)

With the arm on the MEBot wheelchair, run the planner with the chair
config. It does NOT ship in this image: the chair's planner config, its world
and its four layer files describe one chair's geometry and where the arm sits
on it, so they are per-robot data and live with the robot, in rammp-deployments
`config/robots/<robot>/curobo/`, bind-mounted read-only at `/config/robot`
(issue #19). So `config:=/config/robot/curobo/planner.yaml` for the node, or
`CuRoboPlanner.from_config("<that path>")` for the library. Re-measuring the
chair is a commit there, not a release here, and nothing resolves these files
by bare name — a packaged fallback would let an unmounted `/config/robot` plan
the chair against stale geometry.

If your manifest sets the `world` parameter, a chair robot must override it to
`""`: a non-empty `world` replaces whatever the planner config names, so a
bench world left in place would plan the chair against the wrong room. The chair's drive base,
modelled with every joint at zero (layer `platform`, 16 boxes), its seat
(layer `seat`, 5 boxes: base, backrest and the joystick-side armrest, best
guesses) and the person in it (layer `person`, 8 boxes) are then in every
collision world the planner builds. Nothing models the footplate, the
joystick, the brackets on the arm's side, or the plate the arm is mounted
on. Nor do the person and the seat cover the column carved around the
arm's base (for the person x -0.073 to 0.250, y ±0.105, z -0.049 to 0.366 m
in `arm_mount`; the seat's is 30 mm wider): most of it is open space beside
the shoulder that the arm can move through, so the occupant's left hip,
elbow or forearm beside the mount is not kept from the arm there.
See the README's **Obstacle layers** for how the layers are made and what
they cost. For a caller:

- **A pushed scene replaces only your task geometry.** `set_world` and
  `update_world` replace the scene, as before; the layers the config lists
  stay in every world. The start-up scene's floor is task geometry too: the
  first push replaces it, so carry your own floor in the scenes you push.
  The three layers take 29 of the 60-box cache, which leaves 31 for your
  scene. A scene with no obstacles, objects or layers of its own is
  refused, and so is one that overflows the cache; the previous world
  stays.

- **The SetWorld reply lists the layers.** On success it reads
  `world set to <world> (<summary>)`, e.g. for an 8-box scene:

  ```text
  world set to /config/my_world.yaml (37 boxes: scene 8 | config layers: platform 16, seat 5, person 8 | scene layers: none)
  ```

  `platform 16, seat 5, person 8` there is your proof that the chair is in the
  world. A refused push replies with the reason, as before.

- **Start from the arm's measured state**, which `start_joints` requires
  anyway. A failed plan's error now ends with the boxes its start or goal
  reach into and where the world's boxes come from, e.g.
  `[start collides with person/thighs (105 mm), ...] [world: 30 boxes: ...]`;
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

- **A 1.0.0 image refuses the chair config.** Mounting the chair's files
  into one does not help: 1.0.0 ignores `layers:`, finds no obstacles in
  `world.yaml` and refuses it as an empty world at start-up. It never
  plans without the chair; pin a release that carries the layers.

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
