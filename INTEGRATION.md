# Integrating rammp-curobo from another RAMMP module

## Pure Python (no ROS) — the 5 lines

```python
from rammp_curobo import CuRoboPlanner

planner = CuRoboPlanner.from_config("gen3.yaml")   # ~20 s GPU init, keep it alive
res = planner.plan_to_joints([0.2, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571])
if res.success:                                     # res is a PlanResult
    hand_off(res.joint_traj)   # numpy: joint_names, positions (N,7), velocities, dt
```

Also available: `planner.plan_to_pose(pos, quat)` (quat is **xyzw** by
default — pass `quat_order='wxyz'` for cuRobo order),
`planner.update_world(obstacles)` (list of dicts, a Scene, or a world YAML),
`planner.fk(q)`, `planner.check_state_valid(q)`. Slow a plan down for
execution with `res.joint_traj.scaled(0.25)` — exact time dilation, never
faster than planned. `start=` defaults to the configured home pose; pass the
arm's live joints for real use.

Config is one YAML (`core/rammp_curobo/configs/gen3.yaml` — robot file,
world file, planner knobs). Copy it next to your code and point
`from_config` at your copy to customize; bare names resolve to the packaged
defaults. No cuRobo types cross the API.

Requirements: the pinned GPU stack from the README (torch + cuRobo v0.7.8)
and `pip install --user --no-build-isolation -e <repo>/core`.

## Over ROS 2 (plan + execute on the arm)

Depend on `rammp_curobo_interfaces` (dependency-free rosidl package —
mirrors `arm_interfaces` conventions). With the planner node running
(`ros2 launch rammp_curobo_ros planner.launch.py execute:=true` — see the
README for the sim/real bringup on the other side):

```python
from rammp_curobo_interfaces.action import PlanToJoints, ExecuteTrajectory
from rclpy.action import ActionClient

plan_client = ActionClient(node, PlanToJoints, '/rammp_curobo/plan_to_joints')
exec_client = ActionClient(node, ExecuteTrajectory, '/rammp_curobo/execute_trajectory')

goal = PlanToJoints.Goal(target_joints=[0.2, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571])
plan = plan_client.send_goal(goal).result          # sync form; async works too
if plan.success:
    ex = ExecuteTrajectory.Goal(trajectory=plan.trajectory, speed_scale=0.25)
    exec_client.send_goal(ex)                      # cancel this goal = abort+hold
```

The node re-validates every execution goal (limits, continuity, live
start-state match) and refuses anything stale — plan again if the arm moved.
`/rammp_curobo/plan_to_pose` takes a `geometry_msgs/Pose`;
`/rammp_curobo/set_world` swaps the collision world;
`/rammp_curobo/open_gripper` / `close_gripper` are `std_srvs/Trigger`.

## Adopting into Demo-Software

- Clone this repo into the workspace `src/` (or add as a submodule under
  `third_party/`); colcon picks up the two ament packages, the pip core
  installs once per machine (add to setup.sh alongside the other pip deps,
  rosdep-stub pattern like `python3-kortex-api`).
- **Mutual exclusion:** execution drives ros2_kortex's
  `joint_trajectory_controller` — `hardware/arm_driver` must NOT be running
  at the same time (both own the arm at 192.168.1.10). Sequencing that
  handover is a team decision, not something this repo enforces.
- `ExecuteTrajectory` here has the same goal shape as
  `arm_interfaces/ExecuteTrajectory` (a `trajectory_msgs/JointTrajectory`),
  plus `speed_scale` — a future `arm_driver` integration could accept the
  same trajectories, but note arm_driver currently discards trajectory
  timing (0.5 s/waypoint), which defeats cuRobo's parameterization.
