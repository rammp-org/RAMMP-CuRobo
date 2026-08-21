#!/usr/bin/env python3
"""Prove the reactive chain works — live, with NO arm and NO camera.

    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
    python3 scripts/sweep_demo_checks.py

Walks the same A->B stroke sweep_demo flies and forces all three of its
outcomes: clear, blocked-and-routed-around, and the protective HOLD when
an obstacle overlaps the arm itself.

The middle one is what makes the demo safe. Nothing in the execution
gate chain re-checks collision, so ~/check_trajectory is the only thing
that notices an obstacle appearing AFTER a trajectory was planned; if it
is wrong, the demo drives through obstacles.

Plans from explicit start joints, so no bringup and no joint states are
needed. Never sends an execution goal — nothing can move.
"""

import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Vector3
from rclpy.action import ActionClient

from rammp_curobo_interfaces.action import PlanToPose
from rammp_curobo_interfaces.srv import CheckTrajectory, UpdateWorldBoxes

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.96, 1.571]
POSE_A = [0.55, -0.30, 0.35]        # sweep_demo's defaults
POSE_B = [0.55, 0.30, 0.35]
QUAT = [0.5, 0.5, 0.5, 0.5]
# blocks the corridor between A and B without touching either end
BLOCKER = ([0.55, 0.0, 0.30], [0.12, 0.12, 0.30])
# overlaps the arm's own body at A: measured, this sets in around 15 cm
TOO_CLOSE = ([0.55, -0.20, 0.30], [0.12, 0.12, 0.30])
NS = "/rammp_curobo"
fails = []


def check(label, ok, detail=""):
    print("  %-52s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "  <- " + detail))
    if not ok:
        fails.append(label)


def spin_until(node, future, timeout):
    end = time.monotonic() + timeout
    while not future.done() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    return future.result() if future.done() else None


def main():
    rclpy.init()
    node = rclpy.create_node("sweep_demo_checks")
    plan_cli = ActionClient(node, PlanToPose, NS + "/plan_to_pose")
    check_cli = node.create_client(CheckTrajectory, NS + "/check_trajectory")
    world_cli = node.create_client(UpdateWorldBoxes, NS + "/update_world_boxes")
    if not plan_cli.wait_for_server(timeout_sec=30.0):
        sys.exit("no planner node on %s — start planner.launch.py first" % NS)
    for name, cli in (("check_trajectory", check_cli), ("update_world_boxes", world_cli)):
        if not cli.wait_for_service(timeout_sec=10.0):
            sys.exit("%s service missing — rebuild the interfaces package" % name)

    def set_boxes(boxes):
        req = UpdateWorldBoxes.Request()
        for i, (c, d) in enumerate(boxes):
            req.names.append("chk%d" % i)
            req.centers.append(Point(x=c[0], y=c[1], z=c[2]))
            req.dims.append(Vector3(x=d[0], y=d[1], z=d[2]))
        return spin_until(node, world_cli.call_async(req), 10.0)

    def plan(target, start):
        goal = PlanToPose.Goal()
        goal.target = Pose()
        (goal.target.position.x, goal.target.position.y,
         goal.target.position.z) = target
        (goal.target.orientation.x, goal.target.orientation.y,
         goal.target.orientation.z, goal.target.orientation.w) = QUAT
        goal.start_joints = [float(v) for v in start]
        send = spin_until(node, plan_cli.send_goal_async(goal), 15.0)
        if send is None or not send.accepted:
            return None
        res = spin_until(node, send.get_result_async(), 60.0)
        return res.result if res is not None else None

    def verdict(traj, start_index=0):
        req = CheckTrajectory.Request()
        req.trajectory = traj
        req.start_index = int(start_index)
        return spin_until(node, check_cli.call_async(req), 15.0)

    print("\n1. get to A, then plan the A -> B stroke in a clear world")
    set_boxes([])
    to_a = plan(POSE_A, HOME)
    if to_a is None or not to_a.success:
        sys.exit("cannot reach A: %s" % (to_a.message if to_a else "no result"))
    q_a = list(to_a.trajectory.points[-1].positions)
    stroke = plan(POSE_B, q_a)
    if stroke is None or not stroke.success:
        sys.exit("cannot plan A->B: %s" % (stroke.message if stroke else "no result"))
    traj = stroke.trajectory
    n = len(traj.points)
    print("   %d points, %.2f s" % (
        n, traj.points[-1].time_from_start.sec
        + traj.points[-1].time_from_start.nanosec * 1e-9))

    print("\n2. the stroke, checked against the world it was planned in")
    v = verdict(traj)
    check("clear path reports collision_free", v is not None and v.collision_free,
          v.message if v else "no response")
    check("no bad index on a clear path", v is not None and v.first_bad_index == -1)

    print("\n3. an obstacle APPEARS on that path (the whole point)")
    set_boxes([BLOCKER])
    v = verdict(traj)
    check("the SAME trajectory now reports BLOCKED",
          v is not None and not v.collision_free, v.message if v else "no response")
    check("names the first bad point",
          v is not None and 0 <= v.first_bad_index < n,
          "first_bad_index=%s" % (v.first_bad_index if v else "?"))
    check("counts the infeasible points", v is not None and v.n_bad > 0)
    check("reports a finite arm clearance",
          v is not None and np.isfinite(v.min_clearance),
          "min_clearance=%s" % (v.min_clearance if v else "?"))

    print("\n4. start_index skips what has already been driven")
    late = verdict(traj, start_index=n)
    check("an empty remaining span is clear",
          late is not None and late.collision_free, late.message if late else "")

    print("\n5. cuRobo routes AROUND it — the demo's recovery")
    detour = plan(POSE_B, q_a)
    check("replan from A succeeds with the obstacle present",
          detour is not None and detour.success,
          detour.message.strip()[:70] if detour else "no result")
    if detour is not None and detour.success:
        dv = verdict(detour.trajectory)
        check("and the detour itself checks clear",
              dv is not None and dv.collision_free, dv.message if dv else "")
        print("   detour: %d points, arm gap %.3f m"
              % (len(detour.trajectory.points),
                 dv.min_clearance if dv else float("nan")))

    print("\n6. an obstacle ON the arm -> HOLD, not a detour")
    set_boxes([TOO_CLOSE])
    held = plan(POSE_B, q_a)
    check("planner refuses rather than inventing a path",
          held is not None and not held.success, "it planned anyway")
    check("and says the start state is the problem",
          held is not None and "INVALID_START" in held.message,
          held.message.strip()[:70] if held else "no result")

    print("\n7. world restored")
    set_boxes([])
    v = verdict(traj)
    check("original path is clear again", v is not None and v.collision_free,
          v.message if v else "")

    node.destroy_node()
    rclpy.shutdown()
    print("\n%s — %d check(s) failed" % ("FAILED" if fails else "ALL PASS", len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
