#!/usr/bin/env python3
"""Plan-only tour: 4 random reachable points, chained cuRobo plans.

Samples 4 random targets in a +/-45 deg cone in front of the arm inside
its natural reach and PRE-PLANS the whole tour as chained segments
(start -> P1 -> P2 -> P3 -> P4 -> start), each planned from the previous
segment's planned endpoint, then merges them into one continuous
trajectory and reports it.

    ros2 run rammp_curobo_ros tour_demo
    ros2 run rammp_curobo_ros tour_demo --points 6 --seed 3
    ros2 run rammp_curobo_ros tour_demo --start 0 0.26 3.14 -2.27 0 0.96 1.57

NOTHING MOVES and no arm need exist: the tour starts from an explicit
joint configuration rather than a measured state, so it is a pure
exercise of the planner. It is the repo's showcase and its end-to-end smoke test — if
this prints a tour, the node, the world, and chained planning all work.
Executing a trajectory is the arm owner's job (kinova_arm_ros2); see
issue #6 for why that boundary exists.
"""

import argparse
import math
import random
import sys

import rclpy
from rclpy.action import ActionClient

from rammp_curobo.geometry import yaw_about_world_z
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_ros.ros_util import NODE_NAMESPACE, spin_until_done

# The demo's own home. Deliberately hardcoded HERE and not read from the
# planner: where the arm belongs is an arm-layer opinion, and the planner
# holds none (issue #6). A demo is allowed one; a planner is not.
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
# tool_frame orientation at HOME (FK-verified on gen3_real.yaml to 3e-4):
# tool z level along +x, wrist flat. Targets steer THIS attitude toward
# each bearing (Rz(bearing) ⊗ q_home) so the wrist stays flat instead of
# rolling the gripper vertical/straight-down.
HOME_QUAT_XYZW = [0.5, 0.5, 0.5, 0.5]


def sample_targets(
    n, rng, bearing_deg=45.0, r_range=(0.45, 0.85), z_range=(0.25, 0.70), min_sep=0.25
):
    """n random reachable points in the frontal cone, decently spread."""
    pts = []
    guard = 0
    while len(pts) < n and guard < 500:
        guard += 1
        b = math.radians(rng.uniform(-bearing_deg, bearing_deg))
        r = rng.uniform(*r_range)
        z = rng.uniform(*z_range)
        p = [r * math.cos(b), r * math.sin(b), z]
        if pts and math.dist(p, pts[-1]) < min_sep:
            continue
        pts.append(p)
    return pts


class TourPlanner:
    def __init__(self, node, start):
        self.node = node
        # Where the tour returns to: simply where it began.
        self.start = start
        self.plan_pose = ActionClient(
            node, PlanToPose, NODE_NAMESPACE + "/plan_to_pose"
        )
        self.plan_joints = ActionClient(
            node, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints"
        )

    def _call(self, client, goal, timeout_s=120.0):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit(
                "planner node not running — start it first:\n"
                "  ros2 launch rammp_curobo_ros planner.launch.py"
            )
        send = spin_until_done(self.node, client.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(self.node, send.get_result_async(), timeout_s)
        return None if wrapped is None else wrapped.result

    def plan_pose_from(self, pos, quat, start):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = pos
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = quat
        # start_joints is required — this planner has no view of any arm.
        g.start_joints = [float(v) for v in start]
        return self._call(self.plan_pose, g)

    def plan_back_from(self, start):
        g = PlanToJoints.Goal(target_joints=self.start)
        g.start_joints = [float(v) for v in start]
        return self._call(self.plan_joints, g)


def merge_trajectories(plans):
    """Chained per-segment plans -> ONE continuous JointTrajectory.

    Every segment starts (at rest) exactly where the previous one ended
    (chained pre-planning), so concatenation is dynamically valid — and a
    caller executing the tour can send a single goal with no controller
    goal transitions mid-tour.
    """
    from trajectory_msgs.msg import JointTrajectory

    merged = JointTrajectory()
    merged.joint_names = list(plans[0].trajectory.joint_names)
    offset = 0.0
    for plan in plans:
        for pt in plan.trajectory.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 + offset
            q = type(pt)()
            q.positions = list(pt.positions)
            q.velocities = list(pt.velocities)
            q.accelerations = list(pt.accelerations)
            q.time_from_start.sec = int(t)
            q.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            merged.points.append(q)
        last = plan.trajectory.points[-1].time_from_start
        offset += last.sec + last.nanosec * 1e-9
    return merged


def traj_end(plan):
    return list(plan.trajectory.points[-1].positions)


def traj_time(plan):
    p = plan.trajectory.points[-1].time_from_start
    return p.sec + p.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--start",
        type=float,
        nargs=7,
        default=None,
        help="joint configuration to start and end the tour at (rad, "
        "joint_1..7); default: this demo's HOME",
    )
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument(
        "--seed", type=int, default=None, help="random seed for a repeatable tour"
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    origin = list(args.start) if args.start else HOME
    print("start/end configuration: [%s]" % ", ".join("%.3f" % v for v in origin))

    rclpy.init()
    node = rclpy.create_node("rammp_curobo_tour")
    tour = TourPlanner(node, origin)

    print("sampling %d targets and chain-planning the tour..." % args.points)
    plans, points = [], []
    start = origin
    tries = 0
    while len(plans) < args.points and tries < 12:
        tries += 1
        remaining = args.points - len(plans)
        for p in sample_targets(remaining, rng):
            quat = list(yaw_about_world_z(HOME_QUAT_XYZW, math.atan2(p[1], p[0])))
            plan = tour.plan_pose_from(p, quat, start)
            if plan is None or not plan.success:
                print(
                    "  candidate [%.2f %.2f %.2f] unplannable — resampling" % tuple(p)
                )
                continue
            plans.append(plan)
            points.append(p)
            start = traj_end(plan)
            print(
                "  P%d [%.2f %.2f %.2f]  %5.2f s  (%.0f ms to plan)"
                % (
                    len(plans),
                    p[0],
                    p[1],
                    p[2],
                    traj_time(plan),
                    plan.planning_time * 1e3,
                )
            )
    if len(plans) < args.points:
        sys.exit(
            "could not find %d plannable targets — is the world sane?" % args.points
        )

    back = tour.plan_back_from(start)
    if back is None or not back.success:
        sys.exit("cannot plan the return to the start configuration")
    plans.append(back)
    print(
        "  back    %5.2f s  (%.0f ms to plan)"
        % (traj_time(back), back.planning_time * 1e3)
    )

    merged = merge_trajectories(plans)
    total = sum(traj_time(pl) for pl in plans)
    print(
        "\ntour planned: %d segments, %d points, %.2f s of motion at full "
        "speed, merged into one continuous trajectory."
        % (len(plans), len(merged.points), total)
    )
    print("nothing moved — executing this is the arm owner's job.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
