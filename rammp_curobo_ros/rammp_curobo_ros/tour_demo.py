#!/usr/bin/env python3
"""Speed tour: 4 random reachable points, chained cuRobo plans, full tilt.

Samples 4 random targets in a +/-45 deg cone in front of the arm inside
its natural reach, PRE-PLANS the whole tour as chained segments
(home -> P1 -> P2 -> P3 -> P4 -> home, each planned from the previous
segment's planned endpoint), prints it, and on 'go' executes the chain
back-to-back — no planning pauses between segments, cuRobo's full
time-parameterization for fast, fluid motion — then reports the lap time.

    ros2 run rammp_curobo_ros tour_demo            # plan + print only
    ros2 run rammp_curobo_ros tour_demo --execute  # the real thing

SAFETY: at --speed 1.0 the arm moves at its rated joint limits — the
workspace must be COMPLETELY CLEAR of people and objects, a human holds
the physical e-stop, and execution needs the typed 'go'. Every executor
gate (limits, continuity, live start-state match) still applies to every
segment. Ctrl+C mid-tour cancels the active segment; the arm holds.
"""

import argparse
import math
import random
import sys
import time

import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

from rammp_curobo.geometry import ang_diff, euler_deg_to_quat_xyzw
from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose
from rammp_curobo_ros.scan_common import NODE_NAMESPACE, spin_until_done

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]


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


class TourDemo:
    def __init__(self, node):
        self.node = node
        self._q = None
        node.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.plan_pose = ActionClient(
            node, PlanToPose, NODE_NAMESPACE + "/plan_to_pose"
        )
        self.plan_joints = ActionClient(
            node, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints"
        )
        self.execute = ActionClient(
            node, ExecuteTrajectory, NODE_NAMESPACE + "/execute_trajectory"
        )

    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self._q = [float(msg.position[idx[n]]) for n in JOINTS]
        except (KeyError, IndexError):
            pass

    def joints(self):
        t0 = time.monotonic()
        while self._q is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > 10:
                sys.exit(
                    "no /joint_states — start the arm stack first:\n"
                    "  ros2 launch rammp_curobo_ros planner.launch.py "
                    "config:=gen3_real.yaml execute:=true launch_arm:=true"
                )
        return list(self._q)

    def _call(self, client, goal, timeout_s=120.0):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running (execute:=true needed)")
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
        g.start_joints = [float(v) for v in start] if start else []
        return self._call(self.plan_pose, g)

    def plan_home_from(self, start):
        g = PlanToJoints.Goal(target_joints=HOME)
        g.start_joints = [float(v) for v in start] if start else []
        return self._call(self.plan_joints, g)

    def run(self, traj, scale):
        goal = ExecuteTrajectory.Goal(trajectory=traj, speed_scale=float(scale))
        send = spin_until_done(self.node, self.execute.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return False
        future = send.get_result_async()
        try:
            wrapped = spin_until_done(self.node, future, 240.0)
        except KeyboardInterrupt:
            spin_until_done(self.node, send.cancel_goal_async(), 3.0)
            print("\nCtrl+C — segment cancelled, arm holds")
            raise
        return wrapped is not None and wrapped.result.success


def traj_end(plan):
    return list(plan.trajectory.points[-1].positions)


def traj_time(plan, scale):
    p = plan.trajectory.points[-1].time_from_start
    return (p.sec + p.nanosec * 1e-9) / scale


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--execute", action="store_true", help="allow motion")
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="execution scale (1.0 = the arm's full rated speed; the "
        "executor refuses anything above)",
    )
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument(
        "--seed", type=int, default=None, help="random seed for a repeatable tour"
    )
    args = ap.parse_args()
    scale = min(max(args.speed, 0.1), 1.0)
    rng = random.Random(args.seed)

    rclpy.init()
    node = rclpy.create_node("rammp_curobo_tour")
    demo = TourDemo(node)

    q_now = demo.joints()
    if max(abs(ang_diff(a, b)) for a, b in zip(q_now, HOME)) > 0.1:
        if not args.execute:
            sys.exit("arm is not at home — rerun with --execute to home it")
        input(
            "arm is away from home — press ENTER to home it at 25% "
            "(hand on e-stop), Ctrl+C to quit: "
        )
        plan = demo.plan_home_from(None)
        if plan is None or not plan.success:
            sys.exit("cannot plan home")
        if not demo.run(plan.trajectory, 0.25):
            sys.exit("homing failed — see planner log")
        print("homed.")

    # sample + chain-plan the whole tour before anything moves
    print("sampling %d targets and pre-planning the tour..." % args.points)
    plans, points = [], []
    start = None  # live state (= home) for the first segment
    tries = 0
    while len(plans) < args.points and tries < 12:
        tries += 1
        remaining = args.points - len(plans)
        for p in sample_targets(remaining, rng):
            yaw = math.degrees(math.atan2(p[1], p[0]))
            quat = list(euler_deg_to_quat_xyzw([0.0, 90.0, yaw]))
            plan = demo.plan_pose_from(p, quat, start)
            if plan is None or not plan.success:
                print(
                    "  candidate [%.2f %.2f %.2f] unplannable — resampling" % tuple(p)
                )
                continue
            plans.append(plan)
            points.append(p)
            start = traj_end(plan)
            print(
                "  P%d [%.2f %.2f %.2f]  %5.2f s at speed %.2f"
                % (len(plans), p[0], p[1], p[2], traj_time(plan, scale), scale)
            )
    if len(plans) < args.points:
        sys.exit(
            "could not find %d plannable targets — is the world sane?" % args.points
        )

    home_plan = demo.plan_home_from(start)
    if home_plan is None or not home_plan.success:
        sys.exit("cannot plan the return home")
    plans.append(home_plan)
    total = sum(traj_time(pl, scale) for pl in plans)
    print(
        "tour planned: %d segments, %.2f s of motion at speed %.2f"
        % (len(plans), total, scale)
    )

    if not args.execute:
        print("dry-run complete — nothing moved (add --execute)")
        return

    print(
        "\n*** FULL-SPEED TOUR: the workspace must be COMPLETELY CLEAR of "
        "people and objects. Human on the physical e-stop. ***"
    )
    if input("type 'go' to run the tour: ").strip() != "go":
        print("aborted — nothing moved")
        return

    t0 = time.monotonic()
    for i, plan in enumerate(plans):
        label = "P%d" % (i + 1) if i < len(points) else "home"
        seg_t = time.monotonic()
        ok = False
        for attempt in range(3):
            if demo.run(plan.trajectory, scale):
                ok = True
                break
            # The known transient: the arm faults at motion onset without
            # moving; the planner node auto-recovers servoing. If we are
            # still exactly at this segment's start, the SAME pre-planned
            # trajectory is still valid — re-send it. Any real motion
            # means something else went wrong: abort.
            start_q = list(plan.trajectory.points[0].positions)
            moved = max(abs(ang_diff(a, b)) for a, b in zip(demo.joints(), start_q))
            if moved > 0.05 or attempt == 2:
                sys.exit(
                    "segment %s failed (arm %.3f rad from segment start) — "
                    "arm holds; see the planner log" % (label, moved)
                )
            print(
                "  %s: no-motion fault, recovered — retrying (%d/2)"
                % (label, attempt + 1)
            )
            time.sleep(3.0)  # recovery + post-fault arm settling need real time
        if not ok:
            sys.exit("segment %s failed" % label)
        print("  %s reached in %.2f s" % (label, time.monotonic() - seg_t))
    lap = time.monotonic() - t0
    print(
        "\nTOUR COMPLETE: %d points + home in %.2f s wall (%.2f s motion, "
        "%.0f ms/segment overhead)"
        % (len(points), lap, total, (lap - total) * 1000 / len(plans))
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ntour stopped — arm holds")
