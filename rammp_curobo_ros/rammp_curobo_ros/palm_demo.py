#!/usr/bin/env python3
"""Palm-touch demo: detect a presented palm, plan to it, touch it on cue.

Flow (loops until 'q'):
  home -> D405 depth finds the nearest hand-sized blob in the approach
  corridor -> palm point gated against the workspace -> cuRobo plans
  home->standoff and standoff->touch -> operator types 'go' -> FAST
  transit to the standoff, then SLOW monitored final approach that stops
  on wrist-torque contact (or on arrival) -> hold -> retreat home.

    ros2 run rammp_curobo_ros palm_demo                    # detect+plan only
    ros2 run rammp_curobo_ros palm_demo --execute          # full demo

Safety (do not weaken):
  * The final (contact-capable) segment is hard-capped at slow speed and
    torque-monitored — only the transit to the standoff is fast, and
    transit speed is clamped to 0.6.
  * The palm target must sit inside a sane workspace window (above the
    table, inside reach, away from the base) or the round is refused.
  * The person is told to HOLD STILL after 'go' — the plan targets where
    the palm WAS. Moving the hand away is the human abort, on top of the
    operator's Ctrl+C (cancel -> arm holds) and the physical e-stop.
  * Requires the planner node with execute:=true and a real-bench world
    (never the sim kitchen).
"""

import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

from rammp_curobo.geometry import euler_deg_to_quat_xyzw
from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose
from rammp_curobo_ros.scan_common import (
    NODE_NAMESPACE,
    DepthCameraGrabber,
    load_camera_config,
    spin_until_done,
)

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

# Contact-capable segment limits — clamped in code, not just defaults.
FINAL_SCALE_CAP = 0.15
TRANSIT_SCALE_CAP = 0.6


def palm_target_ok(p, table_z=0.10, r_min=0.35, r_max=0.80, z_max=0.85):
    """Workspace gate for a palm point [x, y, z] in the base frame.

    Above the table by a margin, inside the comfortable reach annulus,
    not behind the arm (the demo corridor is the front half-plane).
    Returns (ok, reason).
    """
    x, y, z = (float(v) for v in p)
    r = math.hypot(x, y)
    if z < table_z:
        return False, "palm too low (z=%.2f < %.2f — near the table)" % (z, table_z)
    if z > z_max:
        return False, "palm too high (z=%.2f)" % z
    if r < r_min:
        return False, "palm too close to the base (r=%.2f)" % r
    if r > r_max:
        return False, "palm out of reach (r=%.2f)" % r
    if x < 0.15:
        return False, "palm outside the frontal demo corridor (x=%.2f)" % x
    return True, ""


def detect_palm(points, min_pts=150, cluster_r=0.06):
    """Nearest coherent blob's center from base-frame points (N, 3).

    The person presents an open palm facing the arm inside the corridor;
    the nearest cluster of sufficient size is the hand, its centroid the
    palm. Returns (center [3] or None, n_points, reason).
    """
    if len(points) < min_pts:
        return None, len(points), "not enough points in the corridor"
    r = np.hypot(points[:, 0], points[:, 1])
    order = np.argsort(r)
    seed = points[order[: max(min_pts // 3, 30)]].mean(axis=0)
    for _ in range(4):  # few mean-shift steps around the nearest surface
        d = np.linalg.norm(points - seed, axis=1)
        members = points[d < cluster_r * 2]
        if len(members) < min_pts:
            return None, len(members), "nearest blob too small (%d pts)" % len(members)
        seed = members.mean(axis=0)
    return seed, len(members), ""


class PalmDemo(DepthCameraGrabber):
    def __init__(self, camera_cfg):
        super().__init__(camera_cfg, node_name="rammp_curobo_palm_demo")
        self._js_lock = threading.Lock()
        self._effort = None
        self._q = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.plan_pose = ActionClient(
            self, PlanToPose, NODE_NAMESPACE + "/plan_to_pose"
        )
        self.plan_joints = ActionClient(
            self, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints"
        )
        self.execute = ActionClient(
            self, ExecuteTrajectory, NODE_NAMESPACE + "/execute_trajectory"
        )

    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [float(msg.position[idx[n]]) for n in JOINTS]
            eff = (
                [float(msg.effort[idx[n]]) for n in JOINTS]
                if len(msg.effort) == len(msg.name)
                else None
            )
        except (KeyError, IndexError):
            return
        with self._js_lock:
            self._q = q
            self._effort = eff

    def joints(self):
        t0 = time.monotonic()
        while True:
            with self._js_lock:
                if self._q is not None:
                    return list(self._q)
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - t0 > 10:
                sys.exit("no /joint_states — is the bringup running?")

    def wrist_effort(self):
        with self._js_lock:
            return None if self._effort is None else list(self._effort[3:])

    # ------------------------------------------------------------ planner I/O
    def _result(self, client, goal, timeout_s):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running (execute:=true needed)")
        send = spin_until_done(self, client.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None, None
        wrapped = spin_until_done(self, send.get_result_async(), timeout_s)
        return (None, None) if wrapped is None else (wrapped.result, send)

    def plan_to(self, pos, quat_xyzw):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = pos
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = quat_xyzw
        res, _ = self._result(self.plan_pose, g, 120.0)
        return res

    def plan_home(self):
        res, _ = self._result(
            self.plan_joints, PlanToJoints.Goal(target_joints=HOME), 120.0
        )
        return res

    def run_traj(self, traj, scale, touch_nm=None):
        """Execute; if touch_nm is set, cancel on wrist-torque contact.

        Returns 'arrived' | 'touch' | 'failed'.
        """
        goal = ExecuteTrajectory.Goal(trajectory=traj, speed_scale=float(scale))
        if not self.execute.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running")
        send = spin_until_done(self, self.execute.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return "failed"
        result_future = send.get_result_async()

        baseline, contact = None, False
        t0 = time.monotonic()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if touch_nm is not None:
                eff = self.wrist_effort()
                if eff is not None:
                    if baseline is None and time.monotonic() - t0 > 0.4:
                        baseline = eff
                    elif baseline is not None:
                        dev = max(abs(a - b) for a, b in zip(eff, baseline))
                        if dev > touch_nm:
                            contact = True
                            print(
                                "  contact felt (wrist torque +%.1f Nm) — stopping"
                                % dev
                            )
                            spin_until_done(self, send.cancel_goal_async(), 3.0)
                            spin_until_done(self, result_future, 10.0)
                            return "touch"
            if time.monotonic() - t0 > 240:
                spin_until_done(self, send.cancel_goal_async(), 3.0)
                return "failed"
        wrapped = result_future.result()
        if wrapped is not None and wrapped.result.success:
            return "arrived"
        return "touch" if contact else "failed"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", default="camera_d405_wrist.yaml")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually move (default: detect + plan + print only)",
    )
    ap.add_argument(
        "--transit-scale",
        type=float,
        default=0.5,
        help="speed for home<->standoff transit (clamped to %.1f)" % TRANSIT_SCALE_CAP,
    )
    ap.add_argument(
        "--touch-nm",
        type=float,
        default=3.0,
        help="wrist-torque deviation that counts as palm contact",
    )
    ap.add_argument(
        "--standoff",
        type=float,
        default=0.12,
        help="fast/slow handover distance short of the palm (m)",
    )
    ap.add_argument(
        "--touch-back",
        type=float,
        default=0.015,
        help="stop the fingertip midpoint this short of the palm surface (m)",
    )
    args = ap.parse_args()
    transit = min(max(args.transit_scale, 0.1), TRANSIT_SCALE_CAP)

    rclpy.init()
    node = PalmDemo(load_camera_config(args.camera))

    print(
        "PALM DEMO — person: stand at the table edge, present an OPEN PALM\n"
        "facing the arm, 0.4-0.7 m from the camera, and HOLD STILL once the\n"
        "operator types 'go'. Moving your hand away is your abort.\n"
        "Operator: hand on the physical e-stop; Ctrl+C stops and holds."
    )

    node.joints()  # wait for the bringup
    if node.wrist_effort() is None:
        print(
            "WARNING: /joint_states carries no effort values — torque touch "
            "detection is UNAVAILABLE; the final approach will stop at the "
            "palm plane by position only."
        )

    while True:
        # 1. be at home (the detection vantage)
        q = node.joints()
        if max(abs(a - b) for a, b in zip(q, HOME)) > 0.1:
            print("returning to home vantage...")
            plan = node.plan_home()
            if plan is None or not plan.success:
                sys.exit(
                    "cannot plan home: %s" % (plan.message if plan else "no answer")
                )
            if args.execute and node.run_traj(plan.trajectory, transit) == "failed":
                sys.exit("home move failed — see planner log")

        input("\n[enter] to scan for a palm ('ctrl+c' quits)... ")

        # 2. detect the palm in the frontal corridor
        pts = node.capture_points(
            n_frames=3, min_z=0.05, xy_extent=1.0, max_z=1.1, self_radius=0.13
        )
        # corridor: in front of the arm, inside detection range
        if len(pts):
            m = (pts[:, 0] > 0.15) & (np.hypot(pts[:, 0], pts[:, 1]) < 0.95)
            pts = pts[m]
        palm, n, why = detect_palm(pts) if len(pts) else (None, 0, "no points")
        if palm is None:
            print("no palm found (%s) — try again" % why)
            continue
        ok, reason = palm_target_ok(palm)
        print(
            "palm at [%.2f, %.2f, %.2f] (%d pts)%s"
            % (palm[0], palm[1], palm[2], n, "" if ok else " — REFUSED: " + reason)
        )
        if not ok:
            continue

        # 3. approach geometry: horizontal reach toward the palm
        yaw = math.degrees(math.atan2(palm[1], palm[0]))
        quat = list(euler_deg_to_quat_xyzw([0.0, 90.0, yaw]))
        ux, uy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        touch = [
            palm[0] - args.touch_back * ux,
            palm[1] - args.touch_back * uy,
            palm[2],
        ]
        standoff = [
            palm[0] - (args.standoff + args.touch_back) * ux,
            palm[1] - (args.standoff + args.touch_back) * uy,
            palm[2],
        ]

        plan_a = node.plan_to(standoff, quat)
        if plan_a is None or not plan_a.success:
            print(
                "standoff unreachable (%s) — present the palm elsewhere"
                % (plan_a.message if plan_a else "no answer")
            )
            continue
        print(
            "planned: transit %s + final approach %.0f cm"
            % (plan_a.message, (args.standoff) * 100)
        )

        if not args.execute:
            print("(dry-run: add --execute to move)")
            continue

        # 4. the cue
        if input("type 'go' to touch: ").strip() != "go":
            print("aborted — nothing moved")
            continue

        # 5. fast transit, slow monitored touch
        if node.run_traj(plan_a.trajectory, transit) == "failed":
            print("transit failed — see planner log (auto-recovery may have run)")
            continue
        plan_b = node.plan_to(touch, quat)
        if plan_b is None or not plan_b.success:
            print(
                "final approach unplannable (%s) — retreating"
                % (plan_b.message if plan_b else "no answer")
            )
        else:
            verdict = node.run_traj(
                plan_b.trajectory, FINAL_SCALE_CAP, touch_nm=args.touch_nm
            )
            print(
                {
                    "touch": "TOUCH — hold...",
                    "arrived": "arrived at the palm plane (no contact felt)",
                    "failed": "final approach failed",
                }[verdict]
            )
            time.sleep(0.6)

        # 6. retreat home (from wherever contact stopped us)
        print("retreating home...")
        plan_r = node.plan_home()
        if plan_r is None or not plan_r.success:
            sys.exit("cannot plan retreat: %s" % (plan_r.message if plan_r else "?"))
        if node.run_traj(plan_r.trajectory, transit) == "failed":
            sys.exit("retreat failed — arm holds; see planner log")
        print("round complete — reset.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndemo stopped — arm holds")
