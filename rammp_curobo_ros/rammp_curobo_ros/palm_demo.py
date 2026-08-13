#!/usr/bin/env python3
"""Palm-touch demo with live video — detect, lock, cue, touch, retreat.

One command; watch the annotated live view in a browser at
http://192.168.1.11:8405 through every phase:

    ros2 run rammp_curobo_ros palm_demo --execute

Flow, shown live on the overlay:
  SCANNING   present an open palm 0.35-0.8 m in front of the camera;
             MediaPipe finds it, depth+TF give its 3D point, the
             workspace gate vets it
  LOCKED     the palm held still ~1 s: target frozen, plan computed —
             type  go<enter>  to run (anything else rescans)
  TRANSIT    fast move to a standoff short of the palm (clamped 0.6)
  TOUCH      slow monitored final approach (hard-capped 0.15) that stops
             the instant wrist torque feels contact
  RETREAT    back to home, then scanning again — next person

Safety (do not weaken): explicit 'go' per round; person HOLDS STILL after
'go' (the plan targets where the palm WAS; if it moved >6 cm by then the
round aborts) — moving the hand away is the human abort; Ctrl+C cancels
the active goal (arm stops and holds); a human holds the physical e-stop.
Requires the planner node with execute:=true and a real-bench world.
"""

import argparse
import math
import select
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

from rammp_curobo.geometry import euler_deg_to_quat_xyzw
from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose
from rammp_curobo_ros.palm_common import (
    STREAM_PORT,
    ColorDepthGrabber,
    _MjpegServer,
    depth_at,
    landmark_palms,
    make_hands,
    palm_target_ok,
    push_stream,
)
from rammp_curobo_ros.scan_common import (
    NODE_NAMESPACE,
    load_camera_config,
    spin_until_done,
)

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

FINAL_SCALE_CAP = 0.15
TRANSIT_SCALE_CAP = 0.6


class PalmDemo(ColorDepthGrabber):
    def __init__(self, camera_cfg, color_topic):
        super().__init__(camera_cfg, color_topic)
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
                sys.exit(
                    "no /joint_states — start the arm stack first:\n"
                    "  ros2 launch rammp_curobo_ros planner.launch.py "
                    "config:=gen3_real.yaml execute:=true launch_arm:=true"
                )

    def wrist_effort(self):
        with self._js_lock:
            return None if self._effort is None else list(self._effort[3:])

    def _result(self, client, goal, timeout_s):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running (execute:=true needed)")
        send = spin_until_done(self, client.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(self, send.get_result_async(), timeout_s)
        return None if wrapped is None else wrapped.result

    def plan_to(self, pos, quat_xyzw):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = pos
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = quat_xyzw
        return self._result(self.plan_pose, g, 120.0)

    def plan_home(self):
        return self._result(
            self.plan_joints, PlanToJoints.Goal(target_joints=HOME), 120.0
        )

    def run_traj(self, traj, scale, touch_nm=None, on_tick=None):
        """Execute; cancel on wrist-torque contact when touch_nm is set.

        on_tick(): called every loop so the video keeps streaming during
        motion. Ctrl+C cancels the controller goal (arm stops and holds).
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
        try:
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if on_tick is not None:
                    on_tick()
                if touch_nm is not None:
                    eff = self.wrist_effort()
                    if eff is not None:
                        if baseline is None and time.monotonic() - t0 > 0.4:
                            baseline = eff
                        elif baseline is not None:
                            dev = max(abs(a - b) for a, b in zip(eff, baseline))
                            if dev > touch_nm:
                                contact = True
                                spin_until_done(self, send.cancel_goal_async(), 3.0)
                                spin_until_done(self, result_future, 10.0)
                                return "touch"
                if time.monotonic() - t0 > 240:
                    spin_until_done(self, send.cancel_goal_async(), 3.0)
                    return "failed"
        except KeyboardInterrupt:
            spin_until_done(self, send.cancel_goal_async(), 3.0)
            print("\nCtrl+C — goal cancelled, arm holds")
            raise
        wrapped = result_future.result()
        if wrapped is not None and wrapped.result.success:
            return "arrived"
        return "touch" if contact else "failed"


def read_key_line():
    """Non-blocking: a full line from stdin if one is waiting, else None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None


def main():
    import cv2

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", default="camera_d405_wrist.yaml")
    ap.add_argument("--color-topic", default="/d405/d405/color/image_rect_raw")
    ap.add_argument("--execute", action="store_true", help="allow motion")
    ap.add_argument(
        "--transit-scale",
        type=float,
        default=0.5,
        help="home<->standoff speed (clamped to %.1f)" % TRANSIT_SCALE_CAP,
    )
    ap.add_argument("--touch-nm", type=float, default=3.0)
    ap.add_argument("--standoff", type=float, default=0.12)
    ap.add_argument("--touch-back", type=float, default=0.015)
    ap.add_argument(
        "--stability",
        type=float,
        default=1.0,
        help="seconds the palm must hold still to lock",
    )
    args = ap.parse_args()
    transit = min(max(args.transit_scale, 0.1), TRANSIT_SCALE_CAP)

    rclpy.init()
    node = PalmDemo(load_camera_config(args.camera), args.color_topic)
    hands = make_hands()
    stream = None
    try:
        stream = _MjpegServer(STREAM_PORT)
        print("LIVE VIEW: http://192.168.1.11:%d  (open in any browser)" % STREAM_PORT)
    except OSError as exc:
        print("stream port %d unavailable (%s) — no live view" % (STREAM_PORT, exc))

    state = {"name": "SCANNING", "msg": ""}
    lock = {"target": None, "since": None, "history": []}

    def set_state(name, msg=""):
        """Update the banner AND announce transitions in the terminal, so
        the demo is followable without the browser view."""
        if name != state["name"]:
            print(">> %s%s" % (name, ("  " + msg) if msg else ""))
        state.update(name=name, msg=msg)

    def annotate_and_show():
        """Grab the newest frame, draw hands + state banner, display."""
        if node.color is None or not node.frames or node.info is None:
            return None
        img, enc = node.color
        node.color = None
        depth = node.frames[-1]
        del node.frames[:-1]
        rgb = img if enc == "rgb8" else img[:, :, ::-1]
        frame = np.ascontiguousarray(rgb[:, :, ::-1])
        k = np.array(node.info.k).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        h_img, w_img = rgb.shape[:2]
        sx, sy = depth.shape[1] / w_img, depth.shape[0] / h_img

        palms = []
        seen = {"hands": 0, "why": ""}
        try:
            R, t = node.camera_pose(timeout_s=1.5)
        except SystemExit:
            R = t = None
        for (pu, pv), (x0, y0, x1, y1), _lms in landmark_palms(hands, rgb):
            seen["hands"] += 1
            z = depth_at(depth, pu, pv, sx, sy)
            base = None
            if z is not None and R is not None:
                cam = np.array([(pu - cx) / fx * z, (pv - cy) / fy * z, z])
                base = R @ cam + t
                palms.append(base)
                seen["why"] = palm_target_ok(base)[1]
            elif R is None:
                seen["why"] = "no TF (arm stack down?)"
            else:
                seen["why"] = "no depth on the palm"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.circle(frame, (int(pu), int(pv)), 6, (0, 200, 0), -1)
            if base is not None:
                ok, why = palm_target_ok(base)
                cv2.putText(
                    frame,
                    "[%.2f %.2f %.2f] %s"
                    % (base[0], base[1], base[2], "OK" if ok else why),
                    (x0, max(y0 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 0),
                    1,
                )
        banner = state["name"] + ("  " + state["msg"] if state["msg"] else "")
        cv2.rectangle(frame, (0, 0), (w_img, 26), (40, 40, 40), -1)
        cv2.putText(
            frame,
            banner,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        push_stream(frame, stream)
        return palms, seen

    print(
        "PALM DEMO — person: open palm facing the arm, 0.4-0.7 m out, HOLD\n"
        "STILL once 'go' is typed (moving away = your abort). Operator: hand\n"
        "on the e-stop; type go<enter> when LOCKED; q<enter> quits."
        + ("" if args.execute else "\n(DRY-RUN: no --execute, nothing moves)")
    )
    if node.wrist_effort() is None:
        node.joints()
        if node.wrist_effort() is None:
            print(
                "WARNING: no effort in /joint_states — touch detection by "
                "position only"
            )

    print(">> SCANNING — present an open palm 0.4-0.7 m in front of the camera")
    last_status = [""]

    def scan_status(text):
        """Narrate what the scanner sees, once per change (not per frame)."""
        if text != last_status[0]:
            print("   %s" % text)
            last_status[0] = text

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        key = read_key_line()
        if key == "q":
            break

        # keep the arm at the home vantage while scanning
        q_now = node.joints()
        if (
            state["name"] == "SCANNING"
            and max(abs(a - b) for a, b in zip(q_now, HOME)) > 0.1
        ):
            if not lock.get("home_ok"):
                input(
                    "arm is away from home — press ENTER to home it "
                    "(hand on e-stop), Ctrl+C to quit: "
                )
                lock["home_ok"] = True
            set_state("RETREAT", "returning to home vantage")
            annotate_and_show()
            plan = node.plan_home()
            if plan is None or not plan.success:
                sys.exit("cannot plan home")
            if args.execute:
                node.run_traj(plan.trajectory, transit, on_tick=annotate_and_show)
            set_state("SCANNING")
            continue

        result = annotate_and_show()
        if result is None:
            continue
        palms, seen = result

        if state["name"] == "SCANNING":
            good = [p for p in palms if palm_target_ok(p)[0]]
            if not good:
                if seen["hands"] == 0:
                    scan_status("no hand in view")
                else:
                    scan_status("hand seen — %s" % (seen["why"] or "refused"))
                state["msg"] = "present an open palm"
                lock.update(target=None, since=None, history=[])
                continue
            scan_status("palm OK — hold still...")
            p = good[0]
            hist = lock["history"]
            hist.append((time.monotonic(), p))
            del hist[: max(0, len(hist) - 30)]
            if lock["since"] is None:
                lock.update(since=time.monotonic())
            recent = [h for h in hist if h[0] > time.monotonic() - args.stability]
            drift = (
                max(np.linalg.norm(np.asarray(a[1]) - np.asarray(p)) for a in recent)
                if recent
                else 1.0
            )
            if drift > 0.03:
                lock.update(since=time.monotonic())
                state["msg"] = "hold still..."
                continue
            if time.monotonic() - lock["since"] < args.stability:
                state["msg"] = "hold still..."
                continue
            # locked: freeze target, plan the transit
            target = np.mean([h[1] for h in recent], axis=0)
            lock["target"] = target
            yaw = math.degrees(math.atan2(target[1], target[0]))
            quat = list(euler_deg_to_quat_xyzw([0.0, 90.0, yaw]))
            ux, uy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            touch = [
                target[0] - args.touch_back * ux,
                target[1] - args.touch_back * uy,
                target[2],
            ]
            standoff = [
                target[0] - (args.standoff + args.touch_back) * ux,
                target[1] - (args.standoff + args.touch_back) * uy,
                target[2],
            ]
            set_state("PLANNING")
            annotate_and_show()
            plan_a = node.plan_to(standoff, quat)
            if plan_a is None or not plan_a.success:
                set_state("SCANNING", "unreachable — move the palm")
                lock.update(target=None, since=None, history=[])
                continue
            lock.update(quat=quat, touch=touch, standoff=standoff, plan_a=plan_a)
            set_state("LOCKED", "[%.2f %.2f %.2f] — type go<enter>" % tuple(target))
            continue

        if state["name"] == "LOCKED":
            # unlock if the palm wandered off before the cue
            good = [p for p in palms if palm_target_ok(p)[0]]
            if good and np.linalg.norm(np.asarray(good[0]) - lock["target"]) > 0.06:
                set_state("SCANNING", "palm moved — relocking")
                lock.update(target=None, since=None, history=[])
                continue
            if key != "go":
                continue
            if not args.execute:
                set_state("SCANNING", "dry-run: plan OK (add --execute)")
                lock.update(target=None, since=None, history=[])
                continue
            set_state("TRANSIT", "fast to standoff")
            if (
                node.run_traj(
                    lock["plan_a"].trajectory, transit, on_tick=annotate_and_show
                )
                == "failed"
            ):
                set_state("SCANNING", "transit failed — rescan")
                lock.update(target=None, since=None, history=[])
                continue
            set_state("TOUCH", "slow approach...")
            plan_b = node.plan_to(lock["touch"], lock["quat"])
            verdict = "failed"
            if plan_b is not None and plan_b.success:
                verdict = node.run_traj(
                    plan_b.trajectory,
                    FINAL_SCALE_CAP,
                    touch_nm=args.touch_nm,
                    on_tick=annotate_and_show,
                )
            verdict_msg = {
                "touch": "CONTACT!",
                "arrived": "at palm plane",
                "failed": "approach failed",
            }[verdict]
            print(">> TOUCH  %s" % verdict_msg)
            state.update(msg=verdict_msg)
            end = time.monotonic() + 0.8
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
                annotate_and_show()
            set_state("RETREAT")
            plan_r = node.plan_home()
            if plan_r is None or not plan_r.success:
                sys.exit("cannot plan retreat — arm holds")
            if (
                node.run_traj(plan_r.trajectory, transit, on_tick=annotate_and_show)
                == "failed"
            ):
                sys.exit("retreat failed — arm holds; see planner log")
            set_state("SCANNING", "next!")
            last_status[0] = ""
            lock.update(target=None, since=None, history=[])

    print("demo ended")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndemo stopped — arm holds")
