#!/usr/bin/env python3
"""On-arm bench for the driver's go_to_ee_poses / go_to_joint_configs (kinova-gen3-ros2#65).

Each case runs twice as ONE goal: every junction stopped (blend_radius empty), then
blended. The driver plans the legs (cuRobo, its own world) and executes; this records
its feedback (phase, actual joints) and /joint_states. One npz per run in --out, a
summary line per run on stdout. The GoTo actions return no plan, so this measures what
the arm did: motion time, standstill inside the motion, planning time, final error.

    python3 track_goto.py --out /out --speeds 0.5
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rammp_arm_interfaces.action import GoToEEPoses, GoToJointConfigs
from rammp_arm_interfaces.msg import EEPoseTarget, EeState, JointConfigTarget
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String


JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
HOME = np.radians([0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0])
# The tool frame at HOME in arm_mount, from this arm's URDF (fk.py, earlier runs on
# abra); hard-coded because the driver image this runs in has no scipy.
HOME_TOOL = np.array([0.5766651714706285, 0.00134629487675801, 0.4337242777644498])
HOME_QUAT = [0.5000014238335733, 0.49999857616247306, 0.5000005379280927, 0.49999946207122753]
FREE, LEVEL, FIXED = 0, 1, 2
CONTINUOUS = [0, 2, 4, 6]


def wrap(e):
    e = np.array(e, dtype=float)
    e[..., CONTINUOUS] = (e[..., CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return e


class Bench(Node):
    def __init__(self):
        super().__init__("goto_bench")
        self.lock = threading.Lock()
        self.js, self.fb = [], []
        self.urdf = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.create_subscription(EeState, "/ee_state", lambda m: None, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/arm/robot_description", self._on_urdf, latched)
        self.poses = ActionClient(self, GoToEEPoses, "/go_to_ee_poses")
        self.joints = ActionClient(self, GoToJointConfigs, "/go_to_joint_configs")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_js(self, m):
        if not set(JOINTS) <= set(m.name):
            return
        idx = [list(m.name).index(n) for n in JOINTS]
        with self.lock:
            self.js.append((self.now(), [m.position[i] for i in idx], [m.velocity[i] for i in idx]))

    def _on_urdf(self, m):
        self.urdf = m.data

    def _on_fb(self, m):
        f = m.feedback
        q = list(f.actual.positions)[:7] if len(f.actual.positions) >= 7 else [np.nan] * 7
        with self.lock:
            self.fb.append((self.now(), f.phase, f.planner_state, float(f.fraction_complete), q))

    def joints_now(self):
        with self.lock:
            return np.array(self.js[-1][1])


def wait(fut, timeout):
    t0 = time.monotonic()
    while not fut.done():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("timed out")
        time.sleep(0.002)
    return fut.result()


def run_goal(b, client, goal, label, timeout=90.0):
    if not client.wait_for_server(timeout_sec=10.0):
        raise RuntimeError(f"no server {client._action_name}")
    with b.lock:
        b.fb.clear()
        js0 = len(b.js)
    t_send = b.now()
    h = wait(client.send_goal_async(goal, feedback_callback=b._on_fb), 10.0)
    if not h.accepted:
        raise RuntimeError(f"{label}: goal REJECTED (see the driver's log for why)")
    res = wait(h.get_result_async(), timeout).result
    t_done = b.now()
    time.sleep(0.5)  # the settle in the recording
    with b.lock:
        fb, js = list(b.fb), b.js[js0:]
    return res, fb, js, (t_send, t_done)


def summarize(label, res, fb, js, times, goal_q):
    out = {"label": label, "error_code": int(res.error_code), "error": res.error_string}
    t_exec = [f[0] for f in fb if f[1] == "executing"]
    out["planning_s"] = round((t_exec[0] if t_exec else times[1]) - times[0], 3)
    if js:
        jt = np.array([j[0] for j in js])
        jv = np.array([j[2] for j in js])
        speed = np.abs(jv).max(axis=1)
        moving = np.flatnonzero(speed > 0.02)  # rad/s
        if moving.size:
            inside = slice(moving[0], moving[-1] + 1)
            dt = np.diff(jt[inside], append=jt[inside][-1])
            out["motion_s"] = round(float(jt[moving[-1]] - jt[moving[0]]), 3)
            out["still_inside_s"] = round(float(dt[speed[inside] <= 0.02].sum()), 3)
            out["peak_rad_s"] = round(float(speed.max()), 3)
        if goal_q is not None:
            out["final_err_deg"] = round(float(np.degrees(np.abs(wrap(np.array(js[-1][1]) - goal_q)).max())), 3)
    return out


def pose_goal(targets, holds, radii, scale):
    """GoToEEPoses through `targets` [(pos, quat xyzw)], target k with holds[k] and
    the radius radii[k] into the next leg (none on the last), every leg at `scale`."""
    g = GoToEEPoses.Goal()
    for k, (pos, quat) in enumerate(targets):
        t = EEPoseTarget()
        t.pose.header.frame_id = "arm_mount"
        t.pose.pose.position.x, t.pose.pose.position.y, t.pose.pose.position.z = map(float, pos)
        o = t.pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, quat)
        t.orientation_hold = int(holds[k])
        t.blend_radius = float(radii[k]) if k < len(radii) else 0.0
        t.speed_scale = float(scale)
        g.targets.append(t)
    g.sender_id = "goto_bench"
    return g


def joints_goal(qs, radii, scale):
    g = GoToJointConfigs.Goal()
    for k, q in enumerate(qs):
        t = JointConfigTarget(target_joints=[float(v) for v in q], speed_scale=float(scale))
        t.blend_radius = float(radii[k]) if k < len(radii) else 0.0
        g.targets.append(t)
    g.sender_id = "goto_bench"
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speeds", default="0.5")
    ap.add_argument("--cases", default="square,pose_mix,level_turn,joint_sweep,grab_like")
    ap.add_argument("--home-speed", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    b = Bench()
    ex = MultiThreadedExecutor(4)
    ex.add_node(b)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while not b.js and time.monotonic() - t0 < 15:
        time.sleep(0.05)
    if not b.js:
        raise SystemExit("no /joint_states")
    p0, quat = HOME_TOOL, HOME_QUAT
    print(json.dumps({"start_joints_deg": np.degrees(b.joints_now()).round(1).tolist(), "home_tool": p0.round(3).tolist()}), flush=True)

    def at(dx, dy, dz, yaw_deg=0.0):
        """HOME's tool offset by (dx, dy, dz), turned yaw_deg about arm_mount Z."""
        h = np.radians(yaw_deg) / 2
        qz = np.array([0.0, 0.0, np.sin(h), np.cos(h)])  # xyzw, about Z, applied first (left)
        x1, y1, z1, w1 = qz
        x2, y2, z2, w2 = quat
        q = [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
        return (p0 + [dx, dy, dz]).tolist(), q

    J = lambda *d: (HOME + np.radians(d)).tolist()  # noqa: E731
    cases = {
        # (kind, targets, holds (poses), radii when blended)
        "square": ("poses", [at(0, 0.15, -0.08), at(0, -0.15, -0.08), at(0, -0.15, 0.12), at(0, 0.15, 0.12), at(0, 0.15, -0.08)],
                   [LEVEL] * 5, [0.03] * 4),
        "pose_mix": ("poses", [at(0.08, 0.2, -0.05), at(0.05, -0.2, 0.05), at(0, 0, 0), at(0, 0.15, 0.1)],
                     [LEVEL, FREE, FREE, LEVEL], [0.05, 0.05, 0.03]),
        "joint_sweep": ("joints", [J(30, 0, 0, 0, 0, 0, 0), J(30, 10, 0, 20, 0, -20, 0), J(-30, 10, 0, 20, 0, -20, 40), J(-30, 0, 0, 0, 0, 0, 0), J(0, 0, 0, 0, 0, 0, 0)],
                        None, [0.05] * 4),
        # held level, then still level while the tool turns 40 deg about the vertical,
        # which LEVEL leaves free: the blend may overlap the turn with the first leg
        "level_turn": ("poses", [at(0.05, 0.15, 0.0), at(0.05, -0.1, 0.05, 40.0), at(0, 0, 0)],
                       [LEVEL, LEVEL, LEVEL], [0.04, 0.03]),
        "grab_like": ("poses", [at(-0.05, 0.0, 0.2), at(0.12, 0.1, -0.05), at(0.12, 0.1, -0.13), at(0.12, 0.1, 0.05)],
                      [FREE, LEVEL, FIXED, LEVEL], [0.05, 0.01, 0.0]),
    }

    def go_home(label):
        if np.abs(wrap(b.joints_now() - HOME)).max() < np.radians(0.5):
            return
        res, fb, js, times = run_goal(b, b.joints, joints_goal([HOME], [], args.home_speed), label)
        s = summarize(label, res, fb, js, times, HOME)
        print(json.dumps(s), flush=True)
        if res.error_code != 0:
            raise SystemExit(f"{label}: {res.error_string}")

    go_home("home_start")
    for speed in [float(s) for s in args.speeds.split(",")]:
        for name in args.cases.split(","):
            kind, targets, holds, radii = cases[name]
            for blended in (False, True):
                label = f"{name}_{speed:g}_{'blend' if blended else 'stop'}"
                go_home(f"home_before_{label}")
                r = radii if blended else []
                if kind == "poses":
                    g, client, goal_q = pose_goal(targets, holds, r, speed), b.poses, None
                else:
                    g, client, goal_q = joints_goal(targets, r, speed), b.joints, np.array(targets[-1])
                try:
                    res, fb, js, times = run_goal(b, client, g, label)
                except RuntimeError as e:
                    print(json.dumps({"label": label, "refused": str(e)}), flush=True)
                    continue
                s = summarize(label, res, fb, js, times, goal_q)
                np.savez_compressed(
                    args.out / f"{label}.npz",
                    js_t=np.array([j[0] for j in js]), js_q=np.array([j[1] for j in js]), js_v=np.array([j[2] for j in js]),
                    fb_t=np.array([f[0] for f in fb]), fb_phase=np.array([f[1] for f in fb]),
                    fb_q=np.array([f[4] for f in fb]), times=np.array(times), summary=json.dumps(s),
                )
                print(json.dumps(s), flush=True)
    go_home("home_end")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
