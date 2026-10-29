#!/usr/bin/env python3
"""Tracking bench for blended trajectories on the real arm.

Plans each case's legs on the blend planner (plan_to_pose / plan_to_joints,
each from the one before's end), blends them (/blend: radius 0 = stopped
baseline), checks the tool path stays in the clear box (own FK from
/arm/robot_description), executes on /execute_joint_trajectory at speed
scale 1 (the speeds are in the blend), and records the driver's feedback and
/joint_states. One npz per run in --out, a summary line per run on stdout.

    python3 track.py --out /out --plan-only      # plan and check, nothing moves
    python3 track.py --out /out --cases square --speeds 0.3
"""

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.msg import JointTolerance
from rammp_arm_interfaces.action import ExecuteJointTrajectory
from rammp_arm_interfaces.msg import EeState, ImpedanceGains
from rammp_curobo_interfaces.action import Blend, PlanToJoints, PlanToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from fk import FK

JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
HOME = np.radians([0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0])
PATH_TOL_RAD = 0.35  # as go_through
GAINS = {"soft": ImpedanceGains.PROFILE_SOFT, "medium": ImpedanceGains.PROFILE_MEDIUM, "stiff": ImpedanceGains.PROFILE_STIFF}
# The clear workspace: in front of the arm, above J2. Every planned tool
# position (and the flange) must be inside this, or nothing is sent.
BOX_LO = np.array([-0.15, -0.55, 0.26])
BOX_HI = np.array([0.80, 0.55, 0.95])
LEVEL, FREE, FIXED = 1, 0, 2
CONTINUOUS = [0, 2, 4, 6]  # J1/J3/J5/J7 spin freely: their errors are compared the short way


def wrap(e):
    e = np.array(e, dtype=float)
    e[..., CONTINUOUS] = (e[..., CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return e


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def traj_arrays(msg):
    order = [list(msg.joint_names).index(j) for j in JOINTS]
    t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    q = np.array([p.positions for p in msg.points])[:, order]
    qd = np.array([p.velocities for p in msg.points])[:, order]
    return t, q, qd


class Bench(Node):
    def __init__(self, planner):
        super().__init__("trajectory_bench")
        self.lock = threading.Lock()
        self.js, self.fb, self.ee = [], [], []
        self.urdf = None
        self.fb_names = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.create_subscription(EeState, "/ee_state", self._on_ee, qos_profile_sensor_data)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, "/arm/robot_description", self._on_urdf, latched)
        self.exec = ActionClient(self, ExecuteJointTrajectory, "/execute_joint_trajectory")
        self.pose = ActionClient(self, PlanToPose, f"{planner}/plan_to_pose")
        self.joints = ActionClient(self, PlanToJoints, f"{planner}/plan_to_joints")
        self.blend = ActionClient(self, Blend, f"{planner}/blend")

    def _on_js(self, m):
        if not set(JOINTS) <= set(m.name):
            return
        idx = [list(m.name).index(n) for n in JOINTS]
        with self.lock:
            self.js.append((stamp(m.header), [m.position[i] for i in idx], [m.velocity[i] for i in idx]))

    def _on_ee(self, m):
        p, o = m.pose.position, m.pose.orientation
        self.ee_frame = m.header.frame_id
        with self.lock:
            self.ee.append((stamp(m.header), [p.x, p.y, p.z, o.x, o.y, o.z, o.w]))

    def _on_urdf(self, m):
        self.urdf = m.data

    def _on_fb(self, m):
        f = m.feedback
        names = list(f.joint_names)
        self.fb_names = names
        # the driver's names may be joint_1..7 or none: then its points are in JOINTS order
        if set(JOINTS) <= set(names):
            order = [names.index(j) for j in JOINTS]
        elif {f"joint_{i}" for i in range(1, 8)} <= set(names):
            order = [names.index(f"joint_{i}") for i in range(1, 8)]
        else:
            order = list(range(7))
        with self.lock:
            self.fb.append(
                (
                    self.get_clock().now().nanoseconds * 1e-9,  # the header stamps do not advance
                    np.asarray(f.desired.positions)[order],
                    np.asarray(f.actual.positions)[order],
                    np.asarray(f.desired.velocities)[order] if f.desired.velocities else np.zeros(7),
                    np.asarray(f.actual.velocities)[order] if f.actual.velocities else np.zeros(7),
                    f.fraction_complete,
                    np.asarray(f.error.positions)[order] if len(f.error.positions) >= 7 else np.zeros(7),
                )
            )

    def joints_now(self):
        with self.lock:
            return np.array(self.js[-1][1])

    def ee_now(self):
        with self.lock:
            return np.array(self.ee[-1][1])


def wait(fut, timeout):
    t0 = time.monotonic()
    while not fut.done():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("timed out")
        time.sleep(0.002)
    return fut.result()


def call(client, goal, timeout=60.0):
    if not client.wait_for_server(timeout_sec=10.0):
        raise RuntimeError(f"no server {client._action_name}")
    h = wait(client.send_goal_async(goal), 10.0)
    if not h.accepted:
        raise RuntimeError(f"{client._action_name}: goal rejected")
    return wait(h.get_result_async(), timeout).result


def plan_leg(b, leg, start):
    if "joints" in leg:
        g = PlanToJoints.Goal(target_joints=[float(v) for v in leg["joints"]], start_joints=list(map(float, start)))
        r = call(b.joints, g)
    else:
        g = PlanToPose.Goal(start_joints=list(map(float, start)), hold=leg.get("hold", FREE))
        p, o = g.target.position, g.target.orientation
        p.x, p.y, p.z = map(float, leg["pos"])
        o.x, o.y, o.z, o.w = map(float, leg["quat"])
        r = call(b.pose, g)
    if not r.success:
        raise RuntimeError(f"leg {leg}: {r.message}")
    return r.trajectory, r.planning_time


def plan_case(b, legs, radii, scales, start):
    msgs, q, t_plan = [], start, 0.0
    for leg in legs:
        m, pt = plan_leg(b, leg, q)
        msgs.append(m)
        t_plan += pt
        q = traj_arrays(m)[1][-1]
    g = Blend.Goal(
        legs=msgs,
        blend_radius=[float(r) for r in radii],
        speed_scale=[float(s) for s in scales],
        hold=[0 if "joints" in leg else leg.get("hold", FREE) for leg in legs],  # each leg's own
    )
    t0 = time.monotonic()
    r = call(b.blend, g)
    if not r.success:
        raise RuntimeError(f"blend: {r.message}")
    return r.trajectory, list(r.corner_m), t_plan, time.monotonic() - t0, r.planning_time


def check_box(fk, q):
    tool, _ = fk.tool(q)
    flange, _ = fk.flange(q)
    pts = np.vstack([tool, flange])
    bad = (pts < BOX_LO) | (pts > BOX_HI)
    if bad.any():
        i = int(np.argmax(bad.any(axis=1)))
        return f"point {pts[i].round(3).tolist()} outside the clear box"
    return None


def execute(b, msg, label, gains=None):
    g = ExecuteJointTrajectory.Goal()
    g.trajectory = msg
    g.path_tolerance = [JointTolerance(name=j, position=PATH_TOL_RAD) for j in JOINTS]
    g.goal_tolerance = []
    g.control_mode, g.preemption = (0, 0) if gains is None else (1, 0)  # POSITION / IMPEDANCE, QUEUE
    if gains is not None:
        g.gains = ImpedanceGains(profile=GAINS[gains])
    g.sender_id = "trajectory_bench"
    g.speed_scale = 1.0
    if not b.exec.wait_for_server(timeout_sec=10.0):
        raise RuntimeError("no /execute_joint_trajectory")
    with b.lock:
        b.fb.clear()
        js0, ee0 = len(b.js), len(b.ee)
    t_send = b.get_clock().now().nanoseconds * 1e-9
    h = wait(b.exec.send_goal_async(g, feedback_callback=b._on_fb), 10.0)
    if not h.accepted:
        raise RuntimeError(f"{label}: driver rejected the goal")
    t_acc = b.get_clock().now().nanoseconds * 1e-9
    duration = traj_arrays(msg)[0][-1]
    res = wait(h.get_result_async(), duration + 30.0).result
    t_done = b.get_clock().now().nanoseconds * 1e-9
    time.sleep(0.5)  # the tail of the settle in the recording
    with b.lock:
        fb = list(b.fb)
        js = b.js[js0:]
        ee = b.ee[ee0:]
    return res, fb, js, ee, (t_send, t_acc, t_done)


def summarize(fk, label, msg, fb, js, times, res, corners, extra, fb_names=None):
    t, q, qd = traj_arrays(msg)
    out = {"label": label, "planned_s": float(t[-1]), "error_code": int(res.error_code), "error": res.error_string}
    out.update(extra)
    out["corner_cm"] = [round(c * 100, 2) for c in corners]
    out["exec_s"] = round(times[2] - times[1], 3)
    out["fb_names"] = fb_names
    if fb:
        ft = np.array([f[0] for f in fb])
        des = np.array([f[1] for f in fb])
        act = np.array([f[2] for f in fb])
        err = wrap(des - act)
        out["fb_n"] = len(fb)
        out["fb_hz"] = round((len(ft) - 1) / max(ft[-1] - ft[0], 1e-9), 1)
        out["fb_max_err_deg"] = round(float(np.degrees(np.abs(err).max())), 3)
        out["fb_rms_err_deg"] = round(float(np.degrees(np.sqrt((err**2).mean()))), 3)
        out["fb_max_err_joint"] = int(np.abs(err).max(axis=0).argmax()) + 1
        tool_d, _ = fk.tool(des)
        tool_a, _ = fk.tool(act)
        terr = np.linalg.norm(tool_d - tool_a, axis=1)
        out["tool_max_err_mm"] = round(float(terr.max() * 1000), 2)
        out["tool_rms_err_mm"] = round(float(np.sqrt((terr**2).mean()) * 1000), 2)
    if js:
        jt = np.array([j[0] for j in js])
        jq = np.array([j[1] for j in js])
        out["js_hz"] = round((len(jt) - 1) / max(jt[-1] - jt[0], 1e-9), 1)
        out["final_err_deg"] = round(float(np.degrees(np.abs(wrap(jq[-1] - q[-1])).max())), 3)
        # standstill inside the motion: measured speed below 2 % of the planned peak
        jv = np.array([j[2] for j in js])
        peak = np.abs(qd).max()
        moving = (jt >= times[1]) & (jt <= times[2])
        slow = (np.abs(jv[moving]).max(axis=1) < 0.02 * peak) if moving.any() else np.zeros(0, bool)
        out["slow_frac_in_motion"] = round(float(slow.mean()) if slow.size else 0.0, 3)
    return out


def save(path, label, msg, fb, js, ee, times, res, corners, summary):
    t, q, qd = traj_arrays(msg)
    np.savez_compressed(
        path / f"{label}.npz",
        t=t,
        q=q,
        qd=qd,
        fb_t=np.array([f[0] for f in fb]),
        fb_des=np.array([f[1] for f in fb]),
        fb_act=np.array([f[2] for f in fb]),
        fb_des_v=np.array([f[3] for f in fb]),
        fb_act_v=np.array([f[4] for f in fb]),
        fb_frac=np.array([f[5] for f in fb]),
        fb_err=np.array([f[6] for f in fb]),
        js_t=np.array([j[0] for j in js]),
        js_q=np.array([j[1] for j in js]),
        js_v=np.array([j[2] for j in js]),
        ee_t=np.array([e[0] for e in ee]),
        ee=np.array([e[1] for e in ee]),
        times=np.array(times),
        corners=np.array(corners),
        summary=json.dumps(summary),
    )


def cases(p0, quat):
    """The cases, from the tool's position p0 and orientation quat at HOME."""
    up = lambda dx, dy, dz: {"pos": (p0 + [dx, dy, dz]).tolist(), "quat": quat, "hold": LEVEL}  # noqa: E731
    free = lambda dx, dy, dz: {"pos": (p0 + [dx, dy, dz]).tolist(), "quat": quat, "hold": FREE}  # noqa: E731
    j = lambda *d: {"joints": (HOME + np.radians(d)).tolist()}  # noqa: E731
    return {
        # a vertical square in front, held level, back to its first corner
        "square": (
            [up(0, 0.15, -0.08), up(0, -0.15, -0.08), up(0, -0.15, 0.12), up(0, 0.15, 0.12), up(0, 0.15, -0.08)],
            [0.0, 0.03, 0.03, 0.03, 0.03],
        ),
        # held level, then free, then joints, free back to level, then level
        "mixed_hold": (
            [up(0.08, 0.2, -0.05), free(0.05, -0.2, 0.05), j(-25, 5, 0, 10, 0, 0, 0), free(0, 0, 0), up(0, 0.15, 0.1)],
            [0.0, 0.05, 0.05, 0.03, 0.03],
        ),
        # joint waypoints only: J1, J2, J4 and the wrist
        "joint_sweep": (
            [j(30, 0, 0, 0, 0, 0, 0), j(30, 10, 0, 20, 0, -20, 0), j(-30, 10, 0, 20, 0, -20, 40), j(-30, 0, 0, 0, 0, 0, 0), j(0, 0, 0, 0, 0, 0, 0)],
            [0.0, 0.05, 0.05, 0.05, 0.05],
        ),
        # grab_container's shape: view, pre-grasp, grasp, raise
        "grab_like": (
            [free(-0.05, 0.0, 0.2), up(0.12, 0.1, -0.05), {**up(0.12, 0.1, -0.13), "hold": FIXED}, up(0.12, 0.1, 0.05)],
            [0.0, 0.05, 0.01, 0.0],
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--planner", default="/rammp_curobo_blend")
    ap.add_argument("--cases", default="square,mixed_hold,joint_sweep,grab_like")
    ap.add_argument("--speeds", default="0.3,0.6,1.0")
    ap.add_argument("--mixed", action="store_true", help="also each case at alternating speeds 1.0 / 0.4")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--gains", choices=sorted(GAINS), help="IMPEDANCE with this profile (default: POSITION)")
    ap.add_argument("--home-speed", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    b = Bench(args.planner)
    ex = MultiThreadedExecutor(4)
    ex.add_node(b)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while (b.urdf is None or not b.js or not b.ee) and time.monotonic() - t0 < 15:
        time.sleep(0.05)
    if b.urdf is None or not b.js or not b.ee:
        raise SystemExit("no /arm/robot_description, /joint_states or /ee_state")
    fk = FK(b.urdf)
    (args.out / "arm.urdf").write_text(b.urdf)
    q_now = b.joints_now()
    p_ee = b.ee_now()
    p_fk, rot_fk = fk.flange(q_now)
    fk_err = float(np.linalg.norm(p_fk[0] - p_ee[:3]))
    print(
        json.dumps(
            {
                "start_joints_deg": np.degrees(q_now).round(1).tolist(),
                "fk_vs_ee_state_mm": round(fk_err * 1000, 2),
                "fk_root": fk.root,
                "fk_flange": p_fk[0].round(4).tolist(),
                "fk_quat": rot_fk[0].as_quat().round(4).tolist(),
                "ee_frame": b.ee_frame,
                "ee": np.round(p_ee, 4).tolist(),
            }
        ),
        flush=True,
    )
    if fk_err > 0.005:
        raise SystemExit(f"own FK is {fk_err * 1000:.1f} mm from /ee_state: not trusted")

    p0, rot0 = fk.tool(HOME)
    quat = rot0[0].as_quat().tolist()
    print(json.dumps({"home_tool": p0[0].round(3).tolist(), "home_quat": np.round(quat, 3).tolist()}), flush=True)
    all_cases = cases(p0[0], quat)

    def go_home(label):
        start = b.joints_now()
        if np.abs(start - HOME).max() < math.radians(0.5):
            return
        msg, corners, _, _, _ = plan_case(b, [{"joints": HOME.tolist()}], [], [args.home_speed], start)
        why = check_box(fk, traj_arrays(msg)[1]) if np.abs(start - HOME).max() < 0.8 else None
        if why:
            raise SystemExit(f"{label}: home path {why}")
        if args.plan_only:
            return
        res, fb, js, ee, times = execute(b, msg, label, args.gains)
        s = summarize(fk, label, msg, fb, js, times, res, corners, {}, b.fb_names)
        print(json.dumps(s), flush=True)
        if res.error_code != 0:
            raise SystemExit(f"{label}: {res.error_string}")

    go_home("home_start")
    speeds = [float(s) for s in args.speeds.split(",") if s]
    for name in args.cases.split(","):
        legs, radii = all_cases[name]
        runs = [(f"{s:g}", [s] * len(legs)) for s in speeds]
        if args.mixed:
            runs.append(("mixed", [1.0 if k % 2 == 0 else 0.4 for k in range(len(legs))]))
        for tag, scales in runs:
            for blended in (False, True):
                label = f"{name}_{tag}_{'blend' if blended else 'stop'}"
                go_home(f"home_before_{label}")
                start = HOME if args.plan_only else b.joints_now()
                r = radii[1:] if blended else [0.0] * (len(legs) - 1)
                try:
                    msg, corners, t_legs, t_blend_wall, t_blend = plan_case(b, legs, r, scales, start)
                except RuntimeError as e:
                    print(json.dumps({"label": label, "plan_failed": str(e)}), flush=True)
                    continue
                q = traj_arrays(msg)[1]
                why = check_box(fk, q)
                extra = {"leg_plan_s": round(t_legs, 3), "blend_s": round(t_blend, 3), "scales": scales, "gains": args.gains or "position"}
                if why:
                    print(json.dumps({"label": label, "refused": why}), flush=True)
                    continue
                if args.plan_only:
                    print(json.dumps({"label": label, "planned_s": round(float(traj_arrays(msg)[0][-1]), 3), "corner_cm": [round(c * 100, 2) for c in corners], **extra}), flush=True)
                    continue
                res, fb, js, ee, times = execute(b, msg, label, args.gains)
                s = summarize(fk, label, msg, fb, js, times, res, corners, extra, b.fb_names)
                save(args.out, label, msg, fb, js, ee, times, res, corners, s)
                print(json.dumps(s), flush=True)
                # an abort (the driver stopped the arm) is a result: the next run starts by going home
    go_home("home_end")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
