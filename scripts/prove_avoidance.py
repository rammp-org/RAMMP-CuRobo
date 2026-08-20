#!/usr/bin/env python3
"""Prove cuRobo avoids what the camera sees. Numbers, not vibes.

    python3 scripts/prove_avoidance.py                 # live boxes from /cameras
    python3 scripts/prove_avoidance.py --box 0.55 0.0 0.25 0.12 0.12 0.30

Plans the SAME A->B motion twice — once against the bare baseline world,
once with the perceived obstacle in it — runs forward kinematics on both
trajectories, and reports how close the WHOLE ARM (cuRobo's own collision
spheres, not just tool_frame) comes to the obstacle.

PASS means: the blind path goes through the obstacle, the aware path
clears it. That is the whole claim, measured.

No arm motion. Run the `cameras` node (with the Orbbec config) first if
you want live boxes; otherwise pass --box.
"""

import argparse
import sys

import numpy as np

from rammp_curobo import CuRoboPlanner


def live_boxes(timeout_s=8.0):
    """Perceived boxes from /cameras/world_markers -> [(name, c, dims)]."""
    import rclpy
    from visualization_msgs.msg import MarkerArray

    rclpy.init()
    node = rclpy.create_node("prove_avoidance_listener")
    got = {}

    def cb(msg):
        got["m"] = msg

    node.create_subscription(MarkerArray, "/cameras/world_markers", cb, 1)
    import time as _t

    t0 = _t.monotonic()
    while "m" not in got and _t.monotonic() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    out = []
    for k in got.get("m", type("x", (), {"markers": []})).markers:
        if getattr(k, "action", 0) != 0:
            continue
        p, s = k.pose.position, k.scale
        out.append((k.ns + str(k.id), [p.x, p.y, p.z], [s.x, s.y, s.z]))
    return out


def _box_dist(pts, center, dims):
    """Distance from each point to an axis-aligned box surface (0 inside)."""
    c = np.asarray(center, float)
    h = np.asarray(dims, float) / 2.0
    d = np.maximum(np.abs(np.asarray(pts, float) - c) - h, 0.0)
    return np.linalg.norm(d, axis=1)


def arm_clearance(planner, traj, center, dims):
    """Min gap (m) between the WHOLE arm and the box over a trajectory.

    Uses cuRobo's own collision spheres, so 0.0 means the arm is
    genuinely inside the obstacle — not merely that tool_frame is."""
    worst = 1e9
    for q in traj.positions:
        sph = planner.link_spheres(q)
        gaps = _box_dist(sph[:, :3], center, dims) - sph[:, 3]
        worst = min(worst, float(gaps.min()))
    return worst


def tool_path(planner, traj):
    return np.array([planner.fk(q)[0] for q in traj.positions])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="gen3_real.yaml")
    ap.add_argument("--box", type=float, nargs=6, metavar=("X", "Y", "Z", "DX", "DY", "DZ"),
                    help="obstacle instead of the live perceived ones")
    ap.add_argument("--start", type=float, nargs=3, default=[0.55, -0.30, 0.35])
    ap.add_argument("--goal", type=float, nargs=3, default=[0.55, 0.30, 0.35])
    ap.add_argument("--margin", type=float, default=0.02,
                    help="clearance (m) the aware path must achieve")
    args = ap.parse_args()

    if args.box:
        boxes = [("manual", args.box[:3], args.box[3:])]
    else:
        boxes = live_boxes()
        if not boxes:
            sys.exit("no boxes on /cameras/world_markers — is the cameras "
                     "node running with the Orbbec config? (or pass --box)")
    print("obstacles considered: %d" % len(boxes))
    for n, c, d in boxes:
        print("  %-14s centre %s  dims %s" % (n, np.round(c, 3), np.round(d, 3)))

    planner = CuRoboPlanner.from_config(args.config)
    quat = [0.5, 0.5, 0.5, 0.5]          # wrist flat, tool along +x

    # get joint values for A by planning home -> A in the bare world
    planner.update_world_boxes([])
    to_start = planner.plan_to_pose(args.start, quat, start=None)
    if not to_start.success:
        sys.exit("cannot reach --start %s (%s)" % (args.start, to_start.status))
    start_q = list(to_start.joint_traj.positions[-1])
    print("start pose reachable; planning %s -> %s"
          % (np.round(args.start, 2), np.round(args.goal, 2)))

    # A -> B, blind (baseline world only)
    blind = planner.plan_to_pose(args.goal, quat, start=start_q)
    if not blind.success:
        sys.exit("blind plan failed (%s) — pick a reachable --goal" % blind.status)
    blind_path = tool_path(planner, blind.joint_traj)

    # A -> B, aware (baseline + the perceived obstacle)
    planner.update_world_boxes(
        [{"name": n, "position": list(c), "dims": list(d)} for n, c, d in boxes]
    )
    aware = planner.plan_to_pose(args.goal, quat, start=start_q)
    if not aware.success:
        print("\naware plan FAILED (%s)" % aware.status)
        print("cuRobo refused to route through the obstacle — that is "
              "avoidance too, just the conservative kind. Move the goal or "
              "the obstacle so a way around exists.")
        return
    aware_path = tool_path(planner, aware.joint_traj)

    print("\n%-8s %-7s %-9s %s" % ("path", "points", "arm gap", "verdict"))
    worst_blind, worst_aware = 1e9, 1e9
    for _, c, d in boxes:
        worst_blind = min(worst_blind, arm_clearance(planner, blind.joint_traj, c, d))
        worst_aware = min(worst_aware, arm_clearance(planner, aware.joint_traj, c, d))
    print("%-8s %-7d %-9.3f %s" % ("blind", len(blind_path), worst_blind,
                                   "HITS IT" if worst_blind <= 0.0 else "misses anyway"))
    print("%-8s %-7d %-9.3f %s" % ("aware", len(aware_path), worst_aware,
                                   "CLEARS" if worst_aware >= args.margin else "TOO CLOSE"))
    dev = float(np.abs(
        np.linalg.norm(aware_path[:, None] - blind_path[None], axis=2).min(axis=1)
    ).max())
    print("\naware path deviates up to %.3f m from the blind one" % dev)

    if worst_blind > 0.0:
        print("\nINCONCLUSIVE: the blind path already missed the obstacle. "
              "Put the obstacle between --start and --goal.")
    elif worst_aware >= args.margin:
        print("\nPASS — the blind path goes through the obstacle, the aware "
              "path clears it by %.0f mm. cuRobo is using the camera." % (worst_aware * 1000))
    else:
        print("\nFAIL — the aware path still comes within %.0f mm."
              % (worst_aware * 1000))


if __name__ == "__main__":
    main()
