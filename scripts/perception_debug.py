#!/usr/bin/env python3
"""Where are the perceived boxes, and is the arm seeing ITSELF?

    python3 scripts/perception_debug.py            # 20 s sample
    python3 scripts/perception_debug.py --seconds 60

Needs the arm bringup, the camera, and the cameras node (the planner node
is NOT required — this loads its own kinematics for the arm model).

Answers the two questions a flapping obstacle map raises:

  1. WHERE is each box, and how close is it to the arm's own collision
     spheres? A box with a negative gap overlaps the arm, which is
     exactly what makes cuRobo refuse to plan at all
     (INVALID_START_STATE_WORLD_COLLISION) — the demo's permanent HOLD.
  2. Is the map STABLE? A box present on 40% of ticks is noise being
     confirmed and forgotten over and over, not an object.

Reads only. Never plans, never moves.
"""

import argparse
import collections
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--config", default="gen3_real.yaml")
    ap.add_argument("--near", type=float, default=0.05,
                    help="flag boxes closer than this to the arm (m)")
    ap.add_argument("--camera-config",
                    default="rammp_curobo_ros/config/camera_orbbec_bench.yaml",
                    help="read mount_xyz from here to print a corrected line")
    args = ap.parse_args()

    import yaml

    from rammp_curobo import CuRoboPlanner

    cam_xyz = None
    try:
        with open(args.camera_config) as f:
            cam_xyz = np.array(yaml.safe_load(f)["mount_xyz"], dtype=float)
    except Exception as exc:
        print("could not read %s (%s) — offsets will keep the "
              "near-surface bias" % (args.camera_config, exc))

    print("loading kinematics (%s)..." % args.config)
    planner = CuRoboPlanner.from_config(args.config)
    names = list(planner.joint_names)

    rclpy.init()
    node = rclpy.create_node("perception_debug")
    state = {"q": None, "boxes": None, "stamp": 0.0}

    def js_cb(msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            state["q"] = [float(msg.position[idx[n]]) for n in names]
        except (KeyError, IndexError):
            pass

    def mk_cb(msg):
        out = []
        for m in msg.markers:
            if getattr(m, "action", 0) != 0:
                continue
            p, s = m.pose.position, m.scale
            out.append((m.ns + str(m.id), [p.x, p.y, p.z], [s.x, s.y, s.z]))
        state["boxes"] = out
        state["stamp"] = time.monotonic()

    node.create_subscription(JointState, "/joint_states", js_cb, 10)
    node.create_subscription(MarkerArray, "/cameras/world_markers", mk_cb, 1)

    print("sampling %.0f s — hold still, then wave a hand if you like\n"
          % args.seconds)
    seen = collections.Counter()
    detail = {}
    counts = collections.Counter()
    worst_gap = {}
    offsets = []
    ticks = 0
    last = -1.0
    end = time.monotonic() + args.seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if state["boxes"] is None or state["q"] is None:
            continue
        if state["stamp"] == last:
            continue
        last = state["stamp"]
        ticks += 1
        boxes = state["boxes"]
        counts[len(boxes)] += 1

        spheres = planner.link_spheres(state["q"])
        spheres = spheres[spheres[:, 3] > 0.0]
        line = []
        for name, c, d in boxes:
            gap = planner.trajectory_clearance([state["q"]],
                                               [{"position": c, "dims": d}])
            key = tuple(np.round(np.array(c) / 0.05).astype(int))
            seen[key] += 1
            detail[key] = (c, d)
            worst_gap[key] = min(worst_gap.get(key, 1e9), gap)
            if gap < args.near:
                # A box sitting on the arm IS the arm, mis-placed. Compare
                # it to where the arm's visible surface SHOULD be drawn —
                # not to the axis: depth only ever sees the camera-facing
                # side, so a box centre sits about one arm-radius toward
                # the camera by geometry alone. Subtract that or the
                # estimate over-corrects by ~4 cm.
                j = int(np.argmin(np.linalg.norm(spheres[:, :3] - np.array(c),
                                                 axis=1)))
                axis, radius = spheres[j, :3], spheres[j, 3]
                expected = axis
                if cam_xyz is not None:
                    toward = cam_xyz - axis
                    n = float(np.linalg.norm(toward))
                    if n > 1e-6:
                        expected = axis + radius * toward / n
                offsets.append(np.array(c) - expected)
            tag = ("ON-ARM" if gap < 0 else
                   "near" if gap < args.near else "ok")
            line.append("%s@%s %s(%.3f)" % (name, np.round(c, 2), tag, gap))
        if ticks % 5 == 1:
            print("  [%2d boxes] %s" % (len(boxes), "  ".join(line[:4])))

    node.destroy_node()
    rclpy.shutdown()

    if not ticks:
        sys.exit("no marker updates seen — is the cameras node running, and "
                 "is the arm bringup up so /joint_states publishes?")

    print("\n%d marker updates over %.0f s" % (ticks, args.seconds))
    print("box-count distribution: %s"
          % ", ".join("%d boxes x%d" % (k, v) for k, v in sorted(counts.items())))

    print("\n%-26s %-22s %-8s %-8s %s"
          % ("location (base_link)", "dims", "seen", "duty", "closest to arm"))
    on_arm = 0
    flappy = 0
    for key, n in seen.most_common():
        c, d = detail[key]
        duty = n / float(ticks)
        gap = worst_gap[key]
        note = ""
        if gap < 0:
            note = "<-- OVERLAPS THE ARM"
            on_arm += 1
        elif gap < args.near:
            note = "<-- within %.0f cm of the arm" % (args.near * 100)
        if 0.15 < duty < 0.85:
            flappy += 1
            note += "  UNSTABLE"
        print("%-26s %-22s %-8d %-8.0f%% %.3f m %s"
              % (np.round(c, 3), np.round(d, 3), n, duty * 100, gap, note))

    if offsets:
        off = np.array(offsets)
        mean = off.mean(axis=0)
        spread = off.std(axis=0)
        print("\nAPPARENT CAMERA-POSE ERROR")
        print("  Measured from the arm itself: where its own body is drawn,")
        print("  minus where TF says it is. %d samples." % len(off))
        print("  offset  %s m   (spread %s)"
              % (np.round(mean, 3), np.round(spread, 3)))
        print("  magnitude %.3f m" % float(np.linalg.norm(mean)))
        if spread.max() < 0.05:
            print("  Consistent across samples -> looks like a TRANSLATION")
            print("  error in the camera mount, not a rotation. Subtracting")
            print("  it from mount_xyz should land the arm back on itself.")
            try:
                import yaml

                with open(args.camera_config) as f:
                    cfg = yaml.safe_load(f)
                cur = np.array(cfg["mount_xyz"], dtype=float)
                print("\n  %s" % args.camera_config)
                print("    now:       mount_xyz: [%.5f, %.5f, %.5f]" % tuple(cur))
                print("    corrected: mount_xyz: [%.5f, %.5f, %.5f]"
                      % tuple(cur - mean))
                print("  Re-run this script after editing: the ON-ARM boxes")
                print("  should be gone. This is a measured patch, not a")
                print("  calibration — re-run scripts/calibrate_orbbec.py when")
                print("  you want the rotation checked too.")
            except Exception as exc:
                print("  (could not read %s: %s)" % (args.camera_config, exc))
        else:
            print("  Varies with position -> a ROTATION error is in play;")
            print("  a translation fix will not fully correct it. Re-run")
            print("  scripts/calibrate_orbbec.py.")

    print("\nVERDICT")
    if on_arm:
        print("  %d box(es) OVERLAP the arm — this is why the demo sits in" % on_arm)
        print("  HOLD: cuRobo cannot plan from a start state that is inside an")
        print("  obstacle. The arm is seeing itself. Raise self_radius, or")
        print("  re-run the Orbbec calibration (a pose error moves the mask).")
    else:
        print("  no box overlaps the arm — the self-filter is doing its job")
    if flappy:
        print("  %d box(es) flicker in and out (duty 15-85%%). Perception is" % flappy)
        print("  confirming and forgetting noise: raise occupied_at, lower")
        print("  rate_hz, or raise min_voxels.")
    else:
        print("  the map is stable tick to tick")


if __name__ == "__main__":
    main()
