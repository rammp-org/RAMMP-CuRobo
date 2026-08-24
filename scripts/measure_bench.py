#!/usr/bin/env python3
"""Measure the REAL bench from the registered Orbbec — no tape measure.

    python3 scripts/measure_bench.py            # report only
    python3 scripts/measure_bench.py --apply    # rewrite world_real_bench.yaml

world_real_bench.yaml has carried PLACEHOLDER table geometry copied from
the sim kitchen since the first hardware day, with a banner saying
"MEASURE BEFORE USE". The camera now self-registers off the arm to a few
millimetres, so the depth cloud IS the measurement: the dominant
horizontal plane below the arm is the tabletop.

Needs only the Orbbec driver (fixed camera: the mount comes from the
config, no TF, no arm, nothing moves). Hold the bench clear-ish; objects
on it don't bias the fit unless they cover most of it.
"""

import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from rammp_curobo.perception import depth_to_points, quat_to_mat, transform_points
from rammp_curobo_ros.cameras import load_camera_config

WORLD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core", "rammp_curobo", "configs", "world_real_bench.yaml",
)


def collect(cfg, frames, seconds):
    rclpy.init()
    node = rclpy.create_node("measure_bench")
    state = {"depth": None, "info": None}

    def depth_cb(msg):
        if msg.encoding == "16UC1":
            d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
            state["depth"] = d.astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            state["depth"] = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).copy()

    def info_cb(msg):
        k = np.array(msg.k).reshape(3, 3)
        state["info"] = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    node.create_subscription(Image, cfg["depth_topic"], depth_cb,
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, cfg["info_topic"], info_cb,
                             qos_profile_sensor_data)
    q = cfg["mount_quat_xyzw"]
    rot = quat_to_mat(q[0], q[1], q[2], q[3])
    trans = np.asarray(cfg["mount_xyz"], dtype=float)
    clouds = []
    end = time.monotonic() + seconds
    while len(clouds) < frames and time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if state["depth"] is None or state["info"] is None:
            continue
        pts = depth_to_points(
            state["depth"], state["info"]["fx"], state["info"]["fy"],
            state["info"]["cx"], state["info"]["cy"], stride=2,
            min_range=float(cfg.get("min_range", 0.2)),
            max_range=float(cfg.get("max_range", 2.0)),
        )
        clouds.append(transform_points(pts, rot, trans))
        state["depth"] = None
    node.destroy_node()
    rclpy.shutdown()
    if not clouds:
        sys.exit("no depth frames on %s — is the Orbbec driver running?"
                 % cfg["depth_topic"])
    return np.vstack(clouds)


def find_table(pts):
    """(top_z, x_span, y_span, n_plane) of the dominant horizontal plane
    below the arm's base plane."""
    sel = pts[(np.abs(pts[:, 0]) < 1.2) & (np.abs(pts[:, 1]) < 1.2)
              & (pts[:, 2] > -0.45) & (pts[:, 2] < 0.10)]
    if len(sel) < 2000:
        sys.exit("only %d points in the bench window — is the camera aimed "
                 "at the bench?" % len(sel))
    # the tabletop is the tallest 5 mm z-bin by far; refine with a median
    bins = np.arange(-0.45, 0.10, 0.005)
    hist, edges = np.histogram(sel[:, 2], bins=bins)
    peak = int(np.argmax(hist))
    level = 0.5 * (edges[peak] + edges[peak + 1])
    plane = sel[np.abs(sel[:, 2] - level) < 0.012]
    top = float(np.median(plane[:, 2]))
    frac = len(plane) / float(len(sel))
    x_lo, x_hi = np.percentile(plane[:, 0], [2, 98])
    y_lo, y_hi = np.percentile(plane[:, 1], [2, 98])
    if frac < 0.2 or (x_hi - x_lo) < 0.3 or (y_hi - y_lo) < 0.3:
        sys.exit("dominant plane at z=%.3f covers only %.0f%% of the window "
                 "(%.2f x %.2f m) — that is not a tabletop; clear the bench "
                 "and re-run" % (top, frac * 100, x_hi - x_lo, y_hi - y_lo))
    return top, (float(x_lo), float(x_hi)), (float(y_lo), float(y_hi)), len(plane)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-config", default="camera_orbbec_bench.yaml")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite world_real_bench.yaml with the measurement")
    args = ap.parse_args()

    cfg = load_camera_config(args.camera_config)
    if cfg.get("parent_frame") != "base_link":
        sys.exit("%s is not a fixed camera" % args.camera_config)
    pts = collect(cfg, args.frames, args.seconds)
    top, (x_lo, x_hi), (y_lo, y_hi), n = find_table(pts)

    print("tabletop: z = %+.3f m in base_link  (%d plane points)" % (top, n))
    print("extent seen by the camera: x %.2f..%.2f  y %.2f..%.2f m"
          % (x_lo, x_hi, y_lo, y_hi))
    if top > -0.05:
        print("WARNING: top above -0.05 m — the arm's own base collision")
        print("spheres reach z=-0.015, so with 0.02 padding the HOME pose")
        print("has under %.0f mm of margin. Verify HOME validity after apply."
              % ((-0.015 - (top + 0.02)) * -1000))
    print("recommended cameras min_z: %.3f (table top + 2.5 cm)" % (top + 0.025))

    # err HIGH and WIDE: a table modeled too tall/large costs reachable
    # volume the sweep never uses; too small is the unsafe direction
    x0, x1 = min(x_lo, -0.20) - 0.15, max(x_hi, 0.80) + 0.15
    y0, y1 = min(y_lo, -0.70) - 0.15, max(y_hi, 0.70) + 0.15
    if not args.apply:
        print("\nre-run with --apply to write world_real_bench.yaml "
              "(table top %.3f, %.2f x %.2f m)" % (top, x1 - x0, y1 - y0))
        return

    # TWO-TIER floor. The arm's own base collision spheres reach z=-0.015,
    # so a padded box at the TRUE table height would swallow them and no
    # start state near home could ever be valid (probed: effective top
    # >= -0.01 breaks HOME). So:
    #   - `table` at the true height, in no_pad_names like the pedestal:
    #     honest geometry under the base, no padding.
    #   - four `floor_*` guard boxes at the SAME height, excluding a
    #     +-`KEEP` square around the base column: these take the normal
    #     world_padding, which is exactly the hard clearance over the
    #     REAL surface everywhere the arm can actually swing.
    KEEP = 0.15
    dz = 0.06
    zc = top - dz / 2.0

    def box(name, bx0, bx1, by0, by1):
        return (name, [(bx0 + bx1) / 2.0, (by0 + by1) / 2.0, zc],
                [bx1 - bx0, by1 - by0, dz])

    guards = [
        box("floor_front", KEEP, x1, y0, y1),
        box("floor_back", x0, -KEEP, y0, y1),
        box("floor_left", -KEEP, KEEP, KEEP, y1),
        box("floor_right", -KEEP, KEEP, y0, -KEEP),
    ]
    with open(WORLD, "w") as f:
        f.write(
            "# Real-bench collision world — table MEASURED %s by\n"
            "# scripts/measure_bench.py from the registered Orbbec (top %+0.3f m,\n"
            "# plane of %d points). Re-run after moving the arm or the bench.\n"
            "#\n"
            "# Layout: `table` carries the TRUE surface height and is in the\n"
            "# planner's no_pad_names (padding it would swallow the arm's own\n"
            "# base spheres at z=-0.015 and break every start near home); the\n"
            "# floor_* guards repeat the surface outside a %.2f m square around\n"
            "# the base and take normal world_padding — the hard keep-out over\n"
            "# the real tabletop wherever the arm can swing.\n"
            "#\n"
            "# cuRobo v0.7.8 refuses an EMPTY world (silently keeps the previous\n"
            "# one on update), so at minimum keep the pedestal + table entries.\n"
            "\n"
            "base_frame: base_link\n"
            "\n"
            "obstacles:\n"
            "  - name: pedestal                   # the arm's mounting column (no_pad).\n"
            "    position: [0.0, 0.0, -0.05]\n"
            "    dims: [0.14, 0.14, 0.04]\n"
            "  - name: table                      # TRUE surface height (no_pad).\n"
            "    position: [%.3f, %.3f, %.3f]\n"
            "    dims: [%.2f, %.2f, %.2f]\n"
            % (time.strftime("%Y-%m-%d"), top, n, 2 * KEEP,
               (x0 + x1) / 2.0, (y0 + y1) / 2.0, zc, x1 - x0, y1 - y0, dz)
        )
        for name, pos, dims in guards:
            f.write("  - name: %s\n"
                    "    position: [%.3f, %.3f, %.3f]\n"
                    "    dims: [%.2f, %.2f, %.2f]\n"
                    % (name, pos[0], pos[1], pos[2], dims[0], dims[1], dims[2]))
        f.write("\nobjects: []\n\ntargets: []\n")
    print("\nwrote %s (true-height table + 4 padded floor guards)" % WORLD)
    print("REMINDER: no_pad_names must contain [pedestal, table] (gen3.yaml).")

if __name__ == "__main__":
    main()
