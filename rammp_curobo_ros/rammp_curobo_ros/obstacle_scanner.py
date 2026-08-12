#!/usr/bin/env python3
"""Single-shot obstacle scan from a wrist depth camera (no motion).

Captures from wherever the camera currently points, clusters obstacles
into boxes over the conservative table plane, writes the world YAML.
For full-workspace coverage use sweep_scan (the +/-90 deg station sweep);
this tool is the quick "what's in front of me right now" variant.

    ros2 run rammp_curobo_ros scan_world --camera camera_kinova_wrist.yaml --debug
    ros2 run rammp_curobo_ros scan_world --camera camera_d405_wrist.yaml --apply

Prerequisites: the arm bringup (for TF) and the matching camera driver
(see the camera YAMLs in rammp_curobo_ros/config/).
"""

import argparse
import sys
import time

import numpy as np
import rclpy

from rammp_curobo_ros.scan_common import (
    DEFAULT_OUT,
    DepthCameraGrabber,
    cluster_boxes,
    load_camera_config,
    write_world_yaml,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--camera",
        default="camera_kinova_wrist.yaml",
        help="camera YAML (packaged name or path)",
    )
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="call /rammp_curobo/set_world with the result",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--min-voxels", type=int, default=6)
    ap.add_argument("--max-boxes", type=int, default=25)
    ap.add_argument("--inflate", type=float, default=0.01)
    args = ap.parse_args()

    rclpy.init()
    node = DepthCameraGrabber(load_camera_config(args.camera))
    pts = node.capture_points(args.frames)
    if args.debug:
        R, t = node.camera_pose()
        print(
            "camera at %s, view axis %s, %d points"
            % (np.round(t, 3).tolist(), np.round(R[:, 2], 2).tolist(), len(pts))
        )
    if not len(pts):
        sys.exit("no points in the workspace — is the camera looking at it?")

    boxes, n_found = cluster_boxes(pts, args.voxel, 2, args.min_voxels, args.max_boxes)
    if n_found > len(boxes):
        print("NOTE: %d clusters, keeping the %d largest" % (n_found, len(boxes)))
    write_world_yaml(args.out, boxes, args.inflate, "by scan_world")
    print("%-8s %-24s %s" % ("box", "center [m]", "dims [m]"))
    for i, b in enumerate(boxes):
        print(
            "det_%-4d %-24s %s"
            % (i, np.round(b["center"], 3).tolist(), np.round(b["dims"], 3).tolist())
        )
    print("world written: %s (%d boxes + table plane)" % (args.out, len(boxes)))

    if args.apply:
        from rammp_curobo_interfaces.srv import SetWorld

        client = node.create_client(SetWorld, "/rammp_curobo/set_world")
        if not client.wait_for_service(timeout_sec=3.0):
            sys.exit("planner node not running — world written, not applied")
        fut = client.call_async(SetWorld.Request(world=args.out))
        t0 = time.monotonic()
        while not fut.done():
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.monotonic() - t0 > 20:
                sys.exit("set_world did not answer")
        resp = fut.result()
        print("set_world: %s (%s)" % ("OK" if resp.success else "FAILED", resp.message))
        if not resp.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
