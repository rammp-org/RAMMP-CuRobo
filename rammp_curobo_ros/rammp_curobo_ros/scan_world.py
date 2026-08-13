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

import numpy as np
import rclpy

from rammp_curobo_ros.scan_common import (
    DepthCameraGrabber,
    add_cluster_args,
    apply_world,
    cluster_boxes,
    load_camera_config,
    report_boxes,
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
    # single viewpoint -> fewer hits per voxel than the fused sweep, so the
    # occupancy threshold is lower and the size floor slightly higher
    add_cluster_args(ap, frames=5, min_points_per_voxel=2, min_voxels=6, max_boxes=25)
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

    boxes, n_found = cluster_boxes(
        pts, args.voxel, args.min_points_per_voxel, args.min_voxels, args.max_boxes
    )
    write_world_yaml(args.out, boxes, args.inflate, "by scan_world")
    report_boxes(boxes, n_found, args.out)

    if args.apply:
        apply_world(node, args.out)


if __name__ == "__main__":
    main()
