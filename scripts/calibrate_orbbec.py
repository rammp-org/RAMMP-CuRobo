#!/usr/bin/env python3
"""Locate a FIXED Orbbec relative to the arm, using a tag on the gripper.

    # tape a printed tag to the gripper so it faces roughly outward
    python3 scripts/make_tag.py --id 0 --size 0.06 --out tag0.png
    python3 scripts/calibrate_orbbec.py --marker-size 0.06 --execute

The arm sweeps a set of deliberately ROTATION-DIVERSE poses; wherever
the camera sees the tag, we pair the tag's pose in the camera with the
arm's own FK and solve the eye-to-hand problem. Neither the camera's
pose NOR the tag's placement on the gripper needs measuring — tape it
on crooked and it still solves.

Writes rammp_curobo_ros/config/camera_orbbec_bench.yaml (the DEPTH
optical frame, which is what the cameras node deprojects), then:

    ros2 run rammp_curobo_ros cameras --ros-args \
        -p "cameras:=['camera_orbbec_bench.yaml']"

Needs the planner with execute:=true, the arm bringup, and the Orbbec
driver. Human on the e-stop: this moves the arm.
"""

import argparse
import os
import sys
import time

import numpy as np
import rclpy

from rammp_curobo.geometry import rot_about_world_y, yaw_about_world_z
from rammp_curobo_ros.handeye import (
    invert,
    mat_to_quat_xyzw,
    pose_spread,
    solve_eye_to_hand,
)
from rammp_curobo_ros.seek_core import roll_about_tool_z
from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW, TourDemo

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rammp_curobo_ros", "config", "camera_orbbec_bench.yaml",
)


def sweep_poses(radius, z_lo, z_hi):
    """Rotation-diverse presentation poses. Hand-eye NEEDS varied
    orientation — pure translation is degenerate."""
    out = []
    for bearing in (-0.45, 0.0, 0.45):
        for pitch, z in ((0.35, z_hi), (0.75, z_lo)):
            for roll in (0.0, 1.0, -1.0):
                pos = [radius * np.cos(bearing), radius * np.sin(bearing), z]
                quat = list(roll_about_tool_z(
                    yaw_about_world_z(rot_about_world_y(HOME_QUAT_XYZW, pitch),
                                      bearing), roll))
                out.append((pos, quat))
    return out


class Capture:
    def __init__(self, node, marker_size, dict_name, tag_id):
        import cv2
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener

        self.cv2 = cv2
        self.node = node
        self.tag_id = tag_id
        self.color = self.k = self.dist = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        node.create_subscription(Image, "/camera/color/image_raw",
                                 self._color_cb, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, "/camera/color/camera_info",
                                 self._info_cb, qos_profile_sensor_data)
        a = cv2.aruco
        self.detector = a.ArucoDetector(
            a.getPredefinedDictionary(getattr(a, dict_name)),
            a.DetectorParameters())
        s = marker_size / 2.0
        self.objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                             dtype=np.float64)

    def _color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            self.color = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()

    def _info_cb(self, msg):
        self.k = np.array(msg.k).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=float).ravel()

    def frame_pose(self, target, source, timeout=2.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                tr = self.tf_buffer.lookup_transform(target, source,
                                                     rclpy.time.Time())
            except Exception:
                continue
            from rammp_curobo.perception import quat_to_mat

            q, t = tr.transform.rotation, tr.transform.translation
            m = np.eye(4)
            m[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
            m[:3, 3] = [t.x, t.y, t.z]
            return m
        return None

    def detect(self, tries=12):
        """color_T_tag or None."""
        for _ in range(tries):
            self.color = None
            t0 = time.monotonic()
            while self.color is None or self.k is None:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if time.monotonic() - t0 > 2.0:
                    return None
            corners, ids, _ = self.detector.detectMarkers(self.color)
            if ids is None:
                continue
            for c, i in zip(corners, ids.ravel()):
                if self.tag_id >= 0 and int(i) != self.tag_id:
                    continue
                ok, rvec, tvec = self.cv2.solvePnP(
                    self.objp, c.reshape(4, 2).astype(np.float64), self.k,
                    self.dist, flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                m = np.eye(4)
                m[:3, :3] = self.cv2.Rodrigues(rvec)[0]
                m[:3, 3] = tvec.ravel()
                return m
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marker-size", type=float, required=True, help="metres")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--tag-id", type=int, default=-1)
    ap.add_argument("--radius", type=float, default=0.58)
    ap.add_argument("--z-lo", type=float, default=0.28)
    ap.add_argument("--z-hi", type=float, default=0.45)
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--execute", action="store_true", help="allow motion")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("calibrate_orbbec")
    demo = TourDemo(node)
    cap = Capture(node, args.marker_size, args.dictionary, args.tag_id)

    poses = sweep_poses(args.radius, args.z_lo, args.z_hi)
    print("%d sweep poses; tag %.0f mm, dict %s"
          % (len(poses), args.marker_size * 1000, args.dictionary))
    if not args.execute:
        sys.exit("dry run — add --execute (the arm will move; e-stop in hand)")

    base_T_ee, cam_T_tag = [], []
    for i, (pos, quat) in enumerate(poses):
        plan = demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            print("  pose %2d/%d unplannable — skipped" % (i + 1, len(poses)))
            continue
        if not demo.run(plan.trajectory, args.speed):
            print("  pose %2d/%d motion refused — skipped" % (i + 1, len(poses)))
            continue
        time.sleep(0.6)                      # settle before looking
        tag = cap.detect()
        ee = cap.frame_pose("base_link", "end_effector_link")
        if tag is None or ee is None:
            print("  pose %2d/%d: tag not seen" % (i + 1, len(poses)))
            continue
        base_T_ee.append(ee)
        cam_T_tag.append(tag)
        print("  pose %2d/%d: tag at %s m in camera"
              % (i + 1, len(poses), np.round(tag[:3, 3], 3)))

    if len(base_T_ee) < 4:
        sys.exit("only %d usable views — aim the camera at the arm's "
                 "workspace, or re-tape the tag so it faces the camera"
                 % len(base_T_ee))
    spread = pose_spread(base_T_ee)
    print("\n%d views, rotation spread %.2f rad" % (len(base_T_ee), spread))
    if spread < 0.5:
        sys.exit("poses are too similar in ORIENTATION — the hand-eye "
                 "solve is degenerate here; widen the sweep")

    base_T_color, residual = solve_eye_to_hand(base_T_ee, cam_T_tag)
    print("residual (tag scatter on the gripper): %.4f m" % residual)
    if residual > 0.03:
        print("WARNING: >3 cm — treat the result as rough; inflate obstacles")

    depth_T_color = cap.frame_pose("camera_depth_optical_frame",
                                   "camera_color_optical_frame")
    if depth_T_color is None:
        sys.exit("no camera_depth_optical_frame <- camera_color_optical_frame TF")
    base_T_depth = base_T_color @ invert(depth_T_color)
    xyz = base_T_depth[:3, 3]
    quat = mat_to_quat_xyzw(base_T_depth[:3, :3])
    print("camera (depth optical frame) at base_link %s" % np.round(xyz, 4))

    with open(CONFIG, "w") as f:
        f.write(
            "# Fixed Orbbec Gemini 336L — SOLVED by scripts/calibrate_orbbec.py\n"
            "# (tag-on-gripper eye-to-hand; %d views, spread %.2f rad,\n"
            "#  residual %.4f m). Re-run after ANY camera move.\n"
            "depth_topic: /camera/depth/image_raw\n"
            "info_topic: /camera/depth/camera_info\n"
            "parent_frame: base_link\n"
            "mount_xyz: [%.5f, %.5f, %.5f]\n"
            "mount_quat_xyzw: [%.6f, %.6f, %.6f, %.6f]\n"
            "min_range: 0.2\n"
            "max_range: 2.0\n"
            % (len(base_T_ee), spread, residual, xyz[0], xyz[1], xyz[2],
               quat[0], quat[1], quat[2], quat[3])
        )
    print("wrote %s" % CONFIG)


if __name__ == "__main__":
    main()
