#!/usr/bin/env python3
"""Calibrate a FIXED camera's pose in base_link (eye-to-hand) via ArUco.

Attended, one-time, re-runnable. An ArUco tag rides rigidly on the
gripper (anywhere rigid — its tool offset is solved, not measured). THE
HUMAN jogs the arm to N poses spanning the camera view; at each pose,
press ENTER to record (tag pose from the camera + tool pose from TF).
cv2.calibrateHandEye (eye-to-hand form: base->tool poses inverted) solves
the camera extrinsic; the result is only written if the RMS residual over
the recorded pairs is under --max-residual (default 0.01 m).

    export ROS_LOCALHOST_ONLY=1
    # terminal 1: arm bringup (kortex).  terminal 2:
    ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true
    # terminal 3:
    python3 scripts/calibrate_camera_extrinsics.py \
        --marker-id 0 --marker-size 0.05 --poses 10

Writes rammp_curobo_ros/config/camera_orbbec_bench.yaml (depth aligned to
the color frame by depth_registration, so the color extrinsic IS the
depth extrinsic). Rebuild rammp_curobo_ros afterwards so the share/ copy
updates.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join(
    REPO, "rammp_curobo_ros", "config", "camera_orbbec_bench.yaml"
)


def solve_eye_to_hand(base_T_tool, cam_T_tag):
    """Fixed-camera hand-eye: lists of (R, t) pairs -> (R, t, rms) of
    base_T_camera. Standard eye-to-hand trick: feed calibrateHandEye the
    INVERTED gripper poses so the fixed camera looks like a wrist camera.
    """
    import cv2

    R_g, t_g, R_c, t_c = [], [], [], []
    for (R_bt, t_bt), (R_ct, t_ct) in zip(base_T_tool, cam_T_tag):
        R_g.append(R_bt.T)  # tool_T_base
        t_g.append(-R_bt.T @ t_bt)
        R_c.append(R_ct)
        t_c.append(t_ct)
    R_x, t_x = cv2.calibrateHandEye(
        R_g, t_g, R_c, t_c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    # In the inverted-gripper setup calibrateHandEye's "gripper_T_camera"
    # output IS base_T_camera.
    R_bc, t_bc = R_x, t_x.reshape(3)

    # residual: with X known, the tag-in-tool pose implied by each pair
    # must be constant — its scatter is the honest error metric.
    tags = []
    for (R_bt, t_bt), (R_ct, t_ct) in zip(base_T_tool, cam_T_tag):
        t_tt = R_bt.T @ (R_bc @ t_ct + t_bc - t_bt)
        tags.append(t_tt)
    tags = np.asarray(tags)
    rms = float(np.sqrt(np.mean(np.sum((tags - tags.mean(axis=0)) ** 2, axis=1))))
    return R_bc, t_bc, rms


def mat_to_quat_xyzw(R):
    """3x3 rotation matrix -> xyzw quaternion (Shepperd's method)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        return [float(x), float(y), float(z), float(w)]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4.0
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    w = (R[k, j] - R[j, k]) / s
    return [float(q[0]), float(q[1]), float(q[2]), float(w)]


def main():
    ap = argparse.ArgumentParser(
        description="eye-to-hand extrinsic calibration (ArUco on the gripper)"
    )
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument(
        "--marker-size", type=float, required=True, help="tag side (m), MEASURE IT"
    )
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--depth-info-topic", default="/camera/depth/camera_info")
    ap.add_argument("--tool-frame", default="tool_frame")
    ap.add_argument("--max-residual", type=float, default=0.01)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    import cv2
    import rclpy
    import yaml
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    from rammp_curobo.perception import quat_to_mat

    class Grab(Node):
        def __init__(self):
            super().__init__("calibrate_camera_extrinsics")
            self.img = None
            self.info = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(
                Image, args.color_topic, self._img_cb, qos_profile_sensor_data
            )
            self.create_subscription(
                CameraInfo, args.info_topic, self._info_cb, qos_profile_sensor_data
            )

        def _img_cb(self, msg):
            if msg.encoding in ("rgb8", "bgr8"):
                a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                self.img = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()

        def _info_cb(self, msg):
            self.info = msg

        def tool_pose(self):
            tr = self.tf_buffer.lookup_transform(
                "base_link", args.tool_frame, rclpy.time.Time()
            )
            q, t = tr.transform.rotation, tr.transform.translation
            return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    rclpy.init()
    node = Grab()
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(getattr(aruco, args.dict)),
        aruco.DetectorParameters(),
    )
    base_T_tool, cam_T_tag = [], []
    print(
        "Jog the arm so the tag faces the camera. %d poses; spread them "
        "across the view and VARY THE WRIST ROTATION AXIS pose to pose "
        "(tilt AND twist, not just one axis — single-axis pose sets make "
        "the hand-eye solve degenerate). ENTER to record, 'done' to solve "
        "early, Ctrl-C aborts." % args.poses
    )
    while len(base_T_tool) < args.poses:
        if input("[%d/%d] > " % (len(base_T_tool) + 1, args.poses)).strip() == "done":
            break
        node.img = None
        t0 = node.get_clock().now()
        while node.img is None or node.info is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            if (node.get_clock().now() - t0).nanoseconds > 10e9:
                sys.exit("no color frames on %s" % args.color_topic)
        corners, ids, _ = detector.detectMarkers(node.img)
        if ids is None or args.marker_id not in ids.flatten():
            print("  tag %d NOT visible — repose and retry" % args.marker_id)
            continue
        i = list(ids.flatten()).index(args.marker_id)
        k = np.array(node.info.k).reshape(3, 3)
        d = np.array(node.info.d) if len(node.info.d) else np.zeros(5)
        s = args.marker_size / 2.0
        obj = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            corners[i].reshape(4, 2).astype(np.float32),
            k,
            d,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            print("  PnP failed — retry")
            continue
        try:
            R_bt, t_bt = node.tool_pose()
        except Exception as exc:
            print(
                "  no TF base_link->%s (%s) — is the bringup up?"
                % (args.tool_frame, exc)
            )
            continue
        cam_T_tag.append((cv2.Rodrigues(rvec)[0], tvec.reshape(3)))
        base_T_tool.append((R_bt, t_bt))
        print("  recorded (tag at %.2f m)" % float(np.linalg.norm(tvec)))
    if len(base_T_tool) < 5:
        sys.exit("only %d poses — need at least 5" % len(base_T_tool))
    R, t, rms = solve_eye_to_hand(base_T_tool, cam_T_tag)
    print(
        "base_T_camera t = [%.4f, %.4f, %.4f], RMS residual %.4f m"
        % (t[0], t[1], t[2], rms)
    )
    if rms > args.max_residual:
        sys.exit(
            "RESIDUAL %.4f m > %.3f m — NOT writing. More poses, better "
            "spread, check the marker size, keep the tag rigid."
            % (rms, args.max_residual)
        )
    cfg = {
        "depth_topic": args.depth_topic,
        "info_topic": args.depth_info_topic,
        "parent_frame": "base_link",
        "mount_xyz": [float(v) for v in t],
        "mount_quat_xyzw": mat_to_quat_xyzw(R),
        "min_range": 0.25,
        "max_range": 1.5,
        "calibrated": "eye-to-hand ArUco, RMS %.4f m, %d poses"
        % (rms, len(base_T_tool)),
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote %s — REBUILD rammp_curobo_ros so the share/ copy updates" % args.out)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
