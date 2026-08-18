#!/usr/bin/env python3
"""Calibrate a FIXED camera's pose in base_link — no fiducials.

The target is the arm's own fingertip. Close the gripper (the fingertips
meet at ~tool_frame), THE HUMAN moves the arm to N poses spread through
the camera view IN 3D (vary height too, not just a plane); at each pose
press ENTER, then click the fingertip midpoint in the frozen color frame.
The aligned depth at the click deprojects to a camera-frame point, TF
gives the same point in base_link, and a rigid Kabsch fit over all pairs
solves base_T_camera. The result is only written if the RMS residual is
under --max-residual (default 0.02 m).

    export ROS_LOCALHOST_ONLY=1
    # terminal 1: arm bringup (kortex).  terminal 2:
    ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true
    # terminal 3 (needs a display for the click window):
    python3 scripts/calibrate_camera_extrinsics.py --poses 8

Depth sees the fingertip SURFACE, not its center — the click point is
pushed --surface-bias (default 1 cm) further along the viewing ray.
Writes rammp_curobo_ros/config/camera_orbbec_bench.yaml (depth aligned to
the color frame by depth_registration, so the color extrinsic IS the
depth extrinsic). Rebuild rammp_curobo_ros afterwards so the share/ copy
updates.

Window keys after clicking: y = accept, r = re-click, s = skip this pose.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join(
    REPO, "rammp_curobo_ros", "config", "camera_orbbec_bench.yaml"
)


def solve_rigid(base_pts, cam_pts):
    """Kabsch: paired points -> (R, t, rms) with base_pt ~= R @ cam_pt + t.

    The det() guard rejects the reflection solution that plain SVD can
    return for noisy or near-planar point sets.
    """
    base = np.asarray(base_pts, dtype=float)
    cam = np.asarray(cam_pts, dtype=float)
    cb, cc = base.mean(axis=0), cam.mean(axis=0)
    h = (cam - cc).T @ (base - cb)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = cb - rot @ cc
    rms = float(np.sqrt(np.mean(np.sum((base - (cam @ rot.T + t)) ** 2, axis=1))))
    return rot, t, rms


def spread_check(base_pts):
    """None if the poses span 3D well; else a human-readable complaint.

    Kabsch needs non-collinear points for any rotation at all, and
    non-coplanar spread for a well-conditioned one.
    """
    base = np.asarray(base_pts, dtype=float)
    svals = np.linalg.svd(base - base.mean(axis=0), compute_uv=False)
    if svals[1] < 0.02:
        return "poses are nearly COLLINEAR — the rotation is unsolvable"
    if svals[2] < 0.03:
        return (
            "poses are nearly COPLANAR (3rd axis spread %.0f mm) — vary the "
            "HEIGHT of the fingertip between poses" % (svals[2] * 1000)
        )
    return None


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
        description="fiducial-free extrinsic calibration (click the fingertip)"
    )
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--depth-info-topic", default="/camera/depth/camera_info")
    ap.add_argument("--tool-frame", default="tool_frame")
    ap.add_argument("--max-residual", type=float, default=0.02)
    ap.add_argument(
        "--surface-bias",
        type=float,
        default=0.010,
        help="metres added along the viewing ray (depth sees the fingertip "
        "surface, tool_frame is its center)",
    )
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
            self.depths = []
            self.info = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(
                Image, args.color_topic, self._img_cb, qos_profile_sensor_data
            )
            self.create_subscription(
                Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data
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

        def _depth_cb(self, msg):
            if msg.encoding == "16UC1":
                d = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(msg.height, msg.width)
                    .astype(np.float32)
                    / 1000.0
                )
            elif msg.encoding == "32FC1":
                d = (
                    np.frombuffer(msg.data, dtype=np.float32)
                    .reshape(msg.height, msg.width)
                    .copy()
                )
            else:
                return
            self.depths.append(d)
            del self.depths[:-5]

        def _info_cb(self, msg):
            self.info = msg

        def grab(self, n_depth=3, timeout_s=10.0):
            """Fresh color frame + median of the NEXT n_depth depth frames."""
            self.img = None
            self.depths.clear()
            t0 = self.get_clock().now()
            while (
                self.img is None or len(self.depths) < n_depth or self.info is None
            ):
                rclpy.spin_once(self, timeout_sec=0.2)
                if (self.get_clock().now() - t0).nanoseconds > timeout_s * 1e9:
                    sys.exit(
                        "no frames on %s / %s — is the camera driver running "
                        "with depth_registration:=true?"
                        % (args.color_topic, args.depth_topic)
                    )
            return self.img, np.median(np.stack(self.depths[-n_depth:]), axis=0)

        def tool_position(self):
            tr = self.tf_buffer.lookup_transform(
                "base_link", args.tool_frame, rclpy.time.Time()
            )
            t = tr.transform.translation
            return np.array([t.x, t.y, t.z])

    def click_point(img, depth, k):
        """Show the frozen frame; return the deprojected click or None."""
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        state = {"uv": None}

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["uv"] = (x, y)

        win = "calibrate: click the CLOSED fingertip midpoint (y/r/s)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            frame = img.copy()
            if state["uv"] is not None:
                u, v = state["uv"]
                cv2.drawMarker(frame, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                cv2.destroyWindow(win)
                return None
            if key == ord("r"):
                state["uv"] = None
            if key == ord("y") and state["uv"] is not None:
                u, v = state["uv"]
                patch = depth[
                    max(0, v - 2) : v + 3, max(0, u - 2) : u + 3
                ]
                patch = patch[(patch > 0.05) & np.isfinite(patch)]
                if not len(patch):
                    print("  no valid depth at the click — re-click ('r')")
                    state["uv"] = None
                    continue
                z = float(np.median(patch))
                p = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
                # push from the fingertip SURFACE to its center, along the ray
                p *= (np.linalg.norm(p) + args.surface_bias) / np.linalg.norm(p)
                cv2.destroyWindow(win)
                return p

    rclpy.init()
    node = Grab()
    print(
        "CLOSE the gripper first (fingertips together = tool_frame). "
        "%d poses, spread across the view AND in height (coplanar pose "
        "sets weaken the solve). Per pose: move the arm, ENTER here, then "
        "click the fingertip midpoint in the window (y accept / r re-click "
        "/ s skip). 'done' after >=5 solves early, Ctrl-C aborts."
        % args.poses
    )
    base_pts, cam_pts = [], []
    while len(base_pts) < args.poses:
        if input("[%d/%d] > " % (len(base_pts) + 1, args.poses)).strip() == "done":
            break
        img, depth = node.grab()
        if depth.shape != img.shape[:2]:
            sys.exit(
                "depth %s and color %s sizes differ — run the driver with "
                "depth_registration:=true" % (depth.shape, img.shape[:2])
            )
        k = np.array(node.info.k).reshape(3, 3)
        p_cam = click_point(img, depth, k)
        if p_cam is None:
            print("  skipped")
            continue
        try:
            p_base = node.tool_position()
        except Exception as exc:
            print(
                "  no TF base_link->%s (%s) — is the bringup up?"
                % (args.tool_frame, exc)
            )
            continue
        cam_pts.append(p_cam)
        base_pts.append(p_base)
        print(
            "  recorded (fingertip %.2f m from camera, %.2f m from base)"
            % (float(np.linalg.norm(p_cam)), float(np.linalg.norm(p_base)))
        )
    if len(base_pts) < 5:
        sys.exit("only %d poses — need at least 5" % len(base_pts))
    complaint = spread_check(base_pts)
    if complaint:
        sys.exit("BAD POSE SPREAD: %s. Re-run with better spread." % complaint)
    rot, t, rms = solve_rigid(base_pts, cam_pts)
    print(
        "base_T_camera t = [%.4f, %.4f, %.4f], RMS residual %.4f m over %d poses"
        % (t[0], t[1], t[2], rms, len(base_pts))
    )
    print("sanity: tape-measure the camera against those numbers (base frame).")
    if rms > args.max_residual:
        sys.exit(
            "RESIDUAL %.4f m > %.3f m — NOT writing. Click more carefully, "
            "keep the gripper fully closed, spread the poses wider."
            % (rms, args.max_residual)
        )
    cfg = {
        "depth_topic": args.depth_topic,
        "info_topic": args.depth_info_topic,
        "parent_frame": "base_link",
        "mount_xyz": [float(v) for v in t],
        "mount_quat_xyzw": mat_to_quat_xyzw(rot),
        "min_range": 0.25,
        "max_range": 1.5,
        "calibrated": "fingertip-click Kabsch, RMS %.4f m, %d poses"
        % (rms, len(base_pts)),
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote %s — REBUILD rammp_curobo_ros so the share/ copy updates" % args.out)
    # quat_to_mat imported for parity with the node's YAML consumption —
    # keep the round-trip honest if conventions ever drift
    assert np.allclose(quat_to_mat(*cfg["mount_quat_xyzw"]), rot, atol=1e-6)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
