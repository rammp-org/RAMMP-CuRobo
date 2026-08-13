#!/usr/bin/env python3
"""Live camera view with palm detection overlay — see what the demo sees.

    ros2 run rammp_curobo_ros palm_view      # feed appears RIGHT HERE:
                                             #  - kitty terminal: inline video
                                             #  - any other terminal: ANSI video
                                             #  - X desktop: native window
                                             #  - plus http://<jetson>:8405 always
    ros2 run rammp_curobo_ros palm_view --headless   # JPEG previews only

Green box + landmarks = MediaPipe hand, dot = palm center (3D position
and workspace-gate verdict when the arm bringup provides TF). Blue
crosshair = the depth-blob detector (palm_demo's fallback targeting).
Ctrl+C quits ('q' in the native window, if any).
"""

import argparse
import os
import sys
import time

import numpy as np
import rclpy

from rammp_curobo_ros.palm_common import (
    STREAM_PORT,
    ColorDepthGrabber,
    _MjpegServer,
    close_display,
    depth_at,
    detect_palm,
    landmark_palms,
    make_hands,
    palm_target_ok,
    pick_display,
    show_frame,
)
from rammp_curobo_ros.scan_common import load_camera_config

PREVIEW = os.path.expanduser("~/.ros/rammp_curobo/palm_view.jpg")


def main():
    import cv2

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", default="camera_d405_wrist.yaml")
    ap.add_argument("--color-topic", default="/d405/d405/color/image_rect_raw")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = ColorDepthGrabber(load_camera_config(args.camera), args.color_topic)
    hands = make_hands()

    # display tiers: X window if a desktop exists, else kitty/ANSI inline
    window = False
    if not args.headless and os.environ.get("DISPLAY"):
        import subprocess

        if subprocess.run(["xset", "q"], capture_output=True).returncode == 0:
            try:
                cv2.namedWindow("palm_view", cv2.WINDOW_NORMAL)
                cv2.waitKey(1)
                window = True
            except Exception:
                window = False
    kitty, ansi = pick_display(headless=args.headless or window)

    stream = None
    try:
        stream = _MjpegServer(STREAM_PORT)
        if window or (kitty is None and ansi is None):
            print("live view: http://192.168.1.11:%d" % STREAM_PORT)
    except OSError as exc:
        print("stream port %d unavailable (%s)" % (STREAM_PORT, exc))

    t0 = time.monotonic()
    while node.info is None or not node.frames or node.color is None:
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.monotonic() - t0 > 8.0:
            sys.exit(
                "No camera frames after 8 s — start the RealSense driver:\n"
                "  ros2 launch realsense2_camera rs_launch.py "
                "camera_namespace:=d405 camera_name:=d405"
            )

    have_tf = True
    throttle = [0.0]
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    last_save = 0.0

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.color is None or not node.frames or node.info is None:
            continue
        img, enc = node.color
        node.color = None
        depth = node.frames[-1]
        del node.frames[:-1]
        rgb = img if enc == "rgb8" else img[:, :, ::-1]
        frame = np.ascontiguousarray(rgb[:, :, ::-1])

        R = t = None
        if have_tf:
            try:
                R, t = node.camera_pose(timeout_s=1.5)
            except SystemExit:
                have_tf = False

        k = np.array(node.info.k).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        h_img, w_img = rgb.shape[:2]
        sx, sy = depth.shape[1] / w_img, depth.shape[0] / h_img

        # --- MediaPipe hands (green)
        for (pu, pv), (x0, y0, x1, y1), lms in landmark_palms(hands, rgb):
            z = depth_at(depth, pu, pv, sx, sy)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.circle(frame, (int(pu), int(pv)), 6, (0, 200, 0), -1)
            for u, v in lms:
                cv2.circle(frame, (int(u), int(v)), 2, (0, 150, 0), -1)
            if z is not None:
                cam = np.array([(pu - cx) / fx * z, (pv - cy) / fy * z, z])
                label = "palm %.2f m" % z
                if R is not None:
                    base = R @ cam + t
                    ok, why = palm_target_ok(base)
                    label += "  base[%.2f %.2f %.2f] %s" % (
                        base[0],
                        base[1],
                        base[2],
                        "OK" if ok else "REFUSED: " + why,
                    )
                cv2.putText(
                    frame,
                    label,
                    (x0, max(y0 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 0),
                    1,
                )

        # --- depth-blob target (blue): palm_demo's fallback detector
        if R is not None:
            vv, uu = np.mgrid[0 : depth.shape[0] : 2, 0 : depth.shape[1] : 2]
            z = depth[::2, ::2]
            valid = (z > 0.07) & (z < 0.95) & np.isfinite(z)
            zz, u2, v2 = z[valid], uu[valid], vv[valid]
            cam_pts = np.stack(
                [(u2 / sx - cx) / fx * zz, (v2 / sy - cy) / fy * zz, zz], axis=1
            )
            base_pts = cam_pts @ R.T + t
            m = (
                (base_pts[:, 0] > 0.15)
                & (np.hypot(base_pts[:, 0], base_pts[:, 1]) < 0.95)
                & (base_pts[:, 2] > 0.05)
            )
            blob, _n, _why = detect_palm(base_pts[m]) if m.sum() else (None, 0, "")
            if blob is not None:
                camb = R.T @ (np.asarray(blob) - t)
                if camb[2] > 0.05:
                    ub = int(camb[0] / camb[2] * fx + cx)
                    vb = int(camb[1] / camb[2] * fy + cy)
                    if 0 <= ub < w_img and 0 <= vb < h_img:
                        cv2.drawMarker(
                            frame,
                            (ub, vb),
                            (255, 120, 0),
                            cv2.MARKER_CROSS,
                            26,
                            3,
                        )

        fps_n += 1
        if time.monotonic() - fps_t >= 1.0:
            fps = fps_n / (time.monotonic() - fps_t)
            fps_t, fps_n = time.monotonic(), 0
        cv2.putText(
            frame,
            "%.0f fps | green=hand  blue+=depth target%s"
            % (fps, "" if R is not None else " | NO TF"),
            (8, h_img - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )

        show_frame(frame, kitty, ansi, stream, throttle)
        if window:
            cv2.imshow("palm_view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif kitty is None and ansi is None and time.monotonic() - last_save > 0.5:
            cv2.imwrite(PREVIEW, frame)
            last_save = time.monotonic()

    if window:
        cv2.destroyAllWindows()
    close_display(kitty, ansi)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npalm_view stopped")
