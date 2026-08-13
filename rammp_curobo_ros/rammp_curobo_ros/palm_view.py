#!/usr/bin/env python3
"""Live camera view with palm detection overlay — see what the demo sees.

Opens a window showing the D405 color stream with:
  * a green box + landmarks around each detected hand (MediaPipe Hands),
    the palm center dotted, its 3D position (camera depth + TF -> base
    frame) and distance printed on the box,
  * a blue crosshair where the DEPTH-BLOB detector (what palm_demo
    actually targets) currently lands — if green dot and blue crosshair
    agree, the demo will touch where you think it will,
  * the workspace-gate verdict for the would-be target.

    ros2 run rammp_curobo_ros palm_view      # then open http://<jetson>:8405
                                             # (native window if an X desktop
                                             #  session is available)
    ros2 run rammp_curobo_ros palm_view --headless   # JPEG previews only

The live view is always served at http://192.168.1.11:8405 — open it in
any browser (laptop next to VSCode works). Needs the RealSense driver
running; the arm bringup is optional (without TF the 3D readout stays in
the camera frame). Ctrl+C quits ('q' in the native window, if any).
"""

import argparse
import os
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from rammp_curobo_ros.palm_demo import detect_palm, palm_target_ok
from rammp_curobo_ros.scan_common import DepthCameraGrabber, load_camera_config

MODEL_PATH = os.path.expanduser("~/.ros/rammp_curobo/hand_landmarker.task")
STREAM_PORT = 8405


class _MjpegServer:
    """Minimal multipart-JPEG streamer: open http://<host>:<port> live."""

    def __init__(self, port):
        import http.server
        import socketserver
        import threading

        self._lock = threading.Lock()
        self._jpg = None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            jpg = outer._jpg
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(jpg)))
                            self.end_headers()
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.06)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._srv = Server(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def push(self, jpg_bytes):
        with self._lock:
            self._jpg = jpg_bytes


PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + finger MCP knuckles
PREVIEW = os.path.expanduser("~/.ros/rammp_curobo/palm_view.jpg")


class PalmViewer(DepthCameraGrabber):
    """Grabber (depth + TF) extended with the color stream."""

    def __init__(self, camera_cfg, color_topic):
        super().__init__(camera_cfg, node_name="rammp_curobo_palm_view")
        self.color = None
        self.create_subscription(Image, color_topic, self._color_cb, 5)

    def _color_cb(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().error("unsupported color encoding %s" % msg.encoding)
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.color = (img, msg.encoding)


def make_hands():
    """MediaPipe Hands detector (legacy pipeline: models ship in the wheel)."""
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def main():
    import cv2

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", default="camera_d405_wrist.yaml")
    ap.add_argument(
        "--color-topic",
        default="/d405/d405/color/image_rect_raw",
        help="RealSense color stream (the D405's color IS its left depth "
        "imager, so color and depth pixels are natively aligned)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="no window: write %s about twice a second" % PREVIEW,
    )
    args = ap.parse_args()

    rclpy.init()
    cfg = load_camera_config(args.camera)
    node = PalmViewer(cfg, args.color_topic)
    hands = make_hands()

    window = not args.headless
    if window and not os.environ.get("DISPLAY"):
        # open on the robot's local desktop even when run from SSH
        os.environ["DISPLAY"] = ":0"
        xauth = os.path.expanduser("~/.Xauthority")
        if "XAUTHORITY" not in os.environ and os.path.exists(xauth):
            os.environ["XAUTHORITY"] = xauth
    if window:
        # Verify the X connection FIRST: Qt aborts the whole process on a
        # failed connect (it does not raise), which would kill the stream.
        import subprocess

        probe = subprocess.run(
            ["xset", "q"], capture_output=True, env=os.environ.copy()
        )
        if probe.returncode != 0:
            window = False
        else:
            try:
                cv2.namedWindow("palm_view", cv2.WINDOW_NORMAL)
                cv2.waitKey(1)
            except Exception:
                window = False
    if not window:
        print("(no usable X display — use the browser view)")

    stream = None
    try:
        stream = _MjpegServer(STREAM_PORT)
        print("live view: http://192.168.1.11:%d  (any browser)" % STREAM_PORT)
    except OSError as exc:
        print("stream port %d unavailable (%s) — browser view off" % (STREAM_PORT, exc))

    # fail fast if the camera driver isn't up
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
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    last_save = 0.0
    print(
        "palm_view running — 'q' quits"
        if window
        else "palm_view headless — Ctrl+C quits"
    )

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.color is None or not node.frames or node.info is None:
            continue
        img, enc = node.color
        node.color = None
        depth = node.frames[-1]
        del node.frames[:-1]
        rgb = img if enc == "rgb8" else img[:, :, ::-1]
        frame = np.ascontiguousarray(rgb[:, :, ::-1])  # BGR for OpenCV drawing

        # base_T_camera (needs the bringup for TF; degrade gracefully)
        R = t = None
        if have_tf:
            try:
                R, t = node.camera_pose()
            except SystemExit:
                have_tf = False
                print("(no TF — bringup not running; showing camera-frame only)")

        k = np.array(node.info.k).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        h_img, w_img = rgb.shape[:2]
        sx, sy = depth.shape[1] / w_img, depth.shape[0] / h_img

        def pixel_to_3d(u, v):
            """(u, v) color pixel -> camera / base point via median depth."""
            du, dv = int(u * sx), int(v * sy)
            patch = depth[max(0, dv - 3) : dv + 4, max(0, du - 3) : du + 4]
            patch = patch[(patch > 0.05) & np.isfinite(patch)]
            if len(patch) < 5:
                return None, None
            z = float(np.median(patch))
            cam = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
            base = (R @ cam + t) if R is not None else None
            return cam, base

        # --- MediaPipe hands (green)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                us = [lm.x * w_img for lm in hand.landmark]
                vs = [lm.y * h_img for lm in hand.landmark]
                x0, y0 = int(max(min(us) - 10, 0)), int(max(min(vs) - 10, 0))
                x1, y1 = (
                    int(min(max(us) + 10, w_img - 1)),
                    int(min(max(vs) + 10, h_img - 1)),
                )
                pu = np.mean([us[i] for i in PALM_LANDMARKS])
                pv = np.mean([vs[i] for i in PALM_LANDMARKS])
                cam, base = pixel_to_3d(pu, pv)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
                cv2.circle(frame, (int(pu), int(pv)), 6, (0, 200, 0), -1)
                for u, v in zip(us, vs):
                    cv2.circle(frame, (int(u), int(v)), 2, (0, 150, 0), -1)
                if cam is not None:
                    label = "palm %.2f m" % cam[2]
                    if base is not None:
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

        # --- depth-blob target (blue): what palm_demo would aim at
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
                        ok, why = palm_target_ok(blob)
                        cv2.putText(
                            frame,
                            "demo target %s" % ("OK" if ok else "refused"),
                            (ub + 14, vb - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 140, 0),
                            1,
                        )

        fps_n += 1
        if time.monotonic() - fps_t >= 1.0:
            fps = fps_n / (time.monotonic() - fps_t)
            fps_t, fps_n = time.monotonic(), 0
        cv2.putText(
            frame,
            "%.0f fps | green=MediaPipe hand  blue+=demo depth target%s"
            % (fps, "" if R is not None else " | NO TF (camera frame only)"),
            (8, h_img - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )

        if stream is not None:
            ok_enc, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                stream.push(jpg.tobytes())
        if window:
            cv2.imshow("palm_view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif time.monotonic() - last_save > 0.5:
            cv2.imwrite(PREVIEW, frame)
            last_save = time.monotonic()

    if window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npalm_view stopped")
