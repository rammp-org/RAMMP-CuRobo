"""Shared palm-demo vision and display machinery.

Used by palm_view (the standalone viewer) and palm_demo (the touch demo):
MediaPipe hand detection, the depth-blob fallback detector, the workspace
gate, and the MJPEG browser stream (open http://<jetson>:8405).
"""

import math
import os
import time

import numpy as np
from sensor_msgs.msg import Image

from rammp_curobo_ros.scan_common import DepthCameraGrabber

STREAM_PORT = 8405
MODEL_PATH = os.path.expanduser("~/.ros/rammp_curobo/hand_landmarker.task")
PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + finger MCP knuckles


def palm_target_ok(p, table_z=0.10, r_min=0.35, r_max=0.88, z_max=0.90):
    """Workspace gate for a palm point [x, y, z] in the base frame.

    Above the table by a margin, inside the comfortable reach annulus,
    not behind the arm (the demo corridor is the front half-plane).
    Returns (ok, reason).
    """
    x, y, z = (float(v) for v in p)
    r = math.hypot(x, y)
    if z < table_z:
        return False, "palm too low (z=%.2f < %.2f — near the table)" % (z, table_z)
    if z > z_max:
        return False, "palm too high (z=%.2f)" % z
    if r < r_min:
        return False, "palm too close to the base (r=%.2f)" % r
    if r > r_max:
        return False, "palm out of reach (r=%.2f)" % r
    if x < 0.15:
        return False, "palm outside the frontal demo corridor (x=%.2f)" % x
    return True, ""


def detect_palm(points, min_pts=150, cluster_r=0.06):
    """Nearest coherent blob's center from base-frame points (N, 3).

    The person presents an open palm facing the arm inside the corridor;
    the nearest cluster of sufficient size is the hand, its centroid the
    palm. Returns (center [3] or None, n_points, reason).
    """
    if len(points) < min_pts:
        return None, len(points), "not enough points in the corridor"
    r = np.hypot(points[:, 0], points[:, 1])
    order = np.argsort(r)
    seed = points[order[: max(min_pts // 3, 30)]].mean(axis=0)
    for _ in range(4):  # few mean-shift steps around the nearest surface
        d = np.linalg.norm(points - seed, axis=1)
        members = points[d < cluster_r * 2]
        if len(members) < min_pts:
            return None, len(members), "nearest blob too small (%d pts)" % len(members)
        seed = members.mean(axis=0)
    return seed, len(members), ""


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


class ColorDepthGrabber(DepthCameraGrabber):
    """Grabber (depth + TF) extended with the color stream."""

    def __init__(self, camera_cfg, color_topic):
        super().__init__(camera_cfg, node_name="rammp_curobo_palm_cam")
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
    # quiet the harmless TFLite/glog startup chatter that reads like errors
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def push_stream(frame_bgr, stream, quality=80):
    """Encode and push one annotated BGR frame to the MJPEG stream."""
    import cv2

    if stream is None:
        return
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        stream.push(jpg.tobytes())


def landmark_palms(hands, rgb):
    """MediaPipe: list of (palm_uv, bbox, all_landmark_uv) per hand."""
    h, w = rgb.shape[:2]
    res = hands.process(rgb)
    out = []
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            us = [lm.x * w for lm in hand.landmark]
            vs = [lm.y * h for lm in hand.landmark]
            pu = float(np.mean([us[i] for i in PALM_LANDMARKS]))
            pv = float(np.mean([vs[i] for i in PALM_LANDMARKS]))
            bbox = (
                int(max(min(us) - 10, 0)),
                int(max(min(vs) - 10, 0)),
                int(min(max(us) + 10, w - 1)),
                int(min(max(vs) + 10, h - 1)),
            )
            out.append(((pu, pv), bbox, list(zip(us, vs))))
    return out


def depth_at(depth, u, v, sx, sy, patch_r=3):
    """Median depth (m) around a color pixel; None if unusable."""
    du, dv = int(u * sx), int(v * sy)
    patch = depth[
        max(0, dv - patch_r) : dv + patch_r + 1,
        max(0, du - patch_r) : du + patch_r + 1,
    ]
    patch = patch[(patch > 0.05) & np.isfinite(patch)]
    if len(patch) < 5:
        return None
    return float(np.median(patch))
