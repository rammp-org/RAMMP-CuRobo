"""Pure seek helpers: text -> class, detection, 3-D localization, sanity.

Shared by the seeker node and its tests. No control flow lives here.
"""

import os
import re
import sys
import time

import numpy as np
import rclpy

from rammp_curobo.geometry import rot_about_world_y, yaw_about_world_z
from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

SYNONYMS = {
    "mug": "cup", "glass": "wine glass", "water": "bottle",
    "soda": "bottle", "drink": "bottle", "phone": "cell phone",
    "plant": "potted plant", "ball": "sports ball", "teddy": "teddy bear",
}


def parse_target(text):
    """The COCO class named in free text (longest match wins), or None."""
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = " %s " % " ".join(words.split())
    hits = [c for c in COCO_CLASSES if " %s " % c in t or " %ss " % c in t]
    for word, cls in SYNONYMS.items():
        if " %s " % word in t or " %ss " % word in t:
            hits.append(cls)
    return max(hits, key=len) if hits else None


def box_to_center(xyxy, depth, fx, fy, cx, cy, shrink=0.3, mask=None,
                  min_depth=0.07):
    """Deproject a detection's core -> (center_cam, extent_m), or None.

    Samples the shrunken bbox core intersected with the instance mask
    (an occluder otherwise hijacks the foreground). Anchor = 5th-
    percentile depth, not the min pixel (flying-pixel-proof); cluster
    kept within 6 cm of it, needs >= 10 samples. min_depth 0.16 for the
    seeker: the gripper's own fingers sit 0.10-0.14 m from the lens and
    must never anchor the target (self-chase spiral, field 2026-08-19).
    """
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    dx, dy = int((x2 - x1) * shrink / 2), int((y2 - y1) * shrink / 2)
    ys = slice(max(0, y1 + dy), max(0, y2 - dy))
    xs = slice(max(0, x1 + dx), max(0, x2 - dx))
    core = depth[ys, xs]
    if core.size == 0:
        return None
    h, w = core.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = core
    valid = (z > min_depth) & (z < 0.9) & np.isfinite(z)
    if mask is not None:
        valid &= mask[ys, xs].astype(bool)
    if valid.sum() < 10:
        return None
    z, uu, vv = z[valid], uu[valid], vv[valid]
    anchor = float(np.percentile(z, 5))
    keep = z < anchor + 0.06
    if keep.sum() < 10:
        return None
    z = z[keep]
    uu = (uu + x1 + dx)[keep]
    vv = (vv + y1 + dy)[keep]
    pts = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=1)
    center = np.median(pts, axis=0)
    zc = float(center[2])
    extent = np.array([(x2 - x1) / fx * zc, (y2 - y1) / fy * zc])
    return center, extent


def glance_pose(bearing, pitch, r=0.35, z=0.42):
    """A survey pose: tool (and camera) aimed down-range at the bench."""
    pos = [r * float(np.cos(bearing)), r * float(np.sin(bearing)), z]
    quat = list(yaw_about_world_z(rot_about_world_y(HOME_QUAT_XYZW, pitch), bearing))
    return pos, quat


def roll_about_tool_z(xyzw, rad):
    """Post-multiply a LOCAL z (tool-axis) roll onto quat xyzw — the
    'wrist flat' convention knob (field 2026-08-19)."""
    x, y, z, w = xyzw
    c, s = np.cos(rad / 2.0), np.sin(rad / 2.0)
    return (x * c + y * s, y * c - x * s, z * c + w * s, w * c - z * s)


class D405Grabber:
    """Color + aligned-depth + info + camera pose for one shot."""

    def __init__(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener

        from rammp_curobo_ros.cameras import load_camera_config

        self.node = node
        cfg = load_camera_config("camera_d405_wrist.yaml")
        self.cfg = cfg
        self.mount_xyz = np.asarray(cfg["mount_xyz"], dtype=float)
        self.mount_quat = list(cfg["mount_quat_xyzw"])
        self.parent = cfg["parent_frame"]
        ns = cfg["depth_topic"].rsplit("/depth/", 1)[0]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.color = self.depth = self.info = None
        self.color_stamp = None
        self.last_fail = "no frames yet"
        node.create_subscription(
            Image, ns + "/color/image_raw", self._color_cb, qos_profile_sensor_data
        )
        node.create_subscription(
            Image,
            ns + "/aligned_depth_to_color/image_raw",
            self._depth_cb,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CameraInfo,
            ns + "/color/camera_info",
            self._info_cb,
            qos_profile_sensor_data,
        )

    def _color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            # ultralytics treats numpy input as BGR — feeding RGB runs
            # every inference channel-swapped (verified in 8.4.12)
            self.color = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()
            self.color_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            self.depth = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32)
                / 1000.0
            )

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.info = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    def shot(self, timeout_s=5.0):
        """(color_bgr, depth_m, intr, R, t) or None. Fresh frames only.

        TF at the frame's stamp, latest as fallback — the seeker only
        shoots while parked, where latest is safe. The mount YAML is the
        DEPTH frame; aligned depth is in the COLOR frame ~4 mm away,
        deliberately uncorrected.
        """
        from rammp_curobo.perception import quat_to_mat

        self.color = self.depth = None
        t0 = time.monotonic()
        while self.color is None or self.depth is None or self.info is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > timeout_s:
                self.last_fail = ", ".join(self.missing())
                return None
        if self.depth.shape != self.color.shape[:2]:
            sys.exit(
                "aligned depth %s vs color %s — launch the driver with "
                "align_depth.enable:=true" % (self.depth.shape, self.color.shape[:2])
            )
        tr = None
        for when in (rclpy.time.Time.from_msg(self.color_stamp), rclpy.time.Time()):
            try:
                tr = self.tf_buffer.lookup_transform("base_link", self.parent, when)
                break
            except Exception:
                continue
        if tr is None:
            self.last_fail = "TF base_link->%s" % self.parent
            return None
        self.last_fail = None
        q, t = tr.transform.rotation, tr.transform.translation
        r_p = quat_to_mat(q.x, q.y, q.z, q.w)
        qx, qy, qz, qw = self.mount_quat
        rot = r_p @ quat_to_mat(qx, qy, qz, qw)
        trans = r_p @ self.mount_xyz + np.array([t.x, t.y, t.z])
        return self.color, self.depth, self.info, rot, trans

    def missing(self):
        """Which streams are absent (diagnosis)."""
        out = []
        if self.color is None:
            out.append("color")
        if self.depth is None:
            out.append("ALIGNED depth (align_depth.enable:=true?)")
        if self.info is None:
            out.append("camera_info")
        return out


def load_detector(weights):
    path = os.path.expanduser(weights)
    if not os.path.isfile(path):
        sys.exit(
            "YOLO weights not found at %s — NOT downloading; point "
            "--weights at an existing .pt" % path
        )
    from ultralytics import YOLO

    return YOLO(path)


def detect_all(model, frame_bgr, target, conf):
    """Every (xyxy, conf, mask|None) of `target` in the frame, best-first."""
    res = model(frame_bgr, verbose=False)[0]
    out = []
    for i, b in enumerate(res.boxes):
        if res.names[int(b.cls)] != target or float(b.conf) < conf:
            continue
        m = None
        if res.masks is not None:
            import cv2

            m = res.masks.data[i].cpu().numpy()
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
        out.append(([float(v) for v in b.xyxy[0]], float(b.conf), m))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def track_update(current, sightings, max_jump=0.35, min_move=0.05):
    """Nearest in-leash sighting position, or None.

    The leash keeps a decoy elsewhere in frame from yanking the target;
    min_move=0 turns it into a pure refresh."""
    best = None
    for _, c, _, _ in sightings:
        c = np.asarray(c, dtype=float)
        d = float(np.linalg.norm(c - np.asarray(current, dtype=float)))
        if d <= max_jump and (best is None or d < best[1]):
            best = (c, d)
    if best is None or best[1] < min_move:
        return None
    return best[0]


def stable_fix(samples, tol=0.08, n=3):
    """Median position when the last n samples agree pairwise within tol.

    The acquisition gate: one frame never moves the arm — n agreeing
    frames from a parked camera do. samples = [(pos, t), ...] newest
    last; returns np array or None."""
    if len(samples) < n:
        return None
    pts = np.array([np.asarray(p, dtype=float) for p, _ in samples[-n:]])
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    if float(d.max()) > tol:
        return None
    return np.median(pts, axis=0)


def joint_travel(traj):
    """Per-joint TOTAL travel (rad) — winding detector (net hides flips)."""
    pts = np.array([list(p.positions) for p in traj.points])
    travel = np.abs(np.diff(pts, axis=0)).sum(axis=0)
    return dict(zip(traj.joint_names, travel.tolist()))


def purged_count(msg):
    """Voxels an ignore-region purge removed (None if unparseable) —
    zero means no world change to wait for."""
    m = re.search(r"(\d+) mapped voxels purged", msg or "")
    return int(m.group(1)) if m else None
