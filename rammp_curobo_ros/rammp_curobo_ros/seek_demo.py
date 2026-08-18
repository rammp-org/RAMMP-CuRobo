#!/usr/bin/env python3
"""Seek demo: "go to the bottle" — scan, find, approach, dodge clutter.

    ros2 run rammp_curobo_ros seek_demo --text "go to the bottle"            # dry logic
    ros2 run rammp_curobo_ros seek_demo --text "go to the bottle" --execute  # the real thing

Flow (attended, every gate intact): typed 'seek' -> auto-generated glance
poses until YOLO (wrist D405) sees the object twice -> 3-D localization
via aligned depth + TF -> ignore region set around the target (purges
its mapped voxels; the thing you approach must not be dodged) ->
standoff pose planned through the perceived world -> plan shown -> typed
'go' -> execute at <=0.25 speed.

Needs: planner (execute:=true), cameras node, arm bringup, and the D405
driver WITH ALIGNED DEPTH:

    ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 \
        camera_name:=d405 align_depth.enable:=true

Vocabulary = YOLO's 80 COCO classes (+ a few synonyms). Weights are the
existing ~/yolo11s-seg.pt — nothing is downloaded (disk is at 95%).
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import rclpy

from rammp_curobo.geometry import rot_about_world_y, yaw_about_world_z
from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW, TourDemo, traj_time

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

# Pitches audited against the D405's 58-degree vertical FOV from the
# glance camera position: 40 degrees left a centered bench object OUTSIDE
# every glance's view (audit 2026-08-18) — 55/68 actually paint the
# 0.45-0.65 m band the runbook tells the operator to use.
GLANCES = [
    (0.0, math.radians(55)),
    (math.radians(35), math.radians(55)),
    (math.radians(-35), math.radians(55)),
    (0.0, math.radians(68)),
]


def parse_target(text):
    """The COCO class named in free text (longest match wins), or None.

    Punctuation is stripped first — 'go to the bottle.' must not fail
    (audit 2026-08-18)."""
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = " %s " % " ".join(words.split())
    hits = [c for c in COCO_CLASSES if " %s " % c in t or " %ss " % c in t]
    for word, cls in SYNONYMS.items():
        if " %s " % word in t or " %ss " % word in t:
            hits.append(cls)
    return max(hits, key=len) if hits else None


def box_to_center(xyxy, depth, fx, fy, cx, cy, shrink=0.3, mask=None):
    """Median-deproject the detection's core -> (center_cam, extent_m).

    Samples the bbox's shrunken core (off the silhouette edge, where
    depth holes and background bleed-through live — calibration
    postmortem), intersected with the instance MASK when the model
    provides one (an occluder overlapping the bbox otherwise hijacks the
    foreground cluster). The cluster anchor is the 5th-percentile depth,
    not the single minimum pixel — one near-range flying pixel must not
    outvote the object (audit 2026-08-18) — and the kept cluster needs
    >= 10 samples.
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
    valid = (z > 0.07) & (z < 0.9) & np.isfinite(z)
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


def standoff_pose(obj_xyz, standoff=0.18, min_radius=0.30, min_gap=0.08):
    """Wrist-flat approach pose on the base side of the object.

    Returns (pos, quat, gap) with the ACTUAL tool-to-object horizontal
    gap after clamping — never claim the requested standoff (audit
    2026-08-18: the old radial clamp silently shrank the gap and could
    even place the tool PAST a close object whose collision voxels the
    ignore region had just purged). Returns None when the object is too
    close to the base for a safe approach (gap would fall under
    min_gap).
    """
    ox, oy, oz = (float(v) for v in obj_xyz)
    bearing = math.atan2(oy, ox)
    r_obj = math.hypot(ox, oy)
    r = max(r_obj - standoff, min_radius)
    if r_obj - r < min_gap:
        return None
    r = min(r, 0.72)
    gap = r_obj - r
    z = min(max(oz + 0.03, 0.15), 0.55)
    pos = [r * math.cos(bearing), r * math.sin(bearing), z]
    quat = list(yaw_about_world_z(HOME_QUAT_XYZW, bearing))
    return pos, quat, gap


def glance_pose(bearing, pitch, r=0.35, z=0.42):
    """A scan pose: tool (and camera) aimed down-range at the bench."""
    pos = [r * math.cos(bearing), r * math.sin(bearing), z]
    quat = list(yaw_about_world_z(rot_about_world_y(HOME_QUAT_XYZW, pitch), bearing))
    return pos, quat


class _D405Grabber:
    """Color + aligned-depth + info + camera pose for one shot."""

    def __init__(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener

        from rammp_curobo_ros.cameras import load_camera_config

        self.node = node
        cfg = load_camera_config("camera_d405_wrist.yaml")
        self.mount_xyz = np.asarray(cfg["mount_xyz"], dtype=float)
        self.mount_quat = list(cfg["mount_quat_xyzw"])
        self.parent = cfg["parent_frame"]
        ns = cfg["depth_topic"].rsplit("/depth/", 1)[0]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.color = self.depth = self.info = None
        self.color_stamp = None
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
            # normalize to BGR: ultralytics treats numpy input as BGR (its
            # preprocess does the BGR->RGB flip itself) — feeding RGB ran
            # every inference channel-swapped (audit 2026-08-18, verified
            # in the installed 8.4.12)
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
        """(color_rgb, depth_m, intr, R, t) or None. Fresh frames only."""
        from rammp_curobo.perception import quat_to_mat

        self.color = self.depth = None
        t0 = time.monotonic()
        while self.color is None or self.depth is None or self.info is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > timeout_s:
                return None
        if self.depth.shape != self.color.shape[:2]:
            sys.exit(
                "aligned depth %s vs color %s — launch the driver with "
                "align_depth.enable:=true" % (self.depth.shape, self.color.shape[:2])
            )
        # TF at the frame's stamp when the buffer can serve it (the arm is
        # settled during shots, so latest is a safe fallback). Note: the
        # mount YAML describes the DEPTH optical frame; aligned depth
        # lives in the COLOR frame ~4 mm away — inside the ±3 cm budget,
        # deliberately uncorrected.
        tr = None
        for when in (rclpy.time.Time.from_msg(self.color_stamp), rclpy.time.Time()):
            try:
                tr = self.tf_buffer.lookup_transform("base_link", self.parent, when)
                break
            except Exception:
                continue
        if tr is None:
            return None
        q, t = tr.transform.rotation, tr.transform.translation
        r_p = quat_to_mat(q.x, q.y, q.z, q.w)
        qx, qy, qz, qw = self.mount_quat
        rot = r_p @ quat_to_mat(qx, qy, qz, qw)
        trans = r_p @ self.mount_xyz + np.array([t.x, t.y, t.z])
        return self.color, self.depth, self.info, rot, trans

    def missing(self):
        """Which streams are absent (for the preflight diagnosis)."""
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
            "YOLO weights not found at %s — NOT downloading (disk is "
            "nearly full); point --weights at an existing .pt" % path
        )
    from ultralytics import YOLO

    return YOLO(path)


def detect(model, frame_bgr, target, conf):
    """Best (xyxy, conf, mask|None) for `target` in the frame, or None."""
    res = model(frame_bgr, verbose=False)[0]
    best = None
    for i, b in enumerate(res.boxes):
        if res.names[int(b.cls)] != target or float(b.conf) < conf:
            continue
        if best is not None and float(b.conf) <= best[1]:
            continue
        m = None
        if res.masks is not None:
            import cv2

            m = res.masks.data[i].cpu().numpy()
            m = (
                cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
            )
        best = ([float(v) for v in b.xyxy[0]], float(b.conf), m)
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--text", required=True, help='e.g. "go to the bottle"')
    ap.add_argument("--execute", action="store_true", help="allow motion")
    ap.add_argument("--speed", type=float, default=0.25)
    ap.add_argument("--standoff", type=float, default=0.18)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--weights", default="~/yolo11s-seg.pt")
    args = ap.parse_args()
    scale = min(max(args.speed, 0.1), 0.25)

    target = parse_target(args.text)
    if target is None:
        sys.exit(
            "no known object in %r — vocabulary is YOLO's COCO classes "
            "(bottle, cup, bowl, ...)" % args.text
        )
    print("target class: %r" % target)

    rclpy.init()
    node = rclpy.create_node("rammp_curobo_seek")
    demo = TourDemo(node)
    grab = _D405Grabber(node)

    from rammp_curobo_interfaces.srv import SetIgnoreRegion

    ignore_cli = node.create_client(SetIgnoreRegion, "/cameras/set_ignore_region")

    if not args.execute:
        print(
            "dry-run: would scan %d glances for %r, then approach to "
            "%.2f m standoff. Add --execute." % (len(GLANCES), target, args.standoff)
        )
        return
    # preflight BEFORE any motion: all camera streams + TF must be alive,
    # else a missing align_depth flag would burn a full arm scan and then
    # misdiagnose as "object not seen" (audit 2026-08-18)
    print("preflight: waiting for camera streams + TF...")
    if grab.shot(timeout_s=6.0) is None:
        sys.exit(
            "camera preflight FAILED — missing: %s (and/or TF to %s). "
            "The seek demo needs the driver launched with "
            "align_depth.enable:=true and the arm bringup running."
            % (", ".join(grab.missing()) or "TF only", grab.parent)
        )
    print("preflight OK.")

    print(
        "\n*** SEEK: the arm will SCAN (%d glance poses) and then APPROACH "
        "the %s. Workspace clear, hand on e-stop. ***" % (len(GLANCES), target)
    )
    if input("type 'seek' to start scanning: ").strip() != "seek":
        sys.exit("aborted — nothing moved")

    model = load_detector(args.weights)
    found = []  # (center_base, extent), AT MOST ONE PER GLANCE —
    # two sightings from the same viewpoint share every systematic error,
    # so the consistency gate below would only measure frame noise
    # (audit 2026-08-18); cross-glance sightings actually triangulate
    for i, (bearing, pitch) in enumerate(GLANCES):
        pos, quat = glance_pose(bearing, pitch)
        plan = demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            print("glance %d unplannable — skipping" % (i + 1))
            continue
        print(
            "glance %d/%d (%.1f s)..." % (i + 1, len(GLANCES), traj_time(plan, scale))
        )
        if not demo.run(plan.trajectory, scale):
            sys.exit("glance motion failed — arm holds; see planner log")
        time.sleep(1.0)  # settle; frames while moving are useless anyway
        for _ in range(4):
            shot = grab.shot()
            if shot is None:
                continue
            frame, depth, intr, rot, trans = shot
            hit = detect(model, frame, target, args.conf)
            if hit is None:
                continue
            loc = box_to_center(hit[0], depth, mask=hit[2], **intr)
            if loc is None:
                continue
            center_cam, extent = loc
            center_base = rot @ center_cam + trans
            found.append((center_base, extent))
            print(
                "  saw %s (conf %.2f) at [%.2f, %.2f, %.2f]"
                % (target, hit[1], center_base[0], center_base[1], center_base[2])
            )
            break  # one sighting per viewpoint
        if len(found) >= 2:
            break
    del model  # free the GPU for cuRobo
    if len(found) < 2:
        sys.exit(
            "did not see a %s from two viewpoints — reposition it "
            "0.45-0.65 m in front of the arm and re-run" % target
        )
    centers = np.array([f[0] for f in found])
    if np.linalg.norm(centers[0] - centers[1]) > 0.10:
        sys.exit(
            "sightings from two viewpoints disagree by %.0f mm — moving "
            "object or bad depth/mount; re-run"
            % (np.linalg.norm(centers[0] - centers[1]) * 1000)
        )
    obj = centers.mean(axis=0)
    extent = np.max([f[1] for f in found], axis=0)
    dims = [float(max(extent[0], 0.05)) + 0.04] * 2 + [
        float(max(extent[1], 0.05)) + 0.04
    ]

    so = standoff_pose(obj, standoff=args.standoff)
    if so is None:
        sys.exit(
            "the %s is too close to the arm base (%.2f m out) for a safe "
            "approach — move it outward and re-run"
            % (target, float(np.hypot(obj[0], obj[1])))
        )
    pos, quat, gap = so

    from geometry_msgs.msg import Point, Vector3

    region_set = False
    try:
        if ignore_cli.wait_for_service(timeout_sec=3.0):
            req = SetIgnoreRegion.Request()
            req.center = Point(x=float(obj[0]), y=float(obj[1]), z=float(obj[2]))
            req.dims = Vector3(x=dims[0], y=dims[1], z=dims[2])
            fut = ignore_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
            res = fut.result()
            region_set = bool(res and res.success)
            print("ignore region:", res.message if res else "NO RESPONSE")
            # the purge reaches the PLANNER via the cameras node's next
            # 2 Hz tick + async world push — planning immediately would
            # race the stale world (audit 2026-08-18)
            time.sleep(1.5)
        else:
            print("cameras node not up — no ignore region (approach may be refused)")

        plan = demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            sys.exit("cannot plan the approach — see planner log")
        print(
            "\n%s at [%.2f, %.2f, %.2f]; approach leaves a %.2f m gap "
            "(%.1f s at speed %.2f)"
            % (target, obj[0], obj[1], obj[2], gap, traj_time(plan, scale), scale)
        )
        if input("type 'go' to approach: ").strip() != "go":
            sys.exit("aborted — arm holds at the last glance")
        if not demo.run(plan.trajectory, scale):
            sys.exit("approach failed — arm holds; see planner log")
        print(
            "\nARRIVED — %.2f m from the %s, facing it. Take it from here."
            % (gap, target)
        )
    finally:
        # never leave a permanent blind spot in the perceived world
        # (audit 2026-08-18): zero dims clears the region on EVERY exit
        if region_set and ignore_cli.service_is_ready():
            fut = ignore_cli.call_async(SetIgnoreRegion.Request())
            rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
            print("ignore region cleared — the world watches that spot again")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nseek stopped — arm holds")
