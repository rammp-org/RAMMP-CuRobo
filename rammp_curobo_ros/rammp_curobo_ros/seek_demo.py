#!/usr/bin/env python3
"""Seek demo: "go to the bottle" — scan, find, approach, dodge clutter.

    ros2 run rammp_curobo_ros seek_demo --text "go to the bottle"            # dry logic
    ros2 run rammp_curobo_ros seek_demo --text "go to the bottle" --execute  # the real thing

Flow (attended, every gate intact): typed 'seek' -> auto-generated
glance poses; every YOLO instance (wrist D405) in each glance's frame
is 3-D-localized via aligned depth + TF. The scan ends at the FIRST of:
a confident sighting (conf >= --sure-conf, default 0.80, re-confirmed
by a second same-pose frame within 5 cm) -> go now; a location
confirmed from TWO different glances (10 cm cluster) -> go; all glances
visited -> cluster and decide (a look-alike seen once loses the vote;
two confirmed locations refuse with a listing unless --pick nearest;
--sure-conf 1.1 disables the confident shortcut and always requires
two viewpoints) -> ignore region set around the target (purges its
mapped voxels; the thing you approach must not be dodged) -> standoff
pose planned through the perceived world -> plan shown (with the
largest joint travel — a wind-up warning precedes any joint-family
flip) -> typed 'go' -> execute at <=0.25 speed. With --follow, typed
'follow' then keeps tracking: the wrist camera re-detects the target
and the arm replans to it whenever it moves (~1-2 s reaction — a
replan loop, not millisecond servoing), until Ctrl+C.

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
        self.cfg = cfg
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


def detect_all(model, frame_bgr, target, conf):
    """Every (xyxy, conf, mask|None) of `target` in the frame, best-first.

    ALL instances, not just the best: a second bottle-shaped object in
    view must surface as a competing sighting the cross-viewpoint
    clustering can out-vote — under best-per-frame it silently replaced
    the real target and vetoed two scans (field 2026-08-19, decoy at
    y=+0.6)."""
    res = model(frame_bgr, verbose=False)[0]
    out = []
    for i, b in enumerate(res.boxes):
        if res.names[int(b.cls)] != target or float(b.conf) < conf:
            continue
        m = None
        if res.masks is not None:
            import cv2

            m = res.masks.data[i].cpu().numpy()
            m = (
                cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
            )
        out.append(([float(v) for v in b.xyxy[0]], float(b.conf), m))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def detect(model, frame_bgr, target, conf):
    """Best (xyxy, conf, mask|None) for `target` in the frame, or None."""
    hits = detect_all(model, frame_bgr, target, conf)
    return hits[0] if hits else None


def cluster_sightings(sightings, radius=0.10):
    """3D clustering of (glance_idx, center, extent, conf) tuples.

    Complete linkage: a sighting joins a cluster only when it is within
    `radius` of EVERY member, so a drifting weighted center can never
    chain distant sightings into one "object" (the pairwise 10 cm
    consistency gate survives the rework, audit 2026-08-19); among
    eligible clusters the NEAREST center wins, so a sighting between
    two objects joins the closer one instead of the first-created.
    Returns clusters sorted most-credible-first (distinct viewpoints,
    then summed confidence), each a dict with center (confidence-
    weighted mean), extent (elementwise max), glances (set of glance
    indices), conf (summed), n (sightings), members (raw centers).
    Only a cluster seen from >= 2 DISTINCT glances triangulates a real
    object — same-glance repeats share every systematic error and add
    no independence (audit 2026-08-18). Multi-instance scenes are
    first-class: a decoy seen from one viewpoint loses the vote instead
    of vetoing the scan (field 2026-08-19)."""
    clusters = []
    for gi, center, extent, conf in sightings:
        center = np.asarray(center, dtype=float)
        extent = np.asarray(extent, dtype=float)
        best = None
        for c in clusters:
            if all(np.linalg.norm(center - m) <= radius for m in c["members"]):
                d = float(np.linalg.norm(center - c["center"]))
                if best is None or d < best[1]:
                    best = (c, d)
        if best is None:
            clusters.append(
                {
                    "center": center.copy(),
                    "extent": extent.copy(),
                    "glances": {gi},
                    "conf": float(conf),
                    "n": 1,
                    "members": [center.copy()],
                }
            )
            continue
        home = best[0]
        w = home["conf"] + float(conf)
        home["center"] = (home["center"] * home["conf"] + center * float(conf)) / w
        home["extent"] = np.maximum(home["extent"], extent)
        home["glances"].add(gi)
        home["conf"] = w
        home["n"] += 1
        home["members"].append(center.copy())
    clusters.sort(key=lambda c: (len(c["glances"]), c["conf"]), reverse=True)
    return clusters


def reconfirmed(first_center, second_sightings, tol=0.05):
    """True when a second same-pose frame re-localizes the target within
    `tol` of the confident sighting.

    The confident fast path (--sure-conf) approaches after ONE glance at
    the operator's request, trading the cross-viewpoint gate for speed —
    this same-pose re-check is the remaining guard: it catches depth
    flicker and flying-pixel localization (two frames rarely repeat
    them) but NOT systematic errors or a look-alike object, which only
    the full two-viewpoint scan can."""
    first = np.asarray(first_center, dtype=float)
    return any(
        float(np.linalg.norm(np.asarray(c, dtype=float) - first)) <= tol
        for _, c, _, _ in second_sightings
    )


def track_update(current, sightings, max_jump=0.35, min_move=0.05):
    """Next believed target position while following, or None.

    Picks the sighting NEAREST the current belief — a decoy elsewhere in
    the frame must not yank the arm (35 cm leash; a real object can't
    teleport between ~1 s replans) — and reports it only when it moved
    at least min_move, so localization noise doesn't trigger replans."""
    best = None
    for _, c, _, _ in sightings:
        c = np.asarray(c, dtype=float)
        d = float(np.linalg.norm(c - np.asarray(current, dtype=float)))
        if d <= max_jump and (best is None or d < best[1]):
            best = (c, d)
    if best is None or best[1] < min_move:
        return None
    return best[0]


def joint_travel(traj):
    """Per-joint TOTAL travel (rad) over a trajectory.

    Net displacement hides winding: a joint-family flip travels ~2*pi on
    a wrist joint while ending near where it started (field 2026-08-19:
    the approach did a '360 flip' the operator never saw coming). The
    caller prints/warns on the worst joint before motion is offered."""
    pts = np.array([list(p.positions) for p in traj.points])
    travel = np.abs(np.diff(pts, axis=0)).sum(axis=0)
    return dict(zip(traj.joint_names, travel.tolist()))


def decide(clusters, pick="refuse"):
    """The scan's verdict from cluster_sightings output.

    Returns (status, ranked):
      ("unseen", [])            nothing was localized at all
      ("unconfirmed", clusters) sightings exist, none from 2+ viewpoints
      ("ambiguous", confirmed)  2+ confirmed locations and pick=refuse
      ("ok", confirmed)         approach ranked[0]; nearest-first when
                                pick="nearest" resolved a multi-instance
                                scene
    Lives outside main() so the decision that vetoed the field scans is
    testable on its own (audit 2026-08-19)."""
    if not clusters:
        return "unseen", []
    confirmed = [c for c in clusters if len(c["glances"]) >= 2]
    if not confirmed:
        return "unconfirmed", clusters
    if len(confirmed) > 1 and pick != "nearest":
        return "ambiguous", confirmed
    confirmed.sort(key=lambda c: float(np.hypot(c["center"][0], c["center"][1])))
    return "ok", confirmed


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
    ap.add_argument(
        "--pick",
        choices=["refuse", "nearest"],
        default="refuse",
        help="when 2+ locations are each confirmed from 2+ viewpoints: "
        "refuse (default, lists them) or approach the nearest",
    )
    ap.add_argument(
        "--sure-conf",
        type=float,
        default=0.80,
        help="confidence at which ONE re-confirmed sighting skips the "
        "rest of the scan and goes straight to the approach (default "
        "0.80; set above 1.0 to always require two viewpoints)",
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="after arriving, keep tracking: re-detect the target and "
        "replan to it whenever it moves (>=5 cm, <=35 cm per step; "
        "~1-2 s reaction). Typed 'follow' arms it; Ctrl+C stops.",
    )
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
    # enforce the driver-side depth contract (High Accuracy preset):
    # localization on hallucinated textureless-surface depth put the
    # bottle 25 cm in the air (field 2026-08-19). Non-fatal — the
    # cameras node also asserts it and retries.
    from rammp_curobo_ros.cameras import ensure_sensor_params

    ensure_sensor_params(node, grab.cfg)
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
    sightings = []  # (glance_idx, center_base, extent, conf) — ONE frame
    # per viewpoint (same-frame repeats share every systematic error;
    # audit 2026-08-18 gate rationale) but EVERY instance in that frame.
    # The scan stops EARLY the moment it is sure: a conf >= sure_conf
    # sighting that a second same-pose frame re-confirms goes straight
    # to approach (operator request 2026-08-19 — speed over the cross-
    # viewpoint gate; --sure-conf 1.1 restores strict scanning), and a
    # cluster confirmed from two viewpoints ends the tour too.
    visited = 0
    unlocalized = 0
    fixed = None  # cluster decided before the tour finished
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
        visited += 1
        time.sleep(1.0)  # settle; frames while moving are useless anyway
        seen_but_lost = 0
        glance_got = []
        for _ in range(4):
            shot = grab.shot()
            if shot is None:
                continue
            frame, depth, intr, rot, trans = shot
            hits = detect_all(model, frame, target, args.conf)
            got = []
            for xyxy, cf, mask in hits:
                loc = box_to_center(xyxy, depth, mask=mask, **intr)
                if loc is None:
                    continue
                center_cam, extent = loc
                center_base = rot @ center_cam + trans
                got.append((i, center_base, extent, cf))
                print(
                    "  saw %s (conf %.2f) at [%.2f, %.2f, %.2f]"
                    % (target, cf, center_base[0], center_base[1], center_base[2])
                )
            if got:
                glance_got = got
                sightings.extend(got)
                break  # one frame per viewpoint (except the re-check)
            seen_but_lost += len(hits)
        else:
            if seen_but_lost:
                # detection without localization must be VISIBLE — the
                # High Accuracy preset makes depth sparse on low texture,
                # and "did not see it" would misdirect the operator to
                # repositioning (audit 2026-08-19)
                unlocalized += seen_but_lost
                print(
                    "  %s detected %d time(s) but never depth-localized "
                    "(sparse depth on a low-texture/translucent target?)"
                    % (target, seen_but_lost)
                )
        if not glance_got:
            continue
        # fast path 1 — CONFIDENT: sure_conf sighting + same-pose re-check
        sure = max(
            (s for s in glance_got if s[3] >= args.sure_conf),
            key=lambda s: s[3],
            default=None,
        )
        if sure is not None:
            recheck = []
            for _ in range(3):
                shot = grab.shot()
                if shot is None:
                    continue
                frame, depth, intr, rot, trans = shot
                for xyxy, cf, mask in detect_all(model, frame, target, args.conf):
                    loc = box_to_center(xyxy, depth, mask=mask, **intr)
                    if loc is not None:
                        recheck.append((i, rot @ loc[0] + trans, loc[1], cf))
                if recheck:
                    break
            if reconfirmed(sure[1], recheck):
                cl = cluster_sightings([sure] + recheck)
                fixed = min(
                    cl, key=lambda c: float(np.linalg.norm(c["center"] - sure[1]))
                )
                print(
                    "confident (%.2f >= --sure-conf %.2f) and re-confirmed "
                    "— going now, %d glance(s) skipped"
                    % (sure[3], args.sure_conf, len(GLANCES) - i - 1)
                )
                break
            print(
                "  conf %.2f but the same-pose re-check did not agree — "
                "continuing the scan" % sure[3]
            )
        # fast path 2 — cross-viewpoint confirmation already achieved
        if i < len(GLANCES) - 1:
            st, rk = decide(cluster_sightings(sightings), pick=args.pick)
            if st == "ok" and len(rk) == 1:
                fixed = rk[0]
                print(
                    "confirmed from two viewpoints after glance %d — "
                    "skipping the rest" % (i + 1)
                )
                break
    if not args.follow:
        del model  # free the GPU for cuRobo (follow mode keeps detecting)
    if fixed is not None:
        chosen = fixed
    else:
        if visited < 2:
            sys.exit(
                "only %d of %d glance poses could be planned and executed — "
                "confirmation needs two viewpoints; clear the space around "
                "the arm or check the planner" % (visited, len(GLANCES))
            )
        status, ranked = decide(cluster_sightings(sightings), pick=args.pick)
        if status == "unseen":
            extra = (
                " (YOLO detected it %d time(s) but depth never localized it "
                "— low-texture/translucent target?)" % unlocalized
                if unlocalized
                else ""
            )
            sys.exit(
                "did not localize a %s from any viewpoint%s — reposition it "
                "0.45-0.65 m in front of the arm and re-run" % (target, extra)
            )
        if status == "unconfirmed":
            lone = "; ".join(
                "[%.2f, %.2f, %.2f]" % tuple(c["center"]) for c in ranked
            )
            sys.exit(
                "no %s location was confirmed from two viewpoints (single-"
                "viewpoint sightings at: %s) — moving object, bad depth, or "
                "visible from only one glance; re-run (a conf >= %.2f "
                "sighting would have gone directly)" % (target, lone, args.sure_conf)
            )
        listing = "; ".join(
            "[%.2f, %.2f, %.2f] (%d viewpoints, conf %.2f)"
            % (c["center"][0], c["center"][1], c["center"][2],
               len(c["glances"]), c["conf"])
            for c in ranked
        )
        if status == "ambiguous":
            sys.exit(
                "%d distinct %s locations each confirmed from 2+ viewpoints: "
                "%s — ambiguous scene. Remove the extras or re-run with "
                "--pick nearest." % (len(ranked), target, listing)
            )
        if len(ranked) > 1:
            print(
                "%d confirmed %s locations: %s — approaching the NEAREST "
                "(--pick nearest)" % (len(ranked), target, listing)
            )
        chosen = ranked[0]
    obj = chosen["center"]
    extent = chosen["extent"]
    print(
        "%s fixed at [%.2f, %.2f, %.2f] — %d sighting(s) from %d viewpoint(s)"
        % (target, obj[0], obj[1], obj[2], chosen["n"], len(chosen["glances"]))
    )
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

    def set_region(center):
        if not ignore_cli.service_is_ready():
            return False, "cameras node not up"
        req = SetIgnoreRegion.Request()
        req.center = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        req.dims = Vector3(x=dims[0], y=dims[1], z=dims[2])
        fut = ignore_cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        res = fut.result()
        return bool(res and res.success), (res.message if res else "NO RESPONSE")

    region_set = False
    try:
        if ignore_cli.wait_for_service(timeout_sec=3.0):
            region_set, msg = set_region(obj)
            print("ignore region:", msg)
            # the purge reaches the PLANNER via the cameras node's next
            # 2 Hz tick + async world push — planning immediately would
            # race the stale world (audit 2026-08-18)
            time.sleep(1.5)
        else:
            print("cameras node not up — no ignore region (approach may be refused)")

        plan = demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            sys.exit("cannot plan the approach — see planner log")
        travel = joint_travel(plan.trajectory)
        worst_j = max(travel, key=travel.get)
        wind = ""
        if travel[worst_j] > 3.5:
            wind = (
                "\n*** WARNING: this plan WINDS %s through %.1f rad (joint-"
                "family flip) — the arm will make a large sweeping "
                "reconfiguration. Ctrl+C mid-motion stops it; consider "
                "re-running instead of typing go. ***"
                % (worst_j, travel[worst_j])
            )
        print(
            "\n%s at [%.2f, %.2f, %.2f]; approach leaves a %.2f m gap "
            "(%.1f s at speed %.2f; largest joint travel %s %.1f rad)%s"
            % (target, obj[0], obj[1], obj[2], gap, traj_time(plan, scale),
               scale, worst_j, travel[worst_j], wind)
        )
        if input("type 'go' to approach: ").strip() != "go":
            sys.exit("aborted — arm holds at the last glance")
        if not demo.run(plan.trajectory, scale):
            sys.exit("approach failed — arm holds; see planner log")
        print(
            "\nARRIVED — %.2f m from the %s, facing it."
            % (gap, target)
        )
        if args.follow:
            print(
                "\n*** FOLLOW: after you type 'follow', the arm re-detects "
                "the %s and moves to track it WITHOUT further confirmation "
                "(replan loop, ~1-2 s reaction — move it SLOWLY). Ctrl+C "
                "stops and holds. ***" % target
            )
            if input("type 'follow' to track: ").strip() == "follow":
                obj = np.asarray(obj, dtype=float)

                def localized(shot):
                    frame, depth, intr, rot, trans = shot
                    out = []
                    for xyxy, cf, mask in detect_all(model, frame, target, args.conf):
                        loc = box_to_center(xyxy, depth, mask=mask, **intr)
                        if loc is not None:
                            out.append((0, rot @ loc[0] + trans, loc[1], cf))
                    return out

                try:
                    while True:
                        shot = grab.shot(timeout_s=2.0)
                        if shot is None:
                            continue
                        cand = track_update(obj, localized(shot))
                        if cand is None:
                            continue
                        shot = grab.shot(timeout_s=2.0)
                        if shot is None or not reconfirmed(cand, localized(shot)):
                            continue  # one frame never moves the arm
                        so = standoff_pose(cand, standoff=args.standoff)
                        if so is None:
                            print("  moved too close to the base — holding")
                            continue
                        npos, nquat, ngap = so
                        ok, _ = set_region(cand)
                        region_set = region_set or ok
                        time.sleep(1.0)  # purge propagation (2 Hz world)
                        plan = demo.plan_pose_from(npos, nquat, None)
                        if plan is None or not plan.success:
                            print("  replan failed — holding; see planner log")
                            continue
                        if max(joint_travel(plan.trajectory).values()) > 3.5:
                            print("  replan winds the arm — skipped; move the "
                                  "%s back a little" % target)
                            continue
                        print(
                            "  -> [%.2f, %.2f, %.2f] (gap %.2f m)"
                            % (cand[0], cand[1], cand[2], ngap)
                        )
                        if not demo.run(plan.trajectory, scale):
                            print("  segment refused — holding; see planner log")
                            continue
                        obj = cand
                except KeyboardInterrupt:
                    print("\nfollow stopped — arm holds")
        else:
            print("Take it from here.")
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
