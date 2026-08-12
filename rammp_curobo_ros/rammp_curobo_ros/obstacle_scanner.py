#!/usr/bin/env python3
"""Automatic obstacle detection from the Gen3's built-in wrist depth camera.

One-shot scan: grab depth frames from kinova_vision, project them into the
base frame via TF, drop the arm's own body and the tabletop, cluster what
remains into axis-aligned boxes, and write a complete collision-world YAML
(a conservative zero-measurement table plane + the detected boxes). With
--apply the planner node hot-swaps to it via /rammp_curobo/set_world.

    ros2 run rammp_curobo_ros scan_world --debug          # look, don't touch
    ros2 run rammp_curobo_ros scan_world --apply          # scan + swap world

Prerequisites: the kortex bringup (for TF) and the camera driver:
    ros2 launch kinova_vision kinova_vision.launch.py device:=192.168.1.10

The camera only sees where the hand points — scan from a pose that faces
the workspace you're about to plan in, and re-scan after the scene changes.
The static table plane (top at z=0, above the real surface) is kept because
depth on flat surfaces is noise, not truth; everything the camera can't see
is UNKNOWN, not free.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from rammp_curobo.geometry import euler_deg_to_quat_xyzw

# end_effector_link -> camera_link, from kortex_description gen3_macro.xacro
# (real-hardware branch). The bringup's URDF is built with vision:=false, so
# the scanner broadcasts this itself; kinova_vision publishes the rest
# (camera_link -> camera_depth_frame).
CAMERA_MOUNT_XYZ = (0.0, 0.05639, -0.00305)
CAMERA_MOUNT_RPY_DEG = (180.0, 180.0, 0.0)

ARM_CHAIN = [
    "base_link",
    "shoulder_link",
    "half_arm_1_link",
    "half_arm_2_link",
    "forearm_link",
    "spherical_wrist_1_link",
    "spherical_wrist_2_link",
    "bracelet_link",
    "end_effector_link",
]


def quat_to_mat(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def table_ring(extent=1.5, hole=0.16, thickness=0.08):
    """A conservative 'never go below the tabletop' plane, no tape measure.

    Four boxes with tops at z=0 (ABOVE the real surface — the arm is
    bolted nearly flush, plate ~1.5 cm) covering +/-extent, with a hole
    around the base so the arm's own base spheres keep a valid start.
    """
    t2, zc = thickness, -thickness / 2.0
    mid = (hole + extent) / 2.0
    span = extent - hole
    return [
        {
            "name": "table_front",
            "position": [mid, 0.0, zc],
            "dims": [span, 2 * extent, t2],
        },
        {
            "name": "table_back",
            "position": [-mid, 0.0, zc],
            "dims": [span, 2 * extent, t2],
        },
        {
            "name": "table_left",
            "position": [0.0, mid, zc],
            "dims": [2 * hole, span, t2],
        },
        {
            "name": "table_right",
            "position": [0.0, -mid, zc],
            "dims": [2 * hole, span, t2],
        },
    ]


class Scanner(Node):
    def __init__(self):
        super().__init__("rammp_curobo_scanner")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._static = StaticTransformBroadcaster(self)
        self._broadcast_camera_mount()
        self.frames = []
        self.info = None
        self.depth_frame_id = None
        self.create_subscription(
            CameraInfo, "/camera/depth/camera_info", self._info_cb, 5
        )
        self.create_subscription(Image, "/camera/depth/image_raw", self._depth_cb, 5)

    def _broadcast_camera_mount(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "end_effector_link"
        t.child_frame_id = "camera_link"
        t.transform.translation.x = CAMERA_MOUNT_XYZ[0]
        t.transform.translation.y = CAMERA_MOUNT_XYZ[1]
        t.transform.translation.z = CAMERA_MOUNT_XYZ[2]
        qx, qy, qz, qw = euler_deg_to_quat_xyzw(CAMERA_MOUNT_RPY_DEG)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._static.sendTransform(t)

    def _info_cb(self, msg):
        self.info = msg

    def _depth_cb(self, msg):
        self.depth_frame_id = msg.header.frame_id
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
            self.get_logger().error("unsupported depth encoding %s" % msg.encoding)
            return
        self.frames.append(d)

    def collect(self, n_frames, timeout_s=15.0):
        t0 = time.monotonic()
        while len(self.frames) < n_frames or self.info is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - t0 > timeout_s:
                sys.exit(
                    "No depth frames after %.0f s — is kinova_vision "
                    "running? (ros2 launch kinova_vision "
                    "kinova_vision.launch.py device:=192.168.1.10)" % timeout_s
                )
        return np.median(np.stack(self.frames[-n_frames:]), axis=0)

    def base_from(self, frame, timeout_s=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                tr = self.tf_buffer.lookup_transform(
                    "base_link", frame, rclpy.time.Time()
                )
                q = tr.transform.rotation
                t = tr.transform.translation
                return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        sys.exit("No TF base_link <- %s — is the kortex bringup running?" % frame)

    def link_points(self):
        pts = []
        for name in ARM_CHAIN:
            _R, t = self.base_from(name)
            pts.append(t)
        # gripper extent: ~18 cm past the flange along the tool axis
        R_ee, t_ee = self.base_from("end_effector_link")
        pts.append(t_ee + R_ee @ np.array([0.0, 0.0, 0.18]))
        return pts


def robot_mask(points, link_pts, radius):
    """True where a point is NOT part of the arm (capsule filter)."""
    keep = np.ones(len(points), dtype=bool)
    for a, b in zip(link_pts[:-1], link_pts[1:]):
        ab = b - a
        denom = float(ab @ ab) or 1e-9
        t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
        closest = a[None, :] + t[:, None] * ab[None, :]
        keep &= np.linalg.norm(points - closest, axis=1) > radius
    return keep


def cluster_boxes(points, voxel, min_voxels, max_boxes):
    """Occupied-voxel connected components -> axis-aligned boxes."""
    from scipy import ndimage

    idx = np.floor(points / voxel).astype(np.int64)
    origin = idx.min(axis=0)
    idx -= origin
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.uint8)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    solid = uniq[counts >= 2]
    grid[solid[:, 0], solid[:, 1], solid[:, 2]] = 1
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    boxes = []
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        if len(cells) < min_voxels:
            continue
        lo = (cells.min(axis=0) + origin) * voxel
        hi = (cells.max(axis=0) + origin + 1) * voxel
        boxes.append(
            {
                "center": ((lo + hi) / 2.0).tolist(),
                "dims": (hi - lo).tolist(),
                "voxels": int(len(cells)),
            }
        )
    boxes.sort(key=lambda b: -b["voxels"])
    return boxes[:max_boxes], len(boxes)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frames", type=int, default=5, help="depth frames to median-combine"
    )
    ap.add_argument("--out", default="/home/abra/.ros/rammp_curobo/scanned_world.yaml")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="call /rammp_curobo/set_world with the result",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--min-voxels", type=int, default=6)
    ap.add_argument("--max-boxes", type=int, default=25)
    ap.add_argument(
        "--max-range", type=float, default=1.3, help="ignore depth beyond this (m)"
    )
    ap.add_argument(
        "--min-z",
        type=float,
        default=0.03,
        help="drop points below this base-frame height "
        "(the static table plane owns that region)",
    )
    ap.add_argument(
        "--self-radius",
        type=float,
        default=0.11,
        help="capsule radius for removing the arm's own body",
    )
    ap.add_argument(
        "--inflate",
        type=float,
        default=0.03,
        help="extra metres per side on detected boxes — a "
        "single viewpoint only sees the front of things",
    )
    args = ap.parse_args()

    rclpy.init()
    node = Scanner()
    depth = node.collect(args.frames)
    k = np.array(node.info.k).reshape(3, 3)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    h, w = depth.shape
    vv, uu = np.mgrid[0:h:2, 0:w:2]
    z = depth[::2, ::2]
    valid = (z > 0.15) & (z < args.max_range) & np.isfinite(z)
    z, uu, vv = z[valid], uu[valid], vv[valid]
    cam_pts = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=1)

    R, t = node.base_from(node.depth_frame_id)
    pts = cam_pts @ R.T + t
    n_raw = len(pts)

    if args.debug:
        mid = depth[h // 2, w // 2]
        c = R @ np.array([0.0, 0.0, float(mid)]) + t if mid > 0 else None
        print(
            "camera at base %s | center pixel %.3f m -> %s"
            % (
                np.round(t, 3).tolist(),
                mid,
                None if c is None else np.round(c, 3).tolist(),
            )
        )

    ws = (
        (np.abs(pts[:, 0]) < 1.2)
        & (np.abs(pts[:, 1]) < 1.2)
        & (pts[:, 2] > args.min_z)
        & (pts[:, 2] < 1.3)
    )
    pts = pts[ws]
    n_ws = len(pts)
    if len(pts):
        pts = pts[robot_mask(pts, node.link_points(), args.self_radius)]
    n_free = len(pts)

    boxes, n_found = ([], 0)
    if len(pts):
        boxes, n_found = cluster_boxes(pts, args.voxel, args.min_voxels, args.max_boxes)
    if n_found > len(boxes):
        print("NOTE: %d clusters found, keeping the %d largest" % (n_found, len(boxes)))

    obstacles = table_ring()
    for i, b in enumerate(boxes):
        obstacles.append(
            {
                "name": "det_%d" % i,
                "position": [round(v, 3) for v in b["center"]],
                "dims": [round(v + 2 * args.inflate, 3) for v in b["dims"]],
                "color": [0.9, 0.4, 0.1, 1.0],
            }
        )

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(
            "# GENERATED by scan_world (%s) — re-scan after the scene "
            "changes.\n# Table plane is the conservative "
            "zero-measurement model (top at z=0).\n"
            % time.strftime("%Y-%m-%d %H:%M:%S")
        )
        yaml.safe_dump(
            {
                "base_frame": "base_link",
                "obstacles": obstacles,
                "objects": [],
                "targets": [],
            },
            f,
            sort_keys=False,
            default_flow_style=None,
        )

    print(
        "points: %d raw -> %d in workspace -> %d after self-filter"
        % (n_raw, n_ws, n_free)
    )
    print("%-8s %-24s %s" % ("box", "center [m]", "dims [m]"))
    for i, b in enumerate(boxes):
        print(
            "det_%-4d %-24s %s"
            % (i, np.round(b["center"], 3).tolist(), np.round(b["dims"], 3).tolist())
        )
    print(
        "world written: %s (%d detected boxes + table plane)" % (args.out, len(boxes))
    )

    if args.apply:
        from rammp_curobo_interfaces.srv import SetWorld

        client = node.create_client(SetWorld, "/rammp_curobo/set_world")
        if not client.wait_for_service(timeout_sec=3.0):
            sys.exit("planner node not running — world file written but " "not applied")
        fut = client.call_async(SetWorld.Request(world=args.out))
        t0 = time.monotonic()
        while not fut.done():
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.monotonic() - t0 > 20:
                sys.exit("set_world did not answer")
        resp = fut.result()
        print("set_world: %s (%s)" % ("OK" if resp.success else "FAILED", resp.message))
        if not resp.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
