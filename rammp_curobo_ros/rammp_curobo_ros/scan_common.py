"""Shared machinery for depth-camera world scanning.

Camera definitions are YAML (rammp_curobo_ros/config/camera_*.yaml) so the
same scan code runs against the MuJoCo sim's rendered D405, the real
wrist-mounted D405, or the Gen3's built-in vision module by swapping one
file:

    depth_topic: /d405/depth
    info_topic: /d405/camera_info
    # EITHER: the optical frame exists in TF (kinova_vision case)
    tf_frame: camera_depth_frame
    # OR: a fixed mount on a robot link (sim d405, real d405 bracket):
    parent_frame: bracelet_link
    mount_xyz: [0.0, -0.058, -0.078]
    mount_quat_xyzw: [0.0, 1.0, 0.0, 0.0]   # OPTICAL frame (z = view dir)
    min_range: 0.12
    max_range: 1.2

Point pipeline per capture: median depth over N frames -> deproject with
the camera_info intrinsics -> transform to base_link -> workspace crop ->
capsule self-filter over the TF link chain. Fusion and box clustering are
shared by the single-shot scanner and the sweep scanner.
"""

import os
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

DEFAULT_OUT = os.path.expanduser("~/.ros/rammp_curobo/scanned_world.yaml")

# The planner node's action/service namespace — derived from the node name
# declared in planner_node.py (Node("rammp_curobo")). Defined once here for
# every in-repo client; the standalone examples carry their own copy on
# purpose (they demonstrate integration without importing this package).
NODE_NAMESPACE = "/rammp_curobo"

# end_effector_link -> camera_link for the Gen3's built-in vision module
# (kortex_description gen3_macro.xacro, real-hardware branch); broadcast so
# kinova_vision's camera_depth_frame chains to the robot even though the
# bringup URDF is built with vision:=false.
KINOVA_CAMERA_MOUNT_XYZ = (0.0, 0.05639, -0.00305)
KINOVA_CAMERA_MOUNT_RPY_DEG = (180.0, 180.0, 0.0)

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


def load_camera_config(name_or_path):
    """Resolve a camera YAML by path or packaged name (config/ dir)."""
    candidates = [name_or_path]
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("rammp_curobo_ros")
        candidates.append(os.path.join(share, "config", name_or_path))
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(here, "config", name_or_path))
    for c in candidates:
        if os.path.isfile(c):
            with open(c) as f:
                return yaml.safe_load(f)
    sys.exit("camera config %r not found (tried: %s)" % (name_or_path, candidates))


def table_ring(extent=1.5, hole=0.16, thickness=0.08):
    """Conservative 'never below the tabletop' plane with a base cutout.

    Tops at z=0 — above the real surface (the arm is bolted nearly flush).
    Kept static because depth on flat surfaces is noise, not truth.
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


def _split_cells(cells, min_fill, min_span):
    """Recursively split a voxel component whose AABB is mostly empty.

    One axis-aligned box around an L-shaped wall complex claims huge
    volumes of FREE space (field: the first sim sweep's wall mega-box
    swallowed the home pose). Split along the longest axis until each
    box is reasonably full or small."""
    lo = cells.min(axis=0)
    hi = cells.max(axis=0) + 1
    span = hi - lo
    vol = int(np.prod(span))
    fill = len(cells) / vol
    if fill >= min_fill or span.max() <= min_span:
        return [(lo, hi, len(cells))]
    axis = int(np.argmax(span))
    mid = (lo[axis] + hi[axis]) // 2
    left = cells[cells[:, axis] < mid]
    right = cells[cells[:, axis] >= mid]
    if not len(left) or not len(right):
        return [(lo, hi, len(cells))]
    return _split_cells(left, min_fill, min_span) + _split_cells(
        right, min_fill, min_span
    )


def cluster_boxes(
    points,
    voxel,
    min_points_per_voxel,
    min_voxels,
    max_boxes,
    min_fill=0.25,
    min_span=4,
):
    """Occupied voxels -> connected components -> TIGHT axis-aligned boxes."""
    from scipy import ndimage

    idx = np.floor(points / voxel).astype(np.int64)
    origin = idx.min(axis=0)
    idx -= origin
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.uint8)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    solid = uniq[counts >= min_points_per_voxel]
    grid[solid[:, 0], solid[:, 1], solid[:, 2]] = 1
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    boxes = []
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        if len(cells) < min_voxels:
            continue
        for lo, hi, nvox in _split_cells(cells, min_fill, min_span):
            lo = (lo + origin) * voxel
            hi = (hi + origin) * voxel
            boxes.append(
                {
                    "center": ((lo + hi) / 2.0).tolist(),
                    "dims": (hi - lo).tolist(),
                    "voxels": int(nvox),
                }
            )

    # keep the NEAREST boxes when capped — a small bottle inside reach
    # matters more than the far half of a wall (the cap once silently
    # dropped the bottle while keeping wall slabs)
    def closest_xy(b):
        c = np.abs(np.asarray(b["center"][:2]))
        half = np.asarray(b["dims"][:2]) / 2.0
        return float(np.linalg.norm(np.maximum(c - half, 0.0)))

    boxes.sort(key=closest_xy)
    return boxes[:max_boxes], len(boxes)


def write_world_yaml(path, boxes, inflate, header_note):
    obstacles = table_ring()
    for i, b in enumerate(boxes):
        obstacles.append(
            {
                "name": "det_%d" % i,
                "position": [round(v, 3) for v in b["center"]],
                "dims": [round(v + 2 * inflate, 3) for v in b["dims"]],
                "color": [0.9, 0.4, 0.1, 1.0],
            }
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "# GENERATED %s (%s) — re-scan after the scene changes.\n"
            "# Table plane = conservative zero-measurement model (top z=0).\n"
            % (header_note, time.strftime("%Y-%m-%d %H:%M:%S"))
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
    return obstacles


def add_cluster_args(parser, frames, min_points_per_voxel, min_voxels, max_boxes):
    """The flags shared by every scan tool (per-tool tuned defaults passed
    explicitly so intentional differences stay visible at the call site)."""
    parser.add_argument(
        "--frames",
        type=int,
        default=frames,
        help="depth frames median-combined per capture",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="hot-swap the planner's world via %s/set_world when done" % NODE_NAMESPACE,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--voxel", type=float, default=0.04)
    parser.add_argument(
        "--min-points-per-voxel", type=int, default=min_points_per_voxel
    )
    parser.add_argument("--min-voxels", type=int, default=min_voxels)
    parser.add_argument("--max-boxes", type=int, default=max_boxes)
    parser.add_argument(
        "--inflate",
        type=float,
        default=0.01,
        help="extra metres per side on detected boxes (voxel quantization "
        "already inflates; a single viewpoint only sees front faces)",
    )


def report_boxes(boxes, n_found, out_path):
    """The human-readable scan summary both tools print."""
    if n_found > len(boxes):
        print(
            "NOTE: %d clusters, keeping the %d NEAREST (see cluster_boxes: "
            "a close bottle outranks a far wall)" % (n_found, len(boxes))
        )
    print("%-8s %-24s %s" % ("box", "center [m]", "dims [m]"))
    for i, b in enumerate(boxes):
        print(
            "det_%-4d %-24s %s"
            % (
                i,
                np.round(b["center"], 3).tolist(),
                np.round(b["dims"], 3).tolist(),
            )
        )
    print("world written: %s (%d boxes + table plane)" % (out_path, len(boxes)))


def apply_world(node, world_path):
    """Hot-swap the planner node's collision world to `world_path`.

    Exits the process on failure — a scan that claims --apply worked must
    never leave the planner on the stale world.
    """
    from rammp_curobo_interfaces.srv import SetWorld

    client = node.create_client(SetWorld, NODE_NAMESPACE + "/set_world")
    if not client.wait_for_service(timeout_sec=3.0):
        sys.exit("planner node not running — world written but NOT applied")
    fut = client.call_async(SetWorld.Request(world=world_path))
    t0 = time.monotonic()
    while not fut.done():
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.monotonic() - t0 > 20:
            sys.exit("set_world did not answer")
    resp = fut.result()
    print("set_world: %s (%s)" % ("OK" if resp.success else "FAILED", resp.message))
    if not resp.success:
        sys.exit(1)


def spin_until_done(node, future, timeout_s):
    """Spin `node` until `future` resolves; None on timeout.

    For single-threaded clients (scan tools, scripts). The planner node
    itself uses executor.await_future — an event wait suited to its
    multithreaded executor.
    """
    t0 = time.monotonic()
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


class DepthCameraGrabber(Node):
    """Depth capture + TF projection for one configured camera."""

    def __init__(self, camera_cfg, node_name="rammp_curobo_scanner"):
        super().__init__(node_name)
        self.cfg = camera_cfg
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._static = StaticTransformBroadcaster(self)
        if camera_cfg.get("broadcast_kinova_mount"):
            self._broadcast_kinova_mount()
        self.frames = []
        self.info = None
        self.create_subscription(CameraInfo, camera_cfg["info_topic"], self._info_cb, 5)
        self.create_subscription(Image, camera_cfg["depth_topic"], self._depth_cb, 5)

    def _broadcast_kinova_mount(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "end_effector_link"
        t.child_frame_id = "camera_link"
        t.transform.translation.x = KINOVA_CAMERA_MOUNT_XYZ[0]
        t.transform.translation.y = KINOVA_CAMERA_MOUNT_XYZ[1]
        t.transform.translation.z = KINOVA_CAMERA_MOUNT_XYZ[2]
        qx, qy, qz, qw = euler_deg_to_quat_xyzw(KINOVA_CAMERA_MOUNT_RPY_DEG)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._static.sendTransform(t)

    def _info_cb(self, msg):
        self.info = msg

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
            self.get_logger().error("unsupported depth encoding %s" % msg.encoding)
            return
        self.frames.append(d)

    def capture(self, n_frames, timeout_s=20.0):
        """Median of the NEXT n_frames (discards anything already queued)."""
        self.frames.clear()
        t0 = time.monotonic()
        while len(self.frames) < n_frames or self.info is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - t0 > timeout_s:
                sys.exit(
                    "No depth frames on %s after %.0f s — is the camera "
                    "driver running?" % (self.cfg["depth_topic"], timeout_s)
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
        sys.exit("No TF base_link <- %s — is the arm bringup running?" % frame)

    def camera_pose(self):
        """(R, t): base_T_optical for the configured camera."""
        if self.cfg.get("tf_frame"):
            return self.base_from(self.cfg["tf_frame"])
        R_p, t_p = self.base_from(self.cfg["parent_frame"])
        mx = np.asarray(self.cfg["mount_xyz"], dtype=float)
        qx, qy, qz, qw = self.cfg["mount_quat_xyzw"]
        return R_p @ quat_to_mat(qx, qy, qz, qw), R_p @ mx + t_p

    def link_points(self):
        pts = []
        for name in ARM_CHAIN:
            _R, t = self.base_from(name)
            pts.append(t)
        R_ee, t_ee = self.base_from("end_effector_link")
        pts.append(t_ee + R_ee @ np.array([0.0, 0.0, 0.18]))
        return pts

    def capture_points(
        self, n_frames, stride=2, min_z=0.03, xy_extent=1.2, max_z=1.3, self_radius=0.11
    ):
        """One capture -> filtered base-frame points (N, 3)."""
        depth = self.capture(n_frames)
        k = np.array(self.info.k).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        h, w = depth.shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[::stride, ::stride]
        valid = (
            (z > float(self.cfg.get("min_range", 0.12)))
            & (z < float(self.cfg.get("max_range", 1.2)))
            & np.isfinite(z)
        )
        z, uu, vv = z[valid], uu[valid], vv[valid]
        cam = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=1)
        R, t = self.camera_pose()
        pts = cam @ R.T + t
        ws = (
            (np.abs(pts[:, 0]) < xy_extent)
            & (np.abs(pts[:, 1]) < xy_extent)
            & (pts[:, 2] > min_z)
            & (pts[:, 2] < max_z)
        )
        pts = pts[ws]
        if len(pts):
            pts = pts[robot_mask(pts, self.link_points(), self_radius)]
        return pts
