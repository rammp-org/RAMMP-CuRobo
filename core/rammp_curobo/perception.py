"""Pure perception stages for the perceived collision world.

NO ROS imports (core policy): numpy in, plain lists/dicts out, so every
stage unit-tests offline. The deprojection, capsule self-filter, and the
cluster/split logic are revived from the field-proven 2026-08 scan
pipeline (git 0e0ec5b~1, scan_common.py); the voxel accumulator and box
tracker are new for continuous operation. scipy is imported lazily inside
the one function that needs it, keeping core import-light.
"""

import numpy as np


def quat_to_mat(x, y, z, w):
    """xyzw quaternion -> 3x3 rotation matrix."""
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def depth_to_points(depth, fx, fy, cx, cy, stride=2, min_range=0.12, max_range=1.2):
    """Depth image (metres, HxW) -> (N, 3) camera-optical-frame points."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = (z > float(min_range)) & (z < float(max_range)) & np.isfinite(z)
    z, uu, vv = z[valid], uu[valid], vv[valid]
    return np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=1)


def transform_points(points, rot, trans):
    """Apply base_T_camera (R, t) to (N, 3) points."""
    return points @ np.asarray(rot, dtype=float).T + np.asarray(trans, dtype=float)


def workspace_crop(points, xy_extent=1.2, min_z=0.03, max_z=1.3):
    """Keep points inside the reachable slab around the base."""
    keep = (
        (np.abs(points[:, 0]) < xy_extent)
        & (np.abs(points[:, 1]) < xy_extent)
        & (points[:, 2] > min_z)
        & (points[:, 2] < max_z)
    )
    return points[keep]


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


def in_box_mask(points, center, dims, inflate=0.0):
    """True where a point is inside the (inflated) axis-aligned box."""
    half = np.asarray(dims, dtype=float) / 2.0 + float(inflate)
    return np.all(np.abs(points - np.asarray(center, dtype=float)) <= half, axis=1)
