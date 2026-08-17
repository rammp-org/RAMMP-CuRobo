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


class VoxelAccumulator:
    """Temporal hysteresis over a voxel grid.

    Each tick: voxels holding >= min_points_per_voxel points gain a point
    of score (capped), every other known voxel loses one. A voxel is an
    obstacle at score >= occupied_at. At 2 Hz with the defaults an object
    appears after ~1.5 s and fades ~2-3 s after it leaves — transients
    (a passing hand) never confirm. Known v1 limitation (spec): no
    free-space raycasting, so an occluded obstacle also decays.
    """

    def __init__(self, voxel=0.03, occupied_at=3, max_score=6):
        self.voxel = float(voxel)
        self.occupied_at = int(occupied_at)
        self.max_score = int(max_score)
        self._scores = {}

    def update(self, points, min_points_per_voxel=2):
        hit = set()
        if len(points):
            idx = np.floor(np.asarray(points) / self.voxel).astype(np.int64)
            uniq, counts = np.unique(idx, axis=0, return_counts=True)
            hit = set(map(tuple, uniq[counts >= int(min_points_per_voxel)]))
        for c in hit:
            self._scores[c] = min(self._scores.get(c, 0) + 1, self.max_score)
        for c in [c for c in self._scores if c not in hit]:
            s = self._scores[c] - 1
            if s <= 0:
                del self._scores[c]
            else:
                self._scores[c] = s

    def occupied_cells(self):
        cells = [c for c, s in self._scores.items() if s >= self.occupied_at]
        return np.asarray(cells, dtype=np.int64).reshape(-1, 3)


def _split_cells(cells, min_fill, min_span):
    """Recursively split a voxel component whose AABB is mostly empty.

    One box around an L-shaped complex claims huge volumes of FREE space
    (field: the first sim sweep's wall mega-box swallowed the home pose).
    Split along the longest axis until each box is reasonably full/small.
    """
    lo = cells.min(axis=0)
    hi = cells.max(axis=0) + 1
    span = hi - lo
    fill = len(cells) / int(np.prod(span))
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


def cluster_cells(cells, voxel, min_voxels=8, max_boxes=20, min_fill=0.25, min_span=4):
    """Occupied voxel cells -> connected components -> tight AABBs.

    Returns (boxes, total_found). When capped, the NEAREST boxes win — a
    small bottle inside reach matters more than the far half of a wall
    (the cap once silently dropped the bottle while keeping wall slabs).
    """
    from scipy import ndimage

    if not len(cells):
        return [], 0
    origin = cells.min(axis=0)
    idx = cells - origin
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.uint8)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    boxes = []
    for lab in range(1, n + 1):
        comp = np.argwhere(labels == lab)
        if len(comp) < min_voxels:
            continue
        for lo, hi, nvox in _split_cells(comp, min_fill, min_span):
            lo = (lo + origin) * voxel
            hi = (hi + origin) * voxel
            boxes.append(
                {
                    "center": ((lo + hi) / 2.0).tolist(),
                    "dims": (hi - lo).tolist(),
                    "voxels": int(nvox),
                }
            )

    def closest_xy(b):
        c = np.abs(np.asarray(b["center"][:2]))
        half = np.asarray(b["dims"][:2]) / 2.0
        return float(np.linalg.norm(np.maximum(c - half, 0.0)))

    boxes.sort(key=closest_xy)
    return boxes[:max_boxes], len(boxes)


def _aabb_overlap(a, b):
    """Overlap volume of two center+dims AABBs (0.0 when disjoint)."""
    ac, ad = np.asarray(a["center"]), np.asarray(a["dims"]) / 2.0
    bc, bd = np.asarray(b["center"]), np.asarray(b["dims"]) / 2.0
    lo = np.maximum(ac - ad, bc - bd)
    hi = np.minimum(ac + ad, bc + bd)
    ext = np.maximum(hi - lo, 0.0)
    return float(np.prod(ext))


class BoxTracker:
    """Stable obstacle names across ticks, matched by AABB overlap."""

    def __init__(self):
        self._prev = {}
        self._next_id = 1

    def assign(self, boxes):
        named = {}
        for b in boxes:
            best, best_ov = None, 0.0
            for name, pb in self._prev.items():
                if name in named:
                    continue
                ov = _aabb_overlap(b, pb)
                if ov > best_ov:
                    best, best_ov = name, ov
            if best is not None and best_ov > 0.0:
                named[best] = b
            else:
                named["obs_%d" % self._next_id] = b
                self._next_id += 1
        self._prev = dict(named)
        return named


def boxes_changed(a, b, tol=0.01):
    """Did the named box set change materially (names, or >tol movement)?"""
    if set(a) != set(b):
        return True
    for name in a:
        for key in ("center", "dims"):
            if (
                np.max(np.abs(np.asarray(a[name][key]) - np.asarray(b[name][key])))
                > tol
            ):
                return True
    return False
