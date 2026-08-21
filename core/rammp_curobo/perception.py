"""Pure perception stages for the perceived collision world.

NO ROS imports (core policy): numpy in, plain lists/dicts out, so every
stage unit-tests offline. The deprojection, capsule self-filter, and the
cluster/split logic are revived from the field-proven 2026-08 scan
pipeline (git 0e0ec5b~1, scan_common.py); the voxel accumulator and box
tracker are new for continuous operation. scipy is imported lazily inside
the one function that needs it, keeping core import-light.
"""

import math

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


def load_self_model(path):
    """{link: (k,4) array of [x,y,z,r] in that LINK's frame} from YAML.

    The arm's own collision geometry — the same spheres cuRobo plans
    against — keyed by the TF frame each set rides on. Baked by
    scripts/bake_self_model.py; never hand-edit.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        str(link): np.asarray(rows, dtype=float).reshape(-1, 4)
        for link, rows in raw["spheres"].items()
    }


def self_model_spheres(model, link_tf):
    """Place a self-model in base_link. -> ((N,4) spheres, [missing links]).

    link_tf: {link: (R (3,3), t (3,))} base_T_link for the frames you
    could resolve (stamped at the depth frame, or the mask smears the
    moment the arm moves). Links without a transform are skipped and
    named, so a caller can decide whether a partial model is usable.
    """
    out, missing = [], []
    for link, rows in model.items():
        tf = link_tf.get(link)
        if tf is None:
            missing.append(link)
            continue
        r, t = tf
        centers = rows[:, :3] @ np.asarray(r, dtype=float).T + np.asarray(t, dtype=float)
        out.append(np.column_stack([centers, rows[:, 3]]))
    spheres = np.vstack(out) if out else np.empty((0, 4))
    return spheres, missing


def robot_mask_spheres(points, spheres, margin):
    """True where a point is NOT within (radius + margin) of any sphere.

    The sphere-model self-filter: accurate to the arm's real shape, so
    `margin` only has to absorb camera-pose error and TF/depth skew —
    not the difference between a capsule and a gripper.
    """
    pts = np.asarray(points, dtype=float)
    if not len(pts) or not len(spheres):
        return np.ones(len(pts), dtype=bool)
    c = np.asarray(spheres, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    # chunk the (N, M) distance table so a dense frame stays in cache
    for lo in range(0, len(pts), 8192):
        blk = pts[lo:lo + 8192]
        d = np.linalg.norm(blk[:, None, :] - c[None, :, :3], axis=2) - c[None, :, 3]
        keep[lo:lo + 8192] = (d > margin).all(axis=1)
    return keep


def _surface_terms(p, sph, cam):
    """Per point: signed distance to the nearest sphere surface, the unit
    normal there, and whether that surface faces the camera."""
    diff = p[:, None, :] - sph[None, :, :3]
    dist = np.linalg.norm(diff, axis=2)
    surf = dist - sph[None, :, 3]
    j = np.argmin(np.abs(surf), axis=1)
    rows = np.arange(len(p))
    d = surf[rows, j]
    r = dist[rows, j]
    n = diff[rows, j] / np.maximum(r, 1e-9)[:, None]
    facing = np.ones(len(p), dtype=bool) if cam is None else (
        (n * (cam[None, :] - sph[j, :3])).sum(axis=1) > 0.0)
    return d, n, facing & (r > 1e-4)


def register_points_to_spheres(points, spheres, cam_origin=None, search=0.15,
                               gate=0.15, iters=12, trim=0.2, min_points=200,
                               inlier=0.02):
    """The translation that lands a point cloud on a sphere model.

    -> (shift (3,), n_used, rms_m) or None when too few points sit near
    the model. Add `shift` to the points — and to the camera's mount
    position — to correct a camera-pose TRANSLATION error, using the arm
    as the calibration object: it is the one thing in view whose true
    pose is known exactly.

    Two stages, because the errors this must swallow (7-9 cm on the
    bench) exceed the arm's own radius, which is all a local fit can
    capture: an error that pushes the drawn arm INTO its model leaves
    the points sitting on the far side of the spheres, a perfectly good
    local minimum.

      1. Coarse: slide the cloud over a grid of +-search, scoring each
         candidate by how many points land within `inlier` of a sphere
         surface that FACES the camera — a depth point can only come
         from the camera-facing hemisphere, so the far-surface solution
         scores nothing.
      2. Fine: point-to-plane ICP (translation only) from the winner,
         same facing test, gate shrinking as it converges so obstacles
         near the arm drop out of the fit instead of biasing it.

    The camera origin moves WITH the shift (it is the mount that is
    wrong), and the facing test is evaluated that way. Unbiased where
    the earlier box-centroid estimate was not: that one saw only the
    fringe of the displaced arm that escaped the self-filter.
    """
    pts = np.asarray(points, dtype=float)
    sph = np.asarray(spheres, dtype=float)
    if pts.ndim != 2 or not len(pts) or not len(sph):
        return None
    cam = None if cam_origin is None else np.asarray(cam_origin, dtype=float)

    # --- coarse grid, on a subsample, two resolutions
    rng = np.random.default_rng(0)
    sub = pts if len(pts) <= 1500 else pts[rng.choice(len(pts), 1500, replace=False)]
    best = np.zeros(3)
    for span, step in ((search, 0.03), (0.045, 0.01)):
        axis = np.arange(-span, span + 1e-9, step)
        cands = best + np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        scores = np.empty(len(cands))
        for k, t in enumerate(cands):
            d, _n, ok = _surface_terms(sub + t, sph, None if cam is None else cam + t)
            hit = ok & (np.abs(d) < inlier)
            scores[k] = hit.sum() - 0.25 * np.abs(d[ok]).clip(0, inlier).sum() / inlier
        best = cands[int(np.argmax(scores))]

    # --- fine: ICP from the coarse winner
    shift = best.copy()
    used, rms, g = 0, float("inf"), float(gate)
    for _ in range(int(iters)):
        p = pts + shift
        d, n, ok = _surface_terms(p, sph, None if cam is None else cam + shift)
        sel = ok & (np.abs(d) < g)
        used = int(sel.sum())
        if used < min_points:
            return None
        ds, ns = d[sel], n[sel]
        if trim > 0:
            keep = np.abs(ds) <= np.quantile(np.abs(ds), 1.0 - trim)
            ds, ns = ds[keep], ns[keep]
        a = ns.T @ ns + 1e-9 * np.eye(3)
        step = -np.linalg.solve(a, ns.T @ ds)
        shift += step
        rms = float(np.sqrt(np.mean(ds ** 2)))
        if np.linalg.norm(step) < g / 3.0:
            g = max(g * 0.6, 0.03)
        if np.linalg.norm(step) < 1e-4:
            break
    return shift, used, rms


class SelfRegistrar:
    """Solve a fixed camera's translation error off the arm, once, from a
    few still frames — then get out of the way.

    feed() one frame at a time with the frame's cropped, UNMASKED points
    and the arm's spheres at that frame's stamp. Frames while the arm is
    moving restart the buffer (a moving target smears the fit). When
    `frames` still frames are in hand it registers their union and
    returns an outcome dict; the gates make a bad fit report itself
    rather than shift the world: enough arm points, a tight residual, a
    plausible magnitude. After `attempts` failed solves it gives up.
    """

    def __init__(self, frames=8, still_m=0.01, min_points=300, max_rms=0.02,
                 max_shift=0.20, attempts=3):
        self.frames = int(frames)
        self.still_m = float(still_m)
        self.min_points = int(min_points)
        self.max_rms = float(max_rms)
        self.max_shift = float(max_shift)
        self.attempts = int(attempts)
        self._buf = []
        self._last = None
        self._tries = 0
        self.done = False
        self.result = None

    def feed(self, points, spheres, cam_origin=None):
        if self.done:
            return None
        sph = np.asarray(spheres, dtype=float)
        if self._last is not None and (
            sph.shape != self._last.shape
            or np.abs(sph[:, :3] - self._last[:, :3]).max() > self.still_m
        ):
            self._buf = []                      # the arm moved: start over
        self._last = sph
        self._buf.append(np.asarray(points, dtype=float))
        if len(self._buf) < self.frames:
            return None
        pts = np.vstack([b for b in self._buf if len(b)]) if self._buf else np.empty((0, 3))
        self._buf = []
        self._tries += 1
        out = register_points_to_spheres(pts, sph, cam_origin=cam_origin,
                                         min_points=self.min_points)
        if out is None:
            reason = "too few points near the arm (is it in view?)"
            shift, used, rms = None, 0, float("inf")
        else:
            shift, used, rms = out
            mag = float(np.linalg.norm(shift))
            reason = ("" if rms <= self.max_rms and mag <= self.max_shift else
                      "fit rejected: rms %.3f m (max %.3f), shift %.3f m (max %.3f)"
                      % (rms, self.max_rms, mag, self.max_shift))
        ok = not reason
        if ok or self._tries >= self.attempts:
            self.done = True
        self.result = {"ok": ok, "shift": shift, "used": used, "rms": rms,
                       "reason": reason, "tries": self._tries, "final": self.done}
        return self.result


def in_box_mask(points, center, dims, inflate=0.0):
    """True where a point is inside the (inflated) axis-aligned box."""
    half = np.asarray(dims, dtype=float) / 2.0 + float(inflate)
    return np.all(np.abs(points - np.asarray(center, dtype=float)) <= half, axis=1)


class VoxelAccumulator:
    """Temporal hysteresis over a voxel grid.

    Each tick: voxels holding >= min_points_per_voxel points gain a point
    of score (capped), decaying voxels lose one. A voxel is an obstacle
    at score >= occupied_at. At 2 Hz with the defaults an object appears
    after ~1.5 s and fades ~2-3 s once provably gone — transients (a
    passing hand) never confirm. Which voxels MAY decay is the caller's
    choice via update(decay_cells=...): pass visible_free_cells() output
    to scope forgetting to what the camera saw through (wrist-camera
    mode — occluded/out-of-view voxels are REMEMBERED), or None to decay
    everything unseen each tick.
    """

    def __init__(self, voxel=0.03, occupied_at=3, max_score=6):
        self.voxel = float(voxel)
        self.occupied_at = int(occupied_at)
        self.max_score = int(max_score)
        self._scores = {}

    def update(self, points, min_points_per_voxel=2, decay_cells=None):
        hit = set()
        if len(points):
            idx = np.floor(np.asarray(points) / self.voxel).astype(np.int64)
            uniq, counts = np.unique(idx, axis=0, return_counts=True)
            hit = set(map(tuple, uniq[counts >= int(min_points_per_voxel)]))
        for c in hit:
            self._scores[c] = min(self._scores.get(c, 0) + 1, self.max_score)
        # decay_cells scopes forgetting for narrow-FOV / moving cameras:
        # only voxels the camera PROVABLY saw through may lose confidence
        # (None = legacy fixed-camera behavior: decay everything unseen).
        for c in [c for c in self._scores if c not in hit]:
            if decay_cells is not None and c not in decay_cells:
                continue
            s = self._scores[c] - 1
            if s <= 0:
                del self._scores[c]
            else:
                self._scores[c] = s

    def known_cells(self):
        """Every voxel with ANY confidence (not just confirmed obstacles) —
        the candidate set for visibility-scoped decay."""
        return np.asarray(list(self._scores.keys()), dtype=np.int64).reshape(-1, 3)

    def clear_box(self, center, dims, inflate=0.0):
        """Drop all confidence inside an axis-aligned box (base frame).

        Needed because frustum-scoped decay can NEVER erase an object that
        is still physically present — the ignore region must purge the
        grasp target's already-accumulated voxels, not just mask new hits.
        Returns the number of voxels cleared.
        """
        half = np.asarray(dims, dtype=float) / 2.0 + float(inflate)
        c = np.asarray(center, dtype=float)
        gone = [
            cell
            for cell in self._scores
            if np.all(np.abs((np.asarray(cell) + 0.5) * self.voxel - c) <= half)
        ]
        for cell in gone:
            del self._scores[cell]
        return len(gone)

    def reset(self):
        """Forget everything (e.g. when the arm self-filter first arms —
        voxels accumulated from the unfiltered arm can otherwise never
        decay: the arm still occupies them, so no camera can ever see
        through their location)."""
        n = len(self._scores)
        self._scores.clear()
        return n

    def occupied_cells(self):
        cells = [c for c, s in self._scores.items() if s >= self.occupied_at]
        return np.asarray(cells, dtype=np.int64).reshape(-1, 3)


def visible_free_cells(
    cells,
    voxel,
    rot,
    trans,
    depth,
    fx,
    fy,
    cx,
    cy,
    min_range=0.07,
    max_range=0.9,
    margin=0.015,
):
    """Voxel cells this camera frame proves EMPTY (safe to decay).

    A cell is decayable only when the camera saw THROUGH it: it projects
    inside the frame, its camera-frame depth is within sensor range, the
    measured pixel is valid, and the measured surface lies BEHIND the
    cell's FAR EDGE (center + the voxel's half-diagonal) by at least
    `margin` of sensor noise. Testing the far edge — not a fat fixed
    margin on the center — is what lets the bottom voxel layer of a
    removed tabletop object be proven empty under a top-down look-back
    (audit 2026-08-18: a flat 5 cm margin left a permanent phantom slab
    at every vacated spot). Occluded, out-of-view, out-of-range, and
    depth-hole cells are all 'unknown' — never decayed. rot/trans are
    base_T_camera (the same convention transform_points uses).
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if not len(cells):
        return set()
    clearance = 0.5 * math.sqrt(3.0) * float(voxel) + float(margin)
    centers = (cells + 0.5) * float(voxel)
    r = np.asarray(rot, dtype=float)
    p_cam = (centers - np.asarray(trans, dtype=float)) @ r  # R.T @ (p - t)
    z = p_cam[:, 2]
    h, w = depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 0, fx * p_cam[:, 0] / np.where(z > 0, z, 1.0) + cx, -1.0)
        v = np.where(z > 0, fy * p_cam[:, 1] / np.where(z > 0, z, 1.0) + cy, -1.0)
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    in_view = (
        (z > float(min_range))
        & (z < float(max_range))
        & (ui >= 0)
        & (ui < w)
        & (vi >= 0)
        & (vi < h)
    )
    free = np.zeros(len(cells), dtype=bool)
    if in_view.any():
        d = depth[vi[in_view], ui[in_view]]
        valid = (d > 0.05) & np.isfinite(d)
        free_in_view = valid & (z[in_view] < d - clearance)
        free[np.flatnonzero(in_view)[free_in_view]] = True
    return set(map(tuple, cells[free]))


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
