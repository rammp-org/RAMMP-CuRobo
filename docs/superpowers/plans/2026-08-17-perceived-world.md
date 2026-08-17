# Perceived Collision World (cameras node) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The planner's collision world continuously reflects what the bench
cameras see: a new `cameras` node turns Orbbec (and later wrist-D405) depth
into named cuboid obstacles and streams them to the planner at ~2 Hz.

**Architecture:** Pure numpy/scipy perception stages live in
`core/rammp_curobo/perception.py` (NO ROS — offline-testable). A new ROS
node `cameras` (rammp_curobo_ros) does I/O: depth/camera_info in, TF for
extrinsics and arm self-filtering, `UpdateWorldBoxes` service calls out,
RViz markers for truth. The planner stays sole owner of the cuRobo world;
it gains one service + one small core method. One attended calibration
script recovers the Orbbec→base_link extrinsic via ArUco hand-eye.

**Tech Stack:** Python 3.10, numpy 1.26, scipy 1.15 (installed), OpenCV
4.11 with aruco (installed), ROS 2 Humble, cuRobo v0.7.8 (PINNED — cuboid
worlds only, `collision_cache_obb=60`).

**Spec:** `docs/superpowers/specs/2026-08-17-perceived-world-design.md`

## Global Constraints

- cuRobo stays **v0.7.8**; world = axis-aligned cuboids only; never send an
  empty world (baseline merge guarantees this); stay under
  `collision_cache_obb=60`.
- `core/` keeps **NO ROS imports** (perception.py is numpy/scipy only;
  scipy imported lazily inside functions like the old code did).
- All angle comparisons wrap-aware; quaternions xyzw in ROS, wxyz in cuRobo.
- Camera subscribers use **sensor-data QoS** (best-effort).
- No execution-gate changes. The cameras node never commands motion.
- Style: Ruff v0.3.0 defaults, PEP 8. Tests: repo-root `pytest.ini`
  (`-p no:anyio`); run `python3 -m pytest core/tests -q` and
  `python3 -m pytest rammp_curobo_ros/test -q -p no:anyio`.
- Work on branch `feat/perceived-world`. Commit after every task.
- Anything touching the real arm/camera bench is ATTENDED — the human runs
  it. Tasks below end at "ready for the attended run" with the exact
  commands documented.

---

### Task 1: New service definitions

**Files:**
- Create: `rammp_curobo_interfaces/srv/UpdateWorldBoxes.srv`
- Create: `rammp_curobo_interfaces/srv/SetIgnoreRegion.srv`
- Modify: `rammp_curobo_interfaces/CMakeLists.txt` (srv list, after line 18)

**Interfaces:**
- Produces: `rammp_curobo_interfaces/srv/UpdateWorldBoxes`
  (`names: string[]`, `centers: geometry_msgs/Point[]`,
  `dims: geometry_msgs/Vector3[]`, `baseline: string` →
  `success: bool`, `message: string`) and
  `rammp_curobo_interfaces/srv/SetIgnoreRegion`
  (`center: geometry_msgs/Point`, `dims: geometry_msgs/Vector3` →
  `success: bool`, `message: string`). Tasks 5 and 6 import these.

- [ ] **Step 1: Write the two srv files**

`rammp_curobo_interfaces/srv/UpdateWorldBoxes.srv`:

```
# Replace the planner's PERCEIVED obstacles with these axis-aligned boxes
# (base_link). They are merged on top of the static baseline world, so an
# empty list is legal (v0.7.8's empty-world trap never triggers). Boxes
# REPLACE the previous perceived set — they never accumulate.
string[] names                # stable IDs from the perception tracker
geometry_msgs/Point[] centers # box centers, base_link (m)
geometry_msgs/Vector3[] dims  # full extents (m)
# Baseline world YAML (name or path) to merge under. "" = keep current
# baseline (initial world at node start, or the last non-empty value).
string baseline
---
bool success
string message                # includes box count vs the obb cache cap
```

`rammp_curobo_interfaces/srv/SetIgnoreRegion.srv`:

```
# Axis-aligned box in base_link: depth points inside it are dropped before
# clustering (the manipulation target must not become an obstacle).
# All-zero dims clears the region.
geometry_msgs/Point center
geometry_msgs/Vector3 dims
---
bool success
string message
```

- [ ] **Step 2: Register them in CMakeLists.txt**

In `rammp_curobo_interfaces/CMakeLists.txt`, the `rosidl_generate_interfaces`
list currently ends with `"srv/SetWorld.srv"`. Add below it:

```cmake
  "srv/UpdateWorldBoxes.srv"
  "srv/SetIgnoreRegion.srv"
```

- [ ] **Step 3: Build and verify the types exist**

```bash
cd ~/RAMMP-CuRobo && source /opt/ros/humble/setup.zsh && \
colcon build --symlink-install --packages-select rammp_curobo_interfaces && \
source install/setup.zsh && \
python3 -c "from rammp_curobo_interfaces.srv import UpdateWorldBoxes, SetIgnoreRegion; print('srvs OK')"
```

Expected: `srvs OK`.

- [ ] **Step 4: Commit**

```bash
git add rammp_curobo_interfaces && git commit -m "interfaces: UpdateWorldBoxes + SetIgnoreRegion srvs for the perceived world"
```

---

### Task 2: core perception — geometry stages (revived, pure)

**Files:**
- Create: `core/rammp_curobo/perception.py`
- Test: `core/tests/test_perception.py`

**Interfaces:**
- Produces (exact signatures Tasks 3/6/7 use):
  - `quat_to_mat(x, y, z, w) -> np.ndarray (3,3)`
  - `depth_to_points(depth, fx, fy, cx, cy, stride=2, min_range=0.12, max_range=1.2) -> (N,3) camera-frame`
  - `transform_points(points, rot, trans) -> (N,3)`
  - `workspace_crop(points, xy_extent=1.2, min_z=0.03, max_z=1.3) -> (M,3)`
  - `robot_mask(points, link_pts, radius) -> bool (N,)` (True = KEEP)
  - `in_box_mask(points, center, dims, inflate=0.0) -> bool (N,)` (True = inside)

- [ ] **Step 1: Write the failing tests**

`core/tests/test_perception.py`:

```python
"""Offline tests for the perceived-world pipeline (no ROS, no GPU)."""

import numpy as np

from rammp_curobo.perception import (
    depth_to_points,
    in_box_mask,
    quat_to_mat,
    robot_mask,
    transform_points,
    workspace_crop,
)


def test_quat_to_mat_identity_and_z90():
    assert np.allclose(quat_to_mat(0, 0, 0, 1), np.eye(3))
    r = quat_to_mat(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # Rz(90°)
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_depth_to_points_center_pixel_lands_on_axis():
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    pts = depth_to_points(depth, fx=100.0, fy=100.0, cx=4.0, cy=4.0, stride=1)
    # the pixel at (u=4, v=4) deprojects to (0, 0, 0.5)
    assert any(np.allclose(p, [0.0, 0.0, 0.5], atol=1e-6) for p in pts)
    assert pts.shape[1] == 3


def test_depth_to_points_range_clip():
    depth = np.array([[0.05, 0.5, 5.0]], dtype=np.float32)
    pts = depth_to_points(depth, 10, 10, 1, 0, stride=1, min_range=0.1, max_range=1.0)
    assert len(pts) == 1 and np.isclose(pts[0][2], 0.5)


def test_transform_and_crop():
    pts = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 0.5]])
    moved = transform_points(pts, np.eye(3), [0.1, 0.0, 0.0])
    assert np.allclose(moved[0], [0.1, 0.0, 1.0])
    kept = workspace_crop(moved, xy_extent=1.2, min_z=0.03, max_z=1.3)
    assert len(kept) == 1  # the x=5.1 point is cropped


def test_robot_mask_removes_points_near_the_arm():
    link_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    pts = np.array([[0.05, 0.0, 0.5], [0.5, 0.0, 0.5]])
    keep = robot_mask(pts, link_pts, radius=0.11)
    assert list(keep) == [False, True]


def test_in_box_mask_with_inflation():
    pts = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.2, 0.0, 0.0]])
    inside = in_box_mask(pts, center=[0, 0, 0], dims=[0.1, 0.1, 0.1])
    assert list(inside) == [True, False, False]
    inside = in_box_mask(pts, [0, 0, 0], [0.1, 0.1, 0.1], inflate=0.02)
    assert list(inside) == [True, True, False]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/RAMMP-CuRobo && python3 -m pytest core/tests/test_perception.py -q
```

Expected: collection error `No module named 'rammp_curobo.perception'`.

- [ ] **Step 3: Write the module**

`core/rammp_curobo/perception.py` (header + these functions; the
accumulator/cluster/tracker come in Task 3):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest core/tests/test_perception.py -q
```

Expected: 6 passed. Also run `python3 -m pytest core/tests -q` — no
regressions (GPU smokes may take ~20 s warm).

- [ ] **Step 5: Commit**

```bash
git add core/rammp_curobo/perception.py core/tests/test_perception.py && \
git commit -m "core: revive pure perception stages (deproject, self-filter, crop)"
```

---

### Task 3: core perception — accumulator, clustering, tracker

**Files:**
- Modify: `core/rammp_curobo/perception.py` (append)
- Test: `core/tests/test_perception.py` (append)

**Interfaces:**
- Produces (Task 6 uses exactly these):
  - `VoxelAccumulator(voxel=0.03, occupied_at=3, max_score=6)` with
    `.update(points, min_points_per_voxel=2)` and
    `.occupied_cells() -> (K,3) int64`
  - `cluster_cells(cells, voxel, min_voxels=8, max_boxes=20, min_fill=0.25, min_span=4) -> (boxes, total)`
    where each box is `{"center": [x,y,z], "dims": [dx,dy,dz], "voxels": int}`
  - `BoxTracker()` with `.assign(boxes) -> dict name->box` (stable names)
  - `boxes_changed(a, b, tol=0.01) -> bool` for send-on-change

- [ ] **Step 1: Write the failing tests (append to test_perception.py)**

```python
from rammp_curobo.perception import (  # noqa: E402
    BoxTracker,
    VoxelAccumulator,
    boxes_changed,
    cluster_cells,
)


def _cube_points(center, side, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-side / 2, side / 2, size=(n, 3))


def test_accumulator_hysteresis_appear_and_decay():
    acc = VoxelAccumulator(voxel=0.05, occupied_at=3, max_score=6)
    pts = _cube_points([0.5, 0.0, 0.1], 0.1)
    acc.update(pts)
    acc.update(pts)
    assert len(acc.occupied_cells()) == 0  # 2 hits < occupied_at
    acc.update(pts)
    assert len(acc.occupied_cells()) > 0  # confirmed after 3
    for _ in range(3):
        acc.update(np.empty((0, 3)))
    assert len(acc.occupied_cells()) == 0  # decayed away


def test_accumulator_noise_needs_min_points():
    acc = VoxelAccumulator(voxel=0.05, occupied_at=1)
    lone = np.array([[0.5, 0.5, 0.5]])  # 1 point < min_points_per_voxel=2
    acc.update(lone)
    assert len(acc.occupied_cells()) == 0


def test_cluster_cells_two_separated_boxes():
    a = np.argwhere(np.ones((4, 4, 4)))  # 0..3 cube
    b = a + np.array([20, 0, 0])
    boxes, total = cluster_cells(np.vstack([a, b]), voxel=0.05, min_voxels=8)
    assert total == 2 and len(boxes) == 2
    dims = boxes[0]["dims"]
    assert np.allclose(dims, [0.2, 0.2, 0.2], atol=1e-9)


def test_cluster_cells_cap_keeps_nearest():
    near = np.argwhere(np.ones((3, 3, 3)))  # near origin
    far = near + np.array([40, 0, 0])
    boxes, total = cluster_cells(
        np.vstack([near, far]), voxel=0.05, min_voxels=8, max_boxes=1
    )
    assert total == 2 and len(boxes) == 1
    assert boxes[0]["center"][0] < 1.0  # the near one survived


def test_tracker_names_are_stable():
    t = BoxTracker()
    b1 = {"center": [0.5, 0.0, 0.1], "dims": [0.2, 0.2, 0.2], "voxels": 8}
    named = t.assign([b1])
    (name,) = named.keys()
    b1_moved = {"center": [0.51, 0.0, 0.1], "dims": [0.2, 0.2, 0.2], "voxels": 8}
    named2 = t.assign([b1_moved])
    assert list(named2.keys()) == [name]  # same ID after a 1 cm drift
    named3 = t.assign(
        [b1_moved, {"center": [-0.5, 0.3, 0.2], "dims": [0.1, 0.1, 0.1], "voxels": 8}]
    )
    assert name in named3 and len(named3) == 2


def test_boxes_changed_thresholds():
    a = {"obs_1": {"center": [0.5, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    same = {"obs_1": {"center": [0.505, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    moved = {"obs_1": {"center": [0.55, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    assert not boxes_changed(a, same, tol=0.01)
    assert boxes_changed(a, moved, tol=0.01)
    assert boxes_changed(a, {}, tol=0.01)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest core/tests/test_perception.py -q
```

Expected: ImportError on the new names.

- [ ] **Step 3: Append the implementation to perception.py**

```python
class VoxelAccumulator:
    """Temporal hysteresis over a voxel grid.

    Each tick: voxels holding >= min_points_per_voxel points gain a point of
    score (capped), every other known voxel loses one. A voxel is an
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
            if np.max(np.abs(np.asarray(a[name][key]) - np.asarray(b[name][key]))) > tol:
                return True
    return False
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest core/tests/test_perception.py -q
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add core/rammp_curobo/perception.py core/tests/test_perception.py && \
git commit -m "core: voxel hysteresis accumulator, AABB clustering, stable box tracker"
```

---

### Task 4: core planner — `update_world_boxes` with baseline merge

**Files:**
- Modify: `core/rammp_curobo/scene.py` (add `merged_scene`, after `scene_from_obstacles`)
- Modify: `core/rammp_curobo/planner.py` (import + new method after `update_world`, ~line 300)
- Test: `core/tests/test_perception.py` (append; offline — targets the merge, not the GPU)

**Interfaces:**
- Consumes: `Scene`, `SceneObject`, `load_scene`, `resolve_config` (existing).
- Produces: `merged_scene(baseline_scene, boxes) -> Scene` (scene.py) and
  `Planner.update_world_boxes(boxes, baseline=None)` where `boxes` is a
  list of `{"name": str, "position": [x,y,z], "dims": [dx,dy,dz]}`.
  Task 5's service handler calls the Planner method.

- [ ] **Step 1: Write the failing test (append to test_perception.py)**

```python
def test_merged_scene_keeps_baseline_and_replaces_objects():
    from rammp_curobo.scene import Obstacle, Scene, merged_scene

    base = Scene(
        base_frame="base_link",
        obstacles=[
            Obstacle({"name": "table", "position": [0.5, 0, -0.04], "dims": [1, 1, 0.08]})
        ],
        targets=[],
        objects=[],
    )
    boxes = [{"name": "obs_1", "position": [0.5, 0.2, 0.1], "dims": [0.1, 0.1, 0.2]}]
    m1 = merged_scene(base, boxes)
    assert [o.name for o in m1.obstacles] == ["table"]
    assert [o.name for o in m1.objects] == ["obs_1"]
    # a second merge REPLACES the perceived set — no accumulation
    m2 = merged_scene(base, [])
    assert m2.objects == [] and [o.name for o in m2.obstacles] == ["table"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest core/tests/test_perception.py::test_merged_scene_keeps_baseline_and_replaces_objects -q
```

Expected: ImportError `merged_scene`.

- [ ] **Step 3: Implement**

Append to `core/rammp_curobo/scene.py`:

```python
def merged_scene(baseline, boxes):
    """The baseline scene with its props REPLACED by perceived boxes.

    `boxes`: dicts with name + position + dims (axis-aligned, base frame).
    The baseline's obstacles/targets pass through untouched; perceived
    boxes never accumulate across calls because they land in `objects`
    wholesale each time.
    """
    return Scene(
        base_frame=baseline.base_frame,
        obstacles=baseline.obstacles,
        targets=baseline.targets,
        objects=[SceneObject(dict(b)) for b in boxes],
    )
```

In `core/rammp_curobo/planner.py`: extend the scene import to
`from rammp_curobo.scene import Scene, load_scene, merged_scene, scene_from_obstacles`
and add after `update_world`:

```python
    def update_world_boxes(self, boxes, baseline=None):
        """Continuous-perception entry: merge perceived AABBs onto the
        static baseline and swap the collision world.

        baseline: world YAML name/path to (re)load as the static scene;
        None keeps the current one (the initial world until a baseline is
        ever named). Callers serialize with planning themselves (the ROS
        node holds its plan lock around this).
        """
        if baseline:
            self._baseline_scene = load_scene(
                resolve_config(baseline, relative_to=self._config_dir)
            )
        if getattr(self, "_baseline_scene", None) is None:
            self._baseline_scene = self._scene
        self.update_world(merged_scene(self._baseline_scene, boxes))
```

(`resolve_config` is already imported by planner.py for `update_world`.
Verify with `grep -n resolve_config core/rammp_curobo/planner.py` — if it
is imported inside the method, follow that existing pattern.)

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest core/tests/test_perception.py -q && python3 -m pytest core/tests -q
```

Expected: all pass (GPU smokes included — `update_world` itself is already
covered by `test_update_world_guards_and_round_trip`).

- [ ] **Step 5: Commit**

```bash
git add core/rammp_curobo/scene.py core/rammp_curobo/planner.py core/tests/test_perception.py && \
git commit -m "core: update_world_boxes — perceived AABBs merged onto a sticky baseline"
```

---

### Task 5: planner_node — UpdateWorldBoxes service + plan-lock timeout

**Files:**
- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`
- Test: `rammp_curobo_ros/test/test_world_boxes_handler.py`

**Interfaces:**
- Consumes: `UpdateWorldBoxes` (Task 1), `Planner.update_world_boxes` (Task 4).
- Produces: ROS service `/rammp_curobo/update_world_boxes`. The `_plan()`
  lock acquire becomes `acquire(timeout=0.5)` (was `blocking=False`).

- [ ] **Step 1: Write the failing test**

`rammp_curobo_ros/test/test_world_boxes_handler.py` (mirror the existing
`test_executor_gates.py` style — pure handler-level, no spinning):

```python
"""UpdateWorldBoxes handler: validation + planner call, no ROS spin."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from rammp_curobo_interfaces.srv import UpdateWorldBoxes

from rammp_curobo_ros.planner_node import RammpCuroboNode


def _bare_node():
    """A node instance with only what the handler touches."""
    node = object.__new__(RammpCuroboNode)  # no __init__ — no ROS context
    node._plan_lock = threading.Lock()
    node.planner = MagicMock()
    return node


def _pt(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def test_length_mismatch_is_refused():
    node = _bare_node()
    req = UpdateWorldBoxes.Request()
    req.names = ["a"]
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert not res.success and "length" in res.message
    node.planner.update_world_boxes.assert_not_called()


def test_boxes_are_converted_and_baseline_forwarded():
    node = _bare_node()
    req = UpdateWorldBoxes.Request()
    req.names = ["obs_1"]
    req.centers = [_pt(0.5, 0.0, 0.1)]
    req.dims = [_pt(0.1, 0.2, 0.3)]
    req.baseline = "world_real_bench.yaml"
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert res.success
    (boxes,), kwargs = node.planner.update_world_boxes.call_args
    assert boxes == [
        {"name": "obs_1", "position": [0.5, 0.0, 0.1], "dims": [0.1, 0.2, 0.3]}
    ]
    assert kwargs == {"baseline": "world_real_bench.yaml"}


def test_planner_exception_reports_failure():
    node = _bare_node()
    node.planner.update_world_boxes.side_effect = ValueError("21 boxes > cache")
    req = UpdateWorldBoxes.Request()
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert not res.success and "cache" in res.message
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd ~/RAMMP-CuRobo && source install/setup.zsh && \
python3 -m pytest rammp_curobo_ros/test/test_world_boxes_handler.py -q -p no:anyio
```

Expected: AttributeError `_update_world_boxes_cb`.

- [ ] **Step 3: Implement in planner_node.py**

1. Import: change `from rammp_curobo_interfaces.srv import SetWorld` to
   also import `UpdateWorldBoxes`.
2. Next to the existing `~/set_world` service creation (~line 187), add:

```python
        self.create_service(
            UpdateWorldBoxes,
            "~/update_world_boxes",
            self._update_world_boxes_cb,
            callback_group=self._cb,
        )
```

3. Add the handler next to `_set_world_cb`:

```python
    def _update_world_boxes_cb(self, request, response):
        n = len(request.names)
        if len(request.centers) != n or len(request.dims) != n:
            response.success = False
            response.message = "names/centers/dims length mismatch"
            return response
        boxes = [
            {
                "name": request.names[i],
                "position": [request.centers[i].x, request.centers[i].y, request.centers[i].z],
                "dims": [request.dims[i].x, request.dims[i].y, request.dims[i].z],
            }
            for i in range(n)
        ]
        with self._plan_lock:
            try:
                self.planner.update_world_boxes(
                    boxes, baseline=request.baseline or None
                )
                response.success = True
                response.message = "world: baseline + %d perceived boxes" % n
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response
```

4. In `_plan()` (~line 297), change

```python
        if not self._plan_lock.acquire(blocking=False):
            return None
```

to

```python
        # A 2 Hz world updater briefly holds this lock; wait a beat instead
        # of bouncing the plan (concurrent PLANS still refuse — one at a time).
        if not self._plan_lock.acquire(timeout=0.5):
            return None
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio
```

Expected: all pass (new file + existing executor gates).

- [ ] **Step 5: Commit**

```bash
git add rammp_curobo_ros/rammp_curobo_ros/planner_node.py rammp_curobo_ros/test/test_world_boxes_handler.py && \
git commit -m "planner node: UpdateWorldBoxes service + tolerate the 2 Hz updater on the plan lock"
```

---

### Task 6: the cameras node

**Files:**
- Create: `rammp_curobo_ros/rammp_curobo_ros/cameras.py`
- Create: `rammp_curobo_ros/config/camera_d405_wrist.yaml` (recovered:
  `git show 0e0ec5b~1:rammp_curobo_ros/config/camera_d405_wrist.yaml`,
  already staged in the scratchpad `recovered/` dir)
- Modify: `rammp_curobo_ros/setup.py` (config data_files + entry point)
- Test: `rammp_curobo_ros/test/test_cameras_tick.py`

**Interfaces:**
- Consumes: everything from Tasks 2/3 (`depth_to_points`,
  `transform_points`, `workspace_crop`, `robot_mask`, `in_box_mask`,
  `quat_to_mat`, `VoxelAccumulator`, `cluster_cells`, `BoxTracker`,
  `boxes_changed`), `SetIgnoreRegion`/`UpdateWorldBoxes` srvs, core
  `load_scene`/`resolve_config` for the baseline strip.
- Produces: console script `cameras`; service `/cameras/set_ignore_region`;
  topic `/cameras/world_markers` (MarkerArray); client of
  `/rammp_curobo/update_world_boxes`.
- The testable unit: `process_camera_points(...)` — a module-level pure
  function so the tick logic tests without ROS.

- [ ] **Step 1: Write the failing test**

`rammp_curobo_ros/test/test_cameras_tick.py`:

```python
"""The cameras node's per-tick point processing, no ROS objects."""

import numpy as np

from rammp_curobo_ros.cameras import process_camera_points


def test_process_filters_ignore_region_and_baseline():
    # a 10x10 flat depth wall 0.5 m out, camera looking down +z of base
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    intr = dict(fx=20.0, fy=20.0, cx=5.0, cy=5.0)
    rot, trans = np.eye(3), np.zeros(3)
    baseline_boxes = [{"position": [0.0, 0.0, 0.44], "dims": [2.0, 2.0, 0.1]}]
    pts = process_camera_points(
        depth,
        intr,
        rot,
        trans,
        stride=1,
        min_range=0.1,
        max_range=1.0,
        xy_extent=1.2,
        min_z=0.03,
        max_z=1.3,
        link_pts=None,
        self_radius=0.11,
        ignore_region=None,
        baseline_boxes=baseline_boxes,
    )
    assert len(pts) == 0  # everything sat on the baseline "table"
    pts = process_camera_points(
        depth, intr, rot, trans,
        stride=1, min_range=0.1, max_range=1.0,
        xy_extent=1.2, min_z=0.03, max_z=1.3,
        link_pts=None, self_radius=0.11,
        ignore_region=None, baseline_boxes=[],
    )
    assert len(pts) > 0  # without the baseline the wall survives
    region = {"center": [0.0, 0.0, 0.5], "dims": [5.0, 5.0, 5.0]}
    pts = process_camera_points(
        depth, intr, rot, trans,
        stride=1, min_range=0.1, max_range=1.0,
        xy_extent=1.2, min_z=0.03, max_z=1.3,
        link_pts=None, self_radius=0.11,
        ignore_region=region, baseline_boxes=[],
    )
    assert len(pts) == 0  # the ignore region swallowed it
```

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest rammp_curobo_ros/test/test_cameras_tick.py -q -p no:anyio
```

Expected: ImportError (`rammp_curobo_ros.cameras` missing).

- [ ] **Step 3: Write cameras.py**

`rammp_curobo_ros/rammp_curobo_ros/cameras.py`:

```python
"""cameras — continuous perceived-obstacle world for the planner.

Subscribes depth + camera_info for each configured camera (sensor-data
QoS), runs the pure perception pipeline (core.rammp_curobo.perception) at
`rate_hz`, and replaces the planner's perceived boxes via
/rammp_curobo/update_world_boxes. Publishes ~/world_markers so RViz shows
exactly what the planner believes. Never commands motion.

    ros2 run rammp_curobo_ros cameras
    ros2 run rammp_curobo_ros cameras --ros-args -p cameras:="['camera_orbbec_bench.yaml']"

Camera YAML schema (same as the 2026-08 scan pipeline): depth_topic,
info_topic, min_range, max_range, and EITHER tf_frame (optical frame in
TF) OR parent_frame + mount_xyz + mount_quat_xyzw (fixed mount; the
calibration script writes this form for the Orbbec).
"""

import os
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rammp_curobo.config import resolve_config
from rammp_curobo.perception import (
    BoxTracker,
    VoxelAccumulator,
    boxes_changed,
    cluster_cells,
    depth_to_points,
    in_box_mask,
    quat_to_mat,
    robot_mask,
    transform_points,
    workspace_crop,
)
from rammp_curobo.scene import load_scene
from rammp_curobo_interfaces.srv import SetIgnoreRegion, UpdateWorldBoxes

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


def load_camera_config(name_or_path):
    """Resolve a camera YAML by path or packaged name (config/ dir)."""
    candidates = [os.path.expanduser(name_or_path)]
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
    sys.exit(
        "camera config %r not found (tried: %s). The Orbbec config is "
        "WRITTEN BY scripts/calibrate_camera_extrinsics.py — run the "
        "calibration first." % (name_or_path, candidates)
    )


def process_camera_points(
    depth,
    intr,
    rot,
    trans,
    stride,
    min_range,
    max_range,
    xy_extent,
    min_z,
    max_z,
    link_pts,
    self_radius,
    ignore_region,
    baseline_boxes,
):
    """One camera frame -> filtered base_link points. Pure (testable)."""
    pts = depth_to_points(
        depth,
        intr["fx"],
        intr["fy"],
        intr["cx"],
        intr["cy"],
        stride=stride,
        min_range=min_range,
        max_range=max_range,
    )
    pts = transform_points(pts, rot, trans)
    pts = workspace_crop(pts, xy_extent=xy_extent, min_z=min_z, max_z=max_z)
    if len(pts) and link_pts is not None:
        pts = pts[robot_mask(pts, link_pts, self_radius)]
    if len(pts) and ignore_region is not None:
        pts = pts[~in_box_mask(pts, ignore_region["center"], ignore_region["dims"])]
    for box in baseline_boxes:
        if not len(pts):
            break
        pts = pts[~in_box_mask(pts, box["position"], box["dims"], inflate=0.01)]
    return pts


class _CameraInput:
    """Latest depth frame + intrinsics for one configured camera."""

    def __init__(self, node, cfg):
        self.cfg = cfg
        self.depth = None
        self.stamp = None
        self.info = None
        node.create_subscription(
            CameraInfo,
            cfg["info_topic"],
            self._info_cb,
            qos_profile_sensor_data,
            callback_group=node.cb_group,
        )
        node.create_subscription(
            Image,
            cfg["depth_topic"],
            self._depth_cb,
            qos_profile_sensor_data,
            callback_group=node.cb_group,
        )

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.info = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

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
            return
        self.depth = d
        self.stamp = time.monotonic()

    def fresh(self, max_age):
        return (
            self.depth is not None
            and self.info is not None
            and time.monotonic() - self.stamp < max_age
        )


class CamerasNode(Node):
    def __init__(self):
        super().__init__("cameras")
        self.cb_group = ReentrantCallbackGroup()
        p = self.declare_parameter
        self.rate_hz = float(p("rate_hz", 2.0).value)
        self.baseline = str(p("baseline", "world_real_bench.yaml").value)
        self.voxel = float(p("voxel", 0.03).value)
        self.self_radius = float(p("self_radius", 0.11).value)
        self.max_boxes = int(p("max_boxes", 20).value)
        self.min_voxels = int(p("min_voxels", 8).value)
        self.stride = int(p("stride", 4).value)
        self.xy_extent = float(p("xy_extent", 1.2).value)
        self.min_z = float(p("min_z", 0.03).value)
        self.max_z = float(p("max_z", 1.3).value)
        cam_names = list(p("cameras", ["camera_orbbec_bench.yaml"]).value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cams = [_CameraInput(self, load_camera_config(n)) for n in cam_names]
        self.acc = VoxelAccumulator(voxel=self.voxel)
        self.tracker = BoxTracker()
        self.ignore_region = None
        self._last_sent = None
        self._warned = set()

        scene = load_scene(resolve_config(self.baseline))
        self.baseline_boxes = [
            {"position": o.position, "dims": o.dims} for o in scene.obstacles
        ]
        self.get_logger().info(
            "baseline %s: %d obstacle(s) stripped from depth"
            % (self.baseline, len(self.baseline_boxes))
        )

        self.world_client = self.create_client(
            UpdateWorldBoxes,
            "/rammp_curobo/update_world_boxes",
            callback_group=self.cb_group,
        )
        self.create_service(
            SetIgnoreRegion,
            "~/set_ignore_region",
            self._set_ignore_cb,
            callback_group=self.cb_group,
        )
        self.markers_pub = self.create_publisher(MarkerArray, "~/world_markers", 1)
        self.create_timer(1.0 / self.rate_hz, self._tick, callback_group=self.cb_group)
        self._pending = None
        self.get_logger().info(
            "cameras up: %s @ %.1f Hz, voxel %.0f mm"
            % (cam_names, self.rate_hz, self.voxel * 1000)
        )

    # ------------------------------------------------------------------ TF
    def _base_from(self, frame):
        try:
            tr = self.tf_buffer.lookup_transform("base_link", frame, rclpy.time.Time())
        except Exception:
            return None
        q, t = tr.transform.rotation, tr.transform.translation
        return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _camera_pose(self, cfg):
        """(R, t) base_T_optical, or None while TF is incomplete."""
        if cfg.get("tf_frame"):
            return self._base_from(cfg["tf_frame"])
        parent = self._base_from(cfg["parent_frame"])
        if parent is None and cfg["parent_frame"] == "base_link":
            parent = (np.eye(3), np.zeros(3))  # fixed camera needs no TF
        if parent is None:
            return None
        r_p, t_p = parent
        mx = np.asarray(cfg["mount_xyz"], dtype=float)
        qx, qy, qz, qw = cfg["mount_quat_xyzw"]
        return r_p @ quat_to_mat(qx, qy, qz, qw), r_p @ mx + t_p

    def _link_points(self):
        pts = []
        for name in ARM_CHAIN:
            got = self._base_from(name)
            if got is None:
                return None  # no bringup running — skip self-filtering
            pts.append(got[1])
        r_ee, t_ee = self._base_from("end_effector_link")
        pts.append(t_ee + r_ee @ np.array([0.0, 0.0, 0.18]))  # gripper capsule
        return pts

    # ---------------------------------------------------------------- tick
    def _tick(self):
        all_pts, saw_frame = [], False
        link_pts = self._link_points()
        if link_pts is None and "tf" not in self._warned:
            self._warned.add("tf")
            self.get_logger().warn(
                "no arm TF — self-filter OFF (fine without a bringup; the "
                "arm will cluster as an obstacle if one IS running)"
            )
        for cam in self.cams:
            if not cam.fresh(max_age=2.0 / self.rate_hz):
                continue
            pose = self._camera_pose(cam.cfg)
            if pose is None:
                continue
            saw_frame = True
            all_pts.append(
                process_camera_points(
                    cam.depth,
                    cam.info,
                    pose[0],
                    pose[1],
                    stride=self.stride,
                    min_range=float(cam.cfg.get("min_range", 0.12)),
                    max_range=float(cam.cfg.get("max_range", 1.2)),
                    xy_extent=self.xy_extent,
                    min_z=self.min_z,
                    max_z=self.max_z,
                    link_pts=link_pts,
                    self_radius=self.self_radius,
                    ignore_region=self.ignore_region,
                    baseline_boxes=self.baseline_boxes,
                )
            )
        if not saw_frame:
            if "frames" not in self._warned:
                self._warned.add("frames")
                self.get_logger().warn(
                    "no fresh depth frames — is the camera driver running?"
                )
            return
        self._warned.discard("frames")
        pts = np.vstack(all_pts) if all_pts else np.empty((0, 3))
        self.acc.update(pts)
        boxes, total = cluster_cells(
            self.acc.occupied_cells(),
            self.voxel,
            min_voxels=self.min_voxels,
            max_boxes=self.max_boxes,
        )
        if total > len(boxes):
            self.get_logger().warn(
                "%d clusters found, capped to nearest %d" % (total, len(boxes))
            )
        named = self.tracker.assign(boxes)
        self._publish_markers(named)
        if self._last_sent is not None and not boxes_changed(named, self._last_sent):
            return
        self._send_world(named)

    def _send_world(self, named):
        if self._pending is not None and not self._pending.done():
            return  # previous update still in flight — next tick retries
        if not self.world_client.service_is_ready():
            if "planner" not in self._warned:
                self._warned.add("planner")
                self.get_logger().warn("planner update_world_boxes not available yet")
            return
        self._warned.discard("planner")
        req = UpdateWorldBoxes.Request()
        req.baseline = self.baseline
        for name, b in sorted(named.items()):
            req.names.append(name)
            req.centers.append(
                Point(x=b["center"][0], y=b["center"][1], z=b["center"][2])
            )
            req.dims.append(Vector3(x=b["dims"][0], y=b["dims"][1], z=b["dims"][2]))
        snapshot = {k: dict(v) for k, v in named.items()}

        def _done(fut):
            res = fut.exception() is None and fut.result()
            if res and res.success:
                self._last_sent = snapshot
                self.get_logger().info(res.message)
            else:
                msg = res.message if res else str(fut.exception())
                self.get_logger().error("world update failed: %s" % msg)

        self._pending = self.world_client.call_async(req)
        self._pending.add_done_callback(_done)

    def _publish_markers(self, named):
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        for i, (name, b) in enumerate(sorted(named.items())):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "perceived", i, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = b["center"]
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = b["dims"]
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.3, 0.1, 0.55
            m.text = name
            arr.markers.append(m)
        self.markers_pub.publish(arr)

    # ------------------------------------------------------------- services
    def _set_ignore_cb(self, request, response):
        d = [request.dims.x, request.dims.y, request.dims.z]
        if not any(d):
            self.ignore_region = None
            response.message = "ignore region cleared"
        else:
            self.ignore_region = {
                "center": [request.center.x, request.center.y, request.center.z],
                "dims": d,
            }
            response.message = "ignoring %.2f x %.2f x %.2f m at (%.2f, %.2f, %.2f)" % (
                *d,
                request.center.x,
                request.center.y,
                request.center.z,
            )
        response.success = True
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CamerasNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Restore the D405 config + wire setup.py**

```bash
mkdir -p rammp_curobo_ros/config && \
git show '0e0ec5b~1:rammp_curobo_ros/config/camera_d405_wrist.yaml' \
    > rammp_curobo_ros/config/camera_d405_wrist.yaml
```

In `rammp_curobo_ros/setup.py`: add to `data_files`

```python
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
```

and to `console_scripts`:

```python
            "cameras = rammp_curobo_ros.cameras:main",
```

- [ ] **Step 5: Run tests + build**

```bash
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio && \
source /opt/ros/humble/setup.zsh && \
colcon build --symlink-install --packages-select rammp_curobo_ros && \
source install/setup.zsh && ros2 run rammp_curobo_ros cameras --help 2>&1 | head -3
```

Expected: tests pass; `cameras` resolves as an executable (it will exit
complaining the Orbbec config is missing — correct until calibration).

- [ ] **Step 6: Commit**

```bash
git add rammp_curobo_ros/rammp_curobo_ros/cameras.py rammp_curobo_ros/config \
    rammp_curobo_ros/setup.py rammp_curobo_ros/test/test_cameras_tick.py && \
git commit -m "cameras node: continuous perceived world from depth cameras"
```

---

### Task 7: Orbbec extrinsic calibration script

**Files:**
- Create: `scripts/calibrate_camera_extrinsics.py`
- Test: `core/tests/test_perception.py` (append one synthetic solve test)

**Interfaces:**
- Consumes: `cv2.calibrateHandEye`, `cv2.aruco`, TF (`base_link` →
  `tool_frame`), Orbbec color topics.
- Produces: `rammp_curobo_ros/config/camera_orbbec_bench.yaml` with
  `parent_frame: base_link`, `mount_xyz`, `mount_quat_xyzw`, plus
  depth topics/ranges — exactly the schema `load_camera_config` reads.
  Also produces `solve_eye_to_hand(base_T_tool_list, cam_T_tag_list)`
  (importable pure function) returning `(R, t, rms_residual_m)`.

- [ ] **Step 1: Write the failing synthetic test (append to core tests)**

The solver is pure math — test it with a fabricated ground truth. Import
via the scripts path (pattern: insert the scripts dir into sys.path).

```python
def test_eye_to_hand_solver_recovers_known_extrinsic():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "calib",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts",
            "calibrate_camera_extrinsics.py",
        ),
    )
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)

    rng = np.random.default_rng(7)

    def rot(ax, ang):
        from rammp_curobo.perception import quat_to_mat

        s, c = np.sin(ang / 2), np.cos(ang / 2)
        v = np.zeros(3)
        v[ax] = s
        return quat_to_mat(v[0], v[1], v[2], c)

    # ground truth: camera 1.2 m out, looking back at the bench
    R_bc = rot(2, np.pi) @ rot(0, 2.2)
    t_bc = np.array([1.2, 0.1, 0.6])
    R_tt = rot(1, 0.3)  # tag-in-tool, arbitrary rigid offset
    t_tt = np.array([0.0, 0.03, 0.05])
    base_T_tool, cam_T_tag = [], []
    for _ in range(10):
        R_bt = rot(0, rng.uniform(-0.6, 0.6)) @ rot(1, rng.uniform(-0.6, 0.6))
        t_bt = np.array([0.45, 0.0, 0.35]) + rng.uniform(-0.15, 0.15, 3)
        base_T_tool.append((R_bt, t_bt))
        # cam_T_tag = cam_T_base @ base_T_tool @ tool_T_tag
        R_ct = R_bc.T @ R_bt @ R_tt
        t_ct = R_bc.T @ (R_bt @ t_tt + t_bt - t_bc)
        cam_T_tag.append((R_ct, t_ct))
    R, t, rms = calib.solve_eye_to_hand(base_T_tool, cam_T_tag)
    assert rms < 1e-6
    assert np.allclose(R, R_bc, atol=1e-6) and np.allclose(t, t_bc, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest core/tests/test_perception.py::test_eye_to_hand_solver_recovers_known_extrinsic -q
```

Expected: FileNotFoundError / AttributeError (script absent).

- [ ] **Step 3: Write the script**

`scripts/calibrate_camera_extrinsics.py`:

```python
#!/usr/bin/env python3
"""Calibrate a FIXED camera's pose in base_link (eye-to-hand) via ArUco.

Attended, one-time, re-runnable. An ArUco tag rides rigidly on the
gripper (anywhere rigid — its tool offset is solved, not measured). THE
HUMAN jogs the arm to N poses spanning the camera view; at each pose,
press ENTER to record (tag pose from the camera + tool pose from TF).
cv2.calibrateHandEye (eye-to-hand form: base->tool poses inverted) solves
the camera extrinsic; the result is only written if the RMS residual over
the recorded pairs is under --max-residual (default 0.01 m).

    export ROS_LOCALHOST_ONLY=1
    # terminal 1: arm bringup (kortex).  terminal 2:
    ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true
    # terminal 3:
    python3 scripts/calibrate_camera_extrinsics.py \
        --marker-id 0 --marker-size 0.05 --poses 10

Writes rammp_curobo_ros/config/camera_orbbec_bench.yaml (depth aligned to
the color frame by depth_registration, so the color extrinsic IS the
depth extrinsic).
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join(REPO, "rammp_curobo_ros", "config", "camera_orbbec_bench.yaml")


def solve_eye_to_hand(base_T_tool, cam_T_tag):
    """Fixed-camera hand-eye: lists of (R, t) pairs -> (R, t, rms) of
    base_T_camera. Standard eye-to-hand trick: feed calibrateHandEye the
    INVERTED gripper poses so the fixed camera looks like a wrist camera.
    """
    import cv2

    R_g, t_g, R_c, t_c = [], [], [], []
    for (R_bt, t_bt), (R_ct, t_ct) in zip(base_T_tool, cam_T_tag):
        R_g.append(R_bt.T)                # tool_T_base
        t_g.append(-R_bt.T @ t_bt)
        R_c.append(R_ct)
        t_c.append(t_ct)
    R_x, t_x = cv2.calibrateHandEye(
        R_g, t_g, R_c, t_c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    # calibrateHandEye returns tool(base)_T_camera in the inverted frame
    # setup — i.e. base_T_camera for eye-to-hand.
    R_bc, t_bc = R_x, t_x.reshape(3)

    # residual: with X known, tag-in-tool implied per pair must agree
    tags = []
    for (R_bt, t_bt), (R_ct, t_ct) in zip(base_T_tool, cam_T_tag):
        R_tt = R_bt.T @ R_bc @ R_ct
        t_tt = R_bt.T @ (R_bc @ t_ct + t_bc - t_bt)
        tags.append(t_tt)
    tags = np.asarray(tags)
    rms = float(np.sqrt(np.mean(np.sum((tags - tags.mean(axis=0)) ** 2, axis=1))))
    return R_bc, t_bc, rms


def mat_to_quat_xyzw(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:  # w ~ 0: pick the dominant diagonal term
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = s / 4.0
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        x, y, z = q[0], q[1], q[2]
        w = (R[k, j] - R[j, k]) / s
    return [float(x), float(y), float(z), float(w)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--marker-size", type=float, required=True, help="tag side (m), MEASURE IT")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--tool-frame", default="tool_frame")
    ap.add_argument("--max-residual", type=float, default=0.01)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    import cv2
    import rclpy
    import yaml
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    sys.path.insert(0, os.path.join(REPO, "core"))
    from rammp_curobo.perception import quat_to_mat

    class Grab(Node):
        def __init__(self):
            super().__init__("calibrate_camera_extrinsics")
            self.img = None
            self.info = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(
                Image, args.color_topic, self._img_cb, qos_profile_sensor_data
            )
            self.create_subscription(
                CameraInfo, args.info_topic, self._info_cb, qos_profile_sensor_data
            )

        def _img_cb(self, msg):
            if msg.encoding in ("rgb8", "bgr8"):
                a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                self.img = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()

        def _info_cb(self, msg):
            self.info = msg

        def tool_pose(self):
            tr = self.tf_buffer.lookup_transform(
                "base_link", args.tool_frame, rclpy.time.Time()
            )
            q, t = tr.transform.rotation, tr.transform.translation
            return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    rclpy.init()
    node = Grab()
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(getattr(aruco, args.dict)),
        aruco.DetectorParameters(),
    )
    base_T_tool, cam_T_tag = [], []
    print(
        "Jog the arm so the tag faces the camera. %d poses; spread them "
        "across the view and VARY THE WRIST ANGLE. ENTER to record, "
        "'done' to solve early, Ctrl-C aborts." % args.poses
    )
    while len(base_T_tool) < args.poses:
        if input("[%d/%d] > " % (len(base_T_tool) + 1, args.poses)).strip() == "done":
            break
        node.img = None
        t0 = node.get_clock().now()
        while node.img is None or node.info is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            if (node.get_clock().now() - t0).nanoseconds > 10e9:
                sys.exit("no color frames on %s" % args.color_topic)
        corners, ids, _ = detector.detectMarkers(node.img)
        if ids is None or args.marker_id not in ids.flatten():
            print("  tag %d NOT visible — repose and retry" % args.marker_id)
            continue
        i = list(ids.flatten()).index(args.marker_id)
        k = np.array(node.info.k).reshape(3, 3)
        d = np.array(node.info.d) if len(node.info.d) else np.zeros(5)
        s = args.marker_size / 2.0
        obj = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners[i].reshape(4, 2).astype(np.float32), k, d,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            print("  PnP failed — retry")
            continue
        try:
            R_bt, t_bt = node.tool_pose()
        except Exception as exc:
            print("  no TF base_link->%s (%s) — is the bringup up?" % (args.tool_frame, exc))
            continue
        cam_T_tag.append((cv2.Rodrigues(rvec)[0], tvec.reshape(3)))
        base_T_tool.append((R_bt, t_bt))
        print("  recorded (tag at %.2f m)" % float(np.linalg.norm(tvec)))
    if len(base_T_tool) < 5:
        sys.exit("only %d poses — need at least 5" % len(base_T_tool))
    R, t, rms = solve_eye_to_hand(base_T_tool, cam_T_tag)
    print("base_T_camera t = [%.4f, %.4f, %.4f], RMS residual %.4f m" % (*t, rms))
    if rms > args.max_residual:
        sys.exit(
            "RESIDUAL %.4f m > %.3f m — NOT writing. More poses, better "
            "spread, check the marker size, keep the tag rigid." % (rms, args.max_residual)
        )
    cfg = {
        "depth_topic": args.depth_topic,
        "info_topic": args.info_topic.replace("color", "depth")
        if not args.depth_topic.endswith("image_raw")
        else "/camera/depth/camera_info",
        "parent_frame": "base_link",
        "mount_xyz": [float(v) for v in t],
        "mount_quat_xyzw": mat_to_quat_xyzw(R),
        "min_range": 0.25,
        "max_range": 1.5,
        "calibrated": "eye-to-hand ArUco, RMS %.4f m, %d poses"
        % (rms, len(base_T_tool)),
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote %s — REBUILD rammp_curobo_ros so the share/ copy updates" % args.out)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

Note on the info topic: with `depth_registration:=true` the aligned depth
carries the COLOR intrinsics; verify live which camera_info the aligned
topic publishes (`ros2 topic echo --once /camera/depth/camera_info`) and
fix the two topic fields in the written YAML if the driver differs — the
YAML is data, editing it needs no code change.

- [ ] **Step 4: Run the synthetic test**

```bash
python3 -m pytest core/tests/test_perception.py -q
```

Expected: all pass (solver recovers the fabricated extrinsic to 1e-6).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_camera_extrinsics.py core/tests/test_perception.py && \
git commit -m "scripts: eye-to-hand Orbbec extrinsic calibration (ArUco, residual-gated)"
```

---

### Task 8: live checks, docs, bench runbook

**Files:**
- Create: `scripts/cameras_checks.py`
- Modify: `README.md` (perception section), `INTEGRATION.md` (new
  services), `docs/HARDWARE_BRINGUP.md` (bench procedure), `CLAUDE.md`
  (component list + invariant)

**Interfaces:**
- Consumes: everything above. Produces: the attended stage-1 procedure.

- [ ] **Step 1: Write scripts/cameras_checks.py**

```python
#!/usr/bin/env python3
"""Live sanity checks for the perceived world (planner + cameras up).

Non-destructive: talks services/topics only, never executes motion.

    export ROS_LOCALHOST_ONLY=1
    python3 scripts/cameras_checks.py

Checks: (1) update_world_boxes round-trip incl. empty-perceived-set,
(2) markers flowing, (3) a plan succeeds while the updater hammers the
world at 2 Hz (the lock-timeout path).
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from rammp_curobo_interfaces.srv import UpdateWorldBoxes

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]


def main():
    rclpy.init()
    node = Node("cameras_checks")
    ok = True

    cli = node.create_client(UpdateWorldBoxes, "/rammp_curobo/update_world_boxes")
    if not cli.wait_for_service(timeout_sec=5.0):
        sys.exit("FAIL: planner update_world_boxes service not up")

    def send(names, centers, dims, baseline=""):
        req = UpdateWorldBoxes.Request()
        req.names, req.baseline = list(names), baseline
        req.centers = [Point(x=c[0], y=c[1], z=c[2]) for c in centers]
        req.dims = [Vector3(x=d[0], y=d[1], z=d[2]) for d in dims]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        return fut.result()

    r = send(["chk_box"], [[0.55, 0.25, 0.15]], [[0.1, 0.1, 0.3]])
    print("[1] add box:", r and r.message)
    ok &= bool(r and r.success)
    r = send([], [], [])
    print("[1] empty perceived set (baseline must survive):", r and r.message)
    ok &= bool(r and r.success)

    got = {"n": 0}
    node.create_subscription(
        MarkerArray, "/cameras/world_markers", lambda m: got.update(n=got["n"] + 1), 1
    )
    t0 = time.monotonic()
    while got["n"] == 0 and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    print("[2] markers:", "flowing" if got["n"] else "NONE (cameras node down? OK if planner-only test)")

    try:
        from rclpy.action import ActionClient

        from rammp_curobo_interfaces.action import PlanToJoints

        ac = ActionClient(node, PlanToJoints, "/rammp_curobo/plan_to_joints")
        if not ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("plan_to_joints server not up")
        goal = PlanToJoints.Goal()
        goal.target_joints = [HOME[0] + 0.05] + HOME[1:]
        goal.start_joints = HOME
        t_end = time.monotonic() + 6.0
        sent = ac.send_goal_async(goal)

        # hammer the world while the plan runs — exercises the lock timeout
        while time.monotonic() < t_end and not sent.done():
            send(["chk_churn"], [[0.5, -0.3, 0.2]], [[0.05, 0.05, 0.2]])
            rclpy.spin_once(node, timeout_sec=0.1)
        rclpy.spin_until_future_complete(node, sent, timeout_sec=10.0)
        res_fut = sent.result().get_result_async()
        rclpy.spin_until_future_complete(node, res_fut, timeout_sec=30.0)
        res = res_fut.result().result
        print("[3] plan under churn:", res.success, res.message)
        ok &= res.success
        send([], [], [])  # leave the world clean
    except Exception as exc:
        print("[3] SKIP/FAIL:", exc)
        ok = False

    print("ALL OK" if ok else "FAILURES — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

(Field names verified against `PlanToJoints.action`: `target_joints` +
`start_joints`.)

- [ ] **Step 2: Docs**

- `README.md`: add a "Perceived world (cameras)" subsection under the run
  sections: what the node does, the run line
  (`ros2 run rammp_curobo_ros cameras`), the calibrate-first note, RViz
  marker topic, and the one-line safety note (planning-only, no motion).
- `INTEGRATION.md`: document `/rammp_curobo/update_world_boxes` and
  `/cameras/set_ignore_region` request/response semantics (perceived
  boxes replace, baseline sticky, ignore region for the manipulation
  target) — the box-opening workspace consumes these.
- `docs/HARDWARE_BRINGUP.md`: append "Stage 1 — Orbbec perceived world
  (attended)" with the exact terminal-by-terminal procedure:
  1. arm bringup (kortex), 2. Orbbec driver with
  `depth_registration:=true`, 3. calibration script (human jogs),
  4. rebuild, 5. `cameras` node + RViz acceptance (box appears ≤2 s
  within ±3 cm, disappears ≤3 s), 6. `cameras_checks.py`,
  7. only then: a plan around a placed obstacle, dry-run first.
- `CLAUDE.md`: add `cameras`/perception to the parts list; add the
  invariant "perceived boxes REPLACE, baseline is sticky, updates are
  never empty"; note the camera-scanning code is deliberately BACK as of
  this branch (supersedes the 2026-08-14 removal note).

- [ ] **Step 3: Full test sweep**

```bash
python3 -m pytest core/tests -q && \
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio && \
python3 -m ruff check . 2>/dev/null || ruff check .
```

Expected: all green, no lint errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/cameras_checks.py README.md INTEGRATION.md docs/HARDWARE_BRINGUP.md CLAUDE.md && \
git commit -m "perceived world: live checks + docs + attended bench runbook"
```

---

## Attended follow-up (not agent-executable — the human runs these)

1. Mount/aim the Orbbec at the bench; print/measure the ArUco tag; run the
   calibration per HARDWARE_BRINGUP; rebuild.
2. RViz acceptance loop, `cameras_checks.py`, then the first plan around a
   real obstacle (dry-run, then ≤0.25 speed with e-stop in hand).
3. Only after stage 1 passes on the bench: stage 2 (D405 fusion) gets its
   own plan.
