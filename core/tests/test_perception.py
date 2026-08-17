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


def _cube_points(center, side, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-side / 2, side / 2, size=(n, 3))


def test_accumulator_hysteresis_appear_and_decay():
    from rammp_curobo.perception import VoxelAccumulator

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
    from rammp_curobo.perception import VoxelAccumulator

    acc = VoxelAccumulator(voxel=0.05, occupied_at=1)
    lone = np.array([[0.5, 0.5, 0.5]])  # 1 point < min_points_per_voxel=2
    acc.update(lone)
    assert len(acc.occupied_cells()) == 0


def test_cluster_cells_two_separated_boxes():
    from rammp_curobo.perception import cluster_cells

    a = np.argwhere(np.ones((4, 4, 4)))  # 0..3 cube
    b = a + np.array([20, 0, 0])
    boxes, total = cluster_cells(np.vstack([a, b]), voxel=0.05, min_voxels=8)
    assert total == 2 and len(boxes) == 2
    dims = boxes[0]["dims"]
    assert np.allclose(dims, [0.2, 0.2, 0.2], atol=1e-9)


def test_cluster_cells_cap_keeps_nearest():
    from rammp_curobo.perception import cluster_cells

    near = np.argwhere(np.ones((3, 3, 3)))  # near origin
    far = near + np.array([40, 0, 0])
    boxes, total = cluster_cells(
        np.vstack([near, far]), voxel=0.05, min_voxels=8, max_boxes=1
    )
    assert total == 2 and len(boxes) == 1
    assert boxes[0]["center"][0] < 1.0  # the near one survived


def test_tracker_names_are_stable():
    from rammp_curobo.perception import BoxTracker

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
    from rammp_curobo.perception import boxes_changed

    a = {"obs_1": {"center": [0.5, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    same = {"obs_1": {"center": [0.505, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    moved = {"obs_1": {"center": [0.55, 0.0, 0.1], "dims": [0.2, 0.2, 0.2]}}
    assert not boxes_changed(a, same, tol=0.01)
    assert boxes_changed(a, moved, tol=0.01)
    assert boxes_changed(a, {}, tol=0.01)


def test_in_box_mask_with_inflation():
    pts = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.2, 0.0, 0.0]])
    inside = in_box_mask(pts, center=[0, 0, 0], dims=[0.1, 0.1, 0.1])
    assert list(inside) == [True, False, False]
    inside = in_box_mask(pts, [0, 0, 0], [0.1, 0.1, 0.1], inflate=0.02)
    assert list(inside) == [True, True, False]
