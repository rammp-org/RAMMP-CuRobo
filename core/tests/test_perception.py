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
