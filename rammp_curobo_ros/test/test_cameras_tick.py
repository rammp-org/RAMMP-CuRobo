"""The cameras node's per-tick point processing, no ROS objects."""

import numpy as np

from rammp_curobo_ros.cameras import process_camera_points


def _run(depth, intr, **kw):
    base = dict(
        stride=1,
        min_range=0.1,
        max_range=1.0,
        xy_extent=1.2,
        min_z=0.03,
        max_z=1.3,
        link_pts=None,
        self_radius=0.11,
        ignore_region=None,
        baseline_boxes=[],
    )
    base.update(kw)
    return process_camera_points(depth, intr, np.eye(3), np.zeros(3), **base)


def test_process_filters_ignore_region_and_baseline():
    # a 10x10 flat depth wall 0.5 m out, camera optical z along base z
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    intr = dict(fx=20.0, fy=20.0, cx=5.0, cy=5.0)
    baseline_boxes = [{"position": [0.0, 0.0, 0.45], "dims": [2.0, 2.0, 0.12]}]
    pts = _run(depth, intr, baseline_boxes=baseline_boxes)
    assert len(pts) == 0  # everything sat on the baseline "table"

    pts = _run(depth, intr)
    assert len(pts) > 0  # without the baseline the wall survives

    region = {"center": [0.0, 0.0, 0.5], "dims": [5.0, 5.0, 5.0]}
    pts = _run(depth, intr, ignore_region=region)
    assert len(pts) == 0  # the ignore region swallowed it


def test_decayable_cells_unions_cameras_and_keeps_unseen():
    from rammp_curobo_ros.cameras import decayable_cells

    depth = np.full((10, 10), 0.8, dtype=np.float32)
    intr = dict(fx=10.0, fy=10.0, cx=5.0, cy=5.0)
    cells = np.array([[0, 0, 4], [30, 0, 4]])  # one in view, one far out
    frames = [(np.eye(3), np.zeros(3), depth, intr, 0.07, 0.9)]
    out = decayable_cells(cells, 0.1, frames)
    assert out == {(0, 0, 4)}
    assert decayable_cells(cells, 0.1, []) == set()  # no frames -> keep all


def test_process_self_filter_uses_link_points():
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    intr = dict(fx=20.0, fy=20.0, cx=5.0, cy=5.0)
    # a capsule running right through the wall points erases them
    link_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    pts_all = _run(depth, intr)
    pts_filtered = _run(depth, intr, link_pts=link_pts, self_radius=0.5)
    assert len(pts_filtered) < len(pts_all)


def test_sphere_self_model_is_preferred_over_capsules():
    """With a sphere model the gripper is erased by its real shape, and the
    capsule radius is ignored — a point 30 cm down the old capsule axis
    but far from every sphere survives."""
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    intr = dict(fx=20.0, fy=20.0, cx=5.0, cy=5.0)
    # a fat capsule that would erase everything near z axis...
    link_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.0])]
    pts_capsule = _run(depth, intr, link_pts=link_pts, self_radius=5.0)
    assert len(pts_capsule) == 0
    # ...is overridden by a tiny sphere model that touches nothing
    spheres = np.array([[5.0, 5.0, 5.0, 0.01]])
    pts_spheres = _run(depth, intr, link_pts=link_pts, self_radius=5.0,
                       self_spheres=spheres, self_margin=0.05)
    assert len(pts_spheres) > 0
    # and a sphere model that covers the frame erases it
    big = np.array([[0.0, 0.0, 1.0, 3.0]])
    assert len(_run(depth, intr, link_pts=None, self_radius=0.0,
                    self_spheres=big, self_margin=0.0)) == 0


def test_sphere_margin_is_applied_exactly():
    """The node folds the model YAML's planning buffer into self_margin;
    the pure function must apply the margin exactly as given — a point
    just inside radius+margin is erased, just outside survives."""
    depth = np.full((1, 1), 0.5, dtype=np.float32)
    intr = dict(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    # the single deprojected point lands at (0, 0, 0.5) in base_link
    radius, margin = 0.10, 0.05
    inside = np.array([[0.0, 0.0, 0.5 - (radius + margin) + 0.01, radius]])
    outside = np.array([[0.0, 0.0, 0.5 - (radius + margin) - 0.01, radius]])
    assert len(_run(depth, intr, self_spheres=inside, self_margin=margin)) == 0
    assert len(_run(depth, intr, self_spheres=outside, self_margin=margin)) == 1
