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


def test_process_self_filter_uses_link_points():
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    intr = dict(fx=20.0, fy=20.0, cx=5.0, cy=5.0)
    # a capsule running right through the wall points erases them
    link_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    pts_all = _run(depth, intr)
    pts_filtered = _run(depth, intr, link_pts=link_pts, self_radius=0.5)
    assert len(pts_filtered) < len(pts_all)
