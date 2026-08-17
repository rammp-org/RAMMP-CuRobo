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


def test_merged_scene_keeps_baseline_and_replaces_objects():
    from rammp_curobo.scene import Obstacle, Scene, merged_scene

    base = Scene(
        base_frame="base_link",
        obstacles=[
            Obstacle(
                {"name": "table", "position": [0.5, 0, -0.04], "dims": [1, 1, 0.08]}
            )
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


def test_eye_to_hand_solver_recovers_known_extrinsic():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "calib",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scripts",
            "calibrate_camera_extrinsics.py",
        ),
    )
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)

    rng = np.random.default_rng(7)

    def rand_rot():
        # Tsai's method needs pose pairs whose relative rotation axes span
        # 3D — rotations about only 1-2 axes make the system degenerate
        # (found the hard way writing this test; the script tells the
        # operator to vary the wrist for the same reason).
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        return quat_to_mat(*q)

    # ground truth: camera ~1.2 m out, arbitrary attitude
    R_bc = rand_rot()
    t_bc = np.array([1.2, 0.1, 0.6])
    R_tt = rand_rot()  # tag-in-tool, arbitrary rigid offset
    t_tt = np.array([0.0, 0.03, 0.05])
    base_T_tool, cam_T_tag = [], []
    for _ in range(10):
        R_bt = rand_rot()
        t_bt = np.array([0.45, 0.0, 0.35]) + rng.uniform(-0.15, 0.15, 3)
        base_T_tool.append((R_bt, t_bt))
        # cam_T_tag = cam_T_base @ base_T_tool @ tool_T_tag
        R_ct = R_bc.T @ R_bt @ R_tt
        t_ct = R_bc.T @ (R_bt @ t_tt + t_bt - t_bc)
        cam_T_tag.append((R_ct, t_ct))
    R, t, rms = calib.solve_eye_to_hand(base_T_tool, cam_T_tag)
    assert rms < 1e-6
    assert np.allclose(R, R_bc, atol=1e-6) and np.allclose(t, t_bc, atol=1e-6)


def test_mat_to_quat_round_trip():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "calib",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scripts",
            "calibrate_camera_extrinsics.py",
        ),
    )
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)
    for q in ([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0], [0.7, 0, 0.7, 0.14]):
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        r = quat_to_mat(*q)
        q2 = np.asarray(calib.mat_to_quat_xyzw(r))
        if q2 @ q < 0:
            q2 = -q2  # q and -q are the same rotation
        assert np.allclose(q, q2, atol=1e-9)


def test_in_box_mask_with_inflation():
    pts = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.2, 0.0, 0.0]])
    inside = in_box_mask(pts, center=[0, 0, 0], dims=[0.1, 0.1, 0.1])
    assert list(inside) == [True, False, False]
    inside = in_box_mask(pts, [0, 0, 0], [0.1, 0.1, 0.1], inflate=0.02)
    assert list(inside) == [True, True, False]
