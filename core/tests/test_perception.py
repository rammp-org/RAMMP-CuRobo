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


def test_visible_free_cells_sees_through_only_in_frustum():
    from rammp_curobo.perception import visible_free_cells

    # camera at origin looking along base +z (identity extrinsic),
    # 10x10 frame, wall at 0.8 m everywhere. voxel 0.1 -> clearance =
    # half-diagonal 0.0866 + margin 0.015 = 0.1016: free needs z < 0.698.
    depth = np.full((10, 10), 0.8, dtype=np.float32)
    intr = dict(fx=10.0, fy=10.0, cx=5.0, cy=5.0)
    voxel = 0.1
    free = (0, 0, 4)  # center z 0.45 << 0.698 -> provably empty
    near_wall = (0, 0, 7)  # center z 0.75, within clearance -> kept
    occluded = (0, 0, 8)  # center z 0.85: in range, BEHIND the wall -> kept
    beyond = (0, 0, 9)  # center z 0.95 > max_range -> kept
    outside = (30, 0, 4)  # projects far off-frame -> kept
    cells = np.array([free, near_wall, occluded, beyond, outside])
    out = visible_free_cells(
        cells,
        voxel,
        np.eye(3),
        np.zeros(3),
        depth,
        **intr,
        min_range=0.07,
        max_range=0.9,
    )
    assert out == {free}


def test_removed_object_bottom_layer_is_decayable_top_down():
    """Audit regression: with a flat 5 cm margin, the lowest voxel layer
    of a removed tabletop object (centers ~4.5 cm above the table) could
    never be proven empty under a top-down look-back — a permanent
    phantom slab at every vacated grasp spot. Far-edge clearance fixes
    it: voxel 0.03 -> clearance 0.041 < the 0.455-of-0.5 gap."""
    from rammp_curobo.perception import visible_free_cells

    # camera 0.5 m above the table looking straight down:
    # camera z = -base z (R = diag(1, -1, -1)), table depth 0.5 everywhere
    rot = np.diag([1.0, -1.0, -1.0])
    trans = np.array([0.05, 0.05, 0.5])
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    bottom_layer_cell = (1, 1, 1)  # center (0.045, 0.045, 0.045)
    out = visible_free_cells(
        np.array([bottom_layer_cell]),
        0.03,
        rot,
        trans,
        depth,
        fx=100.0,
        fy=100.0,
        cx=5.0,
        cy=5.0,
    )
    assert out == {bottom_layer_cell}


def test_clear_box_purges_scores_and_reset_forgets_all():
    from rammp_curobo.perception import VoxelAccumulator

    acc = VoxelAccumulator(voxel=0.05, occupied_at=3)
    inside = _cube_points([0.5, 0.0, 0.1], 0.08, seed=1)
    outside = _cube_points([0.2, 0.3, 0.3], 0.08, seed=2)
    for _ in range(3):
        acc.update(np.vstack([inside, outside]))
    assert len(acc.occupied_cells()) > 0
    n = acc.clear_box([0.5, 0.0, 0.1], [0.12, 0.12, 0.12], inflate=0.02)
    assert n > 0
    remaining = acc.occupied_cells() * 0.05  # cell corners, close enough
    assert np.all(np.linalg.norm(remaining - [0.5, 0.0, 0.1], axis=1) > 0.08)
    assert acc.reset() > 0
    assert len(acc.known_cells()) == 0


def test_visible_free_cells_invalid_depth_is_unknown_not_free():
    from rammp_curobo.perception import visible_free_cells

    depth = np.zeros((10, 10), dtype=np.float32)  # depth hole everywhere
    cells = np.array([[0, 0, 4]])
    out = visible_free_cells(
        cells,
        0.1,
        np.eye(3),
        np.zeros(3),
        depth,
        fx=10.0,
        fy=10.0,
        cx=5.0,
        cy=5.0,
    )
    assert out == set()  # cannot prove empty -> keep


def test_accumulator_decay_cells_scopes_forgetting():
    from rammp_curobo.perception import VoxelAccumulator

    acc = VoxelAccumulator(voxel=0.05, occupied_at=3, max_score=6)
    pts = _cube_points([0.5, 0.0, 0.1], 0.1)
    for _ in range(3):
        acc.update(pts)
    occupied = set(map(tuple, acc.occupied_cells()))
    assert occupied
    # camera looked AWAY: nothing decayable -> world must persist
    for _ in range(10):
        acc.update(np.empty((0, 3)), decay_cells=set())
    assert set(map(tuple, acc.occupied_cells())) == occupied
    # camera sees through them -> they fade as before
    for _ in range(6):
        acc.update(np.empty((0, 3)), decay_cells=occupied)
    assert len(acc.occupied_cells()) == 0


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


def _load_calib_module():
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
    return calib


def test_rigid_solver_recovers_known_extrinsic():
    calib = _load_calib_module()
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    R_bc = quat_to_mat(*q)  # ground truth base_T_camera
    t_bc = np.array([1.2, 0.1, 0.6])
    base_pts = np.array([0.45, 0.0, 0.35]) + rng.uniform(-0.25, 0.25, (8, 3))
    cam_pts = (base_pts - t_bc) @ R_bc  # cam = R^T (base - t)
    R, t, rms = calib.solve_rigid(base_pts, cam_pts)
    assert rms < 1e-9
    assert np.allclose(R, R_bc, atol=1e-9) and np.allclose(t, t_bc, atol=1e-9)
    # noisy clicks: still close, honest residual
    noisy = cam_pts + rng.normal(scale=0.004, size=cam_pts.shape)
    R, t, rms = calib.solve_rigid(base_pts, noisy)
    assert rms < 0.02 and np.abs(t - t_bc).max() < 0.02


def test_robust_solver_drops_background_depth_outlier():
    """One click-through-the-silhouette pair (background depth, ~0.5 m off)
    must be identified and dropped, recovering the true transform — the
    exact failure that blew the first bench run to 15 cm RMS."""
    calib = _load_calib_module()
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    R_bc = quat_to_mat(*q)
    t_bc = np.array([0.03, 0.41, 0.52])
    base = np.array([0.45, 0.0, 0.35]) + rng.uniform(-0.25, 0.25, (8, 3))
    cam = (base - t_bc) @ R_bc
    cam += rng.normal(scale=0.003, size=cam.shape)  # honest click noise
    cam[4] *= (np.linalg.norm(cam[4]) + 0.5) / np.linalg.norm(cam[4])  # wall behind
    R, t, rms, residuals, dropped = calib.solve_rigid_robust(base, cam)
    assert dropped == [4]
    assert rms < 0.02
    assert np.abs(t - t_bc).max() < 0.02
    assert residuals[4] > 0.3  # reported against the final fit


def test_spread_check_flags_degenerate_pose_sets():
    calib = _load_calib_module()
    rng = np.random.default_rng(1)
    line = np.array([0.4, 0.0, 0.3]) + np.outer(rng.uniform(-0.3, 0.3, 8), [1, 0, 0])
    assert "COLLINEAR" in calib.spread_check(line)
    plane = np.array([0.4, 0.0, 0.3]) + np.concatenate(
        [rng.uniform(-0.3, 0.3, (8, 2)), np.zeros((8, 1))], axis=1
    )
    assert "COPLANAR" in calib.spread_check(plane)
    volume = np.array([0.4, 0.0, 0.3]) + rng.uniform(-0.25, 0.25, (8, 3))
    assert calib.spread_check(volume) is None


def test_mat_to_quat_round_trip():
    calib = _load_calib_module()
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


# ----------------------------------------------------------- self-model
def _rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_self_model_spheres_places_links_and_names_missing_ones():
    from rammp_curobo.perception import self_model_spheres

    model = {"a": np.array([[0.1, 0.0, 0.0, 0.05]]),
             "b": np.array([[0.0, 0.2, 0.0, 0.03], [0.0, 0.3, 0.0, 0.03]])}
    sph, missing = self_model_spheres(model, {"a": (_rotz(90.0), np.array([1.0, 2.0, 3.0]))})
    assert missing == ["b"]
    assert sph.shape == (1, 4)
    assert np.allclose(sph[0], [1.0, 2.1, 3.0, 0.05], atol=1e-9)
    sph, missing = self_model_spheres(model, {})
    assert sph.shape == (0, 4) and sorted(missing) == ["a", "b"]


def test_robot_mask_spheres_uses_radius_plus_margin():
    from rammp_curobo.perception import robot_mask_spheres

    sph = np.array([[0.0, 0.0, 0.0, 0.05]])
    pts = np.array([[0.04, 0, 0], [0.10, 0, 0], [0.20, 0, 0]])
    keep = robot_mask_spheres(pts, sph, margin=0.08)      # erase within 0.13
    assert keep.tolist() == [False, False, True]
    assert robot_mask_spheres(np.empty((0, 3)), sph, 0.1).shape == (0,)
    assert robot_mask_spheres(pts, np.empty((0, 4)), 0.1).all()


def _synthetic_arm(rng, cam):
    """An L-shaped chain of spheres, sampled on the hemisphere a camera at
    `cam` can see — the geometry the estimator will meet on the bench."""
    centers = [[0.0, 0.0, z] for z in np.linspace(0.1, 0.5, 6)]      # upright
    centers += [[x, 0.0, 0.5] for x in np.linspace(0.1, 0.6, 8)]      # forward
    centers += [[0.6, y, 0.5] for y in (-0.04, 0.04)]                 # "fingers"
    sph = np.array([[c[0], c[1], c[2], 0.05] for c in centers])
    sph[-2:, 3] = 0.02
    pts = []
    for c in sph:
        u = rng.normal(size=(400, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        u = u[(u @ (np.asarray(cam) - c[:3])) > 0.0]                  # facing camera
        pts.append(c[:3] + c[3] * u)
    return sph, np.vstack(pts)


def test_register_recovers_a_camera_translation_error():
    from rammp_curobo.perception import register_points_to_spheres

    rng = np.random.default_rng(3)
    cam = [0.08, 0.70, 0.30]
    sph, true_pts = _synthetic_arm(rng, cam)
    delta = np.array([0.023, 0.078, -0.040])              # the bench's error
    seen = true_pts + delta + rng.normal(scale=0.003, size=true_pts.shape)
    # an obstacle 12 cm off the arm must not bias the fit
    blob = np.array([0.35, 0.17, 0.5]) + rng.uniform(-0.05, 0.05, size=(400, 3))
    out = register_points_to_spheres(np.vstack([seen, blob]), sph, cam_origin=cam)
    assert out is not None
    shift, used, rms = out
    assert np.linalg.norm(shift + delta) < 0.006, (shift, delta)
    assert used > 200
    assert rms < 0.006


def test_register_recovers_an_error_that_pushes_the_arm_into_itself():
    # the drawn arm lands INSIDE its own model (error away from the
    # camera): without the facing filter ICP matches the far surface
    from rammp_curobo.perception import register_points_to_spheres

    rng = np.random.default_rng(11)
    cam = [0.08, 0.70, 0.30]
    sph, true_pts = _synthetic_arm(rng, cam)
    for delta in ([0.0, -0.06, 0.0], [0.02, -0.09, 0.03], [-0.05, 0.01, -0.04]):
        delta = np.array(delta)
        seen = true_pts + delta + rng.normal(scale=0.002, size=true_pts.shape)
        out = register_points_to_spheres(seen, sph, cam_origin=cam)
        assert out is not None, delta
        assert np.linalg.norm(out[0] + delta) < 0.006, (delta, out[0])


def test_register_refuses_without_arm_points():
    from rammp_curobo.perception import register_points_to_spheres

    sph = np.array([[0.0, 0.0, 0.3, 0.05]])
    far = np.random.default_rng(0).uniform(1.0, 2.0, size=(500, 3))
    assert register_points_to_spheres(far, sph) is None
    assert register_points_to_spheres(np.empty((0, 3)), sph) is None


def test_self_registrar_waits_for_still_frames_then_solves():
    from rammp_curobo.perception import SelfRegistrar

    rng = np.random.default_rng(5)
    sph, true_pts = _synthetic_arm(rng, [0.08, 0.70, 0.30])
    delta = np.array([0.03, -0.05, 0.02])
    seen = true_pts + delta
    cam = [0.08, 0.70, 0.30]
    reg = SelfRegistrar(frames=3)
    assert reg.feed(seen, sph, cam) is None
    moved = sph.copy()
    moved[:, 0] += 0.05                          # the arm moved: buffer resets
    assert reg.feed(seen, moved, cam) is None
    assert reg.feed(seen, moved, cam) is None
    out = reg.feed(seen, moved, cam)             # third STILL frame -> solve
    assert out is not None and out["ok"], out
    assert reg.done
    # the registrar solved against the LAST spheres (moved), so the truth
    # is delta minus the move
    assert np.linalg.norm(out["shift"] + delta - np.array([0.05, 0, 0])) < 0.005
    assert reg.feed(seen, moved, cam) is None    # done: silent thereafter


def test_self_registrar_gives_up_and_reports():
    from rammp_curobo.perception import SelfRegistrar

    sph = np.array([[0.0, 0.0, 0.3, 0.05]])
    far = np.random.default_rng(1).uniform(1.0, 2.0, size=(200, 3))
    reg = SelfRegistrar(frames=2, attempts=2)
    assert reg.feed(far, sph) is None
    out = reg.feed(far, sph)
    assert out is not None and not out["ok"] and not out["final"]
    reg.feed(far, sph)
    out = reg.feed(far, sph)
    assert not out["ok"] and out["final"] and reg.done
    assert "too few" in out["reason"]


def test_self_registrar_survives_all_empty_frames():
    # an Orbbec that comes up publishing all-zero depth produces eight
    # empty cropped frames in a row; that must be a "too few points"
    # outcome, not a np.vstack ValueError that kills the cameras node
    from rammp_curobo.perception import SelfRegistrar

    sph = np.array([[0.0, 0.0, 0.3, 0.05]])
    reg = SelfRegistrar(frames=3, attempts=1)
    assert reg.feed(np.empty((0, 3)), sph) is None
    assert reg.feed(np.empty((0, 3)), sph) is None
    out = reg.feed(np.empty((0, 3)), sph)
    assert out is not None and not out["ok"] and reg.done
    assert "too few" in out["reason"]
