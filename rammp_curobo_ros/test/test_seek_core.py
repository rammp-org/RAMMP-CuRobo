"""seek_core: parsing, localization, tracking and motion-sanity helpers."""

import numpy as np

from rammp_curobo_ros.seek_core import (
    COCO_CLASSES,
    box_to_center,
    glance_pose,
    joint_travel,
    parse_target,
    purged_count,
    roll_about_tool_z,
    stable_fix,
    track_update,
)

EXT = (0.06, 0.20)


def test_parse_target_matches_classes_synonyms_and_punctuation():
    assert parse_target("go to the bottle.") == "bottle"
    assert parse_target("Find the CUP please") == "cup"
    assert parse_target("grab the mug") == "cup"
    assert parse_target("approach the wine glass") == "wine glass"
    assert parse_target("where is my phone?") == "cell phone"
    assert parse_target("go to the unicorn") is None
    assert len(COCO_CLASSES) == 80 and COCO_CLASSES[39] == "bottle"


def test_box_to_center_deprojects_the_core():
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    center, extent = box_to_center([40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0)
    assert np.allclose(center, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(extent, [0.1, 0.1], atol=0.02)
    holes = np.zeros((100, 100), dtype=np.float32)
    assert box_to_center([40, 40, 60, 60], holes, 100, 100, 50, 50) is None


def test_box_to_center_ignores_a_single_flying_pixel():
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    depth[50, 50] = 0.2  # near-range artifact must not hijack the anchor
    center, _ = box_to_center([40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0)
    assert np.isclose(center[2], 0.5, atol=0.02)


def test_box_to_center_mask_excludes_the_occluder():
    depth = np.full((100, 100), 0.6, dtype=np.float32)
    depth[:, :50] = 0.3  # occluder
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 50:] = True
    center, _ = box_to_center(
        [40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0, mask=mask
    )
    assert np.isclose(center[2], 0.6, atol=0.02)


def test_box_to_center_min_depth_excludes_the_gripper_fingers():
    # field regression 2026-08-19: fingers 0.10-0.14 m from the lens
    # anchored the target and the arm chased itself into the table
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    depth[45:55, 40:60] = 0.12  # finger in the core
    center, _ = box_to_center(
        [40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0, min_depth=0.16
    )
    assert np.isclose(center[2], 0.5, atol=0.02)


def test_track_update_leash_and_dead_band():
    cur = [0.60, -0.15, 0.02]
    moved = [(0, [0.60, 0.05, 0.02], EXT, 0.8)]
    assert np.allclose(track_update(cur, moved), [0.60, 0.05, 0.02])
    noise = [(0, [0.61, -0.14, 0.02], EXT, 0.8)]
    assert track_update(cur, noise) is None  # < min_move
    assert track_update(cur, noise, min_move=0.0) is not None  # pure refresh
    decoy = [(0, [0.20, 0.60, 0.02], EXT, 0.9)]
    assert track_update(cur, decoy) is None  # outside the leash
    assert np.allclose(track_update(cur, decoy + moved), [0.60, 0.05, 0.02])


def test_stable_fix_needs_n_agreeing_frames():
    a = ([0.60, -0.15, 0.02], 0.0)
    b = ([0.62, -0.14, 0.02], 0.1)
    c = ([0.61, -0.15, 0.03], 0.2)
    far = ([0.75, -0.15, 0.02], 0.3)
    assert stable_fix([a, b]) is None  # too few
    fix = stable_fix([a, b, c])
    assert np.allclose(fix, [0.61, -0.15, 0.02], atol=0.01)
    assert stable_fix([b, c, far]) is None  # disagreement blocks


def test_joint_travel_exposes_winding():
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    traj = JointTrajectory()
    traj.joint_names = ["joint_1", "joint_7"]
    for j1, j7 in [(0.0, 0.0), (0.1, 1.6), (0.2, 3.14), (0.3, 1.6), (0.3, 0.0)]:
        traj.points.append(
            JointTrajectoryPoint(positions=[j1, j7], time_from_start=Duration())
        )
    t = joint_travel(traj)
    assert np.isclose(t["joint_1"], 0.3, atol=1e-6)
    assert t["joint_7"] > 6.0  # net end-start would call this 0


def test_roll_about_tool_z_keeps_the_aim():
    from rammp_curobo.perception import quat_to_mat
    from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW

    q = roll_about_tool_z(HOME_QUAT_XYZW, np.pi / 2)
    rolled = quat_to_mat(*q)
    flat = quat_to_mat(*HOME_QUAT_XYZW)
    assert np.allclose(rolled[:, 2], flat[:, 2], atol=1e-9)  # same tool axis
    assert not np.allclose(rolled[:, 0], flat[:, 0])  # rolled about it


def test_glance_pose_points_camera_down_at_the_bench():
    pos, quat = glance_pose(0.0, np.radians(55))
    assert np.isclose(pos[2], 0.42)
    x, y, z, w = quat
    tool_z2 = 1 - 2 * (x * x + y * y)
    assert tool_z2 < -0.7  # looking steeply down


def test_purged_count_parses_the_cameras_reply():
    assert purged_count("...; 0 mapped voxels purged") == 0
    assert purged_count("...; 17 mapped voxels purged") == 17
    assert purged_count("cameras node not up") is None
    assert purged_count(None) is None


def test_seeker_calls_only_methods_that_exist():
    # regression: tick() kept calling a deleted _heartbeat and the seek
    # loop died with AttributeError on the first quick re-attempt. Pure
    # source check (a Seeker needs a live node, so no instance here).
    import ast
    import importlib.util

    spec = importlib.util.find_spec("rammp_curobo_ros.seeker")
    with open(spec.origin) as f:
        tree = ast.parse(f.read())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Seeker"
    )
    defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    called = {
        node.func.attr
        for node in ast.walk(cls)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }
    assert called <= defined, called - defined
