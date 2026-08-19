"""Seek demo pure logic: text matching, localization, standoff synthesis."""

import numpy as np

from rammp_curobo_ros.seek_demo import (
    COCO_CLASSES,
    box_to_center,
    cluster_sightings,
    decide,
    glance_pose,
    joint_travel,
    parse_target,
    reconfirmed,
    standoff_pose,
    track_update,
)


def test_parse_target_matches_classes_and_synonyms():
    assert parse_target("go to the bottle") == "bottle"
    assert parse_target("Find the CUP please") == "cup"
    assert parse_target("grab the mug") == "cup"  # synonym
    assert parse_target("approach the wine glass") == "wine glass"
    assert parse_target("go to the unicorn") is None


def test_parse_target_survives_punctuation():
    assert parse_target("go to the bottle.") == "bottle"
    assert parse_target("where is my phone?") == "cell phone"
    assert parse_target("hand me the knife, please") == "knife"


def test_coco_has_80_classes_with_bottle_at_39():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[39] == "bottle"


def test_box_to_center_deprojects_the_core():
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    out = box_to_center([40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0)
    assert out is not None
    center, extent = out
    assert np.allclose(center, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(extent, [0.1, 0.1], atol=0.02)  # 20px at 0.5m/f100
    holes = np.zeros((100, 100), dtype=np.float32)
    assert box_to_center([40, 40, 60, 60], holes, 100, 100, 50, 50) is None


def test_box_to_center_ignores_a_single_flying_pixel():
    # audit regression: one near-range artifact must not hijack the
    # foreground cluster away from the true object
    depth = np.full((100, 100), 0.5, dtype=np.float32)
    depth[50, 50] = 0.09  # flying pixel inside the validity window
    out = box_to_center([40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0)
    assert out is not None
    center, _ = out
    assert np.isclose(center[2], 0.5, atol=0.02)


def test_box_to_center_mask_excludes_the_occluder():
    # left half of the core is a nearer occluder; the instance mask keeps
    # only the object's pixels — localization must land on the OBJECT
    depth = np.full((100, 100), 0.6, dtype=np.float32)
    depth[:, :50] = 0.3  # occluder
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 50:] = True  # detector says the object is the right half
    out = box_to_center([40, 40, 60, 60], depth, 100.0, 100.0, 50.0, 50.0, mask=mask)
    assert out is not None
    center, _ = out
    assert np.isclose(center[2], 0.6, atol=0.02)
    # without the mask the occluder's front face wins — documents why the
    # mask matters
    center_nomask, _ = box_to_center([40, 40, 60, 60], depth, 100, 100, 50, 50)
    assert np.isclose(center_nomask[2], 0.3, atol=0.02)


def test_standoff_pose_backs_off_toward_base_and_aims_at_object():
    obj = [0.6, 0.0, 0.25]
    pos, quat, gap = standoff_pose(obj, standoff=0.18)
    assert np.isclose(pos[0], 0.42, atol=1e-6) and abs(pos[1]) < 1e-9
    assert np.isclose(gap, 0.18, atol=1e-9)
    assert np.isclose(pos[2], 0.30, atol=1e-6)  # 5 cm above the object
    # tool z (facing direction) points at the object bearing AND pitches
    # down at its center (field 2026-08-19: the old level pose hovered
    # over a small bottle's cap and read as a miss)
    x, y, z, w = quat
    tool_z = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
    assert np.isclose(np.arctan2(tool_z[1], tool_z[0]), 0.0, atol=1e-6)
    expected_pitch = np.arctan2(pos[2] - obj[2], gap)
    assert np.isclose(tool_z[2], -np.sin(expected_pitch), atol=1e-6)
    # a bench-level object: height floor keeps the tool at 0.12 m
    low, lquat, lgap = standoff_pose([0.62, -0.17, 0.01], standoff=0.18)
    assert np.isclose(low[2], 0.12, atol=1e-6)
    lx, ly, lz, lw = lquat
    ltz = 1 - 2 * (lx * lx + ly * ly)
    assert ltz < -0.3  # still pointing down at the bottle, not over it


def test_standoff_pose_reports_reduced_gap_and_refuses_too_close():
    # inside the min radius the gap shrinks — must be REPORTED, not the
    # requested standoff (audit regression)
    pos, _, gap = standoff_pose([0.40, 0.0, 0.2], standoff=0.18)
    assert np.isclose(np.hypot(pos[0], pos[1]), 0.30, atol=1e-9)
    assert np.isclose(gap, 0.10, atol=1e-9)
    # object so close the tool would land past it: REFUSE
    assert standoff_pose([0.25, 0.05, 0.2], standoff=0.18) is None
    assert standoff_pose([0.36, 0.0, 0.2], standoff=0.18) is None  # gap < 8 cm
    # far object: outer clamp grows the gap, reported honestly
    _, _, far_gap = standoff_pose([1.0, 0.0, 0.2], standoff=0.18)
    assert np.isclose(far_gap, 0.28, atol=1e-9)


EXT = (0.06, 0.20)


def test_decide_confirms_across_viewpoints():
    s = [
        (0, [0.60, -0.15, 0.02], EXT, 0.9),
        (1, [0.62, -0.13, 0.03], EXT, 0.8),
    ]
    status, ranked = decide(cluster_sightings(s))
    assert status == "ok" and len(ranked) == 1
    c = ranked[0]
    assert c["glances"] == {0, 1} and c["n"] == 2
    assert np.allclose(c["center"], [0.609, -0.141, 0.025], atol=0.005)


def test_decide_decoy_from_one_viewpoint_loses_the_vote():
    # field regression 2026-08-19: a second bottle-shaped object seen
    # from ONE glance must not veto the target confirmed from two
    s = [
        (0, [0.60, -0.15, 0.02], EXT, 0.9),
        (1, [0.35, 0.62, 0.04], EXT, 0.83),  # decoy, single viewpoint
        (2, [0.61, -0.14, 0.02], EXT, 0.7),
    ]
    status, ranked = decide(cluster_sightings(s))
    assert status == "ok" and len(ranked) == 1
    assert np.linalg.norm(ranked[0]["center"] - [0.6, -0.15, 0.02]) < 0.05


def test_decide_same_glance_repeats_do_not_confirm():
    # two sightings from the SAME viewpoint share systematics — one
    # distinct glance, never confirmed (audit 2026-08-18 rationale)
    s = [
        (0, [0.60, -0.15, 0.02], EXT, 0.9),
        (0, [0.61, -0.14, 0.02], EXT, 0.8),
    ]
    status, ranked = decide(cluster_sightings(s))
    assert status == "unconfirmed"
    assert ranked[0]["glances"] == {0} and ranked[0]["n"] == 2
    assert decide([]) == ("unseen", [])


def test_decide_two_real_objects_ambiguous_unless_pick_nearest():
    s = [
        (0, [0.60, -0.15, 0.02], EXT, 0.9),
        (0, [0.40, 0.30, 0.03], EXT, 0.9),
        (1, [0.61, -0.14, 0.02], EXT, 0.8),
        (1, [0.41, 0.31, 0.03], EXT, 0.8),
    ]
    status, ranked = decide(cluster_sightings(s))
    assert status == "ambiguous" and len(ranked) == 2  # surfaced, not swallowed
    status, ranked = decide(cluster_sightings(s), pick="nearest")
    assert status == "ok"
    # hypot(0.40, 0.30) = 0.50 < hypot(0.60, -0.15) — nearest first
    assert np.isclose(ranked[0]["center"][0], 0.405, atol=0.01)


def test_cluster_sightings_never_chains_beyond_the_radius():
    # audit 2026-08-19: complete linkage — the drifting weighted center
    # must not bridge sightings whose PAIRWISE spread exceeds 10 cm
    s = [
        (0, [0.60, 0.0, 0.0], EXT, 0.5),
        (1, [0.69, 0.0, 0.0], EXT, 0.95),
        (2, [0.755, 0.0, 0.0], EXT, 0.9),  # 15.5 cm from the first
    ]
    clusters = cluster_sightings(s)
    assert len(clusters) == 2
    for c in clusters:
        m = np.array(c["members"])
        assert np.linalg.norm(m[:, None] - m[None, :], axis=2).max() <= 0.10


def test_cluster_sightings_between_two_objects_joins_the_nearest():
    # audit 2026-08-19: a sighting eligible for two clusters must join
    # the NEAREST, not the first-created (which stole confirmations)
    s = [
        (0, [0.5, 0.00, 0.0], EXT, 0.9),
        (0, [0.5, 0.15, 0.0], EXT, 0.9),
        (1, [0.5, 0.08, 0.0], EXT, 0.8),  # 8 cm from A, 7 cm from B
    ]
    confirmed = [c for c in cluster_sightings(s) if len(c["glances"]) >= 2]
    assert len(confirmed) == 1
    assert np.isclose(
        confirmed[0]["center"][1], (0.15 * 0.9 + 0.08 * 0.8) / 1.7, atol=1e-6
    )


def test_reconfirmed_needs_a_second_frame_within_tolerance():
    # the confident fast path's only remaining guard: a second same-pose
    # frame must re-localize within 5 cm — catches flicker, not decoys
    first = [0.60, -0.15, 0.02]
    near = [(0, [0.62, -0.14, 0.02], EXT, 0.7)]
    far = [(0, [0.70, -0.15, 0.02], EXT, 0.9)]
    assert reconfirmed(first, near)
    assert not reconfirmed(first, far)
    assert not reconfirmed(first, [])  # no re-detection -> no shortcut
    assert reconfirmed(first, far + near)  # any agreeing sighting counts


def test_track_update_follows_the_nearest_and_ignores_noise_and_decoys():
    cur = [0.60, -0.15, 0.02]
    moved = [(0, [0.60, 0.05, 0.02], EXT, 0.8)]  # 20 cm hop
    assert np.allclose(track_update(cur, moved), [0.60, 0.05, 0.02])
    noise = [(0, [0.61, -0.14, 0.02], EXT, 0.8)]  # <5 cm -> no replan
    assert track_update(cur, noise) is None
    decoy = [(0, [0.20, 0.60, 0.02], EXT, 0.9)]  # >35 cm leash -> ignored
    assert track_update(cur, decoy) is None
    both = decoy + moved  # decoy plus the real move: nearest wins
    assert np.allclose(track_update(cur, both), [0.60, 0.05, 0.02])
    assert track_update(cur, []) is None


def test_joint_travel_exposes_winding_a_net_check_would_miss():
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    traj = JointTrajectory()
    traj.joint_names = ["joint_1", "joint_7"]
    # joint_7 winds out pi and back (net 0, travel 2*pi); joint_1 moves 0.3
    for j1, j7 in [(0.0, 0.0), (0.1, 1.6), (0.2, 3.14), (0.3, 1.6), (0.3, 0.0)]:
        p = JointTrajectoryPoint(positions=[j1, j7], time_from_start=Duration())
        traj.points.append(p)
    t = joint_travel(traj)
    assert np.isclose(t["joint_1"], 0.3, atol=1e-6)
    assert t["joint_7"] > 6.0  # the flip a net end-start check would call 0


def test_standoff_pose_z_min_ladder_raises_the_pose():
    # the approach relax ladder passes growing z_min when the low pose
    # IK_FAILs against unpurged perceived boxes (field 2026-08-19)
    obj = [0.62, -0.17, 0.01]
    low = standoff_pose(obj, standoff=0.18, z_min=0.12)
    high = standoff_pose(obj, standoff=0.18, z_min=0.30)
    assert np.isclose(low[0][2], 0.12) and np.isclose(high[0][2], 0.30)
    assert np.isclose(low[2], high[2])  # gap unchanged by the z relax


def test_purged_count_parses_the_cameras_reply():
    from rammp_curobo_ros.seek_demo import _purged_count

    assert _purged_count("ignoring 0.13 x 0.13 x 0.20 m at (0.62, -0.17, "
                         "0.01); 0 mapped voxels purged") == 0
    assert _purged_count("...; 17 mapped voxels purged") == 17
    assert _purged_count("cameras node not up") is None
    assert _purged_count(None) is None


def test_glance_pose_points_camera_down_at_the_bench():
    pos, quat = glance_pose(0.0, np.radians(55))
    assert np.isclose(pos[2], 0.42)
    x, y, z, w = quat
    tool_z = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
    assert tool_z[2] < -0.7  # looking steeply downward (FOV audit)
