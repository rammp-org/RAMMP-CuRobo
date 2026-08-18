"""Seek demo pure logic: text matching, localization, standoff synthesis."""

import numpy as np

from rammp_curobo_ros.seek_demo import (
    COCO_CLASSES,
    box_to_center,
    glance_pose,
    parse_target,
    standoff_pose,
)


def test_parse_target_matches_classes_and_synonyms():
    assert parse_target("go to the bottle") == "bottle"
    assert parse_target("Find the CUP please") == "cup"
    assert parse_target("grab the mug") == "cup"  # synonym
    assert parse_target("approach the wine glass") == "wine glass"
    assert parse_target("go to the unicorn") is None


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


def test_standoff_pose_backs_off_toward_base_and_faces_object():
    obj = [0.6, 0.0, 0.25]
    pos, quat = standoff_pose(obj, standoff=0.18)
    assert np.isclose(pos[0], 0.42, atol=1e-6) and abs(pos[1]) < 1e-9
    assert 0.15 <= pos[2] <= 0.55
    # tool z (facing direction) points at the object bearing
    x, y, z, w = quat
    tool_z = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
    assert np.isclose(np.arctan2(tool_z[1], tool_z[0]), 0.0, atol=1e-6)
    # too-close object: standoff clamps to min radius, never behind base
    pos2, _ = standoff_pose([0.25, 0.05, 0.2], standoff=0.18)
    assert np.hypot(pos2[0], pos2[1]) >= 0.30 - 1e-9


def test_glance_pose_points_camera_down_at_the_bench():
    pos, quat = glance_pose(0.0, np.radians(40))
    assert np.isclose(pos[2], 0.47)
    x, y, z, w = quat
    tool_z = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
    assert tool_z[2] < -0.5  # looking meaningfully downward
