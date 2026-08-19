"""Seeker controller pure logic: viewpoint quantization and the belief
rules it shares with the seek helpers (the loop itself is exercised on
the bench and audited; these pin the arithmetic)."""

import numpy as np

from rammp_curobo_ros.seek_demo import cluster_sightings, track_update
from rammp_curobo_ros.seeker import viewpoint_key

EXT = np.array([0.06, 0.20])


def test_viewpoint_key_quantizes_camera_motion():
    a = viewpoint_key([0.100, 0.000, 0.400])
    same = viewpoint_key([0.101, -0.014, 0.409])  # parked camera jitter
    moved = viewpoint_key([0.180, 0.000, 0.400])  # 8 cm of real motion
    assert a == same and a != moved


def test_parked_camera_never_triangulates_but_a_moved_one_does():
    # the continuous controller has no glance indices — viewpoint keys
    # must carry the same 'two DISTINCT viewpoints' guarantee
    parked = viewpoint_key([0.10, 0.0, 0.40])
    sights = [
        (parked, np.array([0.60, -0.15, 0.02]), EXT, 0.9),
        (parked, np.array([0.61, -0.14, 0.02]), EXT, 0.9),
    ]
    assert all(
        len(c["glances"]) < 2 for c in cluster_sightings(sights)
    )
    sights.append(
        (viewpoint_key([0.30, 0.1, 0.40]), np.array([0.60, -0.15, 0.02]), EXT, 0.8)
    )
    assert any(len(c["glances"]) >= 2 for c in cluster_sightings(sights))


def test_belief_refresh_uses_zero_min_move():
    # a parked target must still REFRESH the belief (min_move=0), or it
    # would go stale while the camera stares straight at it
    cur = [0.60, -0.15, 0.02]
    unmoved = [(0, np.array([0.61, -0.14, 0.02]), EXT, 0.8)]
    assert track_update(cur, unmoved, min_move=0.0) is not None
    assert track_update(cur, unmoved) is None  # action-side dead-band holds
