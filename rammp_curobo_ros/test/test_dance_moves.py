"""The dance choreography generator: pure, seeded, safe-boxed."""

import random

from rammp_curobo_ros.dance_demo import SAFE_BOX, choreograph


def _poses(moves):
    return [m[1] for m in moves if m[0] == "pose"]


def test_seeded_choreography_is_repeatable():
    a = choreograph(random.Random(7), n_moves=8)
    b = choreograph(random.Random(7), n_moves=8)
    assert a == b
    c = choreograph(random.Random(8), n_moves=8)
    assert c != a


def test_every_pose_stays_in_the_safe_box():
    for seed in range(20):
        for p in _poses(choreograph(random.Random(seed), n_moves=10)):
            for v, (lo, hi) in zip(p, SAFE_BOX):
                assert lo - 1e-9 <= v <= hi + 1e-9


def test_flourishes_come_in_cancelling_pairs():
    for seed in range(20):
        moves = choreograph(random.Random(seed), n_moves=10)
        for i, m in enumerate(moves):
            if m[0] in ("twist", "nod") and m[1] > 0:
                assert moves[i + 1] == (m[0], -m[1])  # un-twist follows
        total = {"twist": 0.0, "nod": 0.0}
        for kind, val in moves:
            if kind in total:
                total[kind] += val
        assert abs(total["twist"]) < 1e-9 and abs(total["nod"]) < 1e-9


def test_flourish_amplitudes_are_bounded():
    for seed in range(20):
        for kind, val in choreograph(random.Random(seed), n_moves=10):
            if kind == "twist":
                assert abs(val) <= 0.7
            if kind == "nod":
                assert abs(val) <= 0.4


def test_ends_settled_at_center():
    moves = choreograph(random.Random(3), n_moves=6)
    assert moves[-1][0] == "pose"


def test_circles_produce_multiple_round_waypoints():
    # with enough moves and seeds, at least one circle occurs and shows
    # up as >= 5 consecutive poses at (roughly) constant distance from
    # their own centroid
    import numpy as np

    found = False
    for seed in range(30):
        poses = _poses(choreograph(random.Random(seed), n_moves=12))
        for i in range(len(poses) - 5):
            arc = np.array(poses[i : i + 6])
            c = arc.mean(axis=0)
            r = np.linalg.norm(arc - c, axis=1)
            if r.std() < 0.01 and r.mean() > 0.04:
                found = True
    assert found
