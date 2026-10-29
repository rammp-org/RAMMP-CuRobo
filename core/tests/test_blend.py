"""blend.py: overlapping rest-to-rest legs into one trajectory — no cuRobo."""

import numpy as np
import pytest

from rammp_curobo import blend

DT = 0.02
VMAX = np.full(7, 1.2)
AMAX = np.full(7, 25.0)


def straight_leg(q0, q1, vmax=VMAX, amax=AMAX, dt=DT):
    """(p, v, a) of a straight joint move q0 -> q1 from rest to rest, on a
    minimum-jerk profile timed so no joint passes `vmax` or `amax`."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    d = q1 - q0
    # minimum jerk peaks at 1.875 x the mean speed and 5.774 d / T^2
    duration = max(
        np.max(1.875 * np.abs(d) / vmax), np.max(np.sqrt(5.7735 * np.abs(d) / amax)), dt
    )
    n = int(np.ceil(duration / dt))
    u = np.arange(n + 1)[:, None] / n
    T = n * dt
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    sdd = (60 * u - 180 * u**2 + 120 * u**3) / T**2
    return q0 + s * d, sd * d, sdd * d


def _legs(*points):
    """Straight legs through `points` (7-vectors), each from rest to rest."""
    qs = [np.asarray(q, dtype=float) for q in points]
    return [straight_leg(a, b) for a, b in zip(qs, qs[1:])]


def _corner():
    """Two legs that turn 90 degrees in joints 1 and 2 (joint 1, then 2)."""
    return _legs(np.zeros(7), np.r_[0.6, np.zeros(6)], np.r_[0.6, 0.6, np.zeros(5)])


def test_no_overlap_is_the_legs_end_to_end():
    legs = _corner()
    p, v, a = blend.blend(legs, [0])
    n_a = len(legs[0][0])
    assert len(p) == n_a + len(legs[1][0]) - 1  # the shared waypoint once
    np.testing.assert_allclose(p[:n_a], legs[0][0])
    np.testing.assert_allclose(p[n_a - 1 :], legs[1][0])
    assert np.abs(v[n_a - 1]).max() == 0.0  # stopped on the waypoint


def test_an_overlap_keeps_moving_through_the_waypoint_and_ends_on_the_goal():
    legs = _corner()
    (m,) = blend.max_overlaps([len(leg[0]) for leg in legs], [True])
    assert m == min(len(leg[0]) for leg in legs) - 1  # a whole leg: both ends stop
    p, v, a = blend.blend(legs, [m])
    w = blend.window([len(leg[0]) for leg in legs], [m], 0)
    speed = np.linalg.norm(v, axis=1)
    assert speed[1:-1].min() > 0.0  # never at rest between start and goal
    assert speed[(w.start + w.stop) // 2] > 0.5  # at full speed past the corner
    np.testing.assert_allclose(p[-1], legs[-1][0][-1], atol=1e-12)
    assert len(p) == len(legs[0][0]) + len(legs[1][0]) - 1 - m
    # the corner is cut: the waypoint itself is never reached
    assert np.linalg.norm(p - legs[0][0][-1], axis=1).min() > 0.01


def test_blended_velocities_and_accelerations_are_the_positions_derivatives():
    legs = _legs(np.zeros(7), np.full(7, 0.3), np.r_[0.8, np.zeros(6)], np.zeros(7))
    lengths = [len(leg[0]) for leg in legs]
    overlaps = blend.max_overlaps(lengths, [True, True])
    p, v, a = blend.blend(legs, overlaps)
    np.testing.assert_allclose(np.gradient(p, DT, axis=0)[1:-1], v[1:-1], atol=0.02)
    np.testing.assert_allclose(
        np.gradient(v, DT, axis=0)[1:-1], a[1:-1], atol=0.1 * np.abs(a).max()
    )
    assert np.abs(v[[0, -1]]).max() == 0.0 and np.abs(a[[0, -1]]).max() == 0.0


def test_max_overlaps_gives_a_leg_whole_only_to_a_junction_with_a_stop_beside_it():
    lengths = [41, 61, 21]
    assert blend.max_overlaps(lengths, [True, True]) == [29, 20]
    assert blend.max_overlaps(lengths, [True, False]) == [40, 0]
    assert blend.max_overlaps(lengths, [False, True]) == [0, 20]


@pytest.mark.parametrize("n", [94, 95])
def test_windows_at_their_caps_share_no_state(n):
    lengths = [n, n, n]
    overlaps = blend.max_overlaps(lengths, [True, True])
    w0, w1 = (blend.window(lengths, overlaps, k) for k in (0, 1))
    assert w0.stop <= w1.start  # junction 1's window starts after junction 0's ends
    leg = (np.zeros((n, 7)), np.zeros((n, 7)), np.zeros((n, 7)))
    blend.blend([leg] * 3, overlaps)  # accepted: within every leg's room
    with pytest.raises(ValueError, match="junction 1 is outside"):
        # the middle leg's two overlaps may total n - 2 steps; n - 1 shares a state
        blend.blend([leg] * 3, [overlaps[0], n - 1 - overlaps[0]])


def test_overlapping_windows_or_an_overlap_past_a_leg_are_refused():
    legs = _legs(np.zeros(7), np.full(7, 0.3), np.full(7, 0.6), np.full(7, 0.9))
    n = len(legs[1][0])
    with pytest.raises(ValueError, match="outside"):
        blend.blend(legs[:2], [n])
    with pytest.raises(ValueError, match="junction 1 is outside"):
        blend.blend(legs, [n // 2, n // 2])  # sharing a state of leg 2
    with pytest.raises(ValueError, match="overlaps for"):
        blend.blend(legs, [])


def test_a_leg_without_accelerations_blends_without_them():
    (p0, v0, _), leg = _corner()
    assert blend.blend([(p0, v0, None), leg], [3])[2] is None


def test_choose_overlaps_returns_only_what_accept_allowed():
    caps = [30, 20]
    asked = []

    def accept(k, overlaps):
        asked.append((k, overlaps[k]))
        return overlaps[k] <= (13 if k == 0 else 4)

    assert blend.choose_overlaps(caps, accept) == [13, 4]
    assert all(m > 0 for _, m in asked)  # 0 is never asked: always accepted
    assert blend.choose_overlaps(caps, lambda k, o: True) == [30, 20]
    assert blend.choose_overlaps(caps, lambda k, o: False) == [0, 0]
    assert blend.choose_overlaps([0, 5], lambda k, o: k == 1) == [0, 5]


def test_resample_keeps_the_ends_and_slows_by_under_a_step():
    q0, q1 = np.zeros(7), np.r_[1.0, -0.4, np.zeros(5)]
    p, v, a = straight_leg(q0, q1, VMAX, AMAX, DT)
    slow = 0.3  # a leg at 0.3 of its speed: dt / 0.3 apart
    rp, rv, ra = blend.resample(p, v * slow, DT / slow, DT)
    duration = (len(p) - 1) * DT / slow
    assert duration <= (len(rp) - 1) * DT < duration + DT
    np.testing.assert_allclose(rp[[0, -1]], [q0, q1], atol=1e-12)
    assert np.abs(rv[[0, -1]]).max() < 1e-12
    np.testing.assert_allclose(np.abs(rv[:, 0]).max(), slow * 1.2, rtol=0.02)
    np.testing.assert_allclose(np.gradient(rp, DT, axis=0)[1:-1], rv[1:-1], atol=0.01)
    np.testing.assert_allclose(np.gradient(rv, DT, axis=0)[1:-1], ra[1:-1], atol=0.05)


def test_resample_onto_its_own_grid_is_the_leg():
    p, v, _ = _corner()[0]
    rp, rv, _ = blend.resample(p, v, DT, DT)
    np.testing.assert_allclose(rp, p, atol=1e-12)
    np.testing.assert_allclose(rv, v, atol=1e-12)
