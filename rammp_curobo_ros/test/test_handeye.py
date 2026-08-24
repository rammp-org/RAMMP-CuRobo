"""Eye-to-hand solver, proved against synthetic ground truth."""

import numpy as np

from rammp_curobo_ros.handeye import invert, pose_spread, solve_eye_to_hand


def _rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def _t(r, p):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = p
    return m


def _synthetic(n=8, seed=0):
    rng = np.random.default_rng(seed)
    base_T_cam = _t(_rot([0.2, 1.0, 0.3], 2.1), [0.9, -0.5, 0.7])
    ee_T_tag = _t(_rot([1.0, 0.4, -0.2], 0.6), [0.03, -0.02, 0.05])
    base_T_ee, cam_T_tag = [], []
    for _ in range(n):
        be = _t(_rot(rng.normal(size=3), rng.uniform(0.4, 2.4)),
                [rng.uniform(0.35, 0.7), rng.uniform(-0.3, 0.3),
                 rng.uniform(0.15, 0.5)])
        base_T_ee.append(be)
        cam_T_tag.append(invert(base_T_cam) @ be @ ee_T_tag)
    return base_T_cam, base_T_ee, cam_T_tag


def test_recovers_the_camera_pose_exactly_from_clean_samples():
    truth, be, ct = _synthetic()
    got, residual = solve_eye_to_hand(be, ct)
    assert np.allclose(got, truth, atol=1e-6)
    assert residual < 1e-6


def test_survives_realistic_detection_noise():
    truth, be, ct = _synthetic(n=12, seed=3)
    rng = np.random.default_rng(1)
    noisy = []
    for m in ct:
        m = m.copy()
        m[:3, 3] += rng.normal(scale=0.002, size=3)      # 2 mm
        m[:3, :3] = _rot(rng.normal(size=3), rng.normal(scale=0.01)) @ m[:3, :3]
        noisy.append(m)
    got, residual = solve_eye_to_hand(be, noisy)
    assert np.linalg.norm(got[:3, 3] - truth[:3, 3]) < 0.02   # < 2 cm
    assert residual < 0.02


def test_pose_spread_flags_the_degenerate_single_axis_case():
    # rotating about ONE axis is the classic hand-eye degeneracy
    same = [_t(_rot([0, 0, 1.0], a), [0.5, 0, 0.3]) for a in (0.0, 0.2, 0.4)]
    assert pose_spread(same) < 0.5
    varied = [_t(_rot(a, 1.2), [0.5, 0, 0.3])
              for a in ([1, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]
    assert pose_spread(varied) > 1.0

