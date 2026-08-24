"""Eye-to-hand calibration: fixed camera watching a tag on the gripper.

Unknown: where the camera is in base_link. Known per sample: the arm's
own FK (base -> end_effector_link) and the tag's pose in the camera
(solvePnP on a fiducial). The tag's offset on the gripper is ALSO
unknown and does not need measuring — that is what makes this method
practical: tape the tag on crooked and it still solves.

Poses must ROTATE, not just translate: a set of samples that only
translate (or rotates about one axis) is degenerate and the solver
returns nonsense with no warning. `pose_spread` is the tripwire.
"""

import numpy as np

from rammp_curobo.perception import mat_to_quat_xyzw  # noqa: F401  (re-export: calibrate_orbbec imports it here)


def _rt(t4):
    return np.asarray(t4)[:3, :3], np.asarray(t4)[:3, 3]


def invert(t4):
    r, t = _rt(t4)
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def pose_spread(mats):
    """Largest pairwise relative-rotation angle (rad) across the samples.

    Below ~0.5 rad the hand-eye problem is ill-conditioned — the same
    degeneracy that made a single-axis test 'fail' during the 2026-08
    calibration work when the solver was in fact correct."""
    best = 0.0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            r = np.asarray(mats[i])[:3, :3].T @ np.asarray(mats[j])[:3, :3]
            c = (float(np.trace(r)) - 1.0) / 2.0
            best = max(best, float(np.arccos(np.clip(c, -1.0, 1.0))))
    return best


def solve_eye_to_hand(base_T_ee, cam_T_tag):
    """(4x4 base_T_cam, residual_m) from paired samples.

    base_T_ee: arm FK per sample. cam_T_tag: tag pose in the camera.
    Eye-TO-hand is eye-in-hand with the robot transform inverted, so we
    hand OpenCV base->ee and read the result as camera->base.
    """
    import cv2

    if len(base_T_ee) != len(cam_T_tag) or len(base_T_ee) < 3:
        raise ValueError("need >= 3 paired samples")
    r_b2e, t_b2e, r_t2c, t_t2c = [], [], [], []
    for be, ct in zip(base_T_ee, cam_T_tag):
        eb = invert(be)                      # base -> ee  (the inversion)
        r, t = _rt(eb)
        r_b2e.append(r)
        t_b2e.append(t)
        r, t = _rt(ct)
        r_t2c.append(r)
        t_t2c.append(t)
    r_c2b, t_c2b = cv2.calibrateHandEye(
        r_b2e, t_b2e, r_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK
    )
    base_T_cam = np.eye(4)
    base_T_cam[:3, :3] = r_c2b
    base_T_cam[:3, 3] = np.asarray(t_c2b).ravel()

    # residual: the tag must sit at ONE fixed spot on the gripper, so
    # ee_T_tag should be identical across samples — its scatter is the
    # honest error measure.
    pts = []
    for be, ct in zip(base_T_ee, cam_T_tag):
        pts.append((invert(be) @ base_T_cam @ np.asarray(ct))[:3, 3])
    pts = np.array(pts)
    residual = float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return base_T_cam, residual

