"""Grasp frame conversion: GraspGenX camera-frame pose -> cuRobo goal.

The live client needs the GraspGenX server; these pin the math that
silently ruins grasps when it's wrong.
"""

import numpy as np

from rammp_curobo_ros.grasps import (
    ROBOTIQ_2F85_SWEEP,
    mat_to_quat_xyzw,
    pregrasp,
    quat_to_mat3,
    to_base,
)


def test_mat_to_quat_roundtrips_including_trace_negative_branches():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=(3, 3))
        q_, _ = np.linalg.qr(a)
        if np.linalg.det(q_) < 0:
            q_[:, 0] *= -1
        back = quat_to_mat3(mat_to_quat_xyzw(q_))
        assert np.allclose(back, q_, atol=1e-6)
    # 180 deg rotations exercise every argmax branch
    for m in (np.diag([1.0, -1.0, -1.0]),
              np.diag([-1.0, 1.0, -1.0]),
              np.diag([-1.0, -1.0, 1.0])):
        assert np.allclose(quat_to_mat3(mat_to_quat_xyzw(m)), m, atol=1e-6)


def test_to_base_pushes_the_goal_along_the_grasp_approach_axis():
    # camera 0.5 m up, looking along base +x: cam +Z -> base +x
    rot_cam = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    trans_cam = np.array([0.30, 0.0, 0.50])
    grasp = np.eye(4)
    grasp[:3, 3] = [0.0, 0.0, 0.40]  # 40 cm straight ahead of the camera
    pos, quat = to_base(grasp, rot_cam, trans_cam, tool_offset=0.120)
    # grasp centre lands 0.40 m along base +x; tool_frame goal is pushed
    # 0.12 m further along the grasp's own +Z (also base +x here)
    assert np.allclose(pos, [0.30 + 0.40 + 0.120, 0.0, 0.50], atol=1e-9)
    # tool z of the goal must equal the grasp approach axis in base
    assert np.allclose(quat_to_mat3(quat)[:, 2], rot_cam[:, 2], atol=1e-9)


def test_tool_offset_is_the_fingertip_knob():
    rot_cam, trans_cam = np.eye(3), np.zeros(3)
    grasp = np.eye(4)
    a = to_base(grasp, rot_cam, trans_cam, tool_offset=0.120)[0]
    b = to_base(grasp, rot_cam, trans_cam, tool_offset=0.136)[0]
    # GraspGenX calls the 2F-85 fingertip 0.136; our tool_frame is 0.120
    assert np.isclose(float(b[2] - a[2]), 0.016, atol=1e-9)
    assert np.isclose(ROBOTIQ_2F85_SWEEP["fingertip_depth"], 0.136)


def test_pregrasp_backs_off_along_approach_and_never_sideways():
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    pos = np.array([0.6, 0.0, 0.3])
    pre = pregrasp(pos, rot, standoff=0.10)
    assert np.allclose(pre, pos - rot[:, 2] * 0.10)
    # backing off must increase the distance to the grasp by exactly the
    # standoff, along the approach axis only
    assert np.isclose(float(np.linalg.norm(pre - pos)), 0.10)


def test_sweep_params_match_the_2f85_and_stay_within_its_stroke():
    assert ROBOTIQ_2F85_SWEEP["gripper_type"] == 1  # revolute 2-finger
    # open aperture must not exceed the 2F-85's 85 mm stroke
    assert np.isclose(ROBOTIQ_2F85_SWEEP["extents_open"][0], 0.085)
