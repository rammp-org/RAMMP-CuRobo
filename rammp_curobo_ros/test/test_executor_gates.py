"""Unit tests for the execution safety gates — no ROS runtime, no hardware.

Message classes are plain Python data holders in rclpy, so these run under
pytest with only the message packages on the PYTHONPATH (any sourced Humble
environment has them).
"""

import numpy as np
import pytest

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError:  # pragma: no cover
    pytest.skip('ROS message packages not on PYTHONPATH (source ROS 2 first)',
                allow_module_level=True)

from rammp_curobo_ros.conversions import duration_msg, msg_arrays, scaled_msg
from rammp_curobo_ros.executor import validate_goal_msg

JOINTS = ['joint_%d' % i for i in range(1, 8)]
POS_LIMITS = (np.full(7, -6.0), np.full(7, 6.0))
VEL_LIMITS = np.full(7, 1.4)


def _msg(n=20, dt=0.02, step=0.01):
    msg = JointTrajectory()
    msg.joint_names = list(JOINTS)
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [k * step] * 7
        p.velocities = [step / dt] * 7
        p.time_from_start = duration_msg((k + 1) * dt)
        msg.points.append(p)
    return msg


def test_clean_trajectory_passes():
    msg = _msg()
    q_now = list(msg.points[0].positions)
    assert validate_goal_msg(msg, JOINTS, q_now, POS_LIMITS, VEL_LIMITS,
                             0.05) == []


def test_gate_wrong_joint_names():
    msg = _msg()
    msg.joint_names[3] = 'jointx'
    problems = validate_goal_msg(msg, JOINTS, [0.0] * 7, POS_LIMITS,
                                 VEL_LIMITS, 0.05)
    assert problems and 'joints' in problems[0]


def test_gate_stale_start():
    msg = _msg()
    q_now = [0.5] * 7  # arm nowhere near the trajectory start
    problems = validate_goal_msg(msg, JOINTS, q_now, POS_LIMITS, VEL_LIMITS,
                                 0.05)
    assert any('stale plan' in p for p in problems)


def test_gate_no_joint_state():
    problems = validate_goal_msg(_msg(), JOINTS, None, POS_LIMITS, VEL_LIMITS,
                                 0.05)
    assert any('no current joint state' in p for p in problems)


def test_gate_discontinuity():
    msg = _msg()
    msg.points[10].positions = [2.0] * 7
    problems = validate_goal_msg(msg, JOINTS, list(msg.points[0].positions),
                                 POS_LIMITS, VEL_LIMITS, 0.05)
    assert any('discontinuity' in p for p in problems)


def test_gate_position_limits():
    msg = _msg()
    msg.points[-1].positions = [7.0] * 7
    problems = validate_goal_msg(msg, JOINTS, list(msg.points[0].positions),
                                 POS_LIMITS, VEL_LIMITS, 0.05)
    assert any('position limits' in p for p in problems)


def test_gate_velocity_limits():
    msg = _msg()
    msg.points[5].velocities = [3.0] * 7
    problems = validate_goal_msg(msg, JOINTS, list(msg.points[0].positions),
                                 POS_LIMITS, VEL_LIMITS, 0.05)
    assert any('velocity exceeds' in p for p in problems)


def test_gate_non_monotonic_time():
    msg = _msg()
    msg.points[5].time_from_start = duration_msg(0.01)
    problems = validate_goal_msg(msg, JOINTS, list(msg.points[0].positions),
                                 POS_LIMITS, VEL_LIMITS, 0.05)
    assert any('non-monotonic' in p for p in problems)


def test_scaling_dilates_msg_exactly():
    msg = _msg()
    slow = scaled_msg(msg, 0.25)
    p0, v0, t0 = msg_arrays(msg)
    p1, v1, t1 = msg_arrays(slow)
    np.testing.assert_array_equal(p0, p1)
    np.testing.assert_allclose(t1, t0 / 0.25)
    np.testing.assert_allclose(v1, v0 * 0.25)
    # scaled trajectory still passes the gates it must pass
    assert validate_goal_msg(slow, JOINTS, list(msg.points[0].positions),
                             POS_LIMITS, VEL_LIMITS, 0.05) == []
