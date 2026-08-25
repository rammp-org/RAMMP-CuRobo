"""The plan actions' start_joints contract — no ROS runtime, no hardware.

start_joints is REQUIRED: this planner never sources the arm's state, so a
goal that omits it cannot be planned. The old behaviour (empty = subscribe
to /joint_states and work it out) is what coupled the planner to the robot,
and a wrong-length vector fell through that same path and misreported as
"no fresh joint state" (issue #5).
"""

import pytest

try:
    from rammp_curobo_ros.planner_node import check_start_joints
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )

JOINTS = ["joint_%d" % i for i in range(1, 8)]


def test_correct_length_accepted():
    assert check_start_joints([0.0] * 7, JOINTS) is None


def test_empty_rejected_as_required():
    why = check_start_joints([], JOINTS)
    assert why is not None
    assert "start_joints is required" in why
    # The caller must learn it owns the state — not go looking for a topic.
    assert "joint_states" not in why


def test_wrong_length_reports_the_length():
    """Issue #5: a length mismatch used to surface as 'no fresh joint
    state on /joint_states', sending callers to debug the wrong problem."""
    why = check_start_joints([0.0] * 6, JOINTS)
    assert why is not None
    assert "6" in why and "7" in why
    assert "joint_states" not in why


def test_non_finite_rejected():
    assert check_start_joints([float("nan")] + [0.0] * 6, JOINTS) is not None
    assert check_start_joints([float("inf")] + [0.0] * 6, JOINTS) is not None
