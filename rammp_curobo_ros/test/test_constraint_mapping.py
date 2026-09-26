"""Mapping from PlanToPose goal fields to the core's constraint types.

No ROS runtime and no GPU — same shape as test_start_joints_contract.py.
"""

import pytest

try:
    from rammp_curobo_interfaces.action import PlanToPose
    from rammp_curobo_interfaces.srv import CheckPoseLock

    from rammp_curobo_ros.planner_node import check_reply, constraint_from_goal
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )


def test_default_goal_is_unconstrained():
    """An older client, or any client that sets nothing, must plan exactly
    as it did before these fields existed."""
    c, v, why = constraint_from_goal(PlanToPose.Goal())
    assert why is None
    assert not c.is_active()
    assert not v.is_active()


def test_named_locks_map_to_the_right_axes():
    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.lock_pitch = True
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_roll and c.hold_pitch and not c.hold_yaw
    assert c.hold_vec_weight() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_reference_frame_constant_maps_to_base_frame():
    from rammp_curobo_interfaces.msg import PoseAxisLock

    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.reference_frame = PoseAxisLock.FRAME_BASE
    c, _, _ = constraint_from_goal(g)
    assert c.in_base_frame is True

    g.axis_lock.reference_frame = PoseAxisLock.FRAME_GOAL
    c, _, _ = constraint_from_goal(g)
    assert c.in_base_frame is False


def test_approach_via_maps():
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    c, v, why = constraint_from_goal(g)
    assert why is None and v.is_active()
    assert v.offset_m == pytest.approx(0.1)
    assert v.linear_axis == 2  # the message's own default


def test_bad_fraction_is_reported_not_raised():
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    g.approach_via.at_fraction = 1.5
    _, _, why = constraint_from_goal(g)
    assert why is not None and "tstep_fraction" in why


def test_unknown_reference_frame_is_refused():
    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.reference_frame = 7
    _, _, why = constraint_from_goal(g)
    assert why is not None and "reference_frame" in why


def test_out_of_range_fraction_is_refused_before_planning():
    """A bad fraction must be refused at the boundary — not handed to the
    GPU to come back as an opaque status."""
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    for bad in (1.5, -0.2, 0.0):
        g.approach_via.at_fraction = bad
        _, _, why = constraint_from_goal(g)
        assert why is not None, bad


def test_non_finite_offset_is_refused():
    g = PlanToPose.Goal()
    g.approach_via.offset = float("nan")
    _, _, why = constraint_from_goal(g)
    assert why is not None and "finite" in why


def test_constraint_from_goal_accepts_a_service_request_without_approach_via():
    """Ruling A: CheckPoseLock.Request has no `approach_via` field at all —
    constraint_from_goal must not raise AttributeError on it, and must map
    the locks correctly while returning an inert ViaPoint."""
    req = CheckPoseLock.Request()
    req.axis_lock.lock_roll = True
    req.axis_lock.lock_x = True
    c, v, why = constraint_from_goal(req)
    assert why is None
    assert c.hold_roll and c.hold_x and not c.hold_pitch
    assert not v.is_active()


def test_check_reply_reports_the_offending_axis():
    rep = check_reply(False, "held axis 'roll' differs by 0.1234 (limit 0.050)")
    assert rep["satisfied"] is False
    assert rep["worst_axis"] == "roll"
    assert rep["worst_error"] == pytest.approx(0.1234)
    assert rep["limit"] == pytest.approx(0.05)
    assert rep["marginal"] is False


def test_check_reply_clean():
    rep = check_reply(True, None)
    assert rep["satisfied"] is True
    assert rep["message"] == ""
    assert rep["worst_axis"] == ""
    assert rep["marginal"] is False


def test_check_reply_marginal():
    rep = check_reply(True, "marginal: held axis 'pitch' is 0.0450 from the goal")
    assert rep["satisfied"] is True and rep["marginal"] is True
    assert rep["worst_axis"] == "pitch"


def test_check_reply_goal_frame_not_pre_checked_is_reported_clean():
    """Ruling B: the goal-frame guard's reply carries no quoted axis and no
    numbers. It must map to a clean, non-marginal, satisfied reply — not a
    false axis report — and this must stay true even if the parser changes."""
    reason = (
        "not pre-checked: goal-frame locking is gated in cuRobo's goal "
        "frame, which this check does not reproduce — expect cuRobo to "
        "accept or refuse it"
    )
    rep = check_reply(True, reason)
    assert rep["satisfied"] is True
    assert rep["marginal"] is False
    assert rep["worst_axis"] == ""
    assert rep["worst_error"] == 0.0
    assert rep["limit"] == 0.0
    assert rep["message"] == reason
