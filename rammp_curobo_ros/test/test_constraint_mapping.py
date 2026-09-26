"""Mapping from PlanToPose goal fields to the core's constraint types.

No ROS runtime and no GPU — same shape as test_start_joints_contract.py.
"""

import pytest

try:
    from rammp_curobo_interfaces.action import PlanToPose

    from rammp_curobo_ros.planner_node import constraint_from_goal
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
