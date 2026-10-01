"""Mapping from PlanToPose goal fields to the core's constraint types.

No ROS runtime and no GPU — same shape as test_start_joints_contract.py.
"""

import math
import threading

import pytest

try:
    from rammp_curobo_interfaces.action import PlanToPose
    from rammp_curobo_interfaces.msg import OrientationHold
    from rammp_curobo_interfaces.srv import CheckOrientationHold

    from rammp_curobo_ros.planner_node import (
        RammpCuroboNode,
        check_reply,
        constraint_from_goal,
    )
except ImportError as exc:  # pragma: no cover
    # The skip NAMES what was missing. This file silently skipped through an
    # entire interface rewrite -- every test in it was stale and four were
    # broken, and "1 skipped" looked indistinguishable from "fine". To run it:
    #
    #   source install/setup.bash
    #   PYTHONPATH=<ws>/src/RAMMP-CuRobo/core:<ws>/src/RAMMP-CuRobo/rammp_curobo_ros \
    #     python3 -m pytest .../test_constraint_mapping.py
    #
    # rammp_curobo_ros is NOT in the node workspace's --packages-up-to build,
    # which is why sourcing install/setup.bash alone is not enough.
    pytest.skip(
        "cannot import the planner node or its messages, so NOTHING in this "
        "file ran: %s. Source install/setup.bash and put RAMMP-CuRobo/core "
        "and RAMMP-CuRobo/rammp_curobo_ros on PYTHONPATH." % exc,
        allow_module_level=True,
    )


def test_default_goal_is_unconstrained():
    """An older client, or any client that sets nothing, must plan exactly
    as it did before these fields existed."""
    c, v, why = constraint_from_goal(PlanToPose.Goal())
    assert why is None
    assert not c.is_active()
    assert not v.is_active()


def test_level_maps_to_holding_roll_and_pitch_only():
    g = PlanToPose.Goal()
    g.hold.hold = OrientationHold.HOLD_LEVEL
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_vec_weight() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_fixed_maps_to_holding_all_three_rotations():
    g = PlanToPose.Goal()
    g.hold.hold = OrientationHold.HOLD_FIXED
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_vec_weight() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_no_mode_maps_to_holding_position():
    """Position is never held, whatever the mode. The message has no field
    that could ask for it, and this pins that the mapper cannot invent one."""
    for mode in (
        OrientationHold.HOLD_NONE,
        OrientationHold.HOLD_LEVEL,
        OrientationHold.HOLD_FIXED,
    ):
        g = PlanToPose.Goal()
        g.hold.hold = mode
        c, _, _ = constraint_from_goal(g)
        assert c.hold_vec_weight()[3:] == [0.0, 0.0, 0.0]


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


def test_unknown_hold_mode_is_refused_not_coerced():
    """An out-of-range mode must be REFUSED. Coercing it to HOLD_NONE would
    hand an unconstrained trajectory to a caller who asked for a held one --
    the worst available failure, because it looks like success."""
    g = PlanToPose.Goal()
    g.hold.hold = 7
    c, _, why = constraint_from_goal(g)
    assert why is not None and "unknown hold" in why
    assert not c.is_active()


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
    """CheckOrientationHold.Request has no `approach_via` field at all --
    constraint_from_goal must not raise AttributeError on it, and must map the
    hold correctly while returning an inert ViaPoint."""
    req = CheckOrientationHold.Request()
    req.hold.hold = OrientationHold.HOLD_FIXED
    c, v, why = constraint_from_goal(req)
    assert why is None
    assert c.hold_vec_weight() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert not v.is_active()


def test_check_reply_reports_the_measured_numbers_in_degrees():
    """The numbers are passed in, not scraped from the message. The old
    version regexed its own error string for an axis name and the first two
    numerals -- fragile by construction, and silently wrong once the wording
    and the units changed."""
    rep = check_reply(
        False, "the goal's tilt differs by 7.07 deg", math.radians(7.07),
        math.radians(2.0),
    )
    assert rep["satisfied"] is False
    assert rep["deviation_deg"] == pytest.approx(7.07)
    assert rep["limit_deg"] == pytest.approx(2.0)
    assert rep["marginal"] is False


def test_check_reply_does_not_parse_numbers_out_of_the_message():
    """A message full of misleading numerals must not change the reported
    measurement -- which it would have under the regex version."""
    rep = check_reply(
        True, "marginal: 99.9 deg of nonsense 12345", math.radians(1.8),
        math.radians(2.0),
    )
    assert rep["deviation_deg"] == pytest.approx(1.8)
    assert rep["limit_deg"] == pytest.approx(2.0)
    assert rep["marginal"] is True


def test_check_reply_clean():
    rep = check_reply(True, None, 0.0, math.radians(2.0))
    assert rep["satisfied"] is True
    assert rep["message"] == ""
    assert rep["deviation_deg"] == pytest.approx(0.0)
    assert rep["marginal"] is False


def test_check_reply_marginal():
    """A marginal reply is satisfied AND flagged, and carries the measured
    numbers -- which are passed in, not recovered from the sentence."""
    rep = check_reply(
        True,
        "marginal: the goal's tilt (roll/pitch) is 1.80 deg from the start "
        "(limit 2.00, 90% of it)",
        math.radians(1.8),
        math.radians(2.0),
    )
    assert rep["satisfied"] is True and rep["marginal"] is True
    assert rep["deviation_deg"] == pytest.approx(1.8)
    assert rep["limit_deg"] == pytest.approx(2.0)


class _StubFk:
    """Equivalent to core/tests/test_offline.py's _StubFk — duplicated here
    rather than imported, since this file and core/tests live in separate
    pytest roots (see the hold check's docker test invocation, which only
    puts core/ itself on PYTHONPATH)."""

    def __init__(self, pos, quat_xyzw, tolerance_deg=2.0):
        self._pos, self._quat_xyzw = pos, quat_xyzw
        # The core's pre-check is called unbound with this stub as `self`, so
        # every attribute it reads has to exist here -- including the tolerance.
        self.constraint_tolerance_rad = math.radians(tolerance_deg)

    def fk(self, q, quat_order="xyzw"):
        return self._pos, self._quat_xyzw


def test_check_reply_carries_the_real_producer_numbers_end_to_end():
    """Feed check_reply the numbers the CORE actually computed -- not a
    hand-written stand-in -- for both a marginal start and a rejected one,
    through hold_deviation_at_start rather than by parsing the message."""
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint
    from rammp_curobo.geometry import euler_deg_to_quat_xyzw
    from rammp_curobo.planner import CuRoboPlanner

    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    constraint = PoseConstraint(hold=HOLD_LEVEL)

    # Marginal: 1.8 deg against the 2.0 deg default (90%).
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([1.8, 0.0, 0.0]))
    ok, reason = CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, [0.6, 0.0, 0.4], goal_q, constraint
    )
    dev, lim = CuRoboPlanner.hold_deviation_at_start(
        stub, [0.0] * 7, [0.6, 0.0, 0.4], goal_q, constraint
    )
    rep = check_reply(ok, reason, dev, lim)
    assert rep["satisfied"] is True and rep["marginal"] is True
    assert rep["deviation_deg"] == pytest.approx(1.8, abs=1e-3)
    assert rep["limit_deg"] == pytest.approx(2.0)

    # Rejected: 12 deg is well past the tolerance.
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([12.0, 0.0, 0.0]))
    ok, reason = CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, [0.6, 0.0, 0.4], goal_q, constraint
    )
    dev, lim = CuRoboPlanner.hold_deviation_at_start(
        stub, [0.0] * 7, [0.6, 0.0, 0.4], goal_q, constraint
    )
    rep = check_reply(ok, reason, dev, lim)
    assert rep["satisfied"] is False
    assert rep["deviation_deg"] == pytest.approx(12.0, abs=1e-3)
    assert rep["limit_deg"] == pytest.approx(2.0)


class _FakeNodeForLockTest:
    """Just enough of RammpCuroboNode for the hold check's busy path:
    it must bail out on `self._plan_lock` before touching anything else
    (self.planner, etc.), so no real node/GPU init is needed here."""

    def __init__(self):
        self._plan_lock = threading.Lock()


def test_check_orientation_hold_replies_busy_when_a_plan_is_in_flight():
    """Item 2: the check shares cuRobo's preallocated CUDA buffers with
    planning, so it must serialise against a plan in flight rather than
    race it. Hold the lock (as an in-flight plan would) and confirm the
    callback replies busy instead of touching planner state."""
    node = _FakeNodeForLockTest()
    node._plan_lock.acquire()
    try:
        response = RammpCuroboNode._check_orientation_hold_cb(
            node, CheckOrientationHold.Request(), CheckOrientationHold.Response()
        )
    finally:
        node._plan_lock.release()
    assert response.satisfied is False
    assert response.message == "planner busy — a plan is in flight; retry"
    # every other field is left at its message default
    assert response.deviation_deg == 0.0
    assert response.limit_deg == 0.0
    assert response.marginal is False
