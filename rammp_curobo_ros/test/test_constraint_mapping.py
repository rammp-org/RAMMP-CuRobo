"""Mapping from PlanToPose goal fields to the core's constraint types.

No ROS runtime and no GPU — same shape as test_start_joints_contract.py.
"""

import math

import pytest

try:
    from rammp_curobo_interfaces.action import PlanToPose

    from rammp_curobo_ros.planner_node import constraint_from_goal
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
    g.hold = PlanToPose.Goal.HOLD_LEVEL
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_vec_weight() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_fixed_maps_to_holding_all_three_rotations():
    g = PlanToPose.Goal()
    g.hold = PlanToPose.Goal.HOLD_FIXED
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_vec_weight() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_no_mode_maps_to_holding_position():
    """Position is never held, whatever the mode. The message has no field
    that could ask for it, and this pins that the mapper cannot invent one."""
    for mode in (
        PlanToPose.Goal.HOLD_NONE,
        PlanToPose.Goal.HOLD_LEVEL,
        PlanToPose.Goal.HOLD_FIXED,
    ):
        g = PlanToPose.Goal()
        g.hold = mode
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
    g.hold = 7
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


def test_goal_hold_constants_match_the_core():
    """The mapper passes the wire value straight into PoseConstraint, so the
    goal's constants and the core's must be the same numbers."""
    from rammp_curobo import constraints

    assert PlanToPose.Goal.HOLD_NONE == constraints.HOLD_NONE
    assert PlanToPose.Goal.HOLD_LEVEL == constraints.HOLD_LEVEL
    assert PlanToPose.Goal.HOLD_FIXED == constraints.HOLD_FIXED


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


def test_refusal_message_carries_the_measured_deviation_and_limit():
    """CONSTRAINT_UNSATISFIABLE is the only place a caller learns how far off
    the start was, so the reason must state both numbers."""
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint
    from rammp_curobo.geometry import euler_deg_to_quat_xyzw
    from rammp_curobo.planner import CuRoboPlanner

    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([12.0, 0.0, 0.0]))
    ok, reason = CuRoboPlanner.constraint_satisfied_at_start(
        stub,
        [0.0] * 7,
        [0.6, 0.0, 0.4],
        euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]),
        PoseConstraint(hold=HOLD_LEVEL),
    )
    assert ok is False
    assert "12.00 deg" in reason and "limit 2.00" in reason
