"""Smoke test: real cuRobo planning on the GPU, no hardware, no ROS.

Skips itself cleanly on machines without CUDA/cuRobo. On the Jetson this is
the "is the stack alive" gate: one session-scoped planner init (kernel
warmup takes a minute or two on the Orin the first time) and a handful of
plans in the sim-kitchen world.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from rammp_curobo import CuRoboPlanner  # noqa: E402


@pytest.fixture(scope="session")
def planner():
    return CuRoboPlanner.from_config("gen3.yaml")


@pytest.fixture
def start(planner):
    """Every plan states where it starts — the planner has no default and
    will not guess (issue #6). The retract pose is simply a known-valid
    configuration to use here, not a "home"."""
    return list(planner.retract_pose)


def _pose_error(planner, q, target_pos):
    pos, _ = planner.fk(q)
    return float(np.linalg.norm(np.asarray(pos) - np.asarray(target_pos)))


def test_plan_to_pose_from_retract(planner, start):
    # A guaranteed-reachable goal: the FK of a mild elbow/wrist variation
    # of home. No tool corrections — raw tool_frame pose in, pose reached.
    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    q_target[5] -= 0.3
    pos, quat = planner.fk(q_target)

    res = planner.plan_to_pose(pos, quat, start)
    assert res.success, res.error
    assert res.validated
    traj = res.joint_traj
    assert traj.n_points > 5
    assert traj.dt == pytest.approx(0.02)
    assert traj.duration > 0.1
    # starts where we said we started
    assert np.abs(traj.positions[0] - np.asarray(planner.retract_pose)).max() < 0.15
    # ends at the requested pose (joints may differ from q_target — that is
    # allowed; the POSE is the contract here)
    assert _pose_error(planner, res.final_joints, pos) < 0.005


def test_plan_to_joints_default_is_exact(planner, start):
    # default method 'auto' uses native plan_single_js on this wheel:
    # the JOINT goal is the contract, not just the pose
    q_goal = list(planner.retract_pose)
    q_goal[1] -= 0.25
    q_goal[3] += 0.30

    res = planner.plan_to_joints(q_goal, start)
    assert res.success, res.error
    assert res.goal_mismatch_rad is not None and res.goal_mismatch_rad < 1e-3


def test_plan_to_joints_fk_pose_fallback(planner, start):
    q_goal = list(planner.retract_pose)
    q_goal[1] -= 0.25
    q_goal[3] += 0.30

    res = planner.plan_to_joints(q_goal, start, method="fk_pose")
    assert res.success, res.error
    assert res.goal_mismatch_rad is not None
    goal_pos, _ = planner.fk(q_goal)
    assert _pose_error(planner, res.final_joints, goal_pos) < 0.005


def test_unreachable_goal_fails_cleanly(planner, start):
    res = planner.plan_to_pose([2.5, 0.0, 1.5], [0, 0, 0, 1], start)
    assert not res.success
    assert res.joint_traj is None
    assert res.error


def test_retimed_trajectory_stays_valid(planner, start):
    # +joint_1 swings toward the kitchen's open left side (-0.5 rad heads
    # into the cabinet corner and rightly IK_FAILs on goal collision).
    q_target = list(planner.retract_pose)
    q_target[0] += 0.2
    q_target[4] += 0.4
    ok, detail = planner.check_state_valid(q_target)
    assert ok, "test goal invalid in this world: %s" % detail
    pos, quat = planner.fk(q_target)
    res = planner.plan_to_pose(pos, quat, start)
    assert res.success, res.error

    slow = res.joint_traj.scaled(0.25)
    assert slow.duration == pytest.approx(res.joint_traj.duration * 4)
    lim = planner.joint_limits()
    from rammp_curobo import validate_trajectory

    assert validate_trajectory(slow, lim["position"], lim["velocity"]) == []
    assert (
        np.abs(slow.velocities).max()
        <= np.abs(res.joint_traj.velocities).max() * 0.25 + 1e-9
    )


def test_check_state_valid(planner):
    ok, _ = planner.check_state_valid(planner.retract_pose)
    assert ok
    bad = list(planner.retract_pose)
    bad[1] = 3.0  # joint_2 limit is ±2.41 rad
    ok, _ = planner.check_state_valid(bad)
    assert not ok


def test_update_world_guards_and_round_trip(planner, start):
    with pytest.raises(ValueError, match="empty"):
        planner.update_world([])
    too_many = [
        {"name": "b%d" % i, "position": [2 + i, 5, 5], "dims": [0.01, 0.01, 0.01]}
        for i in range(61)  # cache is 60
    ]
    with pytest.raises(ValueError, match="collision_cache_obb"):
        planner.update_world(too_many)
    try:
        # a fresh world (one distant box) still plans
        planner.update_world(
            [{"name": "crate", "position": [1.5, 1.5, 0.5], "dims": [0.2, 0.2, 0.2]}]
        )
        pos, quat = planner.fk(planner.retract_pose)
        pos[2] -= 0.10
        res = planner.plan_to_pose(pos, quat, start)
        assert res.success, res.error
    finally:
        planner.update_world("world_sim_kitchen.yaml")
    pos, quat = planner.fk(planner.retract_pose)
    res = planner.plan_to_pose(pos, quat, start)
    assert res.success, res.error


def test_park_pose_outside_model_limits_is_clamped(planner):
    # The real Gen3 parks with joint_4 ~0.8 deg past cuRobo's URDF bound
    # (found on first hardware contact) — tiny violations clamp inward,
    # large ones are refused with the joint named.
    park = list(planner.retract_pose)
    park[3] = -2.674  # model limit is ±2.66
    clamped, err = planner._clamp_to_limits(park, "start")
    assert err is None
    assert clamped[3] == pytest.approx(-2.66, abs=1e-6)

    park[3] = -3.0  # far outside
    clamped, err = planner._clamp_to_limits(park, "start")
    assert clamped is None and "joint_4" in err

    res = planner.plan_to_joints(planner.retract_pose, start=park)
    assert not res.success
    assert res.status == "START_OUTSIDE_LIMITS"
    assert "joint_4" in res.error


def test_joint_goal_normalized_to_start_branch(planner):
    # arm reports joint_3 = -pi (wrapped), goal authored at +pi: the plan
    # must NOT wind a full revolution — the goal shifts to the -pi branch
    start = list(planner.retract_pose)
    start[2] = -3.1416
    goal = list(planner.retract_pose)  # joint_3 = +3.142
    goal[0] = 0.15
    res = planner.plan_to_joints(goal, start=start)
    assert res.success, res.error
    j3 = res.joint_traj.positions[:, 2]
    assert abs(j3[-1] - (-3.1416)) < 0.05  # stayed on the start's branch
    assert np.abs(np.diff(j3)).sum() < 0.2  # no winding


def test_constrained_plan_holds_orientation(planner, start):
    """A held-axis plan keeps roll and pitch fixed for the WHOLE path, not
    just at the endpoints — checked by FK over every point."""
    from rammp_curobo import PoseConstraint
    from rammp_curobo.geometry import rotvec_between

    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    pos, quat = planner.fk(q_target)

    # joint_1 is a rotation about the base's own z axis, so it changes
    # neither roll nor pitch — this setup is constructed to satisfy the
    # pre-check, so require that rather than skip past it. A skip here
    # would let the constrained smoke test quietly stop testing itself on
    # exactly the machine it exists to validate.
    ok, why = planner.constraint_satisfied_at_start(
        start, pos, quat, PoseConstraint(hold_roll=True, hold_pitch=True)
    )
    assert ok, "pre-check failed for a goal constructed to satisfy it: %s" % why

    res = planner.plan_to_pose(
        pos, quat, start, constraint=PoseConstraint(hold_roll=True, hold_pitch=True)
    )
    assert res.success, res.error

    _, quat0 = planner.fk(res.joint_traj.positions[0])
    worst = 0.0
    for q in res.joint_traj.positions:
        _, qk = planner.fk(q)
        rot = rotvec_between(quat0, qk)
        worst = max(worst, abs(rot[0]), abs(rot[1]))
    assert worst < 0.05, "roll/pitch drifted %.4f rad along the path" % worst


def test_via_point_does_not_stop_the_arm(planner, start):
    """The whole point of the via point: a blended approach with no
    zero-velocity dip (a chained two-segment plan would show one), AND a
    path that actually differs from the unconstrained plan. The no-stall
    check alone is true of nearly every cuRobo plan — if PoseCostMetric
    accepted the via fields but silently ignored the offset, that check
    would still pass while the feature did nothing, so this also requires
    the joint paths to diverge measurably.

    cuRobo's via point (create_grasp_approach_metric) forces a hold on
    every pose component except its own approach axis — see ViaPoint's
    docstring — so the start must already match the goal on the other
    five. Build the goal as the START's own FK, offset only along z (the
    default linear_axis), rather than perturbing joints the way the other
    smoke tests do: any joint perturbation here would also move x/y/
    orientation and trip the implied hold before the via-blend behaviour
    this test exists to check is even reached."""
    from rammp_curobo import ViaPoint

    pos, quat = planner.fk(start)
    pos = [pos[0], pos[1], pos[2] + 0.15]

    res = planner.plan_to_pose(pos, quat, start, via=ViaPoint(offset_m=0.10))
    assert res.success, res.error
    vel = res.joint_traj.velocities
    assert vel is not None

    speed = np.abs(vel).max(axis=1)
    interior = speed[2:-2]  # ends are legitimately at rest
    assert interior.min() > 1e-3, (
        "commanded motion stalls mid-path (min |qd| = %.5f) — the via point "
        "is behaving like a stop, not a blend" % interior.min()
    )

    res_plain = planner.plan_to_pose(pos, quat, start)
    assert res_plain.success, res_plain.error
    common = min(
        res.joint_traj.positions.shape[0], res_plain.joint_traj.positions.shape[0]
    )
    max_diff = float(
        np.abs(
            res.joint_traj.positions[:common] - res_plain.joint_traj.positions[:common]
        ).max()
    )
    assert max_diff > 1e-2, (
        "via-point path is indistinguishable from the unconstrained plan "
        "(max per-joint diff = %.5f rad over %d shared points) — "
        "PoseCostMetric may be accepting the via fields but ignoring the "
        "offset" % (max_diff, common)
    )


def test_unconstrained_plan_is_unchanged(planner, start):
    """The feature must be inert when nobody asks for it."""
    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    pos, quat = planner.fk(q_target)

    a = planner.plan_to_pose(pos, quat, start)
    b = planner.plan_to_pose(pos, quat, start, constraint=None, via=None)
    assert a.success and b.success
    assert a.joint_traj.dof == b.joint_traj.dof
    assert abs(a.joint_traj.duration - b.joint_traj.duration) < 0.5
