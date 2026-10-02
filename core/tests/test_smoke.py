"""Smoke test: real cuRobo planning on the GPU, no hardware, no ROS.

Skips itself cleanly on machines without CUDA/cuRobo. On the Jetson this is
the "is the stack alive" gate: one session-scoped planner init (kernel
warmup takes a minute or two on the Orin the first time) and a handful of
plans in the sim-kitchen world.
"""

import math
import time

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


def _rotmat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _tilt(q_xyzw, ref_xyzw):
    """Tilt between two orientations, ignoring any spin about base Z: the angle
    between world-vertical as each tool sees it. Deliberately NOT the
    production formula, so a bug there cannot hide behind this."""
    a = _rotmat(q_xyzw).T @ [0.0, 0.0, 1.0]
    b = _rotmat(ref_xyzw).T @ [0.0, 0.0, 1.0]
    return math.acos(max(-1.0, min(1.0, float(a @ b))))


def _yawed_goal(planner, start, d_xyz, yaw):
    """The start's own orientation spun by `yaw` about base Z, at the start's
    position offset by d_xyz and then spun the same way -- so it is level
    exactly as the start is, and a hold is satisfiable from it."""
    spos, squat = planner.fk(start)
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    x, y, z, w = squat
    quat = [c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z]
    px, py, pz = (spos[k] + d_xyz[k] for k in range(3))
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * px - s * py, s * px + c * py, pz], quat


# Found by probing on the Jetson (2026-10-01): unconstrained, cuRobo tilts the
# tool ~19.7 deg on the way; held LEVEL, ~0.5 deg. A goal that a plain base
# rotation reaches would pass whether or not the hold were applied.
_TILTING_GOAL = ([-0.15, 0.30, -0.15], 1.2)


def test_constrained_plan_holds_orientation(planner, start):
    """LEVEL keeps the tilt for the WHOLE path, on a goal where an unconstrained
    plan demonstrably does not -- and still ends on the goal's yaw."""
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint

    pos, quat = _yawed_goal(planner, start, *_TILTING_GOAL)
    level = PoseConstraint(hold=HOLD_LEVEL)
    tol = planner.constraint_tolerance_rad

    def worst(res):
        return max(_tilt(planner.fk(q)[1], quat) for q in res.joint_traj.positions)

    free = planner.plan_to_pose(pos, quat, start)
    assert free.success, free.error
    assert worst(free) > math.radians(5.0), (
        "the unconstrained plan no longer tilts (%.2f deg), so this goal "
        "cannot tell a hold from no hold -- pick another" % math.degrees(worst(free))
    )

    held = planner.plan_to_pose(pos, quat, start, constraint=level)
    assert held.success, held.error
    assert worst(held) < tol, "tilt %.2f deg along the held path" % math.degrees(
        worst(held)
    )

    # LEVEL frees yaw along the way, never at the goal: 1.2 rad of yaw here.
    _, quat_end = planner.fk(held.joint_traj.positions[-1])
    dot = abs(float(np.dot(quat_end, quat)))
    end_err = 2.0 * math.acos(min(1.0, dot))
    assert end_err < tol, "plan ended %.2f deg from the goal" % math.degrees(end_err)


def test_a_raising_plan_does_not_leave_its_hold_behind(planner, start, monkeypatch):
    """cuRobo resets the pose cost metric only on a normal return. If the solve
    raises after the metric is installed, the next UNCONSTRAINED plan must not
    inherit it."""
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint

    mg = planner._motion_gen
    real = mg.plan_single

    def install_then_raise(start_state, goal, config):
        mg.update_pose_cost_metric(config.pose_cost_metric, start_state, goal)
        raise RuntimeError("injected")

    pos, quat = _yawed_goal(planner, start, *_TILTING_GOAL)
    monkeypatch.setattr(mg, "plan_single", install_then_raise)
    res = planner.plan_to_pose(
        pos, quat, start, constraint=PoseConstraint(hold=HOLD_LEVEL)
    )
    assert res.status == "EXCEPTION"
    monkeypatch.setattr(mg, "plan_single", real)

    for rollout in mg.get_all_pose_rollout_instances():
        held = rollout.goal_cost.run_vec_weight
        assert float(held.abs().sum()) == 0.0, "hold leaked: %s" % held.tolist()


def test_hold_verification_cost_is_a_rounding_error_on_planning(planner, start):
    """The hold check runs on EVERY constrained plan, so it has to be cheap.

    What it does: one batched kinematics call over the whole trajectory, then
    numpy. Expected cost is dominated by the single GPU launch plus the
    device->host sync of an (N, 4) quaternion array -- order of a millisecond,
    against planning that measures ~200 ms on the Jetson. So ~1% or less.

    What this guards against is the shape of the computation regressing. An
    earlier version walked the waypoints in a Python loop calling fk() per
    point: ~100 GPU round trips instead of one, on the planning path. That
    would still be CORRECT and would still pass every other test here, which
    is exactly why it needs its own timing gate rather than review.
    """
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint

    level = PoseConstraint(hold=HOLD_LEVEL)
    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    pos, quat = planner.fk(q_target)
    _pos_w, quat_wxyz = planner.fk(q_target, quat_order="wxyz")

    res = planner.plan_to_pose(pos, quat, start, constraint=level)
    assert res.success, res.error
    traj = res.joint_traj
    n = traj.n_points

    def timed(fn, reps):
        # No cuda.is_available() guard: this module skips at import without
        # CUDA, so the device is a given here.
        fn()  # warm: the first call compiles kernels / allocates
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    t_verify = timed(lambda: planner._verify_hold(traj, quat_wxyz, level), 20)
    t_plan = timed(lambda: planner.plan_to_pose(pos, quat, start, constraint=level), 3)

    share = t_verify / t_plan if t_plan > 0 else 1.0
    print(
        "\n  hold verification: %.3f ms over %d waypoints"
        "\n  constrained plan:  %.1f ms"
        "\n  verification is %.2f%% of planning" % (
            t_verify * 1e3, n, t_plan * 1e3, 100.0 * share)
    )
    assert share < 0.05, (
        "hold verification is %.1f%% of planning time (%.2f ms of %.1f ms over "
        "%d waypoints) -- budget is 5%%. The usual cause is the batched "
        "kinematics call having become a per-waypoint loop."
        % (100.0 * share, t_verify * 1e3, t_plan * 1e3, n)
    )


def test_via_point_does_not_stop_the_arm(planner, start):
    """The whole point of the via point: a blended approach with no
    zero-velocity dip (a chained two-segment plan would show one), AND a
    path that actually differs from the unconstrained plan. The no-stall
    check alone is true of nearly every cuRobo plan — if PoseCostMetric
    accepted the via fields but silently ignored the offset, that check
    would still pass while the feature did nothing, so this also requires
    the joint paths to diverge measurably.

    cuRobo's via point (create_grasp_approach_metric) holds every pose
    component except its own approach axis from tstep_fraction onward — see
    ViaPoint's docstring. Build the goal as the START's own FK, offset only
    along z (the default linear_axis), so the approach leg has nothing but
    that one axis to travel and any divergence is the via's doing."""
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
