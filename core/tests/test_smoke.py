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


def _pose_error(planner, q, target_pos):
    pos, _ = planner.fk(q)
    return float(np.linalg.norm(np.asarray(pos) - np.asarray(target_pos)))


def test_plan_to_pose_from_home(planner):
    # A guaranteed-reachable goal: the FK of a mild elbow/wrist variation
    # of home. No tool corrections — raw tool_frame pose in, pose reached.
    q_target = list(planner.home_pose)
    q_target[0] += 0.4
    q_target[5] -= 0.3
    pos, quat = planner.fk(q_target)

    res = planner.plan_to_pose(pos, quat)
    assert res.success, res.error
    assert res.validated
    traj = res.joint_traj
    assert traj.n_points > 5
    assert traj.dt == pytest.approx(0.02)
    assert traj.duration > 0.1
    # starts where we started: home
    assert np.abs(traj.positions[0] - np.asarray(planner.home_pose)).max() < 0.15
    # ends at the requested pose (joints may differ from q_target — that is
    # allowed; the POSE is the contract here)
    assert _pose_error(planner, res.final_joints, pos) < 0.005


def test_plan_to_joints_default_is_exact(planner):
    # default method 'auto' uses native plan_single_js on this wheel:
    # the JOINT goal is the contract, not just the pose
    q_goal = list(planner.home_pose)
    q_goal[1] -= 0.25
    q_goal[3] += 0.30

    res = planner.plan_to_joints(q_goal)
    assert res.success, res.error
    assert res.goal_mismatch_rad is not None and res.goal_mismatch_rad < 1e-3


def test_plan_to_joints_fk_pose_fallback(planner):
    q_goal = list(planner.home_pose)
    q_goal[1] -= 0.25
    q_goal[3] += 0.30

    res = planner.plan_to_joints(q_goal, method="fk_pose")
    assert res.success, res.error
    assert res.goal_mismatch_rad is not None
    goal_pos, _ = planner.fk(q_goal)
    assert _pose_error(planner, res.final_joints, goal_pos) < 0.005


def test_unreachable_goal_fails_cleanly(planner):
    res = planner.plan_to_pose([2.5, 0.0, 1.5], [0, 0, 0, 1])
    assert not res.success
    assert res.joint_traj is None
    assert res.error


def test_retimed_trajectory_stays_valid(planner):
    # +joint_1 swings toward the kitchen's open left side (-0.5 rad heads
    # into the cabinet corner and rightly IK_FAILs on goal collision).
    q_target = list(planner.home_pose)
    q_target[0] += 0.2
    q_target[4] += 0.4
    ok, detail = planner.check_state_valid(q_target)
    assert ok, "test goal invalid in this world: %s" % detail
    pos, quat = planner.fk(q_target)
    res = planner.plan_to_pose(pos, quat)
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
    ok, _ = planner.check_state_valid(planner.home_pose)
    assert ok
    bad = list(planner.home_pose)
    bad[1] = 3.0  # joint_2 limit is ±2.41 rad
    ok, _ = planner.check_state_valid(bad)
    assert not ok


def test_update_world_boxes_reaches_the_live_checker(planner):
    """Perceived boxes must actually land in the GPU collision world.

    Proof by blocking: a goal INSIDE a perceived box must fail, and the
    identical goal must succeed once the perceived set is emptied (the
    sticky baseline surviving the empty update, per the v0.7.8 empty-world
    guard)."""
    pos, quat = planner.fk(planner.home_pose)
    pos = list(pos)
    pos[2] -= 0.10
    try:
        planner.update_world_boxes(
            [{"name": "blocker", "position": pos, "dims": [0.25, 0.25, 0.25]}],
            baseline="world_sim_kitchen.yaml",
        )
        assert [o.name for o in planner.scene.objects] == ["blocker"]
        res = planner.plan_to_pose(pos, quat)
        assert not res.success  # the goal sits inside the perceived box

        planner.update_world_boxes([])  # empty perceived set, baseline kept
        assert planner.scene.objects == []
        assert len(planner.scene.obstacles) > 0
        res = planner.plan_to_pose(pos, quat)
        assert res.success, res.error  # same goal, box gone -> reachable
    finally:
        planner._baseline_scene = None  # don't leak baseline into other tests
        planner.update_world("world_sim_kitchen.yaml")


def test_update_world_guards_and_round_trip(planner):
    with pytest.raises(ValueError, match="empty"):
        planner.update_world([])
    # one past whatever the config's cache is, so raising the cache does
    # not silently stop testing the guard
    too_many = [
        {"name": "b%d" % i, "position": [2 + i, 5, 5], "dims": [0.01, 0.01, 0.01]}
        for i in range(planner.collision_cache_obb + 1)
    ]
    with pytest.raises(ValueError, match="collision_cache_obb"):
        planner.update_world(too_many)
    try:
        # a fresh world (one distant box) still plans
        planner.update_world(
            [{"name": "crate", "position": [1.5, 1.5, 0.5], "dims": [0.2, 0.2, 0.2]}]
        )
        pos, quat = planner.fk(planner.home_pose)
        pos[2] -= 0.10
        res = planner.plan_to_pose(pos, quat)
        assert res.success, res.error
    finally:
        planner.update_world("world_sim_kitchen.yaml")
    pos, quat = planner.fk(planner.home_pose)
    res = planner.plan_to_pose(pos, quat)
    assert res.success, res.error


def test_check_trajectory_sees_an_obstacle_appear(planner):
    """The reactive watchdog's only evidence.

    Nothing in the execution gate chain re-checks collision, so a
    trajectory planned before an obstacle appeared stays "valid" and
    would drive straight through it. This is the check that catches it:
    the SAME trajectory must flip from clear to blocked when a box lands
    on its path."""
    try:
        planner.update_world_boxes([], baseline="world_sim_kitchen.yaml")
        q_goal = list(planner.home_pose)
        q_goal[0] += 0.30
        res = planner.plan_to_joints(q_goal)
        assert res.success, res.error
        traj = res.joint_traj

        ok, first_bad, n_bad = planner.check_trajectory(traj.positions)
        assert ok and first_bad == -1 and n_bad == 0

        mid = list(traj.positions[traj.n_points // 2])
        centre, _ = planner.fk(mid)
        planner.update_world_boxes(
            [{"name": "intruder", "position": list(centre), "dims": [0.2, 0.2, 0.2]}],
            baseline="world_sim_kitchen.yaml",
        )
        ok, first_bad, n_bad = planner.check_trajectory(traj.positions)
        assert not ok
        assert n_bad > 0
        assert 0 <= first_bad < traj.n_points
    finally:
        planner._baseline_scene = None
        planner.update_world("world_sim_kitchen.yaml")


def test_check_trajectory_fails_closed(planner):
    # an empty or malformed span must never be reported as clear
    assert planner.check_trajectory([])[0] is False
    assert planner.check_trajectory(np.zeros((0, 7)))[0] is False


def test_trajectory_clearance_measures_the_whole_arm(planner):
    q = list(planner.home_pose)
    pos, _ = planner.fk(q)
    inside = {"position": list(pos), "dims": [0.3, 0.3, 0.3]}
    far = {"position": [3.0, 0.0, 0.0], "dims": [0.1, 0.1, 0.1]}
    assert planner.trajectory_clearance([q], [inside]) < 0.0
    assert planner.trajectory_clearance([q], [far]) > 1.0
    assert planner.trajectory_clearance([q], []) == float("inf")
    # the minimum over several boxes is the nearest one
    assert planner.trajectory_clearance([q], [inside, far]) == pytest.approx(
        planner.trajectory_clearance([q], [inside])
    )


def test_park_pose_outside_model_limits_is_clamped(planner):
    # The real Gen3 parks with joint_4 ~0.8 deg past cuRobo's URDF bound
    # (found on first hardware contact) — tiny violations clamp inward,
    # large ones are refused with the joint named.
    park = list(planner.home_pose)
    park[3] = -2.674  # model limit is ±2.66
    clamped, err = planner._clamp_to_limits(park, "start")
    assert err is None
    assert clamped[3] == pytest.approx(-2.66, abs=1e-6)

    park[3] = -3.0  # far outside
    clamped, err = planner._clamp_to_limits(park, "start")
    assert clamped is None and "joint_4" in err

    res = planner.plan_to_joints(planner.home_pose, start=park)
    assert not res.success
    assert res.status == "START_OUTSIDE_LIMITS"
    assert "joint_4" in res.error


def test_joint_goal_normalized_to_start_branch(planner):
    # arm reports joint_3 = -pi (wrapped), goal authored at +pi: the plan
    # must NOT wind a full revolution — the goal shifts to the -pi branch
    start = list(planner.home_pose)
    start[2] = -3.1416
    goal = list(planner.home_pose)  # joint_3 = +3.142
    goal[0] = 0.15
    res = planner.plan_to_joints(goal, start=start)
    assert res.success, res.error
    j3 = res.joint_traj.positions[:, 2]
    assert abs(j3[-1] - (-3.1416)) < 0.05  # stayed on the start's branch
    assert np.abs(np.diff(j3)).sum() < 0.2  # no winding
