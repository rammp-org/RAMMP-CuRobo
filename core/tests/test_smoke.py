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


def test_joint_limit_override_bounds_the_base():
    """Workspace sector via a truncated joint_1.

    Must be applied before warmup: cuRobo's solvers cache the limit
    tensors at first solve, so a later edit is seen only by our own
    validator and every plan comes back LIBRARY_VALIDATION_FAILED
    instead of being routed inside the sector.
    """
    p = CuRoboPlanner.from_config(
        "gen3_real.yaml",
        planner_overrides={"joint_limits_deg": {"joint_1": [-75.0, 90.0]},
                           "warmup": False},
    )
    k = p.joint_names.index("joint_1")
    lo, hi = p.joint_limits()["position"][:, k]
    assert lo == pytest.approx(np.radians(-75.0), abs=1e-3)
    assert hi == pytest.approx(np.radians(90.0), abs=1e-3)

    quat = [0.5, 0.5, 0.5, 0.5]
    p.update_world_boxes([])
    res = p.plan_to_pose([0.55, 0.30, 0.35], quat)
    assert res.success, res.error
    j1 = res.joint_traj.positions[:, k]
    assert j1.min() >= np.radians(-75.0) - 1e-3
    assert j1.max() <= np.radians(90.0) + 1e-3
    # straight behind needs a base rotation the sector forbids
    assert not p.plan_to_pose([-0.55, 0.0, 0.35], quat).success


def test_joint_limit_override_refuses_nonsense():
    with pytest.raises(KeyError, match="unknown joint"):
        CuRoboPlanner.from_config(
            "gen3_real.yaml",
            planner_overrides={"joint_limits_deg": {"elbow": [-10.0, 10.0]},
                               "warmup": False},
        )
    # a sector that excludes home would fail every retract seed
    with pytest.raises(ValueError, match="excludes the home pose"):
        CuRoboPlanner.from_config(
            "gen3_real.yaml",
            planner_overrides={"joint_limits_deg": {"joint_1": [10.0, 20.0]},
                               "warmup": False},
        )


def test_tightened_limits_are_enforced_by_the_checkers():
    """cuRobo's BoundCost clones the limit tensors at CONSTRUCTION, so
    check_constraints tests the URDF ranges — a state inside the URDF but
    outside joint_limits_deg must still be flagged, by the planner's own
    numpy limit check."""
    p = CuRoboPlanner.from_config(
        "gen3_real.yaml",
        planner_overrides={"joint_limits_deg": {"joint_1": [-75.0, 90.0]},
                           "warmup": False},
    )
    q = list(p.home_pose)
    q[p.joint_names.index("joint_1")] = np.radians(150.0)
    ok, detail = p.check_state_valid(q)
    assert not ok and "joint_1" in detail
    ok, first_bad, n_bad = p.check_trajectory([q])
    assert not ok and first_bad == 0 and n_bad == 1


def test_winding_plan_is_refused(planner):
    """A plan in which one joint sweeps more than max_joint_span_rad is
    refused as WINDING rather than handed to the executor. Exercised by
    dropping the threshold under an ordinary plan's own span — the real
    code path, not a synthetic array."""
    q_goal = list(planner.home_pose)
    q_goal[0] += 0.4                      # ~23 deg on joint_1
    old = planner.max_joint_span_rad
    try:
        planner.max_joint_span_rad = 0.2  # below that plan's span
        res = planner.plan_to_joints(q_goal)
        assert not res.success
        assert res.status == "WINDING"
        assert "sweeps" in res.error and "refused" in res.error
        planner.max_joint_span_rad = old  # default: same plan is fine
        res = planner.plan_to_joints(q_goal)
        assert res.success, res.error
    finally:
        planner.max_joint_span_rad = old


def test_registration_recovers_camera_error_on_the_real_arm_model(planner):
    """The synthetic-arm tests prove the estimator; this proves it on the
    geometry it will meet: cuRobo's 47 Gen3 spheres at home, sampled on
    the PHYSICAL surface (runtime radius minus the load-time buffer —
    depth cameras see the surface, not the inflated collision set) on the
    hemisphere the bench Orbbec sees, displaced by the bench's own
    measured error. Also pins the baked self-model (raw radii + buffer)
    to cuRobo's sphere set, so a re-bake that drifts is caught here."""
    import yaml

    from rammp_curobo.config import resolve_config
    from rammp_curobo.perception import register_points_to_spheres

    sph = planner.link_spheres(planner.home_pose)
    sph = sph[sph[:, 3] > 0.0]
    baked = yaml.safe_load(open(resolve_config("self_model_gen3_2f85.yaml")))
    buf = float(baked["buffer"])
    baked_r = sorted(round(float(row[3]) + buf, 4)
                     for rows in baked["spheres"].values() for row in rows)
    assert baked_r == sorted(round(float(r), 4) for r in sph[:, 3])

    # register against RAW radii — what perception_debug and the node feed
    sph_raw = sph.copy()
    sph_raw[:, 3] -= buf
    rng = np.random.default_rng(21)
    cam = np.array([0.08, 0.70, 0.30])                # the bench Orbbec
    pts = []
    for i, c in enumerate(sph_raw):
        u = rng.normal(size=(300, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        u = u[(u @ (cam - c[:3])) > 0.0]
        p = c[:3] + c[3] * u
        # union visibility: the real overlapping sphere chains never show
        # points buried inside a neighbour — drop anything >5 mm deep
        others = np.delete(sph_raw, i, axis=0)
        depth_in = np.linalg.norm(p[:, None, :] - others[None, :, :3], axis=2) \
            - others[None, :, 3]
        pts.append(p[depth_in.min(axis=1) > -0.005])
    true_pts = np.vstack(pts)
    for delta in ([0.023, 0.078, -0.040], [0.0, -0.09, 0.0], [-0.06, 0.02, 0.05]):
        delta = np.array(delta)
        seen = true_pts + delta + rng.normal(scale=0.003, size=true_pts.shape)
        blob = np.array([0.45, 0.20, 0.35]) + rng.uniform(-0.05, 0.05, size=(500, 3))
        out = register_points_to_spheres(np.vstack([seen, blob]), sph_raw, cam_origin=cam)
        assert out is not None, delta
        shift, used, rms = out
        assert np.linalg.norm(shift + delta) < 0.005, (delta, shift)
        assert rms < 0.006


def test_measured_floor_keeps_two_centimetres(planner):
    """The lower bound, as a regression: the MEASURED bench world must keep
    home valid (the true-height table sits 12 mm under the arm's own base
    spheres — that is why table is in no_pad_names), and every sweep and
    dodge must clear the REAL tabletop by the padded guards' 2 cm, without
    losing the tight under-the-slab route (path optimality)."""
    import yaml

    from rammp_curobo import CuRoboPlanner
    from rammp_curobo.config import resolve_config

    world = yaml.safe_load(open(resolve_config("world_real_bench.yaml")))
    names = [o["name"] for o in world["obstacles"]]
    assert "table" in names and "floor_front" in names, names
    table = next(o for o in world["obstacles"] if o["name"] == "table")
    top = table["position"][2] + table["dims"][2] / 2.0
    assert -0.10 < top < -0.01, "table top %.3f looks unmeasured" % top

    pl = CuRoboPlanner.from_config(
        "gen3_real.yaml",
        planner_overrides={"collision_activation_distance": 0.07,
                           "warmup": False},
    )
    ok, detail = pl.check_state_valid(pl.home_pose)
    assert ok, detail

    def moving_low(traj):
        alls = np.array([pl.link_spheres(q) for q in traj.positions])
        valid = alls[0, :, 3] > 0.0
        mov = valid & (alls[:, :, :3].std(axis=0).max(axis=1) > 0.002)
        return float((alls[:, mov, 2] - alls[:, mov, 3]).min())

    qa = list(pl.home_pose)
    qa[0] = np.radians(30.0)
    qb = list(pl.home_pose)
    qb[0] = -np.radians(30.0)
    pl.update_world_boxes([])
    res = pl.plan_to_joints(qb, start=qa)
    assert res.success, res.error
    assert moving_low(res.joint_traj) - top > 0.02

    slab = {"name": "s", "position": [0.50, 0.0, 0.62],
            "dims": [0.15, 0.15, 0.25]}
    pl.update_world_boxes([slab])
    res = pl.plan_to_joints(qb, start=qa)
    assert res.success, res.error
    tool_z = np.array([pl.fk(q)[0][2] for q in res.joint_traj.positions])
    assert tool_z.min() < 0.35, "under-the-slab route lost"
    assert moving_low(res.joint_traj) - top > 0.02 - 1e-3


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
