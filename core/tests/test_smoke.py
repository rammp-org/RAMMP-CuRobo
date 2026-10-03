"""Smoke test: real cuRobo planning on the GPU, no hardware, no ROS.

Skips itself cleanly on machines without CUDA/cuRobo. On the Jetson this is
the "is the stack alive" gate: one session-scoped planner init (kernel
warmup takes a minute or two on the Orin the first time) and a handful of
plans in the sim-kitchen world.

It is also where obstacle layers (issue #19) meet cuRobo: paired requests,
the same one with and without a box, that fail if cuRobo ignores the box,
and the generated chair on a second planner (gen3_mebot.yaml), built only
for the tests that use it. Every search is seeded and capped, and fails
naming what it tried. A test that pushes a world puts back the one it
found.
"""

import math
import re
from collections import Counter

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from rammp_curobo import CuRoboPlanner  # noqa: E402
from rammp_curobo.config import PACKAGED_CONFIG_DIR  # noqa: E402
from rammp_curobo.scene import Obstacle, Scene  # noqa: E402
from rammp_curobo.world import sphere_box_penetration  # noqa: E402

# a box out of the arm's reach: a world with nothing in the way
FAR = {"name": "far", "position": [1.5, 1.5, 0.5], "dims": [0.2, 0.2, 0.2]}
FLOOR = "layer_mebot_floor.yaml"
PLATFORM = "layer_mebot_platform.yaml"
PERSON = "layer_mebot_person.yaml"


@pytest.fixture(scope="session")
def planner():
    gen3 = CuRoboPlanner.from_config("gen3.yaml")
    # read back as construction loaded it: the tests before the read-back
    # test push worlds on this planner and put the kitchen back through
    # update_world
    _check_read_back(gen3)
    return gen3


@pytest.fixture(scope="session")
def chair_planner():
    """The Gen3 on the MEBot: the generated platform and person layers in
    every world. Built only when a test asks for it."""
    return CuRoboPlanner.from_config("gen3_mebot.yaml")


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
    before = planner.world_boxes
    with pytest.raises(ValueError, match="empty"):
        planner.update_world([])
    too_many = [
        {"name": "b%d" % i, "position": [2 + i, 5, 5], "dims": [0.01, 0.01, 0.01]}
        for i in range(61)  # cache is 60
    ]
    with pytest.raises(ValueError, match="collision_cache_obb") as exc:
        planner.update_world(too_many)
    # it says where the boxes come from
    assert "scene 61" in str(exc.value)
    # a refused push leaves the world it found
    assert planner.world_boxes == before
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


# ------------------------------------------------------------ obstacle layers


def _scene(path, **sections):
    """Write a world file holding `sections` (obstacles, objects, layers)."""
    path.write_text(yaml.safe_dump(sections))
    return path


def _path(res, start):
    """The joint path a plan flies: its start, then its waypoints."""
    return np.vstack([np.asarray(start, dtype=float), res.joint_traj.positions])


def _dev(path, start, goal):
    """How far a joint path strays from the straight segment start -> goal:
    the largest distance (rad, in joint space) of one of its points from
    the segment."""
    a, b = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    s = np.clip((path - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return float(np.linalg.norm(path - (a + s[:, None] * (b - a)), axis=1).max())


def _curobo_states(planner, states):
    """Joint vectors in controller order as one tensor in cuRobo's order."""
    order = [planner.joint_names.index(n) for n in planner._curobo_joint_names]
    q = np.asarray(states, dtype=float).reshape(-1, 7)[:, order]
    # the column pick can leave numpy's strides, which the tensor keeps, and
    # cuRobo's kinematics kernel takes contiguous memory only
    return planner._tensor(np.ascontiguousarray(q))


def _feasible(planner, states):
    """check_state_valid's verdict on many joint vectors in one call, (N,)
    bools: the searches ask it of tens of thousands."""
    from curobo.types.robot import JointState

    states = np.asarray(states, dtype=float).reshape(-1, 7)
    names = list(planner._curobo_joint_names)
    verdicts = []
    for i in range(0, len(states), 16384):  # bounds one call's GPU memory
        q = _curobo_states(planner, states[i : i + 16384])
        metrics = planner._motion_gen.check_constraints(
            JointState.from_position(q, joint_names=names)
        )
        verdicts.append(metrics.feasible.view(-1).cpu().numpy())
    return np.concatenate(verdicts).astype(bool)


def _spheres(planner, states):
    """cuRobo's collision spheres at each joint vector, (N, n, 4), as
    planner._spheres gives them for one."""
    fk = planner._motion_gen.kinematics.get_state(_curobo_states(planner, states))
    return fk.link_spheres_tensor.cpu().numpy().astype(float)


def _cuboids(planner, layer=None):
    """The installed world's boxes, or those of `layer`, in the form
    sphere_box_penetration takes."""
    return {
        box["name"]: {"dims": box["dims"], "pose": box["pose"]}
        for box in planner.world_boxes
        if layer in (None, box["layer"])
    }


def _search_box(planner, around=None, radius=math.inf):
    """Where the searches sample, (lower, upper) per joint: 0.3 rad inside
    the joint limits, since cuRobo bends a path that comes near one; within
    [-pi, pi], as the driver reports a continuous joint; and within `radius`
    of `around`, if given."""
    lim = planner.joint_limits()["position"]
    lo, hi = np.maximum(lim[0] + 0.3, -math.pi), np.minimum(lim[1] - 0.3, math.pi)
    if around is not None:
        lo = np.maximum(lo, np.asarray(around) - radius)
        hi = np.minimum(hi, np.asarray(around) + radius)
    return lo, hi


def _free_plan(planner, tmp_path):
    """The request the Task-1 tests block, from the retract to the retract
    turned 1 rad on joint_1, planned in a free world (one far box), where it
    goes straight. Returns the start, the goal and the free plan's path."""
    start = list(planner.retract_pose)
    goal = list(start)
    goal[0] += 1.0
    planner.update_world(_scene(tmp_path / "free.yaml", obstacles=[FAR]))
    res = planner.plan_to_joints(goal, start, method="js")
    assert res.success, res.error
    path = _path(res, start)
    assert _dev(path, start, goal) < 0.05
    return start, goal, path


def _assert_blocked(planner, world, start, goal, free):
    """The free plan's request in `world`, which puts a box on its path.
    cuRobo must see the box (start and goal stay valid, and some state of
    the free path no longer is) and so fail or go around."""
    planner.update_world(world)
    assert _feasible(planner, [start, goal]).all(), "premise: start and goal valid"
    assert not _feasible(planner, free).all(), "premise: the box is on the free path"
    res = planner.plan_to_joints(goal, start, method="js")
    assert not res.success or _dev(_path(res, start), start, goal) > 0.10


def test_box_on_the_path_forces_a_detour_and_absent_goes_straight(planner, tmp_path):
    try:
        start, goal, free = _free_plan(planner, tmp_path)
        # 8 cm, at the tool point halfway along the free plan
        tool = planner.fk(free[len(free) // 2])[0]
        box = {"name": "block", "position": tool, "dims": [0.08] * 3}
        world = _scene(tmp_path / "box.yaml", obstacles=[FAR, box])
        _assert_blocked(planner, world, start, goal, free)
    finally:
        planner.update_world("world_sim_kitchen.yaml")


def test_rotated_box_is_read_as_oriented(planner, tmp_path):
    try:
        start, goal, free = _free_plan(planner, tmp_path)
        # a slab across the path halfway, yawed so that its thin axis (y) lies
        # along the way the tool moves there
        k = len(free) // 2
        ahead = np.subtract(planner.fk(free[k + 1])[0], planner.fk(free[k - 1])[0])
        slab = {
            "name": "slab",
            "position": planner.fk(free[k])[0],
            "rpy_deg": [0, 0, math.degrees(math.atan2(-ahead[0], ahead[1]))],
            "dims": [0.30, 0.02, 0.30],
        }
        world = _scene(tmp_path / "slab.yaml", obstacles=[FAR, slab])
        _assert_blocked(planner, world, start, goal, free)

        # one state, judged against the slab and not its bounding box: a slab
        # yawed 45 deg stands 0.15 m from the retract's tool point along its
        # normal, on the side away from the arm's base. Its axis-aligned
        # bounding box holds the tool point; the padded slab is 0.12 m from it
        tool = np.asarray(planner.fk(start)[0])
        c = s = math.sqrt(0.5)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        normal = rot[:, 1] if rot[:2, 1] @ tool[:2] > 0 else -rot[:, 1]
        centre = tool + 0.15 * normal
        dims = np.array([0.40, 0.02, 0.40])
        aabb = np.abs(rot) @ dims
        assert (np.abs(tool - centre) < aabb / 2).all(), "premise: tool in the AABB"
        slab = {
            "name": "slab",
            "position": centre.tolist(),
            "rpy_deg": [0, 0, 45],
            "dims": dims.tolist(),
        }
        planner.update_world(_scene(tmp_path / "oriented.yaml", obstacles=[slab]))
        assert planner.check_state_valid(start) == (True, "")
        box = {"name": "box", "position": centre.tolist(), "dims": aabb.tolist()}
        planner.update_world(_scene(tmp_path / "aligned.yaml", obstacles=[box]))
        assert not planner.check_state_valid(start)[0]
    finally:
        planner.update_world("world_sim_kitchen.yaml")


def test_a_layer_box_on_the_path_is_honoured_and_ignore_removes_it(planner, tmp_path):
    try:
        start, goal, free = _free_plan(planner, tmp_path)
        # the first test's box, delivered as an excludable layer that the
        # scene lists
        tool = planner.fk(free[len(free) // 2])[0]
        layer = {
            "layer": "block",
            "excludable": True,
            "obstacles": [{"name": "box", "position": tool, "dims": [0.08] * 3}],
        }
        (tmp_path / "block.yaml").write_text(yaml.safe_dump(layer))
        world = _scene(
            tmp_path / "layered.yaml", obstacles=[FAR], layers=["block.yaml"]
        )
        _assert_blocked(planner, world, start, goal, free)
        held = {box["name"]: box["layer"] for box in planner.world_boxes}
        assert held == {"far": "scene", "block/box": "block"}

        planner.update_world(world, ignore={"block"})
        assert [box["name"] for box in planner.world_boxes] == ["far"]
        res = planner.plan_to_joints(goal, start, method="js")
        assert res.success, res.error
        assert _dev(_path(res, start), start, goal) < 0.05
    finally:
        planner.update_world("world_sim_kitchen.yaml")


def _platform_pair(planner, with_platform, without):
    """Seeded and capped: a goal near the retract whose straight joint path
    from the retract the platform alone blocks. cuRobo finds the goal valid
    with the platform, and the path valid without the platform and not with
    it. The path keeps the collision activation distance from the floor, so
    that without the platform nothing pushes a plan off it, and it reaches
    40 mm or more into the padded platform, so that going around takes a
    detour. Returns (start, goal, depth_m); fails, naming what it tried, if
    no goal does."""
    start = np.asarray(planner.retract_pose)
    goals = np.random.default_rng(19).uniform(
        *_search_box(planner, around=start, radius=1.6), size=(8000, 7)
    )
    lines = start + np.linspace(0.0, 1.0, 50)[:, None] * (goals - start)[:, None]
    planner.update_world(with_platform)
    assert _feasible(planner, [start]).all(), "premise: retract valid with platform"
    valid = _feasible(planner, goals)
    blocked = ~_feasible(planner, lines).reshape(len(goals), -1).all(axis=1)
    platform = _cuboids(planner, "platform")
    planner.update_world(without)
    clear = _feasible(planner, lines).reshape(len(goals), -1).all(axis=1)
    floor = _cuboids(planner)
    candidates = np.nonzero(valid & blocked & clear)[0]
    near_floor, depths = 0, []
    for i in candidates:
        spheres = _spheres(planner, lines[i])
        spheres = spheres[spheres[..., 3] > 0]
        reach = spheres + [0.0, 0.0, 0.0, planner.collision_activation_distance]
        if sphere_box_penetration(reach, floor):
            near_floor += 1
            continue
        depth = max(sphere_box_penetration(spheres, platform).values(), default=0.0)
        if depth >= 0.04:
            return start, goals[i], depth
        depths.append(depth)
    pytest.fail(
        "no goal near the retract (8000, seed 19) has a straight path the "
        "platform alone blocks by 40 mm: %d goals are valid with the platform, "
        "%d of their paths are blocked by it alone, %d of those come within "
        "the activation distance of the floor, and the rest reach at most "
        "%.1f mm into it"
        % (valid.sum(), len(candidates), near_floor, 1e3 * max(depths, default=0))
    )


def test_platform_layer_pair(planner, tmp_path):
    # the generated platform, as a scene layer of the gen3 planner: the same
    # request with it and without it
    with_platform = _scene(tmp_path / "with.yaml", layers=[FLOOR, PLATFORM])
    without = _scene(tmp_path / "without.yaml", layers=[FLOOR])
    try:
        start, goal, depth = _platform_pair(planner, with_platform, without)
        planner.update_world(without)
        straight = planner.plan_to_joints(goal, start, method="js")
        planner.update_world(with_platform)
        around = planner.plan_to_joints(goal, start, method="js")
    finally:
        planner.update_world("world_sim_kitchen.yaml")
    assert straight.success, straight.error
    assert _dev(_path(straight, start), start, goal) < 0.05
    assert not around.success or _dev(_path(around, start), start, goal) > 0.10, (
        "the straight path reaches %.1f mm into the padded platform" % (1e3 * depth)
    )


def _inverse(pose):
    """The inverse of a pose [x, y, z, qw, qx, qy, qz] with a unit
    quaternion, in the same form."""
    w, x, y, z = pose[3:]
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return [*(-rot.T @ np.asarray(pose[:3])), w, -x, -y, -z]


def _check_read_back(planner):
    """cuRobo's box cache, read back, holds the world the planner reports:
    the boxes of world_boxes, by name, enabled, and every slot after them
    disabled, with their padded dims and the inverse of their poses, which
    is what cuRobo stores. The cache is cuRobo v0.7.8's private state, read
    in tests only."""
    boxes = planner.world_boxes
    checker = planner._motion_gen.world_coll_checker
    n = int(checker._env_n_obbs[0])
    dims, poses, enabled = (t[0].cpu().numpy() for t in checker._cube_tensor_list)
    assert n == len(boxes)
    assert checker._env_obbs_names[0][:n] == [box["name"] for box in boxes]
    assert enabled.tolist() == [1] * n + [0] * (len(enabled) - n)
    np.testing.assert_allclose(dims[:n, :3], [box["dims"] for box in boxes], atol=1e-6)
    inverse = np.array([_inverse(box["pose"]) for box in boxes])
    np.testing.assert_allclose(poses[:n, :3], inverse[:, :3], atol=1e-5)
    # q and -q are one rotation
    sign = np.where((poses[:n, 3:7] * inverse[:, 3:]).sum(axis=1) < 0, -1.0, 1.0)
    np.testing.assert_allclose(
        poses[:n, 3:7], sign[:, None] * inverse[:, 3:], atol=1e-5
    )


def test_world_read_back_matches_composition(planner, chair_planner, tmp_path):
    # the chair as built, since no test before this one asks for it: it
    # warmed up in its floor alone and loaded its layers after. The planner
    # fixture read the gen3 planner back as built
    _check_read_back(chair_planner)
    # a push with a rotated obstacle, a rotated prop and layers, smaller than
    # the world it replaces: the slots it frees are disabled
    obstacle = {
        "name": "slab",
        "position": [0.6, -0.3, 0.2],
        "rpy_deg": [10, 20, 30],
        "dims": [0.3, 0.02, 0.3],
    }
    prop = {
        "name": "crate",
        "position": [0.5, 0.3, 0.1],
        "rpy_deg": [0, 0, 45],
        "dims": [0.1, 0.2, 0.3],
    }
    mixed = _scene(
        tmp_path / "mixed.yaml",
        obstacles=[obstacle],
        objects=[prop],
        layers=[FLOOR, PLATFORM],
    )
    try:
        planner.update_world(mixed)
        _check_read_back(planner)
    finally:
        planner.update_world("world_sim_kitchen.yaml")
    _check_read_back(planner)
    # the chair given the kitchen with its layers, 48 boxes, and then its own
    # world again, 23: the 25 slots in between are disabled again
    try:
        chair_planner.update_world("world_sim_kitchen.yaml")
        _check_read_back(chair_planner)
    finally:
        chair_planner.update_world("world_mebot.yaml")
    _check_read_back(chair_planner)


def test_chair_config_composes_the_layers_and_warms_up_first(chair_planner):
    boxes = chair_planner.world_boxes
    layers = Counter(box["layer"] for box in boxes)
    assert layers == {"floor": 1, "platform": 16, "person": 6}
    # the installed world is the layer files: each box is an entry of its
    # file, padded on every side
    entries = {}
    for name in (FLOOR, PLATFORM, PERSON):
        data = yaml.safe_load((PACKAGED_CONFIG_DIR / name).read_text(encoding="utf-8"))
        for entry in data["obstacles"]:
            entries["%s/%s" % (data["layer"], entry["name"])] = entry
    assert sorted(entries) == sorted(box["name"] for box in boxes)
    pad = 2 * chair_planner.world_padding
    for box in boxes:
        entry = entries[box["name"]]
        assert box["dims"] == pytest.approx([d + pad for d in entry["dims"]])
        assert box["pose"] == pytest.approx(entry["position"] + [1, 0, 0, 0])
    # white-box: warm-up plans from the retract, which is inside the person,
    # so it ran in the scene's own world, the floor alone
    assert chair_planner._warm_boxes == 1
    ok, detail = chair_planner.check_state_valid(chair_planner.retract_pose)
    assert not ok and "person/" in detail, detail


@pytest.fixture(scope="session")
def platform_state(planner, chair_planner, tmp_path_factory):
    """Seeded and capped: a joint vector that the platform alone makes
    invalid on the chair. The gen3 planner finds it valid with the floor
    and invalid with the platform added, and its spheres reach 20 mm or
    more into platform boxes of the chair's start-up world, and into no
    other box. Fails, naming what it tried, if none does."""
    folder = tmp_path_factory.mktemp("platform_state")
    floor = _scene(folder / "floor.yaml", layers=[FLOOR])
    with_platform = _scene(folder / "platform.yaml", layers=[FLOOR, PLATFORM])
    states = np.random.default_rng(19).uniform(*_search_box(planner), size=(20000, 7))
    try:
        planner.update_world(floor)
        free = _feasible(planner, states)
        planner.update_world(with_platform)
        blocked = free & ~_feasible(planner, states)
    finally:
        planner.update_world("world_sim_kitchen.yaml")
    chair = _cuboids(chair_planner)
    layer = {box["name"]: box["layer"] for box in chair_planner.world_boxes}
    candidates = states[blocked]
    # cuRobo's FK refuses an empty batch
    reached = _spheres(chair_planner, candidates) if len(candidates) else []
    elsewhere, depths = 0, []
    for q, spheres in zip(candidates, reached):
        hits = sphere_box_penetration(spheres[spheres[:, 3] > 0], chair)
        if {layer[key] for key in hits} != {"platform"}:
            elsewhere += 1
        elif max(hits.values()) >= 0.02:
            return q.tolist()
        else:
            depths.append(max(hits.values()))
    pytest.fail(
        "no joint vector (20000, seed 19) reaches 20 mm into the platform and "
        "nothing else of the chair: the platform alone blocks %d on the gen3 "
        "planner, %d of those reach into other boxes of the chair, and the rest "
        "at most %.1f mm into the platform"
        % (blocked.sum(), elsewhere, 1e3 * max(depths, default=0))
    )


def test_chair_layers_survive_every_push(chair_planner, platform_state, tmp_path):
    # a push in each form update_world takes, with no layers of its own
    pushes = {
        "list": [FAR],
        "path": _scene(tmp_path / "far.yaml", obstacles=[FAR]),
        "Scene": Scene("base_link", [Obstacle(FAR)], []),
    }
    try:
        for form, world in pushes.items():
            chair_planner.update_world(world)
            layers = Counter(box["layer"] for box in chair_planner.world_boxes)
            assert layers == {"scene": 1, "platform": 16, "person": 6}, form
            # and in cuRobo: a state that only the platform blocks stays invalid
            ok, detail = chair_planner.check_state_valid(platform_state)
            assert not ok and "platform/" in detail, (form, detail)
    finally:
        chair_planner.update_world("world_mebot.yaml")


def test_person_is_excluded_without_touching_the_platform(
    chair_planner, platform_state
):
    retract = chair_planner.retract_pose
    whole = chair_planner.world_boxes
    ok, detail = chair_planner.check_state_valid(retract)
    assert not ok and "person/" in detail, detail
    try:
        chair_planner.update_world("world_mebot.yaml", ignore={"person"})
        kept = chair_planner.world_boxes
        assert not [box for box in kept if box["layer"] == "person"]
        platform = [box for box in whole if box["layer"] == "platform"]
        assert [box for box in kept if box["layer"] == "platform"] == platform
        assert chair_planner.check_state_valid(retract) == (True, "")
        ok, detail = chair_planner.check_state_valid(platform_state)
        assert not ok and "platform/" in detail, detail

        # the platform cannot be left out, and asking leaves the world as it was
        with pytest.raises(ValueError, match="layer 'platform' is not excludable"):
            chair_planner.update_world("world_mebot.yaml", ignore={"platform"})
        assert chair_planner.world_boxes == kept
        _check_read_back(chair_planner)
    finally:
        chair_planner.update_world("world_mebot.yaml")


@pytest.fixture(scope="session")
def chair_states(chair_planner):
    """Seeded and capped: a start and eight goals valid in the chair's
    start-up world, where few joint vectors are (about 2 %). Fails, naming
    what it tried, if fewer are found."""
    states = np.random.default_rng(19).uniform(
        *_search_box(chair_planner), size=(4000, 7)
    )
    valid = states[_feasible(chair_planner, states)]
    if len(valid) < 9:
        pytest.fail(
            "%d of 4000 joint vectors (seed 19) are valid on the chair; 9 are "
            "needed" % len(valid)
        )
    return valid[0].tolist(), valid[1:9].tolist()


def test_invalid_start_names_the_blocking_layer(chair_planner, chair_states):
    # the retract is inside the padded person: cuRobo refuses it as a start
    # before it plans, and the error says what the start collides with
    _, goals = chair_states
    res = chair_planner.plan_to_joints(goals[0], start=chair_planner.retract_pose)
    assert not res.success
    assert "INVALID_START" in res.status, res.status
    assert re.search(
        r"\[start collides with person/\w+ \(\d+ mm\)", res.error
    ), res.error


def test_a_layer_box_over_the_arm_base_is_refused(chair_planner, tmp_path):
    (tmp_path / "block.yaml").write_text(
        "layer: block\nobstacles:\n"
        "  - {name: over_base, position: [0, 0, 0.2], dims: [0.05, 0.05, 0.05]}\n"
    )
    before = chair_planner.world_boxes
    try:
        with pytest.raises(
            ValueError,
            match=r"^layer box block/over_base intrudes [\d.]+ mm into the arm's "
            r"fixed base/shoulder spheres",
        ):
            chair_planner.update_world(
                _scene(tmp_path / "scene.yaml", layers=["block.yaml"])
            )
        # refused before cuRobo saw it: the planner and cuRobo keep the world
        assert chair_planner.world_boxes == before
        _check_read_back(chair_planner)
    finally:
        chair_planner.update_world("world_mebot.yaml")


def _clearance(spheres, cuboids):
    """The least distance, in metres, from the surface of one of `spheres`
    ((..., 4)) to a box of `cuboids`: negative if one reaches in."""
    # grown by more than any distance here, a sphere reaches every box, by
    # `far` less its distance
    far = 10.0
    grown = np.asarray(spheres, dtype=float) + [0.0, 0.0, 0.0, far]
    return far - max(sphere_box_penetration(grown, cuboids).values(), default=0.0)


def test_plans_keep_curobo_spheres_outside_every_padded_box(
    chair_planner, chair_states
):
    # the planner's own re-check of a plan (_recheck_constraints) asks the
    # world cuRobo holds, so it cannot see a box cuRobo does not honour: this
    # asks the boxes the planner composed, on the CPU. It can catch only a box
    # the plans come near. Valid states on the chair sit against the person,
    # so these plans test the person layer, and keep clear of the platform
    # and the floor: test_platform_layer_pair and the platform_state tests
    # show that cuRobo honours the platform
    start, goals = chair_states
    boxes = _cuboids(chair_planner)
    person = _cuboids(chair_planner, "person")
    planned, deep, closest = 0, {}, math.inf
    for i, goal in enumerate(goals):
        res = chair_planner.plan_to_joints(goal, start)
        if not res.success:
            continue
        planned += 1
        spheres = _spheres(chair_planner, _path(res, start))
        hits = sphere_box_penetration(spheres[spheres[..., 3] > 0], boxes)
        deep.update({(i, key): 1e3 * d for key, d in hits.items() if d > 0.0005})
        # how near the person the plan takes the spheres it moves: the base
        # and shoulder spheres, which no joint moves, sit by it in every state
        travel = np.linalg.norm(spheres[..., :3] - spheres[0, :, :3], axis=2)
        moved = spheres[:, (travel.max(axis=0) > 1e-6) & (spheres[0, :, 3] > 0)]
        closest = min(closest, _clearance(moved, person))
    # a goal the chair leaves no path to is not this test's concern, but too
    # few plans would leave it nothing to check
    if planned < 4:
        pytest.fail("%d of the 8 seeded valid goals planned; 4 are needed" % planned)
    # and plans that keep beyond the activation distance of the person are
    # not pushed by it: they would be the same if cuRobo ignored it
    if closest > chair_planner.collision_activation_distance:
        pytest.fail(
            "the spheres the plans move come no closer than %.1f mm to the "
            "person, beyond the %.0f mm activation distance: they do not test it"
            % (1e3 * closest, 1e3 * chair_planner.collision_activation_distance)
        )
    assert not deep, "mm into a padded box, by (goal, box): %s" % deep
