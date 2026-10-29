"""Smoke test: real cuRobo planning on the GPU, no hardware, no ROS.

Skips itself cleanly on machines without CUDA/cuRobo. On the Jetson this is
the "is the stack alive" gate: one session-scoped planner init (kernel
warmup takes a minute or two on the Orin the first time) and a handful of
plans in the sim-kitchen world.

It is also where obstacle layers (issue #19) meet cuRobo: paired requests,
the same one with and without a box, that fail if cuRobo ignores the box,
and the generated chair on a second planner (planner.yaml), built only
for the tests that use it. Every search is seeded and capped, and fails
naming what it tried. A test that pushes a world puts back the one it
found.
"""

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from rammp_curobo import CuRoboPlanner  # noqa: E402
from rammp_curobo.config import resolve_config  # noqa: E402
from rammp_curobo.scene import Obstacle, Scene  # noqa: E402
from rammp_curobo.types import PlanResult, Trajectory  # noqa: E402
from rammp_curobo.world import sphere_box_penetration  # noqa: E402

# a box out of the arm's reach: a world with nothing in the way
FAR = {"name": "far", "position": [1.5, 1.5, 0.5], "dims": [0.2, 0.2, 0.2]}
# The chair is one robot's geometry, not package data: this checkout keeps a
# fixture copy of what rammp-deployments ships (issue #19).
MEBOT = Path(__file__).resolve().parent / "data" / "mebot"
# By path: a scene written to tmp_path names its layers, and a bare name no
# longer resolves against the package.
FLOOR = str(MEBOT / "layer_mebot_floor.yaml")
PLATFORM = str(MEBOT / "layer_mebot_platform.yaml")
PERSON = str(MEBOT / "layer_mebot_person.yaml")


@pytest.fixture(scope="session")
def planner(urdf_level):
    gen3 = CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_level)
    # read back as construction loaded it: the tests before the read-back
    # test push worlds on this planner and put the kitchen back through
    # update_world
    _check_read_back(gen3)
    return gen3


@pytest.fixture(scope="session")
def tilted_planner(urdf_tilt30):
    """The gen3 planner on the chair's mount, tilted 30 degrees, with no
    layers of its own: the chair's layers are pushed onto it as scene
    layers, which unlike the chair config's can be left out. Built only
    when a test asks for it."""
    return CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_tilt30)


@pytest.fixture(scope="session")
def chair_planner(urdf_tilt30):
    """The Gen3 on the MEBot, tilted 30 degrees on its mount as the layers
    were generated for: the platform and person layers in every world.
    Built only when a test asks for it."""
    return CuRoboPlanner.from_config(MEBOT / "planner.yaml", urdf_path=urdf_tilt30)


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
    # +arm_joint_1 swings toward the kitchen's open left side (-0.5 rad heads
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
    bad[1] = 3.0  # arm_joint_2 limit is ±2.24 rad
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
    # a refused push leaves the world it found, in the planner and in cuRobo
    assert planner.world_boxes == before
    _check_read_back(planner)
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


def test_empty_world_constructs_and_plans(urdf_level, tmp_path):
    # Allowed at construction only (flatbot has no table); update_world
    # still refuses an empty world, since cuRobo would keep the old one.
    cfg = yaml.safe_load(resolve_config("gen3.yaml").read_text())
    cfg["world"] = "world_empty.yaml"
    path = tmp_path / "gen3_empty.yaml"
    path.write_text(yaml.safe_dump(cfg))
    empty = CuRoboPlanner.from_config(str(path), urdf_path=urdf_level)

    q_goal = list(empty.retract_pose)
    q_goal[1] -= 0.25
    q_goal[3] += 0.30
    res = empty.plan_to_joints(q_goal, list(empty.retract_pose))
    assert res.success, res.error
    with pytest.raises(ValueError, match="empty"):
        empty.update_world([])

    # A real world on top takes effect: a box on the tool puts the start in
    # collision, and the kitchen replaces it and is planned around.
    pos, _ = empty.fk(empty.retract_pose)
    empty.update_world([{"name": "block", "position": pos, "dims": [0.3, 0.3, 0.3]}])
    assert not empty.check_state_valid(empty.retract_pose)[0]
    res = empty.plan_to_joints(q_goal, list(empty.retract_pose))
    assert not res.success and "WORLD_COLLISION" in str(res.status), res.status
    empty.update_world("world_sim_kitchen.yaml")
    assert empty.check_state_valid(empty.retract_pose)[0]
    res = empty.plan_to_joints(q_goal, list(empty.retract_pose))
    assert res.success, res.error
    assert all(empty.check_state_valid(q)[0] for q in res.joint_traj.positions)


def test_a_start_inside_the_limit_margin_is_refused_and_one_outside_plans(planner):
    # The Gen3 rest preset sits 22 mrad inside arm_joint_4's -2.57: inside
    # the 0.1 rad margin, so refused by name rather than moved. The CPU
    # suite covers the 0.05 rad clamp past the limit, with no margin.
    near = list(planner.retract_pose)
    near[3] = -2.548
    res = planner.plan_to_joints(planner.retract_pose, start=near)
    assert res.status == "START_IN_LIMIT_MARGIN"
    assert "arm_joint_4" in res.error

    near[3] = -2.46
    res = planner.plan_to_joints(planner.retract_pose, start=near)
    assert res.success, res.error


def test_curobo_plans_inside_the_margin(planner):
    lo, hi = planner.joint_limits()["position"]  # the URDF's
    lim = planner._motion_gen.kinematics.get_joint_limits().position
    order = [planner._curobo_joint_names.index(n) for n in planner.joint_names]
    curobo = lim.detach().cpu().numpy()[:, order]
    np.testing.assert_allclose(curobo[0], lo + 0.1, atol=1e-6)
    np.testing.assert_allclose(curobo[1], hi - 0.1, atol=1e-6)


def test_a_goal_on_a_joint_limit_is_not_planned_to_it(planner):
    """2026-10-19: cuRobo's IK put this GoToEEPose goal's arm_joint_4 ON its
    2.57 limit; the quintic resample passed it and the re-check rejected the
    plan (core/tests/data/curobo_at_limit.json). With the margin, every plan
    keeps 0.1 rad from every limit, or the goal fails cleanly."""
    with open(Path(__file__).parent / "data" / "curobo_at_limit.json") as f:
        start = json.load(f)["start"]
    goal = (
        [0.4566828134433112, 0.0013347502139383046, 0.4336047009472797],
        [0.500044674874822, 0.5001091592170074, 0.49989060359678683, 0.499955534455047],
    )
    lo, hi = planner.joint_limits()["position"]
    try:
        planner.update_world("world_real_bench.yaml")  # where it was captured
        for _ in range(8):  # cuRobo hit the limit on ~1 in 7 of these
            res = planner.plan_to_pose(*goal, start)
            assert res.status != "LIBRARY_VALIDATION_FAILED", res.error
            if res.success:
                q = res.joint_traj.positions
                assert np.minimum(q - lo, hi - q).min() > 0.1 - 1e-3
    finally:
        planner.update_world("world_sim_kitchen.yaml")


def test_joint_goal_normalized_to_start_branch(planner):
    # arm reports arm_joint_3 = -pi (wrapped), goal authored at +pi: the plan
    # must NOT wind a full revolution — the goal shifts to the -pi branch
    start = list(planner.retract_pose)
    start[2] = -3.1416
    goal = list(planner.retract_pose)  # arm_joint_3 = +3.142
    goal[0] = 0.15
    res = planner.plan_to_joints(goal, start=start)
    assert res.success, res.error
    j3 = res.joint_traj.positions[:, 2]
    assert abs(j3[-1] - (-3.1416)) < 0.05  # stayed on the start's branch
    assert np.abs(np.diff(j3)).sum() < 0.2  # no winding


def _inject_wound_plan(planner, monkeypatch, start, j, wound_end):
    """Make the next plan_single come back wound: its one _finish result is
    a plan moving joint j alone from start to wound_end. Which branch IK
    picks depends on its random seeds -- #14 did not recur in ~270 plans
    here -- and cuRobo's trajopt will not produce that path on demand. The
    real plan_single is skipped: its result would be discarded, and the
    subject is what plan_to_pose does next. The replan stays real."""
    real_finish = planner._finish
    real_plan_single = planner._motion_gen.plan_single

    def no_plan(*a, **k):
        monkeypatch.setattr(planner._motion_gen, "plan_single", real_plan_single)

    monkeypatch.setattr(planner._motion_gen, "plan_single", no_plan)

    def wound_plan(result, t0, q_goal=None):
        monkeypatch.setattr(planner, "_finish", real_finish)
        pos = np.tile(start, (200, 1))
        pos[:, j] = np.linspace(start[j], wound_end, 200)
        return PlanResult(
            success=True,
            joint_traj=Trajectory(
                list(planner.joint_names),
                pos,
                None,
                None,
                0.05,
                frame_id=planner.base_frame,
            ),
            timing=0.0,
            error=None,
            status="OK",
            validated=True,
            final_joints=pos[-1].copy(),
        )

    monkeypatch.setattr(planner, "_finish", wound_plan)


@pytest.mark.parametrize("move", [0.0, 0.6 * math.pi])
@pytest.mark.parametrize("parked", [-math.pi, math.pi])
@pytest.mark.parametrize("j", [0, 2, 4, 6])  # joints 1/3/5/7, the continuous ones
def test_pose_goal_normalized_to_start_branch(planner, j, parked, move, monkeypatch):
    # #14: the arm parks a continuous joint on the +/-pi boundary, and IK
    # lands that joint on the OTHER 2*pi branch from the goal it should
    # reach (observed live: arm at -pi, a goal at its own pose, IK returned
    # +pi), so the plan goes the long way round. move=0 is that same-pose
    # revolution; move=0.6*pi is a real move whose wound plan travels
    # 1.4*pi, which a path-vs-net-change heuristic misses.
    start = list(planner.retract_pose)
    start[j] = parked
    ok, why = planner.check_state_valid(start)
    assert ok, why
    goal = list(start)
    goal[j] = parked + math.copysign(move, parked)  # inside +/-6.2832
    _inject_wound_plan(
        planner, monkeypatch, start, j, goal[j] - math.copysign(2 * math.pi, parked)
    )
    pos, quat = planner.fk(goal)
    res = planner.plan_to_pose(pos, quat, start=start)
    assert res.success, res.error
    assert res.validated
    assert res.goal_mismatch_rad is None
    assert res.error is None
    assert res.joint_traj.frame_id == "arm_mount"  # the replan is stamped too
    q = res.joint_traj.positions
    assert np.abs(np.diff(q[:, j])).sum() < move + 0.2  # the short way
    assert abs(q[-1, j] - goal[j]) < 0.05  # on the start's branch


def test_a_failed_unwinding_replan_says_so(planner, monkeypatch):
    # the wound plan is still returned (it is valid and collision-free) but
    # not silently: error carries the warning to the consumer
    start = list(planner.retract_pose)
    start[2] = -math.pi
    _inject_wound_plan(planner, monkeypatch, start, 2, math.pi)
    monkeypatch.setattr(
        planner,
        "plan_to_joints",
        lambda *a, **k: PlanResult.failure("TRAJOPT_FAIL", "injected"),
    )
    pos, quat = planner.fk(start)
    res = planner.plan_to_pose(pos, quat, start=start)
    assert res.success
    assert res.error and "2*pi branch" in res.error


def test_winding_under_a_hold_is_warned_not_replanned(planner, monkeypatch):
    from rammp_curobo import planner as planner_module
    from rammp_curobo.constraints import HOLD_LEVEL, PoseConstraint

    start = list(planner.retract_pose)
    start[2] = -math.pi
    _inject_wound_plan(planner, monkeypatch, start, 2, math.pi)
    replans = []
    monkeypatch.setattr(planner, "plan_to_joints", lambda *a, **k: replans.append(a))
    # the module logger's calls are recorded directly: caplog intermittently
    # captured nothing in the full Jetson-image run
    warned = []
    monkeypatch.setattr(
        planner_module.log, "warning", lambda msg, *a: warned.append(msg % a)
    )
    pos, quat = planner.fk(start)
    planner.plan_to_pose(
        pos, quat, start=start, constraint=PoseConstraint(hold=HOLD_LEVEL)
    )
    assert not replans
    assert any("2*pi branch" in w for w in warned), warned


# ------------------------------------------------------------ obstacle layers


def _scene(path, **sections):
    """Write a world file holding `sections` (obstacles, objects, layers),
    in the planning frame; with no obstacles, it says `obstacles: []`."""
    sections = dict({"base_frame": "arm_mount", "obstacles": []}, **sections)
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
    """The request the box-on-the-path tests block, from the retract to the retract
    turned 1 rad on arm_joint_1, planned in a free world (one far box), where it
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


def test_platform_layer_pair(tilted_planner, tmp_path):
    # the generated platform, as a scene layer of the gen3 planner on the
    # chair's mount: the same request with it and without it
    planner = tilted_planner
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
    # the chair given the kitchen with its layers, 50 boxes, and then its own
    # world again, 25: the 25 slots in between are disabled again
    try:
        chair_planner.update_world("world_sim_kitchen.yaml")
        _check_read_back(chair_planner)
    finally:
        chair_planner.update_world(MEBOT / "world.yaml")
    _check_read_back(chair_planner)


def test_chair_config_composes_the_layers_and_warms_up_first(chair_planner):
    boxes = chair_planner.world_boxes
    layers = Counter(box["layer"] for box in boxes)
    assert layers == {"floor": 1, "platform": 16, "person": 9}
    # the installed world is the layer files: each box is an entry of its
    # file, padded on every side
    entries = {}
    for name in (FLOOR, PLATFORM, PERSON):
        data = yaml.safe_load(Path(name).read_text(encoding="utf-8"))
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
def platform_state(tilted_planner, chair_planner, tmp_path_factory):
    """Seeded and capped: a joint vector that the platform alone makes
    invalid on the chair. The gen3 planner on the chair's mount finds it
    valid with the floor
    and invalid with the platform added, and its spheres reach 20 mm or
    more into platform boxes of the chair's start-up world, and into no
    other box. Fails, naming what it tried, if none does."""
    planner = tilted_planner
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
        "Scene": Scene("arm_mount", [Obstacle(FAR)], []),
    }
    try:
        for form, world in pushes.items():
            chair_planner.update_world(world)
            layers = Counter(box["layer"] for box in chair_planner.world_boxes)
            assert layers == {"scene": 1, "platform": 16, "person": 9}, form
            # and in cuRobo: a state that only the platform blocks stays invalid
            ok, detail = chair_planner.check_state_valid(platform_state)
            assert not ok and "platform/" in detail, (form, detail)
    finally:
        chair_planner.update_world(MEBOT / "world.yaml")


def test_person_is_excluded_without_touching_the_platform(
    chair_planner, platform_state
):
    retract = chair_planner.retract_pose
    whole = chair_planner.world_boxes
    ok, detail = chair_planner.check_state_valid(retract)
    assert not ok and "person/" in detail, detail
    try:
        chair_planner.update_world(MEBOT / "world.yaml", ignore={"person"})
        kept = chair_planner.world_boxes
        assert not [box for box in kept if box["layer"] == "person"]
        platform = [box for box in whole if box["layer"] == "platform"]
        assert [box for box in kept if box["layer"] == "platform"] == platform
        assert chair_planner.check_state_valid(retract) == (True, "")
        ok, detail = chair_planner.check_state_valid(platform_state)
        assert not ok and "platform/" in detail, detail

        # the platform cannot be left out, and asking leaves the world as it was
        with pytest.raises(ValueError, match="layer 'platform' is not excludable"):
            chair_planner.update_world(MEBOT / "world.yaml", ignore={"platform"})
        assert chair_planner.world_boxes == kept
        _check_read_back(chair_planner)
    finally:
        chair_planner.update_world(MEBOT / "world.yaml")


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
        chair_planner.update_world(MEBOT / "world.yaml")


def test_plans_keep_curobo_spheres_outside_every_padded_box(
    chair_planner, chair_states
):
    # the planner's own re-check of a plan (_recheck_constraints) asks the
    # world cuRobo holds, so it cannot see a box cuRobo does not honour: this
    # asks the boxes the planner composed, on the CPU. It can catch only a box
    # the plans would reach into if cuRobo ignored it: on the chair, the
    # person, which some of these requests cross when it is left out. They
    # keep clear of the platform and the floor: test_platform_layer_pair and
    # the platform_state tests show that cuRobo honours the platform
    start, goals = chair_states
    boxes = _cuboids(chair_planner)
    person = _cuboids(chair_planner, "person")
    planned, deep = [], {}
    for i, goal in enumerate(goals):
        res = chair_planner.plan_to_joints(goal, start)
        if not res.success:
            continue
        planned.append(goal)
        spheres = _spheres(chair_planner, _path(res, start))
        hits = sphere_box_penetration(spheres[spheres[..., 3] > 0], boxes)
        deep.update({(i, key): 1e3 * d for key, d in hits.items() if d > 0.0005})
    # a goal the chair leaves no path to is not this test's concern, but too
    # few plans would leave it nothing to check
    if len(planned) < 4:
        pytest.fail(
            "%d of the 8 seeded valid goals planned; 4 are needed" % len(planned)
        )
    # and the person must be what keeps them out: planned without it, some of
    # the same requests reach into it, as they would here if cuRobo ignored it
    depths = []
    try:
        chair_planner.update_world(MEBOT / "world.yaml", ignore={"person"})
        for goal in planned:
            res = chair_planner.plan_to_joints(goal, start)
            if res.success:
                spheres = _spheres(chair_planner, _path(res, start))
                hits = sphere_box_penetration(spheres[spheres[..., 3] > 0], person)
                depths.append(max(hits.values(), default=0.0))
    finally:
        chair_planner.update_world(MEBOT / "world.yaml")
    if max(depths, default=0.0) <= 0.0005:
        pytest.fail(
            "none of the %d requests that planned reaches into the person when "
            "it is left out: %d of them planned without it, and reach at most "
            "%.1f mm into it, so these plans would pass if cuRobo ignored it"
            % (len(planned), len(depths), 1e3 * max(depths, default=0.0))
        )
    assert not deep, "mm into a padded box, by (goal, box): %s" % deep


def test_held_and_via_plans_on_the_chair_keep_out_of_its_layers(
    chair_planner, chair_states
):
    # #16's orientation hold and #17's via point plan through plan_to_pose's
    # own paths. On the chair they must plan in the world with its layers:
    # a held or via plan from the retract, which is inside the padded person,
    # is refused and its error names the person box; and the plans that do
    # come back keep cuRobo's spheres out of every padded box
    from rammp_curobo.constraints import HOLD_FIXED, PoseConstraint, ViaPoint

    fixed = PoseConstraint(hold=HOLD_FIXED)
    via = ViaPoint(offset_m=0.05)
    world = re.compile(
        r" \[world: \d+ boxes: .*config layers: platform 16, person 9 .*\]$"
    )
    retract = chair_planner.retract_pose
    pos, quat = chair_planner.fk(retract)
    for kind, kw in (("hold", dict(constraint=fixed)), ("via", dict(via=via))):
        res = chair_planner.plan_to_pose(pos, quat, retract, **kw)
        assert not res.success, kind
        assert "INVALID_START" in res.status, (kind, res.status)
        assert re.search(r"\[start collides with person/\w+ \(\d+ mm\)", res.error), (
            kind,
            res.error,
        )
        assert world.search(res.error), (kind, res.error)

    # from a valid start toward the seeded valid goals' tool points: held, at
    # the start's own orientation, 10 cm toward each (a random goal's point is
    # often out of reach at the start's orientation), and approached through
    # a via point at the goal's own
    start, goals = chair_states
    start_pos, start_quat = chair_planner.fk(start)
    boxes = _cuboids(chair_planner)
    planned, deep, failed = Counter(), {}, []
    for i, goal in enumerate(goals):
        goal_pos, goal_quat = chair_planner.fk(goal)
        toward = np.asarray(goal_pos) - np.asarray(start_pos)
        held_pos = np.asarray(start_pos) + 0.1 * toward / np.linalg.norm(toward)
        requests = (
            ("hold", held_pos.tolist(), start_quat, dict(constraint=fixed)),
            ("via", goal_pos, goal_quat, dict(via=via)),
        )
        for kind, p, q, kw in requests:
            res = chair_planner.plan_to_pose(p, q, start, **kw)
            if not res.success:
                failed.append((kind, i, res.status, res.error))
                continue
            planned[kind] += 1
            spheres = _spheres(chair_planner, _path(res, start))
            hits = sphere_box_penetration(spheres[spheres[..., 3] > 0], boxes)
            deep.update(
                {(kind, i, key): 1e3 * d for key, d in hits.items() if d > 0.0005}
            )
    # a failure cuRobo reports says where the world's boxes come from; one the
    # planner refuses after a plan cuRobo found (the hold check, or
    # post-validation's limits) has its own
    refused = ("CONSTRAINT_VIOLATED", "CONSTRAINT_UNSATISFIABLE")
    for kind, i, status, error in failed:
        if status not in refused + ("LIBRARY_VALIDATION_FAILED",):
            assert world.search(error), (kind, i, status, error)
    print(
        "\n  chair: %d held and %d via plans of 8 each; failures: %s"
        % (planned["hold"], planned["via"], [(k, i, s) for k, i, s, _ in failed])
    )
    assert planned["via"] >= 1 and planned["hold"] >= 1, (
        "too few plans to check: %s; failures: %s"
        % (dict(planned), [(k, i, s) for k, i, s, _ in failed])
    )
    assert not deep, "mm into a padded box, by (kind, goal, box): %s" % deep


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


def test_accelerations_match_positions_and_limits(planner, start):
    """#15: on a plan cuRobo optimized coarser than interpolation_dt, the
    shipped accelerations are the positions' own (LINEAR_CUDA's missed by
    ~95%), within the acceleration limit, and the plan ends on the goal at
    rest."""
    q_goal = list(
        np.asarray(planner.retract_pose) + [1.2, -0.2, 0.3, 0.5, -0.4, 0, -1.0]
    )
    res = planner.plan_to_joints(q_goal, start)
    assert res.success, res.error
    t = res.joint_traj
    assert t.accelerations is not None
    p, v, a, dt = t.positions, t.velocities, t.accelerations, t.dt
    a_fd = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2
    assert np.abs(a_fd - a[1:-1]).max() < 0.1 * np.abs(a).max()
    lim = planner.joint_limits()["acceleration"]
    assert lim.shape == (7,) and (lim > 0).all()
    assert (np.abs(a).max(axis=0) <= lim * 1.01).all()
    np.testing.assert_allclose(v[-1], 0.0, atol=1e-9)
    np.testing.assert_allclose(a[-1], 0.0, atol=1e-9)


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
        "\n  verification is %.2f%% of planning"
        % (t_verify * 1e3, n, t_plan * 1e3, 100.0 * share)
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


def _corner(planner):
    """Two goals with a corner between them, near the retract: a pose
    (arm_joint_1 turned 0.5 rad) and then joints (the elbow bent from there)."""
    q1 = list(planner.retract_pose)
    q1[0] += 0.5
    q2 = list(q1)
    q2[1] -= 0.3
    q2[3] += 0.3
    pos, quat = planner.fk(q1)
    return pos, quat, q2


def test_blend_legs_joins_a_held_leg_and_a_slower_free_one(planner, start):
    from rammp_curobo import PoseConstraint

    pos, quat, q2 = _corner(planner)
    level = planner.plan_to_pose(pos, quat, start, constraint=PoseConstraint(hold=1))
    assert level.success, level.error
    free = planner.plan_to_joints(q2, level.joint_traj.positions[-1])
    assert free.success, free.error
    legs = [level.joint_traj, free.joint_traj]
    hold = PoseConstraint(hold=1)
    res = planner.blend_legs(legs, [0.05], [1.0, 0.5], [hold, None], [quat, None])
    assert res.success and res.validated, res.error
    traj = res.joint_traj
    assert 0.0 < res.corners_m[0] <= 0.05
    # level for as long as the held leg runs, its overlap included
    _, wxyz = planner._tool_poses(traj.positions[: level.joint_traj.n_points])
    tilt = planner._hold_deviation_batch(wxyz, quat, hold)
    assert tilt.max() <= planner.constraint_tolerance_rad
    assert np.abs(traj.positions[-1] - np.asarray(q2)).max() < 1e-3
    assert _feasible(planner, traj.positions).all()
    # the free leg at half its speed takes twice as long; the overlap shortens
    # only the held leg
    assert traj.duration >= 2 * free.joint_traj.duration - traj.dt
