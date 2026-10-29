"""The planner's own safety contracts, on CPU — no torch, no cuRobo.

The real __init__ runs (planner config, robot YAML, base_frame, world) with
_init_curobo stubbed out and the joint limits read from the fixture URDF, so
CI pins what the GPU suite pins only when someone runs it by hand: every
plan is stamped arm_mount, the park pose is refused by name, and an empty
world never reaches cuRobo on update. The GPU tests (test_smoke.py,
test_model_parity.py) still prove cuRobo's side; these prove ours.
"""

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from rammp_curobo import CuRoboPlanner

DATA = Path(__file__).parent / "data"


class _T:
    """A torch-tensor stand-in: .detach().cpu().numpy() and .item()."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def item(self):
        return self.a.item()


def _urdf_limits():
    root = ET.parse(DATA / "arm_level.urdf").getroot()
    lo, hi = [], []
    for i in range(1, 8):
        lim = root.find("joint[@name='arm_joint_%d']/limit" % i)
        lo.append(float(lim.get("lower", -6.2832)))
        hi.append(float(lim.get("upper", 6.2832)))
    return np.array(lo), np.array(hi)


@pytest.fixture
def cpu_planner(monkeypatch, urdf_level):
    monkeypatch.setattr(CuRoboPlanner, "_init_curobo", lambda self, warmup: None)
    p = CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_level)
    lo, hi = _urdf_limits()
    monkeypatch.setattr(
        p,
        "joint_limits",
        lambda: {"position": (lo, hi), "velocity": np.full(7, 10.0)},
    )
    monkeypatch.setattr(p, "_recheck_constraints", lambda plan: [])
    # set by the stubbed _init_curobo, from cuRobo's FK: with none, no layer
    # box is refused for reaching into the arm's fixed spheres
    p._fixed_spheres = np.empty((0, 4))
    return p


def test_finish_stamps_the_models_root(cpu_planner):
    q = np.tile(cpu_planner.retract_pose, (3, 1))
    ordered = types.SimpleNamespace(position=_T(q), velocity=None, acceleration=None)
    plan = types.SimpleNamespace(get_ordered_joint_state=lambda names: ordered)
    result = types.SimpleNamespace(
        success=_T(True), interpolation_dt=0.02, get_interpolated_plan=lambda: plan
    )
    res = cpu_planner._finish(result, 0.0)
    assert res.success, res.error
    assert cpu_planner.base_frame == "arm_mount"
    assert cpu_planner.ee_link == "arm_tool_frame"  # what the node logs
    assert res.joint_traj.frame_id == "arm_mount"
    assert res.joint_traj.scaled(0.5).frame_id == "arm_mount"


def test_a_robot_config_rooted_elsewhere_is_refused(monkeypatch, urdf_level):
    from rammp_curobo import planner as planner_mod

    real = planner_mod.load_robot_config

    def rooted_at_arm_base_link(*args, **kwargs):
        cfg = real(*args, **kwargs)
        cfg["kinematics"]["base_link"] = "arm_base_link"
        return cfg

    monkeypatch.setattr(planner_mod, "load_robot_config", rooted_at_arm_base_link)
    monkeypatch.setattr(CuRoboPlanner, "_init_curobo", lambda self, warmup: None)
    with pytest.raises(
        ValueError, match="base_link is .arm_base_link., expected .arm_mount."
    ):
        CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_level)


def test_park_pose_is_refused_by_name_and_a_near_miss_clamped(cpu_planner):
    cpu_planner.joint_limit_margin_rad = 0.0  # the clamp alone
    park = list(cpu_planner.retract_pose)
    park[3] = -2.674  # the Gen3 park pose, 0.104 rad past the URDF's -2.57
    q, err = cpu_planner._clamp_to_limits(park, "start")
    assert q is None and err[0] == "START_OUTSIDE_LIMITS"
    assert "arm_joint_4" in err[1]
    park[3] = -2.60
    q, err = cpu_planner._clamp_to_limits(park, "start")
    assert err is None and q[3] == pytest.approx(-2.57)


def test_the_joint_limit_margin_defaults_to_a_tenth_of_a_radian(cpu_planner):
    assert cpu_planner.joint_limit_margin_rad == 0.1
    assert cpu_planner._robot_cfg["kinematics"]["cspace"]["position_limit_clip"] == 0.1


@pytest.mark.parametrize("what", ["start", "goal"])
def test_a_joint_inside_the_margin_is_refused_by_name_not_moved(cpu_planner, what):
    q = list(cpu_planner.retract_pose)
    q[3] = -2.548  # the Gen3 rest preset: 22 mrad inside the URDF's -2.57
    out, err = cpu_planner._clamp_to_limits(q, what)
    assert out is None and err[0] == "%s_IN_LIMIT_MARGIN" % what.upper()
    assert "arm_joint_4" in err[1] and "0.022" in err[1] and "0.100" in err[1]
    # Past the limit by less than the clamp: still refused, not clamped onto
    # the limit, because the limit is inside the margin.
    q[3] = -2.60
    assert cpu_planner._clamp_to_limits(q, what)[1][0].endswith("_IN_LIMIT_MARGIN")
    q[3] = -2.46  # 0.11 inside: planned from as given
    assert cpu_planner._clamp_to_limits(q, what) == (q, None)


def test_a_negative_joint_limit_margin_is_refused(monkeypatch, urdf_level):
    from rammp_curobo.config import load_planner_config

    monkeypatch.setattr(CuRoboPlanner, "_init_curobo", lambda self, warmup: None)
    cfg, cfg_dir = load_planner_config("gen3.yaml")
    cfg["planner"]["joint_limit_margin_rad"] = -0.01
    with pytest.raises(ValueError, match="joint_limit_margin_rad"):
        CuRoboPlanner(cfg, config_dir=cfg_dir, urdf_path=urdf_level)


def test_update_world_refuses_before_reaching_curobo(
    cpu_planner, monkeypatch, tmp_path
):
    geom = types.ModuleType("curobo.geom.types")
    # every kind is listed: to_world_config checks what the WorldConfig holds
    kinds = dict(sphere=[], cylinder=[], capsule=[], mesh=[])
    geom.WorldConfig = types.SimpleNamespace(
        from_dict=lambda d: types.SimpleNamespace(cuboid=list(d["cuboid"]), **kinds)
    )
    for name in ("curobo", "curobo.geom"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "curobo.geom.types", geom)
    sent = []
    cpu_planner._motion_gen = types.SimpleNamespace(update_world=sent.append)

    # cuRobo v0.7.8 silently keeps the old world on an empty update
    with pytest.raises(ValueError, match="empty"):
        cpu_planner.update_world([])
    with pytest.raises(ValueError, match="empty"):
        cpu_planner.update_world("world_empty.yaml")
    wrong = tmp_path / "world_base_link.yaml"
    wrong.write_text(
        "base_frame: base_link\nobstacles:\n"
        "  - {name: box, position: [0.5, 0, 0], dims: [0.1, 0.1, 0.1]}\n"
    )
    with pytest.raises(ValueError, match="base_link"):
        cpu_planner.update_world(str(wrong))
    assert not sent

    # a list of obstacles is in the planning frame, so it is not refused
    cpu_planner.update_world(
        [{"name": "box", "position": [0.5, 0, 0], "dims": [0.1, 0.1, 0.1]}]
    )
    assert len(sent) == 1 and cpu_planner.scene.base_frame == "arm_mount"


@pytest.fixture
def through_planner(cpu_planner, monkeypatch):
    """cpu_planner whose legs are straight joint moves (a pose waypoint's
    position is its first three joints), whose "tool" is the first three
    joints, and whose world is empty: records every leg's start."""
    from rammp_curobo import blend
    from rammp_curobo.types import PlanResult, Trajectory

    lo, hi = _urdf_limits()
    lim = {
        "position": (lo, hi),
        "velocity": np.full(7, 1.2),
        "acceleration": np.full(7, 25.0),
    }
    monkeypatch.setattr(cpu_planner, "joint_limits", lambda: lim)
    starts, failing = [], {}

    def leg(goal, start):
        starts.append(np.asarray(start, dtype=float))
        if len(starts) in failing:
            return PlanResult.failure("IK_FAIL", failing[len(starts)])
        p, v, a = blend.straight_leg(
            start, goal, lim["velocity"], lim["acceleration"], 0.02
        )
        traj = Trajectory(
            list(cpu_planner.joint_names), p, v, a, 0.02, frame_id="arm_mount"
        )
        return PlanResult(True, traj, 0.0, None, "OK", True, p[-1].copy())

    monkeypatch.setattr(
        cpu_planner,
        "plan_to_pose",
        lambda pos, quat, start, constraint=None: leg(np.r_[pos, np.zeros(4)], start),
    )
    monkeypatch.setattr(cpu_planner, "plan_to_joints", lambda q, start: leg(q, start))
    monkeypatch.setattr(
        cpu_planner, "_tool_positions", lambda qs: np.asarray(qs)[:, :3]
    )
    cpu_planner.feasible = lambda qs: np.ones(len(qs), dtype=bool)
    monkeypatch.setattr(cpu_planner, "_feasible", lambda qs: cpu_planner.feasible(qs))
    monkeypatch.setattr(cpu_planner, "check_state_valid", lambda q: (False, "hits box"))
    cpu_planner.starts, cpu_planner.failing = starts, failing
    return cpu_planner


def _square(radius):
    """Three waypoints round a 0.4 rad square in joints 1-2, from joints 0."""
    from rammp_curobo import Waypoint

    return [
        Waypoint(
            position=(0.4, 0.0, 0.0), quaternion=(0, 0, 0, 1), blend_radius=radius
        ),
        Waypoint(joints=(0.4, 0.4, 0, 0, 0, 0, 0), blend_radius=radius),
        Waypoint(joints=(0.0, 0.4, 0, 0, 0, 0, 0), straight=True, blend_radius=radius),
    ]


def test_plan_through_stops_where_the_radius_is_zero(through_planner):
    res = through_planner.plan_through(_square(0.0), np.zeros(7))
    assert res.success, res.error
    assert res.corners_m == [0.0, 0.0, 0.0]
    speed = np.linalg.norm(res.joint_traj.velocities, axis=1)
    assert (speed[1:-1] == 0.0).sum() == 2  # at rest on the two inner waypoints
    rest = res.joint_traj.positions[1:-1][speed[1:-1] == 0.0]
    np.testing.assert_allclose(
        rest, [[0.4, 0, 0, 0, 0, 0, 0], [0.4, 0.4, 0, 0, 0, 0, 0]]
    )
    # each leg planned from where the one before it ended (the third is straight)
    np.testing.assert_allclose(through_planner.starts[1], [0.4, 0, 0, 0, 0, 0, 0])


def test_plan_through_blends_within_the_radius_and_the_limits(through_planner):
    stopped = through_planner.plan_through(_square(0.0), np.zeros(7)).joint_traj
    res = through_planner.plan_through(_square(0.05), np.zeros(7))
    assert res.success, res.error
    traj = res.joint_traj
    speed = np.linalg.norm(traj.velocities, axis=1)
    assert speed[1:-1].min() > 0.0  # moving all the way to the goal
    assert 0.0 < max(res.corners_m) <= 0.05 and res.corners_m[-1] == 0.0
    assert (np.abs(traj.velocities) <= 1.2).all()
    assert traj.duration < stopped.duration - 0.2
    np.testing.assert_allclose(traj.positions[-1], [0, 0.4, 0, 0, 0, 0, 0], atol=1e-12)
    assert traj.frame_id == "arm_mount"


def test_plan_through_does_not_cut_a_corner_that_collides(through_planner):
    # infeasible wherever joint 2 moves before joint 1 has arrived: inside
    # the first corner, which the planned legs never enter
    through_planner.feasible = lambda qs: ~(
        (np.asarray(qs)[:, 0] < 0.4 - 1e-9) & (np.asarray(qs)[:, 1] > 1e-9)
    )
    res = through_planner.plan_through(_square(0.05)[:2], np.zeros(7))
    assert res.success, res.error
    assert res.corners_m == [0.0, 0.0]


def test_plan_through_refuses_a_waypoint_that_fails_naming_it(through_planner):
    through_planner.failing[2] = "no IK solution"
    res = through_planner.plan_through(_square(0.05), np.zeros(7))
    assert not res.success and res.joint_traj is None
    assert res.status == "IK_FAIL"
    assert res.error == "waypoint 2/3: no IK solution"
    assert len(through_planner.starts) == 2  # nothing planned past it


def test_plan_through_refuses_a_straight_leg_that_collides(through_planner):
    through_planner.feasible = lambda qs: np.asarray(qs)[:, 1] < 0.2
    res = through_planner.plan_through(_square(0.05)[2:], np.r_[0.4, 0.4, np.zeros(5)])
    assert res.status == "STRAIGHT_LEG_INFEASIBLE"
    assert "waypoint 1/1: the straight joint leg is infeasible at 0.00 s" in res.error
    assert "hits box" in res.error


def test_plan_through_refuses_a_malformed_waypoint(through_planner):
    from rammp_curobo import Waypoint

    for wp, why in [
        (Waypoint(), "a pose (position, quaternion) or joints"),
        (Waypoint(position=(0, 0, 0)), "needs both"),
        (
            Waypoint(position=(0, 0, 0), quaternion=(0, 0, 0, 1), straight=True),
            "straight",
        ),
        (Waypoint(joints=(0,) * 7, blend_radius=-1.0), "blend_radius"),
    ]:
        res = through_planner.plan_through([wp], np.zeros(7))
        assert res.status == "BAD_WAYPOINT" and why in res.error
    assert not through_planner.starts


def _legs(planner, *points):
    """plan_to_joints legs through `points`, each from the one before's end."""
    legs = []
    for a, b in zip(points, points[1:]):
        legs.append(planner.plan_to_joints(np.asarray(b, float), a).joint_traj)
    return legs


CORNER = (np.zeros(7), np.r_[0.4, np.zeros(6)], np.r_[0.4, 0.4, np.zeros(5)])


def test_blend_legs_blends_a_slow_leg_into_a_fast_one(through_planner):
    legs = _legs(through_planner, *CORNER)
    stopped = through_planner.blend_legs(legs, [0.0], [0.25, 1.0])
    res = through_planner.blend_legs(legs, [0.05], [0.25, 1.0])
    assert stopped.success and res.success, res.error
    traj = res.joint_traj
    assert traj.dt == pytest.approx(0.02) and traj.frame_id == "arm_mount"
    # the first leg (joint 1) at a quarter of its speed, the second (joint 2) at full
    assert np.abs(traj.velocities[:, 0]).max() == pytest.approx(
        0.25 * np.abs(legs[0].velocities[:, 0]).max(), rel=0.02
    )
    assert np.abs(traj.velocities[:, 1]).max() == pytest.approx(
        np.abs(legs[1].velocities[:, 1]).max(), rel=0.02
    )
    assert np.linalg.norm(traj.velocities, axis=1)[1:-1].min() > 0.0
    assert 0.0 < res.corners_m[0] <= 0.05 and len(res.corners_m) == 1
    assert traj.duration < stopped.joint_traj.duration
    np.testing.assert_allclose(traj.positions[-1], CORNER[-1], atol=1e-12)


def test_blend_legs_refuses_legs_that_do_not_chain(through_planner):
    from rammp_curobo.types import Trajectory

    a, b = _legs(through_planner, *CORNER)
    c = _legs(through_planner, np.full(7, 0.1), np.zeros(7))[0]
    moving = Trajectory(a.joint_names, a.positions[:-5], a.velocities[:-5], None, a.dt)
    for legs, radii, scales, why in [
        ([], [], None, "no legs"),
        ([a, b], [], None, "2 legs need 1 blend radii"),
        ([a, b], [0.0], [1.0, 0.0], "leg 2/2: speed scale 0 is outside (0, 1]"),
        ([a, c], [0.0], None, "leg 2/2: starts 0.300 rad from where leg 1 ends"),
        ([moving, b], [0.0], None, "leg 1/2: starts or ends moving"),
    ]:
        res = through_planner.blend_legs(legs, radii, scales)
        assert res.status == "BAD_LEG" and why in res.error, res.error
