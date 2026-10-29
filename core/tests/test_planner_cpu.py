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
def blend_planner(cpu_planner, monkeypatch):
    """cpu_planner whose plan_to_joints legs are straight minimum-jerk moves
    at the joint limits, whose "tool" is at the first three joints, tilted
    about x by the fourth and then turned about the vertical by the fifth,
    and whose world is empty unless `feasible` says otherwise."""
    from rammp_curobo.types import PlanResult, Trajectory

    lo, hi = _urdf_limits()
    cpu_planner.lim = {
        "position": (lo, hi),
        "velocity": np.full(7, 1.2),
        "acceleration": np.full(7, 25.0),
    }
    monkeypatch.setattr(cpu_planner, "joint_limits", lambda: cpu_planner.lim)

    def leg(goal, start):
        lim = cpu_planner.lim
        q0, q1 = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        d = q1 - q0
        # minimum jerk peaks at 1.875 x the mean speed and 5.774 d / T^2
        duration = max(
            np.max(1.875 * np.abs(d) / lim["velocity"]),
            np.max(np.sqrt(5.7735 * np.abs(d) / lim["acceleration"])),
            0.02,
        )
        n = int(np.ceil(duration / 0.02))
        u = np.arange(n + 1)[:, None] / n
        T = n * 0.02
        p = q0 + (10 * u**3 - 15 * u**4 + 6 * u**5) * d
        v = (30 * u**2 - 60 * u**3 + 30 * u**4) / T * d
        a = (60 * u - 180 * u**2 + 120 * u**3) / T**2 * d
        traj = Trajectory(
            list(cpu_planner.joint_names), p, v, a, 0.02, frame_id="arm_mount"
        )
        return PlanResult(True, traj, 0.0, None, "OK", True, p[-1].copy())

    def tool_poses(qs):
        qs = np.asarray(qs, dtype=float)
        cx, sx = np.cos(qs[:, 3] / 2), np.sin(qs[:, 3] / 2)  # tilt about x
        cz, sz = np.cos(qs[:, 4] / 2), np.sin(qs[:, 4] / 2)  # then about vertical
        return qs[:, :3], np.stack([cz * cx, cz * sx, sz * sx, sz * cx], axis=1)

    monkeypatch.setattr(cpu_planner, "plan_to_joints", lambda q, start: leg(q, start))
    monkeypatch.setattr(cpu_planner, "_tool_poses", tool_poses)
    cpu_planner.feasible = lambda qs: np.ones(len(qs), dtype=bool)
    monkeypatch.setattr(cpu_planner, "_feasible", lambda qs: cpu_planner.feasible(qs))
    return cpu_planner


def _legs(planner, *points):
    """plan_to_joints legs through `points`, each from the one before's end."""
    legs, q = [], np.asarray(points[0], dtype=float)
    for b in points[1:]:
        legs.append(planner.plan_to_joints(np.asarray(b, float), q).joint_traj)
        q = legs[-1].positions[-1]
    return legs


def _q(*first):
    return np.r_[first, np.zeros(7 - len(first))]


CORNER = (_q(), _q(0.4), _q(0.4, 0.4))
SQUARE = (_q(), _q(0.4), _q(0.4, 0.4), _q(0.0, 0.4))


def test_blend_legs_stops_where_the_radius_is_zero(blend_planner):
    res = blend_planner.blend_legs(_legs(blend_planner, *SQUARE), [0.0, 0.0])
    assert res.success, res.error
    assert res.corners_m == [0.0, 0.0]
    traj = res.joint_traj
    speed = np.linalg.norm(traj.velocities, axis=1)
    rest = traj.positions[1:-1][speed[1:-1] == 0.0]
    np.testing.assert_allclose(rest, [SQUARE[1], SQUARE[2]])  # at rest on both


def test_blend_legs_cuts_the_corners_within_the_radius(blend_planner):
    legs = _legs(blend_planner, *SQUARE)
    stopped = blend_planner.blend_legs(legs, [0.0, 0.0]).joint_traj
    res = blend_planner.blend_legs(legs, [0.05, 0.05])
    assert res.success, res.error
    traj = res.joint_traj
    assert np.linalg.norm(traj.velocities, axis=1)[1:-1].min() > 0.0
    assert all(0.0 < c <= 0.05 for c in res.corners_m)
    for k, c in enumerate(res.corners_m):  # corner_m: the closest pass by the waypoint
        miss = np.linalg.norm(traj.positions[:, :3] - SQUARE[k + 1][:3], axis=1).min()
        assert miss == pytest.approx(c)
    assert traj.duration < stopped.duration - 0.2
    np.testing.assert_allclose(traj.positions[-1], SQUARE[-1], atol=1e-12)
    assert traj.frame_id == "arm_mount"


def test_a_smaller_radius_cuts_less(blend_planner):
    legs = _legs(blend_planner, *CORNER)
    wide = blend_planner.blend_legs(legs, [0.2]).corners_m[0]
    tight = blend_planner.blend_legs(legs, [0.01]).corners_m[0]
    assert 0.0 < tight <= 0.01 < wide


def test_the_tool_stays_within_the_radius_of_the_planned_states(blend_planner):
    """Nearly straight on, at different speeds, the waypoint is passed close
    but the blend runs ahead of the planned path: every state stays within
    the radius of the nearer of the two planned states it was summed from."""
    from rammp_curobo import blend

    legs = _legs(
        blend_planner, _q(), _q(-0.331, 0.157, -0.138), _q(-0.591, 0.297, -0.247)
    )
    res = blend_planner.blend_legs(legs, [0.01], [1.0, 0.5])
    assert res.success, res.error
    p = res.joint_traj.positions[:, :3]
    a = legs[0].positions[:, :3]
    b = blend.resample(legs[1].positions, legs[1].velocities, 0.04, 0.02)[0][:, :3]
    shift = len(p) - len(b)  # leg 2 starts here on the blend's grid
    gaps = [
        min(
            np.linalg.norm(p[i] - a[min(i, len(a) - 1)]),
            np.linalg.norm(p[i] - b[max(i - shift, 0)]),
        )
        for i in range(len(p))
    ]
    assert max(gaps) <= 0.01 + 1e-9
    stopped = blend_planner.blend_legs(legs, [0.0], [1.0, 0.5]).joint_traj.duration
    assert res.joint_traj.duration < stopped  # blended, within that


def test_a_reversal_still_passes_its_waypoint(blend_planner):
    """Out to 0.4 and back: overlapping the legs entirely would never go out
    at all. The radius bounds the distance from the waypoint, not only from
    the path (which the arm, standing still, never leaves)."""
    legs = _legs(blend_planner, _q(), _q(0.4), _q())
    res = blend_planner.blend_legs(legs, [0.05])
    assert res.success, res.error
    assert np.abs(res.joint_traj.positions[:, 0]).max() >= 0.4 - 0.05
    assert res.corners_m[0] <= 0.05


def test_velocities_that_add_stay_in_the_limits(blend_planner):
    legs = _legs(blend_planner, _q(), _q(0.4), _q(0.8))  # both at the limit on joint 1
    res = blend_planner.blend_legs(legs, [1.0])
    assert res.success, res.error
    assert np.abs(res.joint_traj.velocities).max() <= 1.2 + 1e-9
    stopped = blend_planner.blend_legs(legs, [0.0]).joint_traj.duration
    assert res.joint_traj.duration < stopped  # it did blend (on a line: no corner)


@pytest.mark.parametrize("accelerations", [True, False])
def test_accelerations_that_add_stay_in_the_limits(blend_planner, accelerations):
    """A reversal adds the legs' braking; without accelerations on a leg they
    are derived from its velocities, not skipped."""
    from dataclasses import replace

    blend_planner.lim["velocity"] = np.full(7, 100.0)  # only acceleration binds
    legs = _legs(blend_planner, _q(), _q(0.4), _q())
    if not accelerations:
        legs = [replace(leg, accelerations=None) for leg in legs]
    # 0.2: a full overlap (which cancels, the arm never leaving) is out of
    # the radius, so the partial overlaps, where the braking adds, decide
    res = blend_planner.blend_legs(legs, [0.2])
    assert res.success, res.error
    traj = res.joint_traj
    measured = np.gradient(traj.velocities, traj.dt, axis=0)  # not the reported field
    assert np.abs(measured).max() <= 25.0 * 1.05
    assert res.corners_m[0] > 0.0


def test_a_held_leg_keeps_its_hold_through_the_overlap(blend_planner):
    """Leg 1 holds level (joint 4, the tilt, stays 0) while joint 1 moves;
    leg 2 is free and tilts. Unheld, the tilt starts while leg 1 still runs."""
    from rammp_curobo import PoseConstraint

    legs = _legs(blend_planner, _q(), _q(0.4), _q(0.4, 0, 0, 0.5))
    tol = blend_planner.constraint_tolerance_rad

    def tilt_while_leg_1_runs(res):
        q = res.joint_traj.positions
        return np.abs(q[q[:, 0] < 0.4 - 1e-9, 3]).max()

    free = blend_planner.blend_legs(legs, [0.2])
    held = blend_planner.blend_legs(legs, [0.2], holds=[PoseConstraint(hold=1), None])
    assert free.success and held.success, held.error
    assert tilt_while_leg_1_runs(free) > tol  # not vacuous
    assert tilt_while_leg_1_runs(held) <= tol
    stopped = blend_planner.blend_legs(legs, [0.0]).joint_traj.duration
    assert held.joint_traj.duration < stopped  # still blended, less


def test_blend_legs_does_not_cut_a_corner_that_collides(blend_planner):
    # infeasible wherever joint 2 moves before joint 1 has arrived: inside
    # the corner, which the planned legs never enter
    legs = _legs(blend_planner, *CORNER)
    free = blend_planner.blend_legs(legs, [0.05], [1.0, 1.0])
    blend_planner.feasible = lambda qs: ~(
        (np.asarray(qs)[:, 0] < 0.4 - 1e-9) & (np.asarray(qs)[:, 1] > 1e-9)
    )
    res = blend_planner.blend_legs(legs, [0.05])
    assert res.success, res.error
    assert res.corners_m[0] < 1e-3 < free.corners_m[0]  # one shared step at most


def test_blend_legs_blends_a_slow_leg_into_a_fast_one(blend_planner):
    legs = _legs(blend_planner, *CORNER)
    stopped = blend_planner.blend_legs(legs, [0.0], [0.25, 1.0])
    res = blend_planner.blend_legs(legs, [0.05], [0.25, 1.0])
    assert stopped.success and res.success, res.error
    traj = res.joint_traj
    assert traj.dt == pytest.approx(0.02)
    # the first leg (joint 1) at a quarter of its speed, the second (joint 2) at full
    assert np.abs(traj.velocities[:, 0]).max() == pytest.approx(
        0.25 * np.abs(legs[0].velocities[:, 0]).max(), rel=0.02
    )
    assert np.abs(traj.velocities[:, 1]).max() == pytest.approx(
        np.abs(legs[1].velocities[:, 1]).max(), rel=0.02
    )
    assert 0.0 < res.corners_m[0] <= 0.05
    assert traj.duration < stopped.joint_traj.duration
    np.testing.assert_allclose(traj.positions[-1], CORNER[-1], atol=1e-12)


def test_blend_legs_refuses_what_it_cannot_blend(blend_planner):
    from dataclasses import replace

    a, b = _legs(blend_planner, *CORNER)
    far = _legs(blend_planner, np.full(7, 0.1), _q())[0]
    moving = replace(a, positions=a.positions[:-5], velocities=a.velocities[:-5])
    nan = replace(
        b,
        positions=np.where(
            np.arange(len(b.positions))[:, None] == 3, np.nan, b.positions
        ),
    )
    renamed = replace(b, joint_names=list(reversed(b.joint_names)))
    point = replace(
        b, positions=b.positions[:1], velocities=b.velocities[:1], accelerations=None
    )
    for legs, radii, scales, why in [
        ([], [], None, "no legs"),
        ([a, b], [], None, "2 legs need 1 blend radii"),
        ([a, b], [0.0], [1.0, 0.0], "leg 2/2: speed scale 0 is outside (0, 1]"),
        ([a, far], [0.0], None, "leg 2/2: starts 0.3000 rad from where leg 1 ends"),
        ([moving, b], [0.0], None, "leg 1/2: starts or ends moving"),
        ([a, nan], [0.0], None, "leg 2/2: non-finite"),
        ([a, renamed], [0.0], None, "leg 2/2: velocities for joints"),
        ([a, b], [float("nan")], None, "junction 1: blend radius nan"),
        ([a, b], [-0.1], None, "must be finite and >= 0"),
        ([a, b], [float("inf")], None, "must be finite and >= 0"),
        ([a, point], [0.05], None, "leg 2/2: 1 point(s), a leg needs at least 2"),
    ]:
        res = blend_planner.blend_legs(legs, radii, scales)
        assert res.status == "BAD_LEG" and why in res.error, res.error
    res = blend_planner.blend_legs(
        [a, b], [0.05], hold_orientations=[(0, 0, 0, 2), None]
    )
    assert res.status == "BAD_LEG" and "not a unit quaternion" in res.error


def _while(res, moving, joint):
    """|joint| (rad) over the states where `moving` (a predicate on the
    states) holds: a held leg's span."""
    q = res.joint_traj.positions
    return np.abs(q[moving(q), joint]).max()


def test_a_held_second_leg_keeps_its_hold_through_the_overlap(blend_planner):
    """Leg 1 is free and levels the tool (joint 4 back to 0); leg 2 holds
    level while joint 1 moves. Unheld, leg 2 starts before the tool is level."""
    from rammp_curobo import PoseConstraint

    legs = _legs(blend_planner, _q(0, 0, 0, 0.5), _q(), _q(0.4))
    tol = blend_planner.constraint_tolerance_rad
    leg_2 = lambda q: q[:, 0] > 1e-9  # noqa: E731
    free = blend_planner.blend_legs(legs, [0.2])
    held = blend_planner.blend_legs(legs, [0.2], holds=[None, PoseConstraint(hold=1)])
    assert free.success and held.success, held.error
    assert _while(free, leg_2, 3) > tol  # not vacuous
    assert _while(held, leg_2, 3) <= tol
    stopped = blend_planner.blend_legs(legs, [0.0]).joint_traj.duration
    assert held.joint_traj.duration < stopped


def test_level_frees_the_spin_about_vertical_and_fixed_does_not(blend_planner):
    """Leg 1 is held while joint 1 moves; leg 2 turns the tool about the
    vertical (joint 5). LEVEL leaves that turn free in the overlap; FIXED
    keeps it out of leg 1's span."""
    from rammp_curobo import PoseConstraint

    legs = _legs(blend_planner, _q(), _q(0.4), _q(0.4, 0, 0, 0, 0.5))
    tol = blend_planner.constraint_tolerance_rad
    leg_1 = lambda q: q[:, 0] < 0.4 - 1e-9  # noqa: E731
    free = blend_planner.blend_legs(legs, [0.2])
    level = blend_planner.blend_legs(legs, [0.2], holds=[PoseConstraint(hold=1), None])
    fixed = blend_planner.blend_legs(legs, [0.2], holds=[PoseConstraint(hold=2), None])
    assert free.success and level.success and fixed.success, fixed.error
    assert _while(level, leg_1, 4) > tol  # the spin overlaps leg 1 under LEVEL
    assert level.joint_traj.duration == pytest.approx(free.joint_traj.duration)
    assert _while(fixed, leg_1, 4) <= tol
    assert fixed.joint_traj.duration > level.joint_traj.duration


def test_a_hold_is_kept_against_the_goal_not_only_the_legs_end(blend_planner):
    """Leg 1 holds level but ends 1.5 deg tilted (within tolerance of its
    level goal); leg 2 tilts further. Against the leg's end the overlap may
    add a whole tolerance on top; against its goal it stays within one."""
    from rammp_curobo import PoseConstraint

    tilt = np.radians(1.5)
    legs = _legs(
        blend_planner, _q(0, 0, 0, tilt), _q(0.4, 0, 0, tilt), _q(0.4, 0, 0, 0.3)
    )
    tol = blend_planner.constraint_tolerance_rad
    leg_1 = lambda q: q[:, 0] < 0.4 - 1e-9  # noqa: E731
    hold = [PoseConstraint(hold=1), None]
    from_end = blend_planner.blend_legs(legs, [0.2], holds=hold)
    from_goal = blend_planner.blend_legs(
        legs, [0.2], holds=hold, hold_orientations=[(0, 0, 0, 1), None]
    )
    assert from_end.success and from_goal.success, from_goal.error
    assert _while(from_end, leg_1, 3) > tol  # not vacuous: up to tilt + tol
    assert _while(from_goal, leg_1, 3) <= tol


def test_the_final_check_still_refuses(blend_planner):
    """A world every overlap window passes but the whole trajectory does not
    (here the windows are shorter than any whole trajectory)."""
    legs = _legs(blend_planner, *CORNER)
    longest = max(len(leg.positions) for leg in legs)
    blend_planner.feasible = lambda qs: np.full(len(qs), len(qs) <= longest)
    res = blend_planner.blend_legs(legs, [0.05])
    assert res.status == "LIBRARY_VALIDATION_FAILED"
    assert "constraint re-check flagged" in res.error


def test_without_a_goal_orientation_a_hold_is_kept_against_the_legs_end(blend_planner):
    """A level leg that drifts 1.5 deg on its way (its goal, and so its end,
    at 1.5 deg), then a leg that tilts further: with no hold orientation given
    the reference is the leg's end, so the overlap may tilt up to a tolerance
    past it, not past the leg's start."""
    from rammp_curobo import PoseConstraint

    tilt = np.radians(1.5)
    legs = _legs(blend_planner, _q(), _q(0.4, 0, 0, tilt), _q(0.4, 0, 0, 0.3))
    tol = blend_planner.constraint_tolerance_rad
    res = blend_planner.blend_legs(legs, [0.2], holds=[PoseConstraint(hold=1), None])
    assert res.success, res.error
    worst = _while(res, lambda q: q[:, 0] < 0.4 - 1e-9, 3)
    assert tol < worst <= tilt + tol + 1e-9
