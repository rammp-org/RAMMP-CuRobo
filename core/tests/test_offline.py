"""Hardware-free, GPU-free tests: configs, scene parsing, retiming, geometry.

These run anywhere (CI, a laptop) — no torch, no cuRobo, no ROS.
"""

import math

import numpy as np
import pytest
import yaml

from rammp_curobo.config import PACKAGED_CONFIG_DIR, load_planner_config, resolve_config
from rammp_curobo.constraints import (
    HOLD_FIXED,
    HOLD_LEVEL,
    HOLD_NONE,
    PoseConstraint,
    ViaPoint,
)
from rammp_curobo.geometry import (
    euler_deg_to_quat_xyzw,
    spin_about_tool,
    tip_to_tool,
    tool_axis,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from rammp_curobo.retime import scale_trajectory
from rammp_curobo.scene import load_scene, scene_from_obstacles
from rammp_curobo.types import Trajectory
from rammp_curobo.validate import start_state_matches, validate_trajectory
from rammp_curobo.world import world_cuboids


def test_packaged_configs_exist():
    for name in (
        "gen3.yaml",
        "robot_gen3_2f85.yaml",
        "world_sim_kitchen.yaml",
        "world_real_bench.yaml",
    ):
        assert (PACKAGED_CONFIG_DIR / name).is_file(), name


def test_planner_config_loads_and_merges():
    cfg, cfg_dir = load_planner_config("gen3.yaml")
    assert cfg["robot"] == "robot_gen3_2f85.yaml"
    assert cfg["joint_names"] == ["joint_%d" % i for i in range(1, 8)]
    assert cfg["planner"]["enable_graph"] is False
    assert cfg["planner"]["joint_space_method"] == "auto"
    assert cfg["execution"]["speed_scale"] == 0.25
    assert resolve_config(cfg["world"], relative_to=cfg_dir).is_file()


def test_gen3_yaml_matches_planner_defaults():
    # gen3.yaml deliberately restates every default so the file is the
    # single human-readable reference (its own header says so). This test
    # keeps the two copies honest: any drift between config.PLANNER_DEFAULTS
    # and the YAML is a bug in whichever was edited alone.
    from rammp_curobo.config import PLANNER_DEFAULTS

    cfg, _ = load_planner_config("gen3.yaml")
    for section in ("planner", "tool", "execution"):
        assert cfg[section] == PLANNER_DEFAULTS[section], section
    assert cfg["joint_names"] == PLANNER_DEFAULTS["joint_names"]


def test_baked_robot_config_carries_the_patches():
    path = resolve_config("robot_gen3_2f85.yaml")
    kin = yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]
    spheres = kin["collision_spheres"]
    assert isinstance(spheres, dict), "spheres must be inlined, not a ref"
    for link in ("left_inner_finger_pad", "right_inner_finger_pad"):
        assert all(s["radius"] >= 0.02 for s in spheres[link])
    assert len(spheres["robotiq_arg2f_base_link"]) == 19  # stock 1 + shell 18
    assert len(spheres["base_link"]) == 2  # audit-tuned set
    retract = kin["cspace"]["retract_config"]
    assert retract == pytest.approx(
        [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571], abs=1e-6
    )
    assert kin["ee_link"] == "tool_frame"


def test_sim_kitchen_scene_parses():
    scene = load_scene(resolve_config("world_sim_kitchen.yaml"))
    assert scene.base_frame == "base_link"
    names = {o.name for o in scene.obstacles}
    assert {"table", "pedestal", "back_wall"} <= names
    assert len(scene.obstacles) == 11
    assert len(scene.objects) >= 15
    bottle = next(o for o in scene.objects if o.name == "bottle")
    assert bottle.bounding_dims() == pytest.approx([0.066, 0.066, 0.22])


def test_world_cuboids_padding_and_ignore():
    scene = load_scene(resolve_config("world_sim_kitchen.yaml"))
    boxes = world_cuboids(
        scene, padding=0.02, ignore={"bottle"}, no_pad_names={"pedestal"}
    )
    assert "obj_bottle" not in boxes
    assert boxes["pedestal"]["dims"] == pytest.approx([0.14, 0.14, 0.04])
    table = next(o for o in scene.obstacles if o.name == "table")
    assert boxes["table"]["dims"] == pytest.approx([d + 0.04 for d in table.dims])
    # the sim kitchen must leave headroom under the 60-box cache for
    # update_world additions (40 keeps a wide margin)
    assert len(boxes) <= 40


def test_scene_from_obstacle_dicts():
    scene = scene_from_obstacles(
        [
            {"name": "box", "position": [0.5, 0.0, 0.0], "dims": [0.1, 0.2, 0.3]},
            {
                "name": "can",
                "type": "cylinder",
                "position": [0.3, 0.1, 0.0],
                "radius": 0.04,
                "height": 0.12,
            },
        ]
    )
    boxes = world_cuboids(scene, padding=0.0)
    assert boxes["obj_box"]["dims"] == pytest.approx([0.1, 0.2, 0.3])
    assert boxes["obj_can"]["dims"] == pytest.approx([0.08, 0.08, 0.12])


def _traj(n=50, dof=7, dt=0.02, vmax=1.0):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pos = 0.5 * np.sin(t * math.pi) * np.ones((1, dof))
    vel = np.gradient(pos, dt, axis=0).clip(-vmax, vmax)
    return Trajectory(
        joint_names=["joint_%d" % i for i in range(1, dof + 1)],
        positions=pos,
        velocities=vel,
        accelerations=None,
        dt=dt,
    )


def test_retime_dilates_exactly():
    traj = _traj()
    slow = scale_trajectory(traj, 0.25)
    assert slow.dt == pytest.approx(traj.dt / 0.25)
    assert slow.duration == pytest.approx(traj.duration * 4)
    np.testing.assert_allclose(slow.velocities, traj.velocities * 0.25)
    np.testing.assert_array_equal(slow.positions, traj.positions)
    assert slow.speed_scale == 0.25
    with pytest.raises(ValueError):
        scale_trajectory(traj, 1.5)
    with pytest.raises(ValueError):
        scale_trajectory(traj, 0.0)


def test_validate_catches_limit_and_discontinuity():
    limits_pos = np.array([[-1.0] * 7, [1.0] * 7])
    limits_vel = np.array([2.0] * 7)
    ok = validate_trajectory(_traj(), limits_pos, limits_vel)
    assert ok == []
    bad = _traj()
    bad.positions = bad.positions.copy()
    bad.positions[10, 3] = 5.0  # out of limits AND a discontinuity
    problems = validate_trajectory(bad, limits_pos, limits_vel)
    assert any("position limits" in p for p in problems)
    assert any("discontinuity" in p for p in problems)


def test_start_state_matches():
    traj = _traj()
    ok, err = start_state_matches(traj, traj.positions[0], tol_rad=0.05)
    assert ok and err == pytest.approx(0.0)
    ok, err = start_state_matches(traj, traj.positions[0] + 0.2)
    assert not ok


def test_quaternion_round_trip_and_tool_math():
    q = euler_deg_to_quat_xyzw([180, 0, 0])
    assert q == pytest.approx([1, 0, 0, 0], abs=1e-9)  # xyzw
    wxyz = xyzw_to_wxyz(q)
    assert wxyz_to_xyzw(wxyz) == pytest.approx(list(q))
    # tool-down: tool z points along -z in base frame
    assert tool_axis(wxyz) == pytest.approx([0, 0, -1], abs=1e-9)
    # pulling a tool-down fingertip goal back by 21 mm RAISES the command
    assert tip_to_tool([0.4, 0.0, 0.10], wxyz, 0.021) == pytest.approx(
        [0.4, 0.0, 0.121]
    )
    # spinning about tool z never moves the tool axis
    spun = spin_about_tool(wxyz, 90.0)
    assert tool_axis(spun) == pytest.approx([0, 0, -1], abs=1e-9)


def test_yaw_about_world_z_keeps_attitude_level():
    from rammp_curobo.geometry import yaw_about_world_z

    # the Gen3 home tool attitude (FK-verified): tool z level along +x
    home = [0.5, 0.5, 0.5, 0.5]  # xyzw
    assert yaw_about_world_z(home, 0.0) == pytest.approx(home)
    # steering about WORLD z swings the heading but the wrist stays level:
    # tool z ends at [cos b, sin b, 0] for every bearing b
    for deg in (-45.0, -20.0, 10.0, 45.0):
        b = math.radians(deg)
        q = yaw_about_world_z(home, b)
        assert sum(v * v for v in q) == pytest.approx(1.0)
        assert tool_axis(xyzw_to_wxyz(q)) == pytest.approx(
            [math.cos(b), math.sin(b), 0.0], abs=1e-9
        )


def test_real_arm_config_loads():
    cfg, cfg_dir = load_planner_config("gen3_real.yaml")
    assert cfg["world"] == "world_real_bench.yaml"
    assert cfg["robot"] == "robot_gen3_2f85.yaml"
    # inherits the same planner defaults gen3.yaml declares explicitly
    assert cfg["planner"]["enable_graph"] is False
    assert cfg["execution"]["speed_scale"] == 0.25
    assert resolve_config(cfg["world"], relative_to=cfg_dir).is_file()


def test_wrap_aware_start_match():
    # joint_3 at home sits exactly on the +/-pi boundary: the driver can
    # report -3.142 for a commanded +3.142 (observed on the real Gen3 —
    # a perfect first move read as "settled 6.283 rad away").
    from rammp_curobo.geometry import ang_diff

    assert abs(ang_diff(3.142, -3.141)) < 0.01
    assert abs(ang_diff(0.15, 0.0)) == pytest.approx(0.15)
    traj = _traj()
    traj.positions = traj.positions.copy()
    traj.positions[0, 2] = 3.1416
    current = traj.positions[0].copy()
    current[2] = -3.1416  # same physical angle, wrapped report
    ok, err = start_state_matches(traj, current, tol_rad=0.05)
    assert ok and err < 0.01


def test_pose_constraint_inactive_by_default():
    c = PoseConstraint()
    assert not c.is_active()
    assert c.hold_vec_weight() == [0.0] * 6


def test_hold_vec_weight_is_orientation_first():
    """cuRobo's vec_weight is [rx, ry, rz, x, y, z] — orientation FIRST.
    Getting this backwards silently constrains position instead of
    orientation, which still plans, so no test but this one would catch it."""
    assert PoseConstraint(hold=HOLD_LEVEL).hold_vec_weight() == [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ]
    assert PoseConstraint(hold=HOLD_FIXED).hold_vec_weight() == [
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    ]


def test_level_frees_exactly_yaw_and_fixed_frees_nothing():
    """The single difference between the two modes, stated as a test: LEVEL
    leaves rz free so the tool may spin about the vertical, FIXED does not."""
    assert PoseConstraint(hold=HOLD_LEVEL).hold_vec_weight()[2] == 0.0
    assert PoseConstraint(hold=HOLD_FIXED).hold_vec_weight()[2] == 1.0


def test_no_mode_ever_holds_position():
    """Position holds are gone by design — they could only express "travel
    along one base axis", and a straight line in an arbitrary direction is
    not expressible through a DIAGONAL hold_vec_weight at all."""
    for mode in (HOLD_NONE, HOLD_LEVEL, HOLD_FIXED):
        assert PoseConstraint(hold=mode).hold_vec_weight()[3:] == [0.0, 0.0, 0.0]


def test_pose_constraint_is_active_for_both_hold_modes():
    assert PoseConstraint(hold=HOLD_LEVEL).is_active()
    assert PoseConstraint(hold=HOLD_FIXED).is_active()
    assert not PoseConstraint(hold=HOLD_NONE).is_active()


def test_pose_constraint_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="HOLD_NONE"):
        PoseConstraint(hold=7).validate()


def test_via_point_inactive_at_zero_offset():
    assert not ViaPoint().is_active()
    assert not ViaPoint(offset_m=0.0, linear_axis=2).is_active()
    assert ViaPoint(offset_m=0.1).is_active()


def test_via_point_rejects_out_of_range_fraction():
    for bad in (1.5, -0.2, 0.0, 1.0):
        with pytest.raises(ValueError, match="tstep_fraction"):
            ViaPoint(offset_m=0.1, tstep_fraction=bad).validate()


def test_via_point_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        ViaPoint(offset_m=float("nan")).validate()
    with pytest.raises(ValueError, match="finite"):
        ViaPoint(offset_m=0.1, tstep_fraction=float("inf")).validate()


def test_via_point_rejects_bad_axis():
    with pytest.raises(ValueError, match="linear_axis"):
        ViaPoint(offset_m=0.1, linear_axis=7).validate()


def test_inactive_via_point_validates_clean():
    """An all-defaults ViaPoint arrives from every caller that wants
    nothing; it must never raise."""
    ViaPoint().validate()
    PoseConstraint().validate()


def test_pose_cost_kwargs_none_when_nothing_requested():
    from rammp_curobo.planner import CuRoboPlanner

    assert CuRoboPlanner._pose_cost_kwargs(None, None) is None
    assert CuRoboPlanner._pose_cost_kwargs(PoseConstraint(), ViaPoint()) is None


def test_pose_cost_kwargs_for_a_held_constraint():
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(PoseConstraint(hold=HOLD_LEVEL), None)
    assert kw["hold_partial_pose"] is True
    assert kw["hold_vec_weight"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # Always the base frame now: "level" means level with the WORLD, and
    # only the base frame can express that.
    assert kw["project_to_goal_frame"] is False
    assert "offset_position" not in kw


def test_via_point_forces_a_full_hold_except_the_approach_axis():
    """cuRobo's grasp-approach metric (create_grasp_approach_metric, read
    from real cuRobo v0.7.8 on the Jetson) holds ALL FIVE non-approach pose
    components at the goal's value — not just the axes a PoseConstraint
    happened to ask for. A caller who only locked roll/pitch still gets
    yaw/x/y held too, because that is the only shape cuRobo's via point
    supports; there is no way to combine "hold roll/pitch only" with an
    approach the way an earlier, never-run version of this code assumed."""
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(
        PoseConstraint(hold=HOLD_LEVEL),
        ViaPoint(offset_m=0.10, linear_axis=2, tstep_fraction=0.8),
    )
    # everything held except z (3 + linear_axis=2), regardless of which
    # axes the constraint explicitly asked to lock
    assert kw["hold_vec_weight"] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert kw["offset_position"] == 0.10
    assert kw["linear_axis"] == 2
    assert kw["offset_tstep_fraction"] == 0.8


def test_via_point_ignores_the_constraint_when_via_is_inactive():
    """An inactive ViaPoint must not perturb a plain PoseConstraint plan —
    the five-axis hold is a via-point-only behaviour."""
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(
        PoseConstraint(hold=HOLD_LEVEL), ViaPoint()
    )
    assert kw["hold_vec_weight"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert "offset_position" not in kw


def test_a_constraint_can_no_longer_contradict_a_via():
    """There used to be a contradiction to catch — holding z while asking to
    approach along z. With position holds removed a constraint only ever
    holds rotations and a via only ever frees a LINEAR axis, so the two
    cannot collide. Kept as a test so the removal of that guard is
    deliberate and visible rather than looking like an oversight."""
    from rammp_curobo.planner import CuRoboPlanner

    for mode in (HOLD_LEVEL, HOLD_FIXED):
        for axis in (0, 1, 2):
            kw = CuRoboPlanner._pose_cost_kwargs(
                PoseConstraint(hold=mode), ViaPoint(offset_m=0.10, linear_axis=axis)
            )
            assert kw["hold_vec_weight"][3 + axis] == 0.0


def test_rotvec_between_is_zero_for_equal_quaternions():
    from rammp_curobo.geometry import rotvec_between

    q = euler_deg_to_quat_xyzw([10.0, -20.0, 35.0])
    assert max(abs(v) for v in rotvec_between(q, q)) < 1e-9


def test_rotvec_between_recovers_a_single_axis_rotation():
    from rammp_curobo.geometry import rotvec_between

    a = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    b = euler_deg_to_quat_xyzw([12.0, 0.0, 0.0])
    v = rotvec_between(a, b)
    assert abs(v[0] - math.radians(12.0)) < 1e-6
    assert abs(v[1]) < 1e-6 and abs(v[2]) < 1e-6


class _StubFk:
    """Minimal stand-in for CuRoboPlanner: only fk() is exercised.

    The constructor takes the orientation in xyzw; fk() converts to wxyz
    on request, mirroring the real CuRoboPlanner.fk. This is deliberate,
    not incidental: a stub that ignored quat_order and always handed back
    the same fixed quaternion could not reproduce the bug where fk was
    asked for wxyz and the caller silently got mismatched conventions.
    """

    def __init__(self, pos, quat_xyzw, tolerance_deg=2.0):
        self._pos, self._quat_xyzw = pos, quat_xyzw
        # constraint_satisfied_at_start is called unbound with this stub as
        # `self`, so every attribute it reads has to exist here. Forgetting
        # this one turns the whole pre-check suite into AttributeError.
        self.constraint_tolerance_rad = math.radians(tolerance_deg)

    def fk(self, q, quat_order="xyzw"):
        if quat_order == "wxyz":
            return self._pos, xyzw_to_wxyz(self._quat_xyzw)
        return self._pos, self._quat_xyzw


def _check(stub, goal_pos, goal_quat, constraint):
    from rammp_curobo.planner import CuRoboPlanner

    return CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, goal_pos, goal_quat, constraint
    )


def test_pre_check_passes_when_held_axes_already_match():
    q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], q)
    ok, why = _check(stub, [0.6, 0.2, 0.4], q, PoseConstraint(hold=HOLD_LEVEL))
    assert ok and why is None


def test_pre_check_rejects_a_tilted_start():
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([12.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_LEVEL))
    assert not ok
    assert "tilt" in why
    assert "2.00" in why  # the tolerance is named, in degrees, so the caller can act
    assert "12" in why  # and so is the measured deviation


def test_pre_check_passes_a_tilted_start_when_the_goal_is_equally_tilted():
    """A held axis is held at the GOAL'S value, not at zero. A tool at 45
    degrees planning to a 45-degree goal is perfectly legal and stays at 45
    the whole way — 'held' means unchanged, not level."""
    tilted = euler_deg_to_quat_xyzw([45.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], tilted)
    ok, why = _check(
        stub, [0.6, 0.2, 0.4], tilted, PoseConstraint(hold=HOLD_LEVEL)
    )
    assert ok and why is None


def test_pre_check_flags_a_marginal_start():
    """1.8 deg passes the 2.0 deg tolerance but only just. Silence here turns
    into an unreproducible planning failure later.

    The fixture is 1.8 and not the old 2.6 because the gate moved: it used to
    be a hard-coded 0.05 rad (2.86 deg), and is now the configurable
    constraint_tolerance_deg, defaulting to 2.0. A test fixture pinned to the
    old number would have silently started asserting the refusal branch."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([1.8, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_LEVEL))
    assert ok
    assert why is not None and "marginal" in why


def test_pre_check_passes_fixed_when_orientations_are_identical():
    """FIXED holds all three rotations, so an identical goal orientation is
    exactly what it wants. Position is never held, so the goal being 0.3 m
    away is irrelevant — which is the whole point of dropping position
    holds."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(
        stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_FIXED)
    )
    assert ok and why is None  # identical orientations: nothing to disagree about


def test_pre_check_agrees_across_quat_order_for_the_goal():
    """quat_order describes the caller's goal argument only — the start
    pose is always read from fk in xyzw. Passing the goal as wxyz must
    reach the same verdict as the xyzw-equivalent call, not silently mix
    conventions inside rotvec_between."""
    from rammp_curobo.planner import CuRoboPlanner

    start_quat_xyzw = euler_deg_to_quat_xyzw([12.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], start_quat_xyzw)
    goal_quat_xyzw = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    constraint = PoseConstraint(hold=HOLD_LEVEL)

    ok_xyzw, why_xyzw = CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, [0.6, 0.0, 0.4], goal_quat_xyzw, constraint, quat_order="xyzw"
    )
    ok_wxyz, why_wxyz = CuRoboPlanner.constraint_satisfied_at_start(
        stub,
        [0.0] * 7,
        [0.6, 0.0, 0.4],
        xyzw_to_wxyz(goal_quat_xyzw),
        constraint,
        quat_order="wxyz",
    )
    assert ok_xyzw == ok_wxyz
    assert why_xyzw == why_wxyz
    assert "tilt" in why_xyzw  # same measure reported on both paths


def test_pre_check_stays_inert_when_nothing_is_held():
    """HOLD_NONE must be a no-op even from a tilted start: an inactive
    constraint has nothing to disagree about."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([12.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_NONE))
    assert ok and why is None


def test_level_tolerates_a_pure_yaw_difference_but_fixed_does_not():
    """The one behavioural difference between the modes, from the caller's
    side. A start and goal differing ONLY in yaw is fine under LEVEL — the
    spin about vertical is exactly the freedom it leaves — and must be
    refused under FIXED, which holds yaw too."""
    start_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 40.0])
    stub = _StubFk([0.3, 0.0, 0.4], start_q)

    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_LEVEL))
    assert ok and why is None, "LEVEL must not care about yaw: %s" % why

    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_FIXED))
    assert not ok
    assert "orientation" in why and "40" in why


def test_pre_check_ignores_an_active_via_point():
    """A via must NOT be folded into the start check, and this is a
    regression test for a real false refusal.

    The old behaviour treated an active via as implying a five-axis hold AT
    THE START, and refused accordingly. Measured against real cuRobo: a via's
    hold engages at tstep_fraction, not at the start, so the same goal plans
    perfectly well. The pre-check was refusing plans that work — through the
    CheckPoseLock service, to callers who had no way to tell it was wrong.

    With no constraint and only a via, there is nothing for this check to
    judge, so it must pass regardless of how tilted the start is."""
    from rammp_curobo.planner import CuRoboPlanner

    tilted = euler_deg_to_quat_xyzw([12.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], tilted)
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, [0.3, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_NONE)
    )
    assert ok and why is None


def test_pre_check_rejects_a_zero_quaternion_goal():
    """geometry_msgs/Pose defaults orientation to (0,0,0,0). A zero
    quaternion must be refused, not silently accepted as 'satisfied' —
    plan_to_pose itself rejects this same shape as BAD_GOAL."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]))
    ok, why = _check(
        stub, [0.6, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0], PoseConstraint(hold=HOLD_LEVEL)
    )
    assert not ok
    assert why is not None and "zero-length" in why


def test_pre_check_rejects_a_nan_quaternion_goal():
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]))
    ok, why = _check(
        stub,
        [0.6, 0.0, 0.4],
        [float("nan"), 0.0, 0.0, 1.0],
        PoseConstraint(hold=HOLD_LEVEL),
    )
    assert not ok
    assert why is not None and "non-finite" in why


def test_pre_check_rejects_a_nan_goal_position():
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(
        stub, [0.6, float("nan"), 0.4], goal_q, PoseConstraint(hold=HOLD_LEVEL)
    )
    assert not ok
    assert why is not None and "non-finite" in why
