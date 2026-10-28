"""Hardware-free, GPU-free tests: configs, scene parsing, retiming, geometry.

These run anywhere (CI, a laptop) — no torch, no cuRobo, no ROS.
"""

import importlib.util
import math
import os
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

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
from rammp_curobo.robot_config import load_robot_config
from rammp_curobo.scene import load_scene, scene_from_obstacles
from rammp_curobo.types import Trajectory
from rammp_curobo.urdf_model import (
    ARM_JOINTS,
    _fill_mimic_limits,
    prepare_urdf,
    write_urdf,
)
from rammp_curobo.validate import start_state_matches, validate_trajectory
from rammp_curobo.world import make_world_config, world_cuboids

DATA = Path(__file__).parent / "data"
BAKE = Path(__file__).resolve().parents[2] / "scripts" / "bake_robot_config.py"
_spec = importlib.util.spec_from_file_location("bake_robot_config", BAKE)
_bake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bake)
ARM_LINKS = _bake.ARM_LINKS
LINK_MAP = _bake.LINK_MAP


def read(name):
    return (DATA / name).read_text()


def load_yaml_robot(name_or_path):
    return yaml.safe_load(resolve_config(name_or_path).read_text())["robot_cfg"]


def test_conftest_refuses_only_another_checkouts_core(tmp_path, monkeypatch):
    # loaded by path: `import conftest` names whichever conftest.py pytest
    # imported last
    spec = importlib.util.spec_from_file_location(
        "wrong_tree_guard", Path(__file__).with_name("conftest.py")
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    def imported_from(parent):
        stub = types.ModuleType("rammp_curobo")
        stub.__file__ = str(parent / "rammp_curobo" / "__init__.py")
        monkeypatch.setitem(sys.modules, "rammp_curobo", stub)
        guard.pytest_configure(None)

    here = Path(__file__).resolve().parents[1]
    with pytest.raises(pytest.UsageError) as exc:
        imported_from(tmp_path / "core")
    assert str(exc.value) == (
        "rammp_curobo is imported from %s, not this checkout (%s); "
        "run with PYTHONPATH=%s" % (tmp_path / "core", here, here)
    )
    # a non-editable install (the image) is not another checkout
    imported_from(tmp_path / "site-packages")


def test_the_gpu_tests_cannot_skip_where_they_must_run():
    # On the Jetson they are the gate, and a run that skipped them would pass
    # on the offline tests alone: with RAMMP_CUROBO_GPU=1, which make verify
    # sets, a skip of test_smoke.py fails. Without it, as in CI, they skip.
    # The child pytest runs an offline test and test_smoke.py, as a machine
    # without torch would
    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if k != "RAMMP_CUROBO_GPU"}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo / "core"), env.get("PYTHONPATH")])
    )
    code = (
        "import sys; sys.modules['torch'] = None; import pytest; "
        "sys.exit(pytest.main(sys.argv[1:]))"
    )
    tests = ["core/tests/test_offline.py::test_packaged_configs_exist"]
    tests += ["core/tests/test_smoke.py"]

    def run(**switch):
        return subprocess.run(
            [sys.executable, "-c", code, *tests, "-q", "-p", "no:cacheprovider"],
            cwd=repo,
            env=dict(env, **switch),
            capture_output=True,
            text=True,
        )

    skipped = run()
    assert skipped.returncode == 0, skipped.stdout + skipped.stderr
    assert "1 passed, 1 skipped" in skipped.stdout
    # a module that cannot be collected stops the whole run
    failed = run(RAMMP_CUROBO_GPU="1")
    assert failed.returncode != 0
    assert "1 error" in failed.stdout and "passed" not in failed.stdout
    assert (
        "RAMMP_CUROBO_GPU=1 skips no GPU test: Skipped: could not import 'torch'"
        in failed.stdout
    ), failed.stdout


def test_packaged_configs_exist():
    for name in (
        "gen3.yaml",
        "robot_gen3_2f85.yaml",
        "world_sim_kitchen.yaml",
        "world_real_bench.yaml",
        "world_empty.yaml",
    ):
        assert (PACKAGED_CONFIG_DIR / name).is_file(), name


def test_the_chair_is_not_packaged_and_resolves_to_no_stale_copy():
    """The chair's five files are ONE robot's geometry, not module config, so
    they ship with that robot (rammp-deployments, config/robots/<robot>/curobo)
    and this checkout keeps only a fixture copy under core/tests/data/mebot
    (issue #19).

    Nothing may leave a packaged copy behind. The planner container bind-mounts
    config/robots/current, and Docker makes that an empty directory when it
    starts before set_robot.py has run: a packaged fallback would let a bare
    layer name resolve against the image and plan the chair against stale
    geometry, with no warning. Absent, the same start fails instead.
    """
    chair = (
        "planner.yaml",
        "world.yaml",
        "layer_mebot_platform.yaml",
        "layer_mebot_person.yaml",
        "layer_mebot_floor.yaml",
    )
    fixtures = Path(__file__).resolve().parent / "data" / "mebot"
    for name in chair:
        assert not (PACKAGED_CONFIG_DIR / name).exists(), name
        with pytest.raises(FileNotFoundError):
            resolve_config(name, search_cwd=False)
        assert (fixtures / name).is_file(), name


def test_planner_config_loads_and_merges():
    cfg, cfg_dir = load_planner_config("gen3.yaml")
    assert cfg["robot"] == "robot_gen3_2f85.yaml"
    assert cfg["joint_names"] == ["arm_joint_%d" % i for i in range(1, 8)]
    assert cfg["planner"]["enable_graph"] is False
    assert cfg["planner"]["joint_space_method"] == "auto"
    assert "execution" not in cfg
    assert resolve_config(cfg["world"], relative_to=cfg_dir).is_file()


def test_gen3_yaml_matches_planner_defaults():
    # gen3.yaml deliberately restates every default so the file is the
    # single human-readable reference (its own header says so). This test
    # keeps the two copies honest: any drift between config.PLANNER_DEFAULTS
    # and the YAML is a bug in whichever was edited alone.
    from rammp_curobo.config import PLANNER_DEFAULTS

    cfg, _ = load_planner_config("gen3.yaml")
    for section in ("planner", "tool"):
        assert cfg[section] == PLANNER_DEFAULTS[section], section
    assert cfg["joint_names"] == PLANNER_DEFAULTS["joint_names"]
    assert cfg["joint_names"] == ["arm_joint_%d" % i for i in range(1, 8)]
    assert cfg["layers"] == PLANNER_DEFAULTS["layers"]
    # merged, a line deleted from the file reads as its default: `layers` is
    # also read from the file itself, where its comment explains the key
    text = (PACKAGED_CONFIG_DIR / "gen3.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)["layers"] == []


def test_baked_robot_config_is_the_upstream_model():
    kin = load_yaml_robot("robot_gen3_2f85.yaml")["kinematics"]
    old = load_yaml_robot(DATA / "robot_gen3_2f85_nvidia.yaml")["kinematics"]
    assert kin["base_link"] == "arm_mount" and kin["ee_link"] == "arm_tool_frame"
    assert kin["urdf_path"] == "from_robot_description" and kin["mesh_link_names"] == []
    assert kin["lock_joints"] == {"arm_robotiq_85_left_knuckle_joint": 0.0}
    assert kin["cspace"]["joint_names"] == list(ARM_JOINTS)
    assert kin["cspace"]["retract_config"] == old["cspace"]["retract_config"]
    for nv, up in LINK_MAP.items():  # moved, not re-tuned
        if nv in old["collision_spheres"]:
            assert [s["radius"] for s in kin["collision_spheres"][up]] == [
                s["radius"] for s in old["collision_spheres"][nv]
            ]
    assert set(kin["collision_link_names"]) == {
        LINK_MAP[n] for n in old["collision_link_names"]
    }
    for link in ARM_LINKS:  # identity transforms: centres carry over exactly
        assert kin["collision_spheres"]["arm_" + link] == old["collision_spheres"][link]
    # a stale NVIDIA name here matches nothing in cuRobo and is skipped
    # silently, leaving an adjacent pair always in self collision
    assert kin["self_collision_ignore"] == {
        LINK_MAP[k]: [LINK_MAP[x] for x in v]
        for k, v in old["self_collision_ignore"].items()
    }
    assert kin["self_collision_buffer"] == {
        LINK_MAP[k]: v for k, v in old["self_collision_buffer"].items()
    }


def test_bake_rebuilds_the_committed_robot_config():
    """The bake's rename / lock / sphere move, re-run offline: CI catches a
    bake change that `make bake` would only catch by hand, and a hand edit
    of a gripper sphere (the shell nearest the user)."""
    nv_cfg = yaml.safe_load((DATA / "robot_gen3_2f85_nvidia.yaml").read_text())
    moves = {n: np.eye(4) for n in ARM_LINKS}
    moves.update(yaml.safe_load((DATA / "robot_gen3_2f85_moves.yaml").read_text()))
    assert set(moves) == set(LINK_MAP)
    got = _bake.build_robot_config(nv_cfg, moves)["robot_cfg"]["kinematics"]
    want = load_yaml_robot("robot_gen3_2f85.yaml")["kinematics"]
    spheres = (got.pop("collision_spheres"), want.pop("collision_spheres"))
    assert got == want  # names, lock_joints, self_collision maps, cspace
    assert list(spheres[0]) == list(spheres[1])
    for link, ss in spheres[1].items():
        assert len(spheres[0][link]) == len(ss), link
        for a, b in zip(spheres[0][link], ss):
            assert a["radius"] == b["radius"], link
            assert np.allclose(a["center"], b["center"], rtol=0, atol=1e-6), link


def test_sim_kitchen_scene_parses():
    scene = load_scene(resolve_config("world_sim_kitchen.yaml"))
    assert scene.base_frame == "arm_mount"
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


def test_empty_world_only_when_allowed(monkeypatch):
    # cuRobo stubbed so CI's offline job runs the guard; test_smoke covers
    # the real WorldConfig
    geom = types.ModuleType("curobo.geom.types")
    # every kind is listed: to_world_config checks what the WorldConfig holds
    kinds = dict(sphere=[], cylinder=[], capsule=[], mesh=[])
    geom.WorldConfig = types.SimpleNamespace(
        from_dict=lambda d: types.SimpleNamespace(cuboid=list(d["cuboid"]), **kinds)
    )
    for name in ("curobo", "curobo.geom"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "curobo.geom.types", geom)
    scene = load_scene(resolve_config("world_empty.yaml"))
    assert scene.base_frame == "arm_mount"
    assert make_world_config(scene, allow_empty=True).cuboid == []
    with pytest.raises(ValueError, match="empty"):
        make_world_config(scene)


@pytest.mark.parametrize(
    "text, match",
    [
        ("base_frame: base_link\ntargets: []\n", "obstacles"),  # missing key
        (
            "obstacle:\n  - {name: t, position: [0, 0, 0], dims: [1, 1, 1]}\n",
            "obstacle",
        ),
        ("targets:\n  - {name: cup, position: [0.5, 0, 0]}\n", "obstacles"),
        ("obstacles:\n", "obstacles: \\[\\]"),  # null
        ("", "empty"),
        # the boxes would be applied in arm_mount whatever the file says
        ("base_frame: base_link\nobstacles: []\n", "'base_link'"),
        ("obstacles: []\n", "base_frame is None"),
        ("base_frame: arm_mount\nobstacles: []\nobjects:\n", "'objects' is null"),
        (
            "base_frame: arm_mount\nobstacles: []\nobjects: {name: a}\n",
            "'objects' is dict",
        ),
        ("base_frame: arm_mount\nobstacles: []\ntargets: 3\n", "'targets' is int"),
    ],
    ids=[
        "missing",
        "typo",
        "only-targets",
        "null",
        "empty-file",
        "base-link-frame",
        "no-frame",
        "null-objects",
        "mapping-objects",
        "scalar-targets",
    ],
)
def test_load_scene_refuses_mis_keyed_world(tmp_path, text, match):
    path = tmp_path / "world_bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match) as e:
        load_scene(path)
    assert "world_bad.yaml" in str(e.value)


def test_packaged_worlds_load():
    for name in ("world_sim_kitchen.yaml", "world_real_bench.yaml", "world_empty.yaml"):
        load_scene(resolve_config(name))
    assert load_scene(resolve_config("world_empty.yaml")).obstacles == []


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
    assert scene.base_frame == "arm_mount"  # update_world(list) must not be refused
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
    assert "execution" not in cfg
    assert resolve_config(cfg["world"], relative_to=cfg_dir).is_file()


def test_legacy_execution_block_is_dropped_with_a_warning(tmp_path, caplog):
    # A config copied from <= v1.1.0 gen3.yaml still carries execution:
    # speed_scale: 0.25, which nothing ever read (issue #22). It must keep
    # loading, but say out loud that it limits nothing, and not survive into
    # the config where it could be mistaken for a live setting.
    path = tmp_path / "old_gen3.yaml"
    path.write_text(
        "robot: robot_gen3_2f85.yaml\n"
        "world: world_sim_kitchen.yaml\n"
        "execution:\n"
        "  speed_scale: 0.25\n"
        "  max_speed_scale: 1.0\n"
    )
    with caplog.at_level("WARNING", logger="rammp_curobo"):
        cfg, _ = load_planner_config(str(path))
    assert "execution" not in cfg
    (rec,) = [r for r in caplog.records if r.name == "rammp_curobo"]
    assert rec.levelname == "WARNING"
    assert "ignoring the 'execution' block" in rec.getMessage()

    # the packaged config carries no such block, so it must load silently
    caplog.clear()
    with caplog.at_level("WARNING", logger="rammp_curobo"):
        load_planner_config("gen3.yaml")
    assert not [r for r in caplog.records if r.name == "rammp_curobo"]


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
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert PoseConstraint(hold=HOLD_FIXED).hold_vec_weight() == [
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
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


def test_a_hold_and_a_via_together_are_refused():
    """cuRobo's approach metric zeroes the per-step pose cost before
    tstep_fraction (pose_cost.py update_run_weight), so a hold sent with a
    via would not be applied for most of the path. Refused, not dropped."""
    from rammp_curobo.planner import CuRoboPlanner

    for mode in (HOLD_LEVEL, HOLD_FIXED):
        for axis in (0, 1, 2):
            with pytest.raises(ValueError, match="cannot be combined"):
                CuRoboPlanner._pose_cost_kwargs(
                    PoseConstraint(hold=mode),
                    ViaPoint(offset_m=0.10, linear_axis=axis),
                )


def test_a_via_alone_carries_its_offset():
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(
        None, ViaPoint(offset_m=0.10, linear_axis=2, tstep_fraction=0.8)
    )
    assert kw["offset_position"] == 0.10
    assert kw["linear_axis"] == 2
    assert kw["offset_tstep_fraction"] == 0.8


def test_via_point_ignores_the_constraint_when_via_is_inactive():
    """An inactive ViaPoint must not perturb a plain PoseConstraint plan —
    the five-axis hold is a via-point-only behaviour."""
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(PoseConstraint(hold=HOLD_LEVEL), ViaPoint())
    assert kw["hold_vec_weight"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert "offset_position" not in kw


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


def quat_mul_xyzw(a, b):
    """Hamilton product in xyzw, for composing a tilt with a yaw in tests."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


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
    ok, why = _check(stub, [0.6, 0.2, 0.4], tilted, PoseConstraint(hold=HOLD_LEVEL))
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
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold=HOLD_FIXED))
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


def test_level_deviation_is_the_true_tilt_regardless_of_yaw():
    """REGRESSION. The LEVEL measure must report the real tilt even when the
    orientations also differ by a large yaw -- which is the NORMAL case, since
    LEVEL frees yaw without bound.

    The previous measure, hypot(rx, ry) of the rotation vector, over-reported
    by (psi/2)/sin(psi/2): a genuine 1.3 deg tilt with a 180 deg yaw measured
    2.04 deg and was refused as CONSTRAINT_VIOLATED. The old tests could not
    catch it because they only ever probed a PURE yaw (tilt = 0), the single
    case where hypot happens to be exact.

    Every row with a non-zero yaw fails under the old formula and passes
    under the geodesic distance to the yaw orbit.
    """
    from rammp_curobo.planner import CuRoboPlanner

    level = PoseConstraint(hold=HOLD_LEVEL)
    for tilt_deg in (1.3, 2.0, 5.0):
        for yaw_deg in (0.0, 45.0, 90.0, 150.0, 179.0):
            # tilt about base X, then yaw about base Z
            ref = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
            q = euler_deg_to_quat_xyzw([tilt_deg, 0.0, 0.0])
            q = quat_mul_xyzw(euler_deg_to_quat_xyzw([0.0, 0.0, yaw_deg]), q)
            got, what = CuRoboPlanner._hold_deviation(ref, q, level)
            assert what == "tilt"
            assert math.degrees(got) == pytest.approx(tilt_deg, abs=1e-6), (
                "tilt %.1f deg with %.0f deg of yaw measured %.4f deg"
                % (tilt_deg, yaw_deg, math.degrees(got))
            )


def test_batch_and_scalar_deviation_agree_on_combined_tilt_and_yaw():
    """_verify_hold uses the batch form, the start check the scalar one; one
    tolerance only means one thing if they agree."""
    from rammp_curobo.planner import CuRoboPlanner

    goal = euler_deg_to_quat_xyzw([3.0, -2.0, 25.0])
    qs = [
        euler_deg_to_quat_xyzw([r, p, y])
        for r, p, y in ((1.3, 0.0, 90.0), (-4.0, 2.5, 170.0), (0.0, 7.0, -60.0))
    ]
    wxyz = np.array([[q[3], q[0], q[1], q[2]] for q in qs])
    for mode in (HOLD_LEVEL, HOLD_FIXED):
        c = PoseConstraint(hold=mode)
        batch = CuRoboPlanner._hold_deviation_batch(wxyz, goal, c)
        for k, q in enumerate(qs):
            scalar, _ = CuRoboPlanner._hold_deviation(q, goal, c)
            assert batch[k] == pytest.approx(scalar, abs=1e-9)


def test_fixed_deviation_is_the_full_angle():
    """FIXED holds yaw too, so a pure yaw is a full deviation under it. This is
    the companion to the test above: the two modes must disagree about yaw."""
    from rammp_curobo.planner import CuRoboPlanner

    fixed = PoseConstraint(hold=HOLD_FIXED)
    ref = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    for yaw_deg in (10.0, 90.0, 170.0):
        q = euler_deg_to_quat_xyzw([0.0, 0.0, yaw_deg])
        got, what = CuRoboPlanner._hold_deviation(ref, q, fixed)
        assert what == "orientation"
        assert math.degrees(got) == pytest.approx(yaw_deg, abs=1e-6)


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


class _StubTrajFk:
    """Just enough of CuRoboPlanner for _verify_hold, called unbound: each
    trajectory row's single joint value IS the index of the waypoint
    orientation the stub kinematics hands back."""

    joint_names = _curobo_joint_names = ["j"]

    def __init__(self, quats_xyzw, tolerance_deg=2.0):
        from rammp_curobo.planner import CuRoboPlanner

        self._hold_deviation_batch = CuRoboPlanner._hold_deviation_batch
        self.constraint_tolerance_rad = math.radians(tolerance_deg)
        wxyz = np.array([[q[3], q[0], q[1], q[2]] for q in quats_xyzw])

        class _Arr:
            def __init__(self, a):
                self.a = a

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        class _Kin:
            def get_state(self, rows):
                state = type("S", (), {})()
                state.ee_quaternion = _Arr(wxyz[rows[:, 0].astype(int)])
                return state

        self._motion_gen = type("M", (), {"kinematics": _Kin()})()
        self.traj = Trajectory(
            joint_names=["j"],
            positions=np.arange(len(quats_xyzw), dtype=float)[:, None],
            velocities=None,
            accelerations=None,
            dt=0.01,
        )

    @staticmethod
    def _tensor(x):
        return x


def _verify(stub, goal_xyzw, constraint):
    from rammp_curobo.planner import CuRoboPlanner

    goal_wxyz = [goal_xyzw[3], goal_xyzw[0], goal_xyzw[1], goal_xyzw[2]]
    return CuRoboPlanner._verify_hold(stub, stub.traj, goal_wxyz, constraint)


def test_level_must_still_reach_the_goal_yaw():
    """LEVEL frees yaw ALONG THE WAY, never at the goal. A level path that
    ends 10 deg short of the requested yaw is refused, not accepted because
    the hold itself was respected -- cuRobo's own success test would let up
    to ~5.7 deg through, so this cannot be left to it."""
    level = PoseConstraint(hold=HOLD_LEVEL)
    goal = euler_deg_to_quat_xyzw([0.0, 0.0, 30.0])
    path = [euler_deg_to_quat_xyzw([0.0, 0.0, y]) for y in (0.0, 10.0, 20.0)]
    why = _verify(_StubTrajFk(path), goal, level)
    assert why is not None
    assert "ends 10.00 deg from the goal orientation" in why

    path.append(goal)
    assert _verify(_StubTrajFk(path), goal, level) is None


def test_a_path_violation_is_reported_before_the_endpoint():
    level = PoseConstraint(hold=HOLD_LEVEL)
    goal = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    path = [goal, euler_deg_to_quat_xyzw([8.0, 0.0, 0.0]), goal]
    why = _verify(_StubTrajFk(path), goal, level)
    assert why is not None and "tilt (roll/pitch)" in why


class _StubLimits:
    """Just enough of CuRoboPlanner for _nearest_branch, called unbound:
    joint a is continuous (no <limit> in the URDF, so +/-6.2832), joint b
    is not."""

    joint_names = ["a", "b"]

    def joint_limits(self):
        return {"position": np.array([[-6.2832, -2.0], [6.2832, 2.0]])}


def _off_start_branch(final, start):
    """plan_to_pose's #14 trigger: the plan ends on a 2*pi branch other
    than the start's."""
    from rammp_curobo.planner import CuRoboPlanner

    final = [float(v) for v in final]
    return CuRoboPlanner._nearest_branch(_StubLimits(), final, start) != final


def test_a_revolution_back_to_the_same_angle_is_off_branch():
    # #14: -pi -> +pi through zero is the same physical angle, a full turn
    assert _off_start_branch([math.pi, 0.0], [-math.pi, 0.0])


def test_a_move_the_long_way_round_is_off_branch():
    # 216 deg the long way instead of 144 deg the short way: the #14 bug
    # class, not only the pure revolution
    assert _off_start_branch([1.2 * math.pi, 0.0], [0.0, 0.0])


def test_a_direct_move_is_on_branch():
    assert not _off_start_branch([-math.pi + 0.3, 0.0], [-math.pi, 0.0])
    assert not _off_start_branch([0.9 * math.pi, 0.0], [0.0, 0.0])


def test_a_detour_back_to_the_start_is_on_branch():
    # an out-and-back detour ends where it began: nothing to shift
    assert not _off_start_branch([0.4, 0.0], [0.4, 0.0])


def test_only_continuous_joints_are_shifted():
    assert not _off_start_branch([0.0, 1.9], [0.0, -1.9])


def test_meshes_are_stripped_and_missing_files_load():
    xml = prepare_urdf(read("arm_level.urdf"))
    root = ET.fromstring(xml)
    assert not root.findall(".//visual") and not root.findall(".//collision")
    assert not root.findall("ros2_control")
    assert root.find("link[@name='arm_mount']") is not None


def test_ros2_control_is_stripped():
    xml = read("arm_level.urdf").replace(
        "</robot>", "<ros2_control name='x' type='system'/></robot>"
    )
    assert not ET.fromstring(prepare_urdf(xml)).findall("ros2_control")


def test_mimic_joints_get_their_leaders_limit_scaled():
    # cuRobo v0.7.8 dereferences <limit> on every moving joint; the 2F-85's
    # mimic joints carry none in the published model.
    root = ET.fromstring(prepare_urdf(read("arm_level.urdf")))
    lead = root.find("joint[@name='arm_robotiq_85_left_knuckle_joint']/limit")
    assert (lead.get("lower"), lead.get("upper")) == ("0.0", "0.8")
    for name, lo, hi in [
        ("arm_robotiq_85_left_inner_knuckle_joint", 0.0, 0.8),  # multiplier 1
        ("arm_robotiq_85_right_inner_knuckle_joint", -0.8, 0.0),  # multiplier -1
        ("arm_robotiq_85_left_finger_tip_joint", -0.8, 0.0),
        ("arm_robotiq_85_right_finger_tip_joint", 0.0, 0.8),
    ]:
        lim = root.find("joint[@name='%s']/limit" % name)
        assert lim is not None, name
        assert float(lim.get("lower")) == lo and float(lim.get("upper")) == hi, name
        assert float(lim.get("velocity")) == float(lead.get("velocity")), name
    # a mimic with a <limit> of its own keeps it
    right = root.find("joint[@name='arm_robotiq_85_right_knuckle_joint']")
    assert len(right.findall("limit")) == 1


def _mimic_model(mimic, lead_limit='<limit lower="0" upper="0.8" velocity="0.5"/>'):
    return ET.fromstring(
        '<robot><joint name="lead" type="revolute">%s</joint>'
        '<joint name="m" type="revolute">%s</joint></robot>' % (lead_limit, mimic)
    )


def test_mimic_offset_and_multiplier_map_the_range_and_scale_velocity():
    root = _mimic_model('<mimic joint="lead" multiplier="-2" offset="0.1"/>')
    _fill_mimic_limits(root)
    lim = root.find("joint[@name='m']/limit")
    assert float(lim.get("lower")) == pytest.approx(-1.5)
    assert float(lim.get("upper")) == pytest.approx(0.1)
    assert float(lim.get("velocity")) == pytest.approx(1.0)


def test_mimic_range_that_excludes_0_is_refused():
    root = _mimic_model('<mimic joint="lead" offset="1.0"/>')
    with pytest.raises(ValueError, match=r"'m' range \[1, 1.8\] excludes 0"):
        _fill_mimic_limits(root)


def test_mimic_of_a_limitless_leader_is_refused():
    root = _mimic_model('<mimic joint="lead"/>', lead_limit="")
    root.find("joint[@name='lead']").set("type", "fixed")
    with pytest.raises(ValueError, match="'m' mimics 'lead', which has no <limit>"):
        _fill_mimic_limits(root)


def test_limitless_joint_that_mimics_nothing_is_refused():
    xml = read("arm_level.urdf").replace(
        '<limit effort="39" velocity="1.3963" /></joint>', "</joint>", 1
    )
    with pytest.raises(ValueError, match="arm_joint_1"):
        prepare_urdf(xml)


def test_foreign_model_is_refused_by_name():
    for broken, why in [
        (read("arm_level.urdf").replace('"arm_mount"', '"base_link"'), "arm_mount"),
        (
            read("arm_level.urdf").replace("arm_tool_frame", "tool_frame"),
            "arm_tool_frame",
        ),
        (
            read("arm_level.urdf").replace(
                "</robot>",
                '<link name="world"/><joint name="w" type="fixed"><parent link="world"/>'
                '<child link="arm_mount"/></joint></robot>',
            ),
            "arm_mount",
        ),
        (read("arm_level.urdf").replace("arm_joint_4", "joint_4"), "arm_joint_4"),
        (
            read("arm_level.urdf").replace(
                "arm_robotiq_85_left_knuckle_joint", "left_knuckle_joint"
            ),
            "arm_robotiq_85_left_knuckle_joint",
        ),
        ("", "empty"),
        ("<robot", "parse"),
    ]:
        with pytest.raises(ValueError, match=why):
            prepare_urdf(broken)


def test_the_tilted_mount_matches_the_kinematic_reference():
    # arm_base_joint is the mount: it may differ, nothing below it may
    prepare_urdf(read("arm_tilt30.urdf"))


def test_kinematic_drift_is_refused_by_name():
    knuckle = '<origin rpy="0 0 0" xyz="0.03060114 0.0 0.05490452" />'
    assert read("arm_level.urdf").count(knuckle) == 1
    for broken, why in [
        # a 2 mm gripper offset: the spheres would sit 2 mm off the links
        (
            read("arm_level.urdf").replace(
                knuckle, '<origin rpy="0 0 0" xyz="0.03260114 0.0 0.05490452" />'
            ),
            "'arm_robotiq_85_left_knuckle_joint' is 2.000 mm",
        ),
        (
            read("arm_level.urdf").replace(
                knuckle, '<origin rpy="0 0 0.0035" xyz="0.03060114 0.0 0.05490452" />'
            ),
            "'arm_robotiq_85_left_knuckle_joint' is 0.000 mm / 0.201 deg",
        ),
        (
            read("arm_level.urdf").replace(
                '<axis xyz="0 -1 0" /><origin rpy="0 0 0" xyz="0.0127 ',
                '<axis xyz="0 1 0" /><origin rpy="0 0 0" xyz="0.0127 ',
            ),
            "'arm_robotiq_85_left_inner_knuckle_joint'.*axis 180",
        ),
        (
            read("arm_level.urdf").replace(
                "</robot>",
                '<link name="cam"/><joint name="cam_joint" type="fixed">'
                '<parent link="arm_bracelet_link"/><child link="cam"/></joint></robot>',
            ),
            "extra \\['cam_joint'\\]",
        ),
        (
            read("arm_level.urdf").replace(
                '<mimic joint="arm_robotiq_85_left_knuckle_joint" /></joint></robot>',
                "</joint></robot>",
            ),
            "'arm_robotiq_85_right_finger_tip_joint' mimic",
        ),
    ]:
        assert broken != read("arm_level.urdf")
        with pytest.raises(ValueError, match=why):
            prepare_urdf(broken)


def test_kinematic_reference_limit_differences_are_logged_not_refused(caplog):
    xml = read("arm_level.urdf").replace(
        'lower="-2.57" upper="2.57"', 'lower="-2.6" upper="2.6"'
    )
    with caplog.at_level("WARNING", logger="rammp_curobo"):
        prepare_urdf(xml)
    assert "'arm_joint_4' limit" in caplog.text


def test_robot_config_takes_the_urdf_path(tmp_path, monkeypatch):
    p = write_urdf(read("arm_level.urdf"), tmp_path)
    kin = load_robot_config("robot_gen3_2f85.yaml", urdf_path=p)["kinematics"]
    assert kin["urdf_path"] == str(p) and kin["asset_root_path"] == str(tmp_path)
    monkeypatch.chdir(tmp_path)
    kin = load_robot_config("robot_gen3_2f85.yaml", urdf_path=p.name)["kinematics"]
    assert Path(kin["urdf_path"]).is_absolute() and kin["urdf_path"] == str(p)


def test_robot_config_without_a_urdf_is_refused():
    with pytest.raises(ValueError, match="/arm/robot_description"):
        load_robot_config("robot_gen3_2f85.yaml")
