"""Hardware-free, GPU-free tests: configs, scene parsing, retiming, geometry.

These run anywhere (CI, a laptop) — no torch, no cuRobo, no ROS.
"""

import importlib.util
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from rammp_curobo.config import PACKAGED_CONFIG_DIR, load_planner_config, resolve_config
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
        "gen3_mebot.yaml",
        "world_mebot.yaml",
        "layer_mebot_platform.yaml",
        "layer_mebot_person.yaml",
        "layer_mebot_floor.yaml",
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
    assert cfg["layers"] == PLANNER_DEFAULTS["layers"]
    # merged, a line deleted from the file reads as its default: `layers` is
    # also read from the file itself, where its comment explains the key
    text = (PACKAGED_CONFIG_DIR / "gen3.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)["layers"] == []


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
