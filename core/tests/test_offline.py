"""Hardware-free, GPU-free tests: configs, scene parsing, retiming, geometry.

These run anywhere (CI, a laptop) — no torch, no cuRobo, no ROS.
"""

import math

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
    # single human-readable reference (its own header says so). Compare the
    # RAW file, not the merged config: merging fills omitted keys from the
    # defaults, which hid two missing entries until 2026-08.
    from rammp_curobo.config import PLANNER_DEFAULTS

    raw = yaml.safe_load(resolve_config("gen3.yaml").read_text())
    for section in ("planner", "tool", "execution"):
        assert raw[section] == PLANNER_DEFAULTS[section], section
    cfg, _ = load_planner_config("gen3.yaml")
    assert cfg["joint_names"] == PLANNER_DEFAULTS["joint_names"]
    assert cfg["home_pose_rad"] == pytest.approx(PLANNER_DEFAULTS["home_pose_rad"])


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
    # the sim kitchen must leave wide headroom under collision_cache_obb
    # (120) for live perceived-world additions
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


def test_rot_about_world_y_pitches_the_home_tool_down():
    import numpy as np

    from rammp_curobo.geometry import rot_about_world_y, yaw_about_world_z

    home = [0.5, 0.5, 0.5, 0.5]  # tool z along world +x

    def tool_z(q):
        x, y, z, w = q
        return np.array(
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
        )

    assert np.allclose(tool_z(home), [1, 0, 0], atol=1e-9)
    pitched = rot_about_world_y(home, np.radians(40))
    d = tool_z(pitched)
    assert np.isclose(d[2], -np.sin(np.radians(40)), atol=1e-9)  # tips DOWN
    assert np.isclose(d[0], np.cos(np.radians(40)), atol=1e-9)
    both = yaw_about_world_z(pitched, np.radians(30))
    d2 = tool_z(both)
    assert np.isclose(d2[2], d[2], atol=1e-9)  # yaw keeps the pitch
    assert np.isclose(np.arctan2(d2[1], d2[0]), np.radians(30), atol=1e-9)
