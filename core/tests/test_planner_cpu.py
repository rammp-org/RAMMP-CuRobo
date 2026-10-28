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
