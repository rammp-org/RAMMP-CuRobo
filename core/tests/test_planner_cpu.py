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
    assert res.joint_traj.frame_id == "arm_mount"
    assert res.joint_traj.scaled(0.5).frame_id == "arm_mount"


def test_park_pose_is_refused_by_name_and_a_near_miss_clamped(cpu_planner):
    park = list(cpu_planner.retract_pose)
    park[3] = -2.674  # the Gen3 park pose, 0.104 rad past the URDF's -2.57
    q, err = cpu_planner._clamp_to_limits(park, "start")
    assert q is None and "arm_joint_4" in err
    park[3] = -2.60
    q, err = cpu_planner._clamp_to_limits(park, "start")
    assert err is None and q[3] == pytest.approx(-2.57)


def test_update_world_refuses_before_reaching_curobo(
    cpu_planner, monkeypatch, tmp_path
):
    geom = types.ModuleType("curobo.geom.types")
    geom.WorldConfig = types.SimpleNamespace(
        from_dict=lambda d: types.SimpleNamespace(cuboid=list(d["cuboid"]))
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
