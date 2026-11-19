"""plan_log — every plan request and its result, one JSON line each, for
bench/sweep_bench.py to replay. No ROS runtime: the node's _plan path runs on
a stand-in for the node, with a planner whose world_boxes is a property, as
CuRoboPlanner's is."""

import json
import threading
import types

import numpy as np
import pytest

try:
    from rammp_curobo_ros.planner_node import RammpCuroboNode
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )

from rammp_curobo.types import PlanResult, Trajectory

JOINTS = ["arm_joint_%d" % i for i in range(1, 8)]
BOX = {
    "name": "table",
    "layer": "scene",
    "dims": [1, 1, 0.1],
    "pose": [0.5, 0, 0, 1, 0, 0, 0],
}


class _Planner:
    joint_names = JOINTS

    def __init__(self, world):
        self._world = world

    @property
    def world_boxes(self):
        if isinstance(self._world, Exception):
            raise self._world
        return self._world


def _node(path, world=(BOX,)):
    warnings = []
    logger = types.SimpleNamespace(warning=warnings.append, info=lambda _: None)
    node = types.SimpleNamespace(
        plan_log=str(path),
        planner=_Planner(world if isinstance(world, Exception) else list(world)),
        get_logger=lambda: logger,
        warnings=warnings,
        _plan_lock=threading.Lock(),
    )
    for name in ("_log_result", "_log_plan"):
        method = getattr(RammpCuroboNode, name)
        setattr(node, name, lambda *a, m=method: m(node, *a))
    return node


def _plan(node, res, kind="pose", goal=None, start=(0.0,) * 7):
    return RammpCuroboNode._plan(
        node, lambda q: res, list(start), log=(kind, goal or {})
    )


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def _ok():
    traj = Trajectory(JOINTS, np.zeros((3, 7)), None, None, 0.02)
    return PlanResult(
        success=True, joint_traj=traj, timing=0.1, error=None, status="OK"
    )


def test_a_plan_and_a_failure_are_logged_with_their_world(tmp_path):
    path = tmp_path / "plan_log.jsonl"
    node = _node(path)
    goal = {"position": [0.5, 0.0, 0.3], "quat_xyzw": [0, 0, 0, 1]}
    ok = _ok()
    assert _plan(node, ok, goal=goal) is ok
    bad = PlanResult.failure("IK_FAIL", "no IK", timing=0.2)
    assert _plan(node, bad, kind="joints", goal={"joints": [0.1] * 7}) is bad
    first, second = _lines(path)
    assert first["kind"] == "pose" and first["goal"] == goal
    assert first["start"] == [0.0] * 7
    assert first["world"] == [BOX] and first["success"] is True
    assert first["positions"] == [[0.0] * 7] * 3 and first["dt"] == 0.02
    assert second["status"] == "IK_FAIL" and "positions" not in second
    assert node.warnings == []


def test_off_busy_or_a_bad_start_logs_nothing(tmp_path):
    path = tmp_path / "plan_log.jsonl"
    node = _node(path)
    # a bad start never reaches the planner
    assert isinstance(_plan(node, _ok(), start=[0.0] * 6), str)
    # nor does a plan while another holds the lock
    node._plan_lock.acquire()
    assert _plan(node, _ok()) is None
    node._plan_lock.release()
    node.plan_log = ""
    _plan(node, _ok())
    assert not path.exists()


def test_a_log_that_cannot_be_written_never_fails_the_plan(tmp_path):
    unwritable = _node(tmp_path / "missing_dir" / "plan_log.jsonl")
    ok = _ok()
    assert _plan(unwritable, ok) is ok
    assert len(unwritable.warnings) == 1 and "plan_log" in unwritable.warnings[0]
    # nor one whose record cannot be built
    broken = _node(tmp_path / "plan_log.jsonl", world=RuntimeError("no world"))
    assert _plan(broken, ok) is ok
    assert len(broken.warnings) == 1 and "no world" in broken.warnings[0]
