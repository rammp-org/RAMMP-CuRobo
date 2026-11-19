"""plan_log — every plan request and its result, one JSON line each, for
bench/sweep_bench.py to replay. No ROS runtime: the node's two methods run on
a stand-in for the node."""

import json
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

BOX = {
    "name": "table",
    "layer": "scene",
    "dims": [1, 1, 0.1],
    "pose": [0.5, 0, 0, 1, 0, 0, 0],
}


def _node(path):
    warnings = []
    node = types.SimpleNamespace(
        plan_log=str(path),
        planner=types.SimpleNamespace(world_boxes=lambda: [BOX]),
        get_logger=lambda: types.SimpleNamespace(warning=warnings.append),
        warnings=warnings,
    )
    node._log_plan = lambda rec: RammpCuroboNode._log_plan(node, rec)
    return node


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_a_plan_and_a_failure_are_logged_with_their_world(tmp_path):
    path = tmp_path / "plan_log.jsonl"
    node = _node(path)
    traj = Trajectory(["j%d" % i for i in range(7)], np.zeros((3, 7)), None, None, 0.02)
    ok = PlanResult(success=True, joint_traj=traj, timing=0.1, error=None, status="OK")
    goal = {"position": [0.5, 0.0, 0.3], "quat_xyzw": [0, 0, 0, 1]}
    RammpCuroboNode._log_result(node, "pose", [0.0] * 7, goal, ok)
    bad = PlanResult.failure("IK_FAIL", "no IK", timing=0.2)
    RammpCuroboNode._log_result(node, "joints", [0.0] * 7, {"joints": [0.1] * 7}, bad)
    first, second = _lines(path)
    assert first["kind"] == "pose" and first["goal"] == goal
    assert first["world"] == [BOX] and first["success"] is True
    assert first["positions"] == [[0.0] * 7] * 3 and first["dt"] == 0.02
    assert second["status"] == "IK_FAIL" and "positions" not in second


def test_off_or_a_sentinel_result_logs_nothing(tmp_path):
    path = tmp_path / "plan_log.jsonl"
    node = _node(path)
    # _plan's sentinels: None (busy) and a string (bad start)
    RammpCuroboNode._log_result(node, "pose", [0.0] * 7, {}, None)
    RammpCuroboNode._log_result(node, "pose", [0.0] * 7, {}, "bad start")
    node.plan_log = ""
    ok = PlanResult.failure("IK_FAIL", "no IK")
    RammpCuroboNode._log_result(node, "pose", [0.0] * 7, {}, ok)
    assert not path.exists()


def test_an_unwritable_log_warns_instead_of_failing_the_plan(tmp_path):
    node = _node(tmp_path / "missing_dir" / "plan_log.jsonl")
    RammpCuroboNode._log_result(
        node, "pose", [0.0] * 7, {}, PlanResult.failure("IK_FAIL", "no IK")
    )
    assert len(node.warnings) == 1 and "plan_log" in node.warnings[0]
