"""UpdateWorldBoxes handler: validation + planner call, no ROS spin."""

import threading
from unittest.mock import MagicMock

from geometry_msgs.msg import Point, Vector3

from rammp_curobo_interfaces.srv import UpdateWorldBoxes

from rammp_curobo_ros.planner_node import RammpCuroboNode


def _bare_node():
    """A node instance with only what the handler touches."""
    node = object.__new__(RammpCuroboNode)  # no __init__ — no ROS context
    node._plan_lock = threading.Lock()
    node.planner = MagicMock()
    return node


def test_length_mismatch_is_refused():
    node = _bare_node()
    req = UpdateWorldBoxes.Request()
    req.names = ["a"]
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert not res.success and "length" in res.message
    node.planner.update_world_boxes.assert_not_called()


def test_boxes_are_converted_and_baseline_forwarded():
    node = _bare_node()
    req = UpdateWorldBoxes.Request()
    req.names = ["obs_1"]
    req.centers = [Point(x=0.5, y=0.0, z=0.1)]
    req.dims = [Vector3(x=0.1, y=0.2, z=0.3)]
    req.baseline = "world_real_bench.yaml"
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert res.success
    (boxes,), kwargs = node.planner.update_world_boxes.call_args
    assert boxes == [
        {"name": "obs_1", "position": [0.5, 0.0, 0.1], "dims": [0.1, 0.2, 0.3]}
    ]
    assert kwargs == {"baseline": "world_real_bench.yaml"}


def test_planner_exception_reports_failure():
    node = _bare_node()
    node.planner.update_world_boxes.side_effect = ValueError("21 boxes > cache")
    req = UpdateWorldBoxes.Request()
    res = node._update_world_boxes_cb(req, UpdateWorldBoxes.Response())
    assert not res.success and "cache" in res.message
