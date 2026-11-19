"""AddObstacles — boxes into the collision world over ROS, no file.

ROS runtime, no GPU: the planner is stubbed as in test_example_mode.py, so
these pin the node's side (frame check, message -> obstacle dicts, the
reply); test_planner_cpu.py pins that the planner adds without replacing.
"""

import math

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3  # noqa: E402

import rammp_curobo  # noqa: E402
from rammp_arm_interfaces.msg import CollisionBox  # noqa: E402
from rammp_arm_interfaces.srv import AddObstacles  # noqa: E402

from rammp_curobo_ros import planner_node  # noqa: E402


class _FakePlanner:
    def __init__(self, cfg, cfg_dir, *, urdf_path=None):
        self.base_frame = "arm_mount"
        self.ee_link = "arm_tool_frame"
        self.joint_names = list(cfg.get("joint_names", []))
        self.world_name = cfg["world"]
        self._config_dir = cfg_dir
        self.added = []

    def world_summary(self):
        return "%d boxes" % len(self.added)

    def add_obstacles(self, obstacles):
        self.added.extend(obstacles)


@pytest.fixture
def node(monkeypatch, domain_id):
    # no config: example mode, the example's own model, no model wait
    monkeypatch.setattr(rammp_curobo, "CuRoboPlanner", _FakePlanner)
    rclpy.init(args=[], domain_id=domain_id)
    n = planner_node.RammpCuroboNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _box(name, xyz=(0.5, 0.0, 0.3), q=(0.0, 0.0, 0.0, 1.0), dims=(0.1, 0.2, 0.3)):
    return CollisionBox(
        name=name,
        pose=Pose(
            position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        ),
        dims=Vector3(x=dims[0], y=dims[1], z=dims[2]),
    )


def _call(node, frame_id, boxes):
    return node._add_obstacles_cb(
        AddObstacles.Request(frame_id=frame_id, boxes=boxes), AddObstacles.Response()
    )


def test_the_service_is_advertised(node):
    names = [n for n, _ in node.get_service_names_and_types()]
    assert "/rammp_curobo/add_obstacles" in names


def test_boxes_reach_the_planner_as_obstacles(node):
    s = math.sqrt(0.5)
    resp = _call(node, "arm_mount", [_box("mug"), _box("door", q=(0.0, 0.0, s, s))])
    assert resp.success, resp.message
    assert "2 boxes" in resp.message
    mug, door = node.planner.added
    assert mug["name"] == "mug"
    assert mug["position"] == pytest.approx([0.5, 0.0, 0.3])
    assert mug["dims"] == pytest.approx([0.1, 0.2, 0.3])
    assert mug["rpy_deg"] == pytest.approx([0, 0, 0])
    assert door["rpy_deg"] == pytest.approx([0, 0, 90])


def test_another_frame_is_refused_before_the_planner(node):
    resp = _call(node, "base_link", [_box("mug")])
    assert not resp.success
    assert "base_link" in resp.message and "arm_mount" in resp.message
    assert not node.planner.added


def test_a_zero_quaternion_is_refused_whole(node):
    resp = _call(node, "arm_mount", [_box("mug"), _box("bad", q=(0.0, 0.0, 0.0, 0.0))])
    assert not resp.success and "unit quaternion" in resp.message
    assert not node.planner.added


def test_a_planner_refusal_is_the_reply(node):
    def refuse(obstacles):
        raise ValueError("two collision boxes are named 'mug'")

    node.planner.add_obstacles = refuse
    resp = _call(node, "arm_mount", [_box("mug")])
    assert not resp.success and "named 'mug'" in resp.message
