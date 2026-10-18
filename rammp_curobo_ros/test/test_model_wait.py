"""The node's wait for the latched model — ROS runtime, no GPU.

arm_state_publisher normally comes up first, so the model is latched before
the planner subscribes: only a RELIABLE + TRANSIENT_LOCAL subscription gets
it. With no model the node must exit 1, so a supervisor sees a failure.
_wait_for_model uses only Node API, so it runs here on a plain node.
"""

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from rammp_curobo_ros.planner_node import (  # noqa: E402
    ROBOT_DESCRIPTION_TOPIC,
    RammpCuroboNode,
)


@pytest.fixture
def node():
    rclpy.init()
    n = rclpy.create_node("model_wait_test")
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_model_latched_before_the_wait_arrives(node):
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    pub = node.create_publisher(String, ROBOT_DESCRIPTION_TOPIC, qos)
    pub.publish(String(data="<robot/>"))
    assert RammpCuroboNode._wait_for_model(node, 5.0) == "<robot/>"


def test_no_model_exits_1(node):
    with pytest.raises(SystemExit) as exc:
        RammpCuroboNode._wait_for_model(node, 0.5)
    assert exc.value.code == 1
