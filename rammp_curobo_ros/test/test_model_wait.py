"""The node's wait for the latched model — ROS runtime, no GPU.

arm_state_publisher normally comes up first, so the model is latched before
the planner subscribes: only a RELIABLE + TRANSIENT_LOCAL subscription gets
it. With no model the node must exit 1, so a supervisor sees a failure. Later,
a different model is reported, every time, and the one in use is not.
_wait_for_model and _watch_model use only Node API, so they run here on a
plain node.
"""

import random
import time

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from rammp_curobo_ros.planner_node import (  # noqa: E402
    MODEL_QOS,
    ROBOT_DESCRIPTION_TOPIC,
    RammpCuroboNode,
)


@pytest.fixture
def node():
    # a private domain: a real arm_state_publisher nearby must not answer
    rclpy.init(domain_id=random.randint(100, 200))
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


def test_a_different_model_is_reported_every_time(node):
    errors = []
    node.get_logger = lambda: type("L", (), {"error": staticmethod(errors.append)})
    pub = node.create_publisher(String, ROBOT_DESCRIPTION_TOPIC, MODEL_QOS)
    pub.publish(String(data="<robot name='in_use'/>"))
    RammpCuroboNode._watch_model(node, "<robot name='in_use'/>")

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    spin(1.0)  # the latched copy of the model in use arrives: not reported
    assert errors == []
    for n in (1, 2):
        pub.publish(String(data="<robot name='re_measured'/>"))
        spin(1.0)
        assert len(errors) == n
    assert "restart the planner" in errors[0]
    assert ROBOT_DESCRIPTION_TOPIC in errors[0]


def test_no_model_exits_1(node):
    with pytest.raises(SystemExit) as exc:
        RammpCuroboNode._wait_for_model(node, 0.5)
    assert exc.value.code == 1
