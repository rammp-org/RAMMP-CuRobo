"""The node's model intake and frame stamp — no ROS runtime, no GPU.

The planner plans in the model's root frame (arm_mount), and the consumer
executes a pose plan only if the trajectory says so in its header. The
`world` parameter goes into the config before the planner is built, because
update_world refuses the empty world a deployment may choose.
"""

import numpy as np
import pytest

# Skip only without ROS itself. A missing name in the node must FAIL, not
# skip: a guard around these imports would read "1 skipped" through exactly
# the regression this file exists to catch.
pytest.importorskip("rclpy", reason="ROS 2 not sourced")
pytest.importorskip("trajectory_msgs", reason="ROS 2 not sourced")

from rammp_curobo import Trajectory  # noqa: E402
from rammp_curobo.config import load_planner_config  # noqa: E402

from rammp_curobo_ros.conversions import trajectory_to_msg  # noqa: E402
from rammp_curobo_ros.planner_node import planner_config  # noqa: E402


def _traj(frame_id):
    return Trajectory(
        joint_names=["arm_joint_%d" % i for i in range(1, 8)],
        positions=np.zeros((2, 7)),
        velocities=None,
        accelerations=None,
        dt=0.02,
        frame_id=frame_id,
    )


def test_message_header_carries_the_planning_frame():
    assert trajectory_to_msg(_traj("arm_mount")).header.frame_id == "arm_mount"


def test_world_parameter_is_applied_before_construction():
    cfg, cfg_dir = planner_config("gen3.yaml", "world_empty.yaml")
    assert cfg["world"] == "world_empty.yaml"
    assert (cfg_dir / "gen3.yaml").is_file()


def test_empty_world_parameter_keeps_the_configs_world():
    cfg, _ = planner_config("gen3.yaml", "")
    assert cfg["world"] == load_planner_config("gen3.yaml")[0]["world"]
