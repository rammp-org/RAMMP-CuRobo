"""The test client's own logic (mebot_sim/client.py), without a ROS graph."""

import pytest

msgs = pytest.importorskip("trajectory_msgs.msg")

from mebot_sim import client, stack  # noqa: E402


def test_a_plan_is_flown_slower_by_dilating_time():
    traj = msgs.JointTrajectory(joint_names=list(stack.JOINTS))
    point = msgs.JointTrajectoryPoint(
        positions=[0.0] * 7, velocities=[1.0] * 7, accelerations=[2.0] * 7
    )
    point.time_from_start.sec, point.time_from_start.nanosec = 1, 500000000
    traj.points.append(point)
    client.dilate(traj, 0.25)
    (p,) = traj.points
    # t / s, v * s, a * s^2: the same path, slower
    assert (p.time_from_start.sec, p.time_from_start.nanosec) == (6, 0)
    assert list(p.velocities) == [0.25] * 7
    assert list(p.accelerations) == [0.125] * 7


def test_reports_in_words():
    assert client.contact_text([]) == "no contact"
    hit = {"group": "standin", "link": "half_arm_1_link", "part": "lower_arm_right"}
    assert client.contact_text([dict(hit, force_n=833.4)]) == (
        "half_arm_1_link hit the person (lower_arm_right), 833 N"
    )
    closest = {"mebot": {"mm": 53.4}, "standin": {"mm": -6.7}, "floor": {"mm": 405.0}}
    assert client.closest_text(closest) == (
        "floor 405.0 mm, wheelchair 53.4 mm, person -6.7 mm"
    )
