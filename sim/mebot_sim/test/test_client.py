"""The test client's own logic (mebot_sim/client.py), without a ROS graph."""

import types

import pytest

msgs = pytest.importorskip("trajectory_msgs.msg")

from mebot_sim import client, stack  # noqa: E402

STATUS = "/execute_joint_trajectory/_action/status"


class Sent(Exception):
    """A goal reached the driver interface."""


def _tester(publishers):
    """A Tester, made without a ROS graph, whose node sees `publishers`
    ({topic: [(namespace, node)]}) and whose driver client raises Sent."""
    tester = object.__new__(client.Tester)
    tester.node = types.SimpleNamespace(
        get_name=lambda: "test",
        get_publishers_info_by_topic=lambda topic: [
            types.SimpleNamespace(node_namespace=ns, node_name=name)
            for ns, name in publishers.get(topic, [])
        ],
    )
    tester.q, tester.sim = [0.0] * 7, {"move": 0}
    tester._Goal = lambda **fields: fields

    def send_goal_async(goal):
        raise Sent()

    tester.driver = types.SimpleNamespace(send_goal_async=send_goal_async)
    return tester


def test_nothing_is_flown_unless_the_sim_alone_serves_the_driver():
    # an action goal reaches every server of its name, and the sim looked
    # for another driver only when it started
    traj = msgs.JointTrajectory(joint_names=list(stack.JOINTS))
    traj.points.append(msgs.JointTrajectoryPoint(positions=[0.0] * 7))
    sim, driver = ("/", "mebot_sim"), ("/", "kortex_driver")
    with pytest.raises(Sent):
        _tester({"/joint_states": [sim], STATUS: [sim]}).fly(traj, 1.0)
    for publishers in (
        {"/joint_states": [sim, driver], STATUS: [sim]},
        {"/joint_states": [sim], STATUS: [sim, driver]},
        {"/joint_states": [driver], STATUS: [driver]},
        {"/joint_states": [sim]},
        {"/joint_states": [("/bench", "mebot_sim")], STATUS: [sim]},
    ):
        with pytest.raises(RuntimeError, match="nothing is flown"):
            _tester(publishers).fly(traj, 1.0)


class Done:
    """An rclpy future that is done already."""

    def __init__(self, value):
        self.value = value

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self.value


def test_a_flight_says_how_far_the_arm_was_from_its_last_point(monkeypatch):
    # when the flight ended (the tracking error) and once at rest (G6)
    monkeypatch.setattr(client, "SETTLE_S", 0.0)
    traj = msgs.JointTrajectory(joint_names=list(stack.JOINTS))
    traj.points.append(msgs.JointTrajectoryPoint(positions=[0.1] * 7))
    sim = ("/", "mebot_sim")
    tester = _tester({"/joint_states": [sim], STATUS: [sim]})
    tester.q = [0.1] * 7  # at the plan's start
    result = types.SimpleNamespace(
        error_code=0,
        error_string="",
        final_error=types.SimpleNamespace(positions=[0.02, -0.03, 0, 0, 0, 0, 0]),
    )

    def finished():
        tester.sim = {"move": 1, "worst": [], "closest": {}}
        tester.q = [0.1] * 6 + [0.14]
        return Done(types.SimpleNamespace(result=result))

    handle = types.SimpleNamespace(accepted=True, get_result_async=finished)
    tester.driver = types.SimpleNamespace(send_goal_async=lambda goal: Done(handle))
    report = tester.fly(traj, 1.0)
    assert report["executed"] and report["contacts"] == []
    assert report["tracking_rad"] == pytest.approx(0.03)
    assert report["end_error_rad"] == pytest.approx(0.04)


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
