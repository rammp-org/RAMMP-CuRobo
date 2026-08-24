"""The _execute gate CHAIN and current_q staleness — no ROS spin.

test_executor_gates.py covers validate_goal_msg in isolation; this file
proves the node actually consults it (and the other gates) in order, and
that nothing reaches executor_helper.run when any gate refuses. Built the
test_world_boxes_handler way: object.__new__ + only what _execute touches.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo_ros.conversions import duration_msg
from rammp_curobo_ros.planner_node import RammpCuroboNode

JOINTS = ["joint_%d" % i for i in range(1, 8)]


def _bare_node(execute=True):
    node = object.__new__(RammpCuroboNode)  # no __init__ — no ROS context
    node._abort = threading.Event()
    node._exec_lock = threading.Lock()
    node._inflight = 0
    node._inflight_lock = threading.Lock()
    node._state_lock = threading.Lock()
    node._joint_msg = None
    node._joint_msg_time = None
    node.speed_scale = 0.25
    node.max_speed_scale = 1.0
    node.start_tolerance = 0.05
    node.tracking_tolerance = 0.08
    node.planner = MagicMock(joint_names=list(JOINTS))
    node._pos_limits = (np.full(7, -6.0), np.full(7, 6.0))
    node._vel_limits = np.full(7, 1.4)
    node.executor_helper = MagicMock()
    node.executor_helper.run.return_value = ("succeeded", "trajectory executed")
    params = {"execute": execute}
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node.get_logger = MagicMock()
    return node


def _seed_joint_state(node, q, age_s=0.0, names=None):
    msg = JointState()
    msg.name = list(names) if names is not None else list(JOINTS)
    msg.position = [float(v) for v in q]
    node._joint_msg = msg
    node._joint_msg_time = time.monotonic() - age_s


def _traj(n=20, dt=0.02, step=0.01):
    msg = JointTrajectory()
    msg.joint_names = list(JOINTS)
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [k * step] * 7
        p.velocities = [step / dt] * 7
        p.time_from_start = duration_msg((k + 1) * dt)
        msg.points.append(p)
    return msg


class FakeGoalHandle:
    def __init__(self, msg=None, speed_scale=0.0):
        self.request = SimpleNamespace(
            trajectory=msg if msg is not None else _traj(), speed_scale=speed_scale
        )
        self.is_cancel_requested = False
        self.aborted = self.succeeded = self.cancelled = False

    def abort(self):
        self.aborted = True

    def succeed(self):
        self.succeeded = True

    def canceled(self):
        self.cancelled = True

    def publish_feedback(self, fb):
        pass


# ------------------------------------------------------------------ the chain
def test_execute_param_false_refuses_before_anything_else():
    node = _bare_node(execute=False)
    gh = FakeGoalHandle()
    _seed_joint_state(node, [0.0] * 7)  # everything else valid — gate alone refuses
    res = node._execute(gh)
    assert not res.success and "execution disabled" in res.message
    assert gh.aborted and not gh.succeeded
    node.executor_helper.run.assert_not_called()


def test_abort_flag_refuses_as_shutting_down():
    node = _bare_node()
    node._abort.set()
    res = node._execute(FakeGoalHandle())
    assert not res.success and "shutting down" in res.message
    node.executor_helper.run.assert_not_called()


def test_speed_scale_zero_takes_the_node_default():
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 7)
    gh = FakeGoalHandle(speed_scale=0.0)
    res = node._execute(gh)
    assert res.success
    assert node.executor_helper.run.call_args.args[1] == node.speed_scale


def test_speed_scale_above_max_is_refused():
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 7)
    res = node._execute(FakeGoalHandle(speed_scale=1.5))
    assert not res.success and "outside" in res.message
    node.executor_helper.run.assert_not_called()


def test_no_joint_state_refuses():
    node = _bare_node()  # nothing seeded -> current_q() is None
    res = node._execute(FakeGoalHandle())
    assert not res.success and "no current joint state" in res.message
    node.executor_helper.run.assert_not_called()


def test_start_state_mismatch_refuses():
    node = _bare_node()
    _seed_joint_state(node, [0.5] * 7)  # arm nowhere near the trajectory start
    res = node._execute(FakeGoalHandle())
    assert not res.success and "stale plan" in res.message
    node.executor_helper.run.assert_not_called()


def test_valid_goal_runs_once_and_succeeds():
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 7)
    gh = FakeGoalHandle(speed_scale=0.5)
    res = node._execute(gh)
    assert res.success and gh.succeeded and not gh.aborted
    node.executor_helper.run.assert_called_once()
    call = node.executor_helper.run.call_args
    assert call.args[0] is gh.request.trajectory
    assert call.args[1] == 0.5
    assert call.kwargs["goal_handle"] is gh
    assert call.kwargs["get_current_q"] == node.current_q
    assert call.kwargs["tracking_tolerance_rad"] == node.tracking_tolerance
    assert call.kwargs["abort_cb"] == node._abort.is_set


def test_shutdown_cancel_reports_aborted_not_canceled():
    # a shutdown abort stopped the arm like a cancel, but only a goal the
    # CLIENT cancelled may transition to CANCELED (rclpy raises otherwise)
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 7)
    node.executor_helper.run.return_value = ("canceled", "canceled; holds")
    gh = FakeGoalHandle()
    gh.is_cancel_requested = False
    res = node._execute(gh)
    assert not res.success and gh.aborted and not gh.cancelled


def test_client_cancel_reports_canceled():
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 7)
    node.executor_helper.run.return_value = ("canceled", "canceled; holds")
    gh = FakeGoalHandle()
    gh.is_cancel_requested = True
    res = node._execute(gh)
    assert not res.success and gh.cancelled and not gh.aborted


# ----------------------------------------------------------------- current_q
def test_current_q_none_when_stale():
    # a stale state must never gate-pass an execution
    node = _bare_node()
    _seed_joint_state(node, [0.1] * 7, age_s=5.0)
    assert node.current_q() is None
    assert node.current_q(max_age_s=10.0) is not None


def test_current_q_reorders_to_controller_order():
    node = _bare_node()
    # publisher order (reversed, plus a gripper joint) must not leak through
    names = list(reversed(JOINTS)) + ["finger_joint"]
    _seed_joint_state(node, [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 9.9], names=names)
    assert node.current_q() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_current_q_none_on_missing_joint():
    node = _bare_node()
    _seed_joint_state(node, [0.0] * 6, names=JOINTS[:6])
    assert node.current_q() is None
