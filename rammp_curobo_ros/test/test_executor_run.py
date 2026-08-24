"""TrajectoryExecutor.run state machine — fake client, no ROS spin.

await_future blocks on future.add_done_callback, so every fake future
fires its callbacks the moment it is (or becomes) done; a fake that only
flips done() would hang the test 5 s per await.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo_ros.conversions import duration_msg
from rammp_curobo_ros.executor import TrajectoryExecutor, await_future

JOINTS = ["joint_%d" % i for i in range(1, 8)]


class DoneFuture:
    def __init__(self, result=None):
        self._result = result

    def done(self):
        return True

    def add_done_callback(self, cb):
        cb(self)

    def result(self):
        return self._result


class PendingFuture:
    def __init__(self):
        self._done = False
        self._result = None
        self._cbs = []

    def done(self):
        return self._done

    def add_done_callback(self, cb):
        if self._done:
            cb(self)
        else:
            self._cbs.append(cb)

    def set_result(self, result):
        self._result = result
        self._done = True
        for cb in self._cbs:
            cb(self)

    def result(self):
        return self._result


class _T:
    def __init__(self, ns):
        self.nanoseconds = ns

    def __sub__(self, other):
        return _T(self.nanoseconds - other.nanoseconds)


class SteppingClock:
    """Every now() jumps step_s forward — drives the deadline backstop."""

    def __init__(self, step_s):
        self._ns = -int(step_s * 1e9)
        self._step_ns = int(step_s * 1e9)

    def now(self):
        self._ns += self._step_ns
        return _T(self._ns)


class FakeClient:
    def __init__(self, handle=None, ready=True):
        self.handle = handle
        self.ready = ready
        self.sent = None

    def wait_for_server(self, timeout_sec):
        return self.ready

    def send_goal_async(self, goal):
        self.sent = goal
        return DoneFuture(self.handle)


def _traj(n=5, dt=0.1, step=0.01):
    msg = JointTrajectory()
    msg.joint_names = list(JOINTS)
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [k * step] * 7
        p.velocities = [step / dt] * 7
        p.time_from_start = duration_msg((k + 1) * dt)
        msg.points.append(p)
    return msg


def _executor(client, clock=None):
    ex = TrajectoryExecutor.__new__(TrajectoryExecutor)  # no ActionClient
    ex._node = MagicMock()
    ex._node.get_clock = lambda: clock if clock is not None else SteppingClock(0.0)
    ex._log = MagicMock()
    ex._client = client
    ex._action_name = "/jtc/follow_joint_trajectory"
    return ex


def _handle(result_future, accepted=True, cancel_calls=None, cancel_completes=True):
    def cancel_goal_async():
        if cancel_calls is not None:
            cancel_calls.append(1)
        if cancel_completes and not result_future.done():
            # the JTC answers a cancel by finishing the goal
            result_future.set_result(_wrapped(FollowJointTrajectory.Result.SUCCESSFUL))
        return DoneFuture(None)

    return SimpleNamespace(
        accepted=accepted,
        get_result_async=lambda: result_future,
        cancel_goal_async=cancel_goal_async,
    )


def _wrapped(code, error_string=""):
    return SimpleNamespace(
        result=SimpleNamespace(error_code=code, error_string=error_string)
    )


def test_await_future_honours_an_already_done_future():
    assert await_future(DoneFuture(42), 0.5) == 42
    assert await_future(PendingFuture(), 0.1) is None  # timeout -> None


def test_server_not_ready_fails_without_sending():
    client = FakeClient(ready=False)
    status, msg = _executor(client).run(_traj(), 1.0)
    assert status == "failed" and "not available" in msg
    assert client.sent is None


def test_goal_not_accepted_fails():
    client = FakeClient(_handle(PendingFuture(), accepted=False))
    status, msg = _executor(client).run(_traj(), 1.0)
    assert status == "failed" and "rejected" in msg


def test_abort_cb_cancels_the_controller_goal():
    calls = []
    fut = PendingFuture()
    client = FakeClient(_handle(fut, cancel_calls=calls))
    status, msg = _executor(client).run(_traj(), 1.0, abort_cb=lambda: True)
    assert status == "canceled" and "holds position" in msg
    assert calls  # the arm was told to stop, not just abandoned


def test_goal_handle_cancel_cancels_the_controller_goal():
    calls = []
    fut = PendingFuture()
    client = FakeClient(_handle(fut, cancel_calls=calls))
    gh = SimpleNamespace(is_cancel_requested=True)
    status, _msg = _executor(client).run(_traj(), 1.0, goal_handle=gh)
    assert status == "canceled" and calls


def test_deadline_overrun_cancels_and_fails():
    calls = []
    fut = PendingFuture()  # controller never answers
    client = FakeClient(_handle(fut, cancel_calls=calls, cancel_completes=False))
    # traj is 0.5 s -> deadline 5.75 s; each clock read jumps 20 s
    status, msg = _executor(client, clock=SteppingClock(20.0)).run(_traj(), 1.0)
    assert status == "failed" and "timed out" in msg and calls


def test_controller_error_code_fails_with_the_code():
    fut = DoneFuture(_wrapped(-4, "PATH_TOLERANCE_VIOLATED"))
    client = FakeClient(_handle(fut))
    status, msg = _executor(client).run(_traj(), 1.0)
    assert status == "failed" and "-4" in msg and "PATH_TOLERANCE" in msg


def test_success_far_from_endpoint_is_a_tracking_failure():
    msg_traj = _traj()
    fut = DoneFuture(_wrapped(FollowJointTrajectory.Result.SUCCESSFUL))
    client = FakeClient(_handle(fut))
    # arm still at the start: the "controller success without motion" lie
    start = list(msg_traj.points[0].positions)
    status, msg = _executor(client).run(
        msg_traj, 1.0, get_current_q=lambda: start, tracking_tolerance_rad=0.02
    )
    assert status == "failed"
    assert "TRACKING FAILURE" in msg and "never left the start" in msg


def test_success_partway_reports_possible_contact():
    msg_traj = _traj(step=0.2)
    fut = DoneFuture(_wrapped(FollowJointTrajectory.Result.SUCCESSFUL))
    client = FakeClient(_handle(fut))
    midway = list(msg_traj.points[2].positions)
    status, msg = _executor(client).run(
        msg_traj, 1.0, get_current_q=lambda: midway, tracking_tolerance_rad=0.02
    )
    assert status == "failed" and "partway" in msg


def test_success_at_endpoint_succeeds_and_scales_the_goal():
    msg_traj = _traj()
    fut = DoneFuture(_wrapped(FollowJointTrajectory.Result.SUCCESSFUL))
    client = FakeClient(_handle(fut))
    end = list(msg_traj.points[-1].positions)
    status, msg = _executor(client).run(msg_traj, 0.5, get_current_q=lambda: end)
    assert status == "succeeded"
    # the controller got the time-dilated trajectory: 0.5 s plan / 0.5 = 1.0 s
    sent = client.sent.trajectory.points[-1].time_from_start
    assert abs(sent.sec + sent.nanosec * 1e-9 - 1.0) < 1e-9
