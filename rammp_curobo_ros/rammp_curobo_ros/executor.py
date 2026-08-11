"""Safety-gated trajectory execution via the FollowJointTrajectory ACTION.

Deliberate departure from RAMMP-Kinova (which streams on the JTC topic):
the action gives a goal handle — so a software abort is `cancel goal`, and
the controller stops and holds instead of chasing the rest of a stale
trajectory. ros2_kortex's real bringup and the MuJoCo sim expose the same
action under the same controller name, so this path is identical in both.

Every goal passes these gates before anything reaches the controller:
  * the node's `execute` parameter is true (dry-run is the default),
  * joint names match the configured controller joints exactly,
  * finite positions, position limits, velocity limits, monotonic timing,
  * step-to-step continuity consistent with each interval's dt,
  * the arm's CURRENT position matches the trajectory start (a stale plan
    means the catch-up sweep would be unplanned, collision-unchecked motion),
  * speed scale clamped to (0, max_speed_scale] and applied by exact time
    dilation — execution can only ever be slower than planned.
"""

import threading

import numpy as np
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

from rammp_curobo_ros.conversions import msg_arrays, scaled_msg

CONTINUITY_SLACK = 3.0


def await_future(future, timeout_s):
    """Block on an rclpy future from inside a reentrant callback."""
    ev = threading.Event()
    future.add_done_callback(lambda _f: ev.set())
    if not ev.wait(timeout_s):
        return None
    return future.result()


def validate_goal_msg(msg, joint_names, current_q, position_limits,
                      velocity_limits, start_tolerance_rad):
    """All the reasons NOT to execute this trajectory (empty list = go)."""
    problems = []
    if not msg.points:
        return ['trajectory has no points']
    if list(msg.joint_names) != list(joint_names):
        return ['trajectory joints %s != controller joints %s'
                % (list(msg.joint_names), list(joint_names))]
    pos, vel, times = msg_arrays(msg)
    if pos.shape[1] != len(joint_names):
        return ['%d position columns for %d joints'
                % (pos.shape[1], len(joint_names))]
    if not np.isfinite(pos).all():
        problems.append('non-finite positions')

    lower = np.asarray(position_limits[0]) - 1e-3
    upper = np.asarray(position_limits[1]) + 1e-3
    if (pos < lower).any() or (pos > upper).any():
        j = int(np.argmax(((pos < lower) | (pos > upper)).any(axis=0)))
        problems.append('%s exceeds position limits' % joint_names[j])

    vmax = np.asarray(velocity_limits)
    if vel is not None and np.abs(vel).max(axis=0).max() > 0:
        over = np.abs(vel).max(axis=0) > vmax * 1.01
        if over.any():
            j = int(np.argmax(over))
            problems.append('%s velocity exceeds limit %.2f rad/s'
                            % (joint_names[j], vmax[j]))

    dts = np.diff(times, prepend=0.0)
    if (dts <= 0).any():
        problems.append('non-monotonic time_from_start')
    elif pos.shape[0] > 1:
        steps = np.abs(np.diff(pos, axis=0))
        allowed = CONTINUITY_SLACK * vmax[None, :] * dts[1:, None]
        if (steps > allowed).any():
            k = int(np.argmax((steps > allowed).any(axis=1)))
            problems.append(
                'discontinuity between points %d->%d — stale-buffer '
                'failure mode, refusing' % (k, k + 1))

    if current_q is None:
        problems.append('no current joint state — is the arm bringup running?')
    else:
        err = float(np.abs(pos[0] - np.asarray(current_q)).max())
        if err > start_tolerance_rad:
            problems.append(
                'arm is %.3f rad from the trajectory start (tolerance %.3f) '
                '— stale plan; re-plan from the current state'
                % (err, start_tolerance_rad))
    return problems


class TrajectoryExecutor:
    """Owns the FollowJointTrajectory client and the execution state machine."""

    def __init__(self, node, controller_action, callback_group):
        self._node = node
        self._client = ActionClient(
            node, FollowJointTrajectory, controller_action,
            callback_group=callback_group)
        self._log = node.get_logger()

    def server_ready(self, timeout_s=2.0):
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def run(self, msg, speed_scale, goal_handle=None, feedback_cb=None,
            get_current_q=None, tracking_tolerance_rad=0.08):
        """Execute `msg` scaled by `speed_scale`. Blocks until terminal.

        Returns (status, message) with status in {'succeeded', 'canceled',
        'failed'}. goal_handle (ours, not the controller's) is polled for
        cancel requests; cancelling maps to controller-goal cancel — the
        JTC stops and holds position.
        """
        scaled = scaled_msg(msg, speed_scale)
        duration = (scaled.points[-1].time_from_start.sec
                    + scaled.points[-1].time_from_start.nanosec * 1e-9)
        if not self.server_ready():
            return 'failed', ('controller action server %s not available'
                              % self._client._action_name)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = scaled
        send = await_future(self._client.send_goal_async(goal), 5.0)
        if send is None:
            return 'failed', 'controller did not answer the goal in 5 s'
        if not send.accepted:
            return 'failed', 'controller rejected the trajectory goal'
        self._log.info('Executing: %d points over %.1f s (scale %.2f)'
                       % (len(scaled.points), duration, speed_scale))

        result_future = send.get_result_async()
        done = threading.Event()
        result_future.add_done_callback(lambda _f: done.set())

        clock = self._node.get_clock()
        t0 = clock.now()
        # generous deadline: scheduled time + margin; the controller is the
        # authority on completion, this is only a hang backstop
        deadline = duration * 1.5 + 5.0
        while not done.wait(0.1):
            elapsed = (clock.now() - t0).nanoseconds * 1e-9
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._log.warning('Cancel requested — cancelling controller '
                                  'goal (arm stops and holds).')
                await_future(send.cancel_goal_async(), 2.0)
                await_future(result_future, 5.0)
                return 'canceled', 'canceled; controller holds position'
            if feedback_cb is not None:
                feedback_cb(min(1.0, elapsed / max(duration, 1e-6)))
            if elapsed > deadline:
                self._log.error('Execution timed out (%.1f s > %.1f s) — '
                                'cancelling controller goal.'
                                % (elapsed, deadline))
                await_future(send.cancel_goal_async(), 2.0)
                return 'failed', 'execution timed out; controller cancelled'

        wrapped = result_future.result()
        code = wrapped.result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            return 'failed', ('controller error %d: %s'
                              % (code, wrapped.result.error_string))

        # Arrival check: a "successful" goal that settled far from the
        # commanded endpoint means physical contact or saturation.
        if get_current_q is not None:
            q = get_current_q()
            if q is not None:
                target = np.asarray(scaled.points[-1].positions)
                err = float(np.abs(target - np.asarray(q)).max())
                if err > tracking_tolerance_rad:
                    return 'failed', (
                        'TRACKING FAILURE: settled %.3f rad from the '
                        'endpoint — physical contact or controller '
                        'saturation' % err)
        return 'succeeded', 'trajectory executed'
