"""Safety-gated trajectory execution via the kinova_arm_ros2 driver.

The arm side is rammp-org/kinova_arm_ros2's `kinova_arm_node`: a thin ROS 2
shell over the kinova-gen3-driver 1 kHz real-time core (this replaced the
ros2_kortex JTC stack 2026-08-14). Execution goes through its
`/execute_joint_trajectory` action; a software abort is `cancel goal` — the
driver's Supervisor resets its trajectory executor and the position mode
keeps commanding its last reference, so the arm stops and holds.

Every goal passes these gates before anything reaches the driver:
  * the node's `execute` parameter is true (dry-run is the default),
  * joint names match the configured joints exactly,
  * finite positions, position limits, velocity limits, monotonic timing,
  * step-to-step continuity consistent with each interval's dt,
  * the arm's CURRENT position matches the trajectory start (a stale plan
    means the catch-up sweep would be unplanned, collision-unchecked motion),
  * speed scale clamped to (0, max_speed_scale] and applied by exact time
    dilation — execution can only ever be slower than planned.

Driver facts this module encodes (verified in the core source, 2026-08-14):
  * the driver consumes positions + time_from_start ONLY (velocities are
    dropped; it lerps between waypoints at 250 Hz — dense cuRobo plans make
    that exact enough);
  * the commanded reference is rate-limited to the driver's max_ref_speed
    (0.5 rad/s default, deliberately conservative) — a plan whose scaled
    peak velocity exceeds it LAGS its timestamps, and the driver completes
    on the TIMER with no goal-tolerance check, so the wrap-aware arrival
    check below is the only real goal gate. Keep speed_scale low enough
    that peak velocity stays under the cap (warned about below);
  * measured q is wrapped to (-pi, pi] while the driver's path-tolerance
    guard compares RAW |q_meas - q_desired| (not wrap-aware) — near ±pi an
    enabled guard false-aborts, and joint_3 LIVES at +pi in the home elbow
    family. The guard is therefore armed only on the bounded joints
    (joint_2/4/6), and only when the plan stays under the speed cap.
"""

import threading

import numpy as np
from rclpy.action import ActionClient

from rammp_curobo.geometry import ang_diff
from rammp_curobo.validate import CONTINUITY_SLACK
from rammp_curobo_ros.conversions import msg_arrays, scaled_msg

# kinova_arm_interfaces (the driver's rosidl package) is imported lazily
# inside build_driver_goal / TrajectoryExecutor: a planning-only deployment
# (the Docker image) never constructs the executor and must not need the
# driver workspace installed.

# kinova-gen3-driver JointPositionParams.max_ref_speed default: commanded
# references cannot move faster than this, whatever the trajectory says.
DRIVER_REF_SPEED_CAP = 0.5
# Divergence guard armed on the bounded joints when tracking is feasible.
PATH_TOLERANCE_RAD = 0.30
# Gen3 continuous joints (joint_1/3/5/7): reported wrapped to (-pi, pi],
# which the driver's raw-difference path guard cannot handle near ±pi.
CONTINUOUS_JOINTS = (True, False, True, False, True, False, True)

_DRIVER_ERROR_NAMES = {
    -1: "INVALID_GOAL",
    -4: "PATH_TOLERANCE_VIOLATED",
    -5: "GOAL_TOLERANCE_VIOLATED",
    -6: "PREEMPTED",
}


def await_future(future, timeout_s):
    """Block on an rclpy future from inside a reentrant callback."""
    ev = threading.Event()
    future.add_done_callback(lambda _f: ev.set())
    if not ev.wait(timeout_s):
        return None
    return future.result()


def validate_goal_msg(
    msg, joint_names, current_q, position_limits, velocity_limits, start_tolerance_rad
):
    """All the reasons NOT to execute this trajectory (empty list = go)."""
    problems = []
    if not msg.points:
        return ["trajectory has no points"]
    if list(msg.joint_names) != list(joint_names):
        return [
            "trajectory joints %s != driver joints %s"
            % (list(msg.joint_names), list(joint_names))
        ]
    pos, vel, times = msg_arrays(msg)
    if pos.shape[1] != len(joint_names):
        return ["%d position columns for %d joints" % (pos.shape[1], len(joint_names))]
    if not np.isfinite(pos).all():
        problems.append("non-finite positions")

    lower = np.asarray(position_limits[0]) - 1e-3
    upper = np.asarray(position_limits[1]) + 1e-3
    if (pos < lower).any() or (pos > upper).any():
        j = int(np.argmax(((pos < lower) | (pos > upper)).any(axis=0)))
        problems.append("%s exceeds position limits" % joint_names[j])

    vmax = np.asarray(velocity_limits)
    if vel is not None and np.abs(vel).max(axis=0).max() > 0:
        over = np.abs(vel).max(axis=0) > vmax * 1.01
        if over.any():
            j = int(np.argmax(over))
            problems.append(
                "%s velocity exceeds limit %.2f rad/s" % (joint_names[j], vmax[j])
            )

    dts = np.diff(times, prepend=0.0)
    if (dts <= 0).any():
        problems.append("non-monotonic time_from_start")
    elif pos.shape[0] > 1:
        steps = np.abs(np.diff(pos, axis=0))
        allowed = CONTINUITY_SLACK * vmax[None, :] * dts[1:, None]
        if (steps > allowed).any():
            k = int(np.argmax((steps > allowed).any(axis=1)))
            problems.append(
                "discontinuity between points %d->%d — stale-buffer "
                "failure mode, refusing" % (k, k + 1)
            )

    if current_q is None:
        problems.append("no current joint state — is the arm driver running?")
    else:
        err = float(max(abs(ang_diff(a, b)) for a, b in zip(pos[0], current_q)))
        if err > start_tolerance_rad:
            problems.append(
                "arm is %.3f rad from the trajectory start (tolerance %.3f) "
                "— stale plan; re-plan from the current state"
                % (err, start_tolerance_rad)
            )
    return problems


def _peak_velocity(msg):
    """Max joint speed of the (scaled) plan, from POSITIONS as well as the
    claimed velocities — position steps can't be spoofed by a zeroed
    velocity array, and the guard-arming decision keys off this."""
    pos, vel, times = msg_arrays(msg)
    peak = 0.0
    if vel is not None and vel.size:
        peak = float(np.abs(vel).max())
    if pos.shape[0] > 1:
        dts = np.maximum(np.diff(times), 1e-9)
        peak = max(peak, float((np.abs(np.diff(pos, axis=0)) / dts[:, None]).max()))
    return peak


def build_driver_goal(scaled, log=None):
    """A driver goal for `scaled`, with the path guard armed only when safe.

    The guard is disabled entirely when the plan's peak speed exceeds the
    driver's reference-speed cap (lag, not contact, would trip it). On the
    continuous joints it is always disabled — the driver's guard is not
    wrap-aware. Raises ValueError unless joint_names is exactly
    joint_1..joint_7 in order: the driver is POSITIONAL (it ignores
    joint_names — message_mapping copies by index), so any other order
    would execute transposed joints.
    """
    from control_msgs.msg import JointTolerance
    from kinova_arm_interfaces.action import ExecuteJointTrajectory

    expected = ["joint_%d" % i for i in range(1, 8)]
    if list(scaled.joint_names) != expected:
        raise ValueError(
            "driver contract is positional joint_1..joint_7; got %s"
            % list(scaled.joint_names)
        )
    goal = ExecuteJointTrajectory.Goal()
    goal.trajectory = scaled
    goal.control_mode = 0  # POSITION
    # LATEST_WINS: this node serializes intentional executions itself, so
    # preemption only matters against a ZOMBIE goal (a send/deadline
    # timeout that left one live). QUEUE would silently line a fresh goal
    # up behind it — starting motion later, after its start-state check
    # went stale. Replacement is the safe failure mode.
    goal.preemption = 1
    goal.sender_id = "rammp_curobo"
    peak = _peak_velocity(scaled)
    if peak <= 0.9 * DRIVER_REF_SPEED_CAP:
        for wrapped in CONTINUOUS_JOINTS:
            tol = JointTolerance()
            tol.position = -1.0 if wrapped else PATH_TOLERANCE_RAD
            goal.path_tolerance.append(tol)
    elif log is not None:
        log.warning(
            "path guard disabled: plan peak speed %.2f rad/s exceeds the "
            "driver's %.2f rad/s reference cap — the arm will lag the "
            "timestamps; lower the speed scale for tracked motion"
            % (peak, DRIVER_REF_SPEED_CAP)
        )
    # goal_tolerance stays empty (= disabled): the driver defines the field
    # but does not check it yet — the arrival check in run() is the goal gate.
    return goal


class TrajectoryExecutor:
    """Owns the driver action client and the execution state machine."""

    def __init__(self, node, arm_action, callback_group):
        from kinova_arm_interfaces.action import ExecuteJointTrajectory

        self._node = node
        self._action_name = arm_action
        self._client = ActionClient(
            node,
            ExecuteJointTrajectory,
            arm_action,
            callback_group=callback_group,
        )
        self._log = node.get_logger()

    def server_ready(self, timeout_s=2.0):
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def run(
        self,
        msg,
        speed_scale,
        goal_handle=None,
        feedback_cb=None,
        get_current_q=None,
        tracking_tolerance_rad=0.08,
    ):
        """Execute `msg` scaled by `speed_scale`. Blocks until terminal.

        Returns (status, message) with status in {'succeeded', 'canceled',
        'failed'}. goal_handle (ours, not the driver's) is polled for
        cancel requests; cancelling maps to driver-goal cancel — the arm
        stops and holds its position.
        """
        scaled = scaled_msg(msg, speed_scale)
        duration = (
            scaled.points[-1].time_from_start.sec
            + scaled.points[-1].time_from_start.nanosec * 1e-9
        )
        if not self.server_ready():
            return "failed", (
                "driver action server %s not available — is kinova_arm_node "
                "running?" % self._action_name
            )

        try:
            goal = build_driver_goal(scaled, log=self._log)
        except ValueError as exc:
            return "failed", str(exc)
        send = await_future(self._client.send_goal_async(goal), 5.0)
        if send is None:
            return "failed", (
                "driver did not answer the goal in 5 s — the goal may still "
                "arrive late and execute; verify the driver before retrying"
            )
        if not send.accepted:
            return "failed", "driver rejected the trajectory goal"
        self._log.info(
            "Executing: %d points over %.1f s (scale %.2f)"
            % (len(scaled.points), duration, speed_scale)
        )

        result_future = send.get_result_async()
        done = threading.Event()
        result_future.add_done_callback(lambda _f: done.set())

        clock = self._node.get_clock()
        t0 = clock.now()
        # generous deadline: scheduled time + margin; the driver is the
        # authority on completion, this is only a hang backstop
        deadline = duration * 1.5 + 5.0
        while not done.wait(0.1):
            elapsed = (clock.now() - t0).nanoseconds * 1e-9
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._log.warning(
                    "Cancel requested — cancelling driver goal "
                    "(arm stops and holds)."
                )
                cancel_ack = await_future(send.cancel_goal_async(), 2.0)
                wrapped = await_future(result_future, 5.0)
                if wrapped is not None and int(wrapped.result.error_code) == 0:
                    # the driver's timer completed the goal before the
                    # cancel landed — the FULL trajectory executed
                    return "canceled", (
                        "cancel arrived after completion — the full "
                        "trajectory executed; the driver holds the endpoint"
                    )
                if cancel_ack is None and wrapped is None:
                    return "failed", (
                        "cancel UNCONFIRMED — the driver answered neither "
                        "the cancel nor the result; VERIFY THE ARM STOPPED "
                        "before doing anything else"
                    )
                return "canceled", "canceled; the driver holds position"
            if feedback_cb is not None:
                feedback_cb(min(1.0, elapsed / max(duration, 1e-6)))
            if elapsed > deadline:
                self._log.error(
                    "Execution timed out (%.1f s > %.1f s) — "
                    "cancelling driver goal." % (elapsed, deadline)
                )
                cancel_ack = await_future(send.cancel_goal_async(), 2.0)
                wrapped = await_future(result_future, 3.0)
                if cancel_ack is None and wrapped is None:
                    return "failed", (
                        "execution timed out and the cancel is UNCONFIRMED "
                        "— driver unresponsive; VERIFY THE ARM STOPPED"
                    )
                return "failed", "execution timed out; driver goal cancelled"

        wrapped = result_future.result()
        code = int(wrapped.result.error_code)
        if code != 0:
            return "failed", (
                "driver error %d (%s): %s"
                % (
                    code,
                    _DRIVER_ERROR_NAMES.get(code, "?"),
                    wrapped.result.error_string or "no detail",
                )
            )

        # Arrival check: the driver completes on the TIMER (no goal-
        # tolerance check in the core yet), so success with the arm far
        # from the endpoint means lag, physical contact, or saturation.
        # Wrap-aware: at home, joint_3 sits exactly on the +/-pi boundary
        # and its REPORTED angle can flip by 2*pi mid-move — a perfectly
        # tracked first hardware run read as "settled 6.283 rad away".
        if get_current_q is not None:
            q = get_current_q()
            if q is not None:
                target = scaled.points[-1].positions
                errs = [abs(ang_diff(a, b)) for a, b in zip(target, q)]
                j = int(np.argmax(errs))
                err = float(errs[j])
                if err > tracking_tolerance_rad:
                    start = scaled.points[0].positions
                    from_start = float(
                        max(abs(ang_diff(a, b)) for a, b in zip(start, q))
                    )
                    hint = (
                        "the arm never left the start — the driver reported "
                        "success without motion. Check the kinova_arm_node "
                        "terminal for faults and restart it if needed"
                        if from_start < 0.05
                        else "stopped partway — physical contact, or the "
                        "plan out-ran the driver's %.2f rad/s reference cap "
                        "(lower the speed scale)" % DRIVER_REF_SPEED_CAP
                    )
                    return "failed", (
                        "TRACKING FAILURE: %s settled %.3f rad from the "
                        "endpoint; %s" % (scaled.joint_names[j], err, hint)
                    )
        return "succeeded", "trajectory executed"
