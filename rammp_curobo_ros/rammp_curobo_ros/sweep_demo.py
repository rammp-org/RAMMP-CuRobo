"""sweep_demo — sweep left and right; plan around whatever the camera sees.

    ros2 launch rammp_curobo_ros sweep_demo.launch.py                  # dry run
    ros2 launch rammp_curobo_ros sweep_demo.launch.py execute:=true    # it moves

The arm ping-pongs between two tool poses. While a stroke is in flight a
watchdog asks the planner, at watchdog_hz, whether the REMAINING part of
that trajectory is still collision-free in the world the cameras node is
publishing. Nothing in the execution gate chain re-checks collision, so
this service call is the only thing standing between a trajectory planned
before an obstacle appeared and the obstacle.

Three outcomes, and the third is the one a hand actually triggers:

  clear    keep going.
  blocked  cancel (controller stops and holds), wait for the arm to be
           still, replan from there — cuRobo routes over or around.
  HOLD     the planner refuses outright with INVALID_START_STATE_WORLD_
           COLLISION: the obstacle overlaps the arm's own body, so there
           is no path to plan. Measured on this arm, that begins around
           15 cm. Stop and wait for it to clear. This is the protective
           stop, not a failure.

Dry run (execute:=false) never sends an execution goal: it plans a stroke
and polls the same watchdog, printing CLEAR/BLOCKED. Holding something
into the corridor and watching it flip to BLOCKED — then watching the
replan route around it — proves the whole chain without moving the arm.

Needs the arm bringup (for TF and joint states), the Orbbec, the cameras
node, and the planner node. Human on the e-stop for execute:=true.
"""

import signal
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToPose
from rammp_curobo_interfaces.srv import CheckTrajectory
from rammp_curobo_ros.conversions import msg_arrays

STILL_RAD_S = 0.02          # below this the arm counts as stopped


def traj_index(times, elapsed_s, speed_scale):
    """Which trajectory point the arm has reached after `elapsed_s` of wall
    clock. The executor dilates time by 1/speed_scale, so trajectory time
    advances at speed_scale x wall clock — get this backwards and the
    watchdog checks a span the arm has already driven through."""
    return int(np.searchsorted(times, elapsed_s * float(speed_scale)))


class SweepDemo(Node):
    def __init__(self):
        super().__init__("sweep_demo")
        cb = ReentrantCallbackGroup()
        p = self.declare_parameter
        self.pose_a = [float(v) for v in p("pose_a", [0.55, -0.30, 0.35]).value]
        self.pose_b = [float(v) for v in p("pose_b", [0.55, 0.30, 0.35]).value]
        # wrist flat, tool along +x. Keep both waypoints outside ~0.5 m
        # radius: tool_frame sits 0.12 m ahead of the flange, so a flat
        # wrist closer in folds the arm into itself and IK_FAILs.
        self.quat = [float(v) for v in p("orientation", [0.5, 0.5, 0.5, 0.5]).value]
        self.speed_scale = float(p("speed_scale", 0.25).value)
        self.watchdog_hz = float(p("watchdog_hz", 5.0).value)
        # Extra standoff on top of cuRobo's hard verdict. That verdict
        # flips at world_padding — MEASURED 0.020 m, not the
        # collision_activation_distance (0.03) you might expect: that knob
        # feeds the IK/trajopt COST terms and never reaches
        # check_constraints, so raising it buys nothing here. min_clearance
        # is measured against the UNPADDED boxes, so this is a real margin.
        # 0 = trust cuRobo alone, and only 0 is livelock-free: a margin
        # wider than a legitimate plan's own clearance would trip on every
        # fresh plan, so run() drops it for any stroke it would block
        # outright.
        self.margin = float(p("clearance_margin", 0.0).value)
        self.hold_retry_s = float(p("hold_retry_s", 0.5).value)
        self.settle_timeout = float(p("settle_timeout_s", 3.0).value)
        self.execute = bool(p("execute", False).value)
        ns = str(p("planner_ns", "/rammp_curobo").value)

        self.stop = False
        self._vel = None
        self._active = None
        self._last_report = ("", 0.0)

        self.create_subscription(
            JointState, "/joint_states", self._js_cb,
            qos_profile_sensor_data, callback_group=cb,
        )
        self.plan_cli = ActionClient(
            self, PlanToPose, ns + "/plan_to_pose", callback_group=cb
        )
        self.exec_cli = ActionClient(
            self, ExecuteTrajectory, ns + "/execute_trajectory", callback_group=cb
        )
        self.check_cli = self.create_client(
            CheckTrajectory, ns + "/check_trajectory", callback_group=cb
        )

    # ------------------------------------------------------------- plumbing
    def _js_cb(self, msg):
        if msg.velocity:
            self._vel = float(np.abs(np.asarray(msg.velocity, dtype=float)).max())

    def report(self, level, message):
        """Collapse repeats: a demo that cannot plan should say so once and
        then every 5 s, not twice a second for as long as it is wrong."""
        now = time.monotonic()
        if message == self._last_report[0] and now - self._last_report[1] < 5.0:
            return
        self._last_report = (message, now)
        getattr(self.get_logger(), level)(message)

    def request_stop(self):
        self.stop = True

    def _nap(self, seconds):
        """Sleep, but wake early on Ctrl+C."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop and rclpy.ok():
            time.sleep(0.01)

    def _await(self, future, timeout, honor_stop=True):
        """The executor spins on its own thread — just watch the future.

        honor_stop=False is for the one case where giving up early is
        unsafe: finding out whether an execution goal was accepted.
        """
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end or not rclpy.ok():
                return None
            if honor_stop and self.stop:
                return None
            time.sleep(0.005)
        return future.result()

    def _wait(self, ready_fn, timeout):
        """Poll in short slices so Ctrl+C is honoured while waiting.

        A single wait_for_server(30.0) ignores the stop flag, so launch
        escalates SIGINT to SIGTERM after 5 s and the node dies ugly while
        the planner is still warming up.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.stop and rclpy.ok():
            if ready_fn(timeout_sec=0.25):
                return True
        return False

    def wait_for_servers(self, timeout=120.0):
        # the planner spends ~20 s in cuRobo warmup before its servers
        # appear, and longer on a cold GPU — wait it out rather than
        # failing the demo on a slow start
        for name, fn in (
            ("plan_to_pose", self.plan_cli.wait_for_server),
            ("check_trajectory", self.check_cli.wait_for_service),
        ):
            if not self._wait(fn, timeout):
                if self.stop:
                    return False
                self.get_logger().error(
                    "%s not available — is the planner node running? "
                    "(check_trajectory is new: rebuild the interfaces)" % name
                )
                return False
        if self.execute and not self._wait(self.exec_cli.wait_for_server, 10.0):
            self.get_logger().error("execute_trajectory not available")
            return False
        return True

    # ---------------------------------------------------------------- pieces
    def plan(self, xyz):
        """(trajectory, message). trajectory is None on failure."""
        goal = PlanToPose.Goal()
        goal.target = Pose()
        goal.target.position.x, goal.target.position.y, goal.target.position.z = xyz
        (goal.target.orientation.x, goal.target.orientation.y,
         goal.target.orientation.z, goal.target.orientation.w) = self.quat
        send = self._await(self.plan_cli.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None, "plan goal not accepted"
        res = self._await(send.get_result_async(), 30.0)
        if res is None:
            return None, "plan timed out"
        if not res.result.success:
            return None, res.result.message
        return res.result.trajectory, res.result.message

    def check(self, traj, start_index):
        req = CheckTrajectory.Request()
        req.trajectory = traj
        req.start_index = int(start_index)
        return self._await(self.check_cli.call_async(req), 2.0)

    def tripped(self, verdict, use_margin=True):
        """Should the watchdog interrupt this stroke?"""
        if verdict is None:
            return False                      # a dropped check is not evidence
        return not verdict.collision_free or (
            use_margin and self.margin > 0.0 and verdict.min_clearance < self.margin
        )

    def wait_until_still(self):
        """Seconds from now until the arm stops, or None if it never does.

        This is the cancel-unwind cost — the one term in the reaction
        budget that cannot be measured off-hardware, so it is logged
        every time rather than assumed.
        """
        t0 = time.monotonic()
        end = t0 + self.settle_timeout
        while time.monotonic() < end and rclpy.ok():
            if self._vel is not None and self._vel < STILL_RAD_S:
                return time.monotonic() - t0
            time.sleep(0.01)
        return None

    # ----------------------------------------------------------------- modes
    def watch(self, traj):
        """Dry run: poll the watchdog over one stroke's worth of time."""
        _pos, _vel, times = msg_arrays(traj)
        end = time.monotonic() + float(times[-1]) / self.speed_scale
        last = None
        while time.monotonic() < end and not self.stop and rclpy.ok():
            v = self.check(traj, 0)
            if v is not None:
                state = "CLEAR" if not self.tripped(v) else "BLOCKED"
                if state != last:
                    self.get_logger().info("dry run: %s — %s" % (state, v.message))
                    last = state
                if state == "BLOCKED":
                    return
            self._nap(1.0 / self.watchdog_hz)

    def drive(self, traj, use_margin=True):
        """Execute one stroke under the watchdog. -> arrived|blocked|<error>."""
        _pos, _vel, times = msg_arrays(traj)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        goal.speed_scale = self.speed_scale
        sent = self.exec_cli.send_goal_async(goal)
        send = self._await(sent, 10.0)
        if send is None:
            # We stopped waiting — but the server may have accepted it and
            # the arm may already be moving. Never walk away from a goal
            # whose fate we do not know.
            send = self._await(sent, 3.0, honor_stop=False)
            if send is not None and send.accepted:
                self.get_logger().warning(
                    "execution goal was accepted late — cancelling it"
                )
                self._active = send
                send.cancel_goal_async()
                self.wait_until_still()
                self._active = None
            return "gave up waiting for the execution goal"
        if not send.accepted:
            return "execution goal rejected (is execute:=true on the planner?)"
        self._active = send
        result_future = send.get_result_async()
        t0 = time.monotonic()
        try:
            while not result_future.done():
                if self.stop or not rclpy.ok():
                    return self._cancel(send, result_future, "stopped")
                idx = traj_index(times, time.monotonic() - t0, self.speed_scale)
                if self.tripped(self.check(traj, idx), use_margin):
                    self.get_logger().warning("watchdog: path blocked ahead")
                    return self._cancel(send, result_future, "blocked")
                self._nap(1.0 / self.watchdog_hz)
            res = result_future.result()
            if res is not None and res.result.success:
                return "arrived"
            return res.result.message if res is not None else "no execution result"
        finally:
            self._active = None

    def _cancel(self, handle, result_future, outcome):
        handle.cancel_goal_async()
        self._await(result_future, 5.0)
        settle = self.wait_until_still()
        self.get_logger().info(
            "cancelled; arm still after %s"
            % ("%.2f s" % settle if settle is not None else
               "NEVER (>%.1f s) — check the controller" % self.settle_timeout)
        )
        return outcome

    # ------------------------------------------------------------------ loop
    def run(self):
        if not self.wait_for_servers():
            return
        self.get_logger().info(
            "sweeping %s <-> %s at speed %.2f, watchdog %.1f Hz%s"
            % (np.round(self.pose_a, 2), np.round(self.pose_b, 2),
               self.speed_scale, self.watchdog_hz,
               "" if self.execute else " (DRY RUN — nothing will move)")
        )
        targets = [self.pose_a, self.pose_b]
        i = 0
        while rclpy.ok() and not self.stop:
            traj, message = self.plan(targets[i % 2])
            if traj is None:
                if "INVALID_START" in message:
                    self.report(
                        "warning",
                        "HOLD — something is too close to plan around; "
                        "waiting for it to clear",
                    )
                else:
                    self.report("error", "plan failed: %s" % message)
                self._nap(self.hold_retry_s)
                continue
            if not self.execute:
                self.watch(traj)
                i += 1
                continue
            # A margin wider than what this plan actually achieves would
            # trip the instant the stroke starts, cancel, replan, and trip
            # again — forever. Check the fresh plan against its own margin
            # and drop the margin rather than livelock.
            use_margin = True
            if self.margin > 0.0 and self.tripped(self.check(traj, 0)):
                self.report(
                    "warning",
                    "clearance_margin %.3f m is wider than this plan's own "
                    "clearance — ignoring it for this stroke" % self.margin,
                )
                use_margin = False
            outcome = self.drive(traj, use_margin)
            if outcome == "arrived":
                i += 1
            elif outcome == "blocked":
                self.get_logger().info("replanning around it")
            elif outcome == "stopped":
                break
            else:
                self.get_logger().error("execution: %s" % outcome)
                self._nap(1.0)

    def shutdown(self):
        handle = self._active
        if handle is not None:
            self.get_logger().info("stopping the arm")
            handle.cancel_goal_async()
            self.wait_until_still()


def main(args=None):
    from rclpy.signals import SignalHandlerOptions

    # own the SIGINT so an in-flight stroke gets cancelled (arm stops and
    # holds) instead of the context dying under it
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SweepDemo()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    signal.signal(signal.SIGINT, lambda *_: node.request_stop())
    try:
        node.run()
    finally:
        try:
            node.shutdown()
            executor.shutdown()
            # Join BEFORE destroying. Tearing the node down while the
            # executor thread is still in spin() aborts the process
            # ("terminate called without an active exception", SIGABRT) —
            # reproduced 1 run in 3 with execute:=true. If the thread will
            # not stop, leave the node alone and just exit: a tidy
            # destroy_node is not worth a crash.
            spin.join(timeout=5.0)
            if not spin.is_alive():
                node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
