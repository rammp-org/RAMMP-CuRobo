"""The rammp_curobo ROS 2 node: plan/execute action servers.

    ros2 launch rammp_curobo_ros planner.launch.py                 # dry-run node
    ros2 launch rammp_curobo_ros planner.launch.py execute:=true   # can move the arm

Actions (node name rammp_curobo):
    /rammp_curobo/plan_to_pose        rammp_curobo_interfaces/PlanToPose
    /rammp_curobo/plan_to_joints      rammp_curobo_interfaces/PlanToJoints
    /rammp_curobo/execute_trajectory  rammp_curobo_interfaces/ExecuteTrajectory
Services:
    /rammp_curobo/set_world           rammp_curobo_interfaces/srv/SetWorld
    /rammp_curobo/open_gripper, /rammp_curobo/close_gripper (std_srvs/Trigger)

Planning NEVER moves the arm — results carry the trajectory for inspection.
Execution is a separate action, gated by the `execute` parameter (default
false), start-state matching, and limit re-validation (executor.py).
Identical against the MuJoCo sim and the real ros2_kortex bringup: both
expose the same controller names (do not run both — they share
/controller_manager and the same physical arm).
"""

import threading
import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld, UpdateWorldBoxes
from rammp_curobo_ros.conversions import trajectory_to_msg
from rammp_curobo_ros.executor import (
    TrajectoryExecutor,
    await_future,
    validate_goal_msg,
)


class RammpCuroboNode(Node):
    def __init__(self):
        super().__init__("rammp_curobo")
        self._cb = ReentrantCallbackGroup()

        self.config = self.declare_parameter("config", "gen3.yaml").value
        self.world = self.declare_parameter("world", "").value
        self.joint_states_topic = self.declare_parameter(
            "joint_states_topic", "/joint_states"
        ).value
        self.controller_action = self.declare_parameter(
            "controller_action", "/joint_trajectory_controller/follow_joint_trajectory"
        ).value
        # Master enable for real motion. False = dry-run node: planning
        # works, execution goals are refused with instructions.
        self.declare_parameter("execute", False)
        # 0.0 = take the library config's execution.speed_scale (0.25).
        self.speed_scale = float(self.declare_parameter("speed_scale", 0.0).value)
        self.start_tolerance = float(
            self.declare_parameter("start_tolerance_rad", 0.05).value
        )
        self.tracking_tolerance = float(
            self.declare_parameter("tracking_tolerance_rad", 0.08).value
        )
        self.gripper_enabled = bool(
            self.declare_parameter("gripper_enabled", True).value
        )
        gripper_action = self.declare_parameter(
            "gripper_action", "/robotiq_gripper_controller/gripper_cmd"
        ).value
        self.gripper_open_pos = float(self.declare_parameter("gripper_open", 0.0).value)
        self.gripper_closed_pos = float(
            self.declare_parameter("gripper_closed", 0.8).value
        )

        # Refuse to double-serve. Two planner nodes on the same action names
        # answer every goal twice and race each other's controller goals —
        # on hardware this produced phantom TRACKING FAILUREs and an aborted
        # scan (2026-08-13). Fail loudly instead.
        time.sleep(1.0)  # let discovery see an already-running peer
        peers = [
            name
            for name, _ns in self.get_node_names_and_namespaces()
            if name == self.get_name()
        ]
        if len(peers) > 1:
            raise SystemExit(
                "another '%s' node is already running — two planner nodes "
                "race each other's goals. Stop the other one first "
                "(ros2 node list)." % self.get_name()
            )

        self._state_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._exec_lock = threading.Lock()
        self._joint_msg = None
        self._joint_msg_time = None

        # BEST-EFFORT, not the default reliable depth-10. Joint state is sensor
        # data: kinova_arm_ros2 publishes /joint_states with SensorDataQoS
        # (best-effort), and a RELIABLE subscriber is QoS-incompatible with a
        # BEST_EFFORT publisher, so it receives nothing at all:
        #   New publisher discovered on topic '/joint_states', offering
        #   incompatible QoS. Last incompatible policy: RELIABILITY
        # The visible symptom is every plan failing with
        #   'no fresh joint state on /joint_states'
        # even though the driver is plainly publishing it.
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_state_cb,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )

        # Heavy GPU init BEFORE the servers exist: a goal sent during the
        # (possibly minutes-long) warmup must not silently queue and then
        # move the arm long after the client gave up (RAMMP-Kinova lesson).
        self.get_logger().info("Loading cuRobo planner (%s)..." % self.config)
        from rammp_curobo import CuRoboPlanner

        self.planner = CuRoboPlanner.from_config(self.config)
        if self.world:
            self.planner.update_world(self.world)
            self.get_logger().info("World overridden: %s" % self.world)
        if self.speed_scale == 0.0:
            self.speed_scale = float(self.planner.execution["speed_scale"])
        self.max_speed_scale = float(self.planner.execution["max_speed_scale"])
        lim = self.planner.joint_limits()
        self._pos_limits = lim["position"]
        self._vel_limits = lim["velocity"]

        self.executor_helper = TrajectoryExecutor(
            self, self.controller_action, self._cb
        )
        # Servoing-recovery clients (see _try_servoing_recovery): transient
        # arm faults at motion onset knock the Gen3 out of low-level
        # servoing; recovery = clear faults, then bounce the JTC so its
        # hold state resyncs to the true joint positions and the driver
        # re-enters low-level mode via its controller-switch path.
        from controller_manager_msgs.srv import SwitchController
        from example_interfaces.srv import Trigger as ExampleTrigger

        self._reset_fault_type = ExampleTrigger
        self._reset_fault = self.create_client(
            ExampleTrigger, "/fault_controller/reset_fault", callback_group=self._cb
        )
        self._switch_type = SwitchController
        self._switch_ctrl = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
            callback_group=self._cb,
        )
        self._gripper = (
            ActionClient(self, GripperCommand, gripper_action, callback_group=self._cb)
            if self.gripper_enabled
            else None
        )

        ActionServer(
            self,
            PlanToPose,
            "~/plan_to_pose",
            self._plan_to_pose_cb,
            callback_group=self._cb,
        )
        ActionServer(
            self,
            PlanToJoints,
            "~/plan_to_joints",
            self._plan_to_joints_cb,
            callback_group=self._cb,
        )
        ActionServer(
            self,
            ExecuteTrajectory,
            "~/execute_trajectory",
            self._execute_cb,
            callback_group=self._cb,
            cancel_callback=lambda _req: CancelResponse.ACCEPT,
        )
        self.create_service(
            SetWorld, "~/set_world", self._set_world_cb, callback_group=self._cb
        )
        self.create_service(
            UpdateWorldBoxes,
            "~/update_world_boxes",
            self._update_world_boxes_cb,
            callback_group=self._cb,
        )
        if self.gripper_enabled:
            self.create_service(
                Trigger,
                "~/open_gripper",
                self._open_gripper_cb,
                callback_group=self._cb,
            )
            self.create_service(
                Trigger,
                "~/close_gripper",
                self._close_gripper_cb,
                callback_group=self._cb,
            )

        # Real-arm misconfiguration tripwire: without sim time this node is
        # presumably talking to the REAL Gen3, and the sim kitchen's
        # obstacles do not exist on the real bench — cuRobo would dodge
        # phantom furniture and happily route through real objects.
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        world_name = self.world or self.planner.world_name
        if not use_sim_time and "sim" in world_name:
            self.get_logger().warning(
                "=== Planning against the SIM world (%s) WITHOUT sim time — "
                "if this node is driving the real arm, relaunch with "
                "world:=world_real_bench.yaml (MEASURED first; see "
                "docs/HARDWARE_BRINGUP.md). ===" % world_name
            )

        exec_on = bool(self.get_parameter("execute").value)
        self.get_logger().info(
            "rammp_curobo ready — execute=%s, speed_scale=%.2f, controller=%s"
            % (exec_on, self.speed_scale, self.controller_action)
        )
        if not exec_on:
            self.get_logger().info(
                "DRY-RUN mode: execution goals will be refused. Relaunch "
                "with execute:=true to allow motion."
            )

    # ------------------------------------------------------------------- state
    def _joint_state_cb(self, msg):
        with self._state_lock:
            self._joint_msg = msg
            self._joint_msg_time = time.monotonic()

    def current_q(self, max_age_s=2.0):
        """Latest joint positions in configured order, or None if missing
        or stale (a stale state must never gate-pass an execution)."""
        with self._state_lock:
            msg, stamp = self._joint_msg, self._joint_msg_time
        if msg is None or time.monotonic() - stamp > max_age_s:
            return None
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            return [float(msg.position[idx[n]]) for n in self.planner.joint_names]
        except (KeyError, IndexError):
            return None

    def _joint_state_snapshot(self):
        with self._state_lock:
            return self._joint_msg

    # ---------------------------------------------------------------- planning
    def _plan_to_pose_cb(self, goal_handle):
        result = PlanToPose.Result()
        req = goal_handle.request.target
        pos = [req.position.x, req.position.y, req.position.z]
        quat = [
            req.orientation.x,
            req.orientation.y,
            req.orientation.z,
            req.orientation.w,
        ]
        res = self._plan(
            lambda q: self.planner.plan_to_pose(pos, quat, start=q),
            start_override=list(goal_handle.request.start_joints) or None,
        )
        return self._finish_plan(goal_handle, result, res)

    def _plan_to_joints_cb(self, goal_handle):
        result = PlanToJoints.Result()
        target = list(goal_handle.request.target_joints)
        if len(target) != len(self.planner.joint_names):
            result.success = False
            result.message = "%d target joints for %d-joint arm" % (
                len(target),
                len(self.planner.joint_names),
            )
            goal_handle.abort()
            return result
        res = self._plan(
            lambda q: self.planner.plan_to_joints(target, start=q),
            start_override=list(goal_handle.request.start_joints) or None,
        )
        out = self._finish_plan(goal_handle, result, res)
        # _plan returns None (busy) or False (no joint state) as sentinels —
        # only a real PlanResult carries goal_mismatch_rad.
        mismatch = getattr(res, "goal_mismatch_rad", None)
        if mismatch is not None:
            out.goal_mismatch_rad = float(mismatch)
        return out

    def _plan(self, plan_fn, start_override=None):
        """One library planning call — from the live state, or from an
        explicit start (chained pre-planning of multi-segment tours).
        The EXECUTION gate still checks every trajectory against the live
        arm, so a pre-planned segment can only run once the arm is there.
        """
        # A 2 Hz world updater briefly holds this lock; wait a beat instead
        # of bouncing the plan (concurrent PLANS still refuse — one at a time).
        if not self._plan_lock.acquire(timeout=0.5):
            return None
        try:
            if start_override is not None:
                if len(start_override) != len(self.planner.joint_names):
                    return False
                q = [float(v) for v in start_override]
            else:
                q = self.current_q()
                if q is None:
                    return False
            self.get_logger().info("Planning...")
            return plan_fn(q)
        finally:
            self._plan_lock.release()

    def _finish_plan(self, goal_handle, result, res):
        if res is None:
            result.success = False
            result.message = "planner busy — one plan at a time"
            goal_handle.abort()
            return result
        if res is False:
            result.success = False
            result.message = (
                "no fresh joint state on %s — is the arm/sim "
                "bringup running?" % self.joint_states_topic
            )
            goal_handle.abort()
            return result
        result.success = bool(res.success)
        result.planning_time = float(res.timing)
        if res.success:
            result.message = "planned %d points, %.2f s at full speed" % (
                res.joint_traj.n_points,
                res.joint_traj.duration,
            )
            result.trajectory = trajectory_to_msg(res.joint_traj)
            self.get_logger().info(result.message)
            goal_handle.succeed()
        else:
            result.message = "%s: %s" % (res.status, res.error)
            self.get_logger().error("Plan failed — " + result.message)
            goal_handle.abort()
        return result

    # --------------------------------------------------------------- execution
    def _execute_cb(self, goal_handle):
        result = ExecuteTrajectory.Result()

        def refuse(message, canceled=False):
            result.success = False
            result.message = message
            self.get_logger().error("Execution refused/failed: %s" % message)
            if canceled:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return result

        if not bool(self.get_parameter("execute").value):
            return refuse(
                "execution disabled (dry-run node). Relaunch with "
                "execute:=true — and only with a human on the e-stop."
            )
        if not self._exec_lock.acquire(blocking=False):
            return refuse("an execution is already running")
        try:
            msg = goal_handle.request.trajectory
            scale = float(goal_handle.request.speed_scale) or self.speed_scale
            if not 0.0 < scale <= self.max_speed_scale:
                return refuse(
                    "speed_scale %.3f outside (0, %.2f]" % (scale, self.max_speed_scale)
                )
            q = self.current_q()
            problems = validate_goal_msg(
                msg,
                self.planner.joint_names,
                q,
                self._pos_limits,
                self._vel_limits,
                self.start_tolerance,
            )
            if problems:
                return refuse("; ".join(problems))

            fb = ExecuteTrajectory.Feedback()

            def feedback(progress):
                fb.progress = float(progress)
                js = self._joint_state_snapshot()
                if js is not None:
                    fb.joint_states = js
                goal_handle.publish_feedback(fb)

            status, message = self.executor_helper.run(
                msg,
                scale,
                goal_handle=goal_handle,
                feedback_cb=feedback,
                get_current_q=self.current_q,
                tracking_tolerance_rad=self.tracking_tolerance,
            )
        finally:
            self._exec_lock.release()

        if status == "succeeded":
            result.success = True
            result.message = message
            self.get_logger().info(message)
            goal_handle.succeed()
            return result
        if "never left the start" in message:
            # No-motion signature: the arm is exactly where it was, so a
            # servoing reset is benign — do it now so the client's retry
            # lands on a live arm instead of a dead one. Partial-motion
            # failures (possible contact) never auto-reset.
            self._try_servoing_recovery()
        return refuse(message, canceled=(status == "canceled"))

    def _try_servoing_recovery(self):
        """Clear faults, then bounce the JTC to resync + restore servoing.

        Safe by construction: only called when the arm never left its
        start; deactivation stops all writes before anything re-enables
        streaming, and activation re-reads the true joint positions as
        the controller's hold state.
        """
        self.get_logger().warning(
            "no-motion failure: recovering servoing (fault reset + "
            "trajectory-controller restart; the arm does not move)"
        )
        if self._reset_fault.wait_for_service(timeout_sec=2.0):
            await_future(
                self._reset_fault.call_async(self._reset_fault_type.Request()), 5.0
            )
        if not self._switch_ctrl.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                "controller_manager switch service unavailable — cannot "
                "bounce the trajectory controller; restart the bringup"
            )
            return
        jtc = "joint_trajectory_controller"
        for field in ("deactivate_controllers", "activate_controllers"):
            req = self._switch_type.Request()
            getattr(req, field).append(jtc)
            req.strictness = self._switch_type.Request.STRICT
            res = await_future(self._switch_ctrl.call_async(req), 5.0)
            if res is None or not res.ok:
                self.get_logger().error("JTC %s failed — restart the bringup" % field)
                return
            time.sleep(1.0)
        # The arm needs settling time after a fault clear + servoing-mode
        # re-entry: an immediate retry died again in the field (2026-08-13)
        # while the same sequence with generous pauses recovered fine.
        time.sleep(2.5)
        self.get_logger().info(
            "servoing recovery complete — the next execution attempt " "should move"
        )

    # ---------------------------------------------------------------- services
    def _set_world_cb(self, request, response):
        with self._plan_lock:
            try:
                self.planner.update_world(request.world)
                response.success = True
                response.message = "world set to %s" % request.world
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response

    def _update_world_boxes_cb(self, request, response):
        n = len(request.names)
        if len(request.centers) != n or len(request.dims) != n:
            response.success = False
            response.message = "names/centers/dims length mismatch"
            return response
        boxes = [
            {
                "name": request.names[i],
                "position": [
                    request.centers[i].x,
                    request.centers[i].y,
                    request.centers[i].z,
                ],
                "dims": [request.dims[i].x, request.dims[i].y, request.dims[i].z],
            }
            for i in range(n)
        ]
        with self._plan_lock:
            try:
                self.planner.update_world_boxes(
                    boxes, baseline=request.baseline or None
                )
                response.success = True
                response.message = "world: baseline + %d perceived boxes" % n
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response

    def _gripper_cmd(self, position, response):
        if not bool(self.get_parameter("execute").value):
            response.success = False
            response.message = "execution disabled (dry-run node)"
            return response
        if not self._gripper.wait_for_server(timeout_sec=2.0):
            response.success = False
            response.message = "gripper action server not available"
            return response
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 100.0
        send = await_future(self._gripper.send_goal_async(goal), 5.0)
        if send is None or not send.accepted:
            response.success = False
            response.message = "gripper goal not accepted"
            return response
        wrapped = await_future(send.get_result_async(), 10.0)
        if wrapped is None:
            response.success = False
            response.message = "gripper did not finish in 10 s"
            return response
        response.success = True
        response.message = "gripper at %.3f (stalled=%s)" % (
            wrapped.result.position,
            wrapped.result.stalled,
        )
        return response

    def _open_gripper_cb(self, request, response):
        return self._gripper_cmd(self.gripper_open_pos, response)

    def _close_gripper_cb(self, request, response):
        return self._gripper_cmd(self.gripper_closed_pos, response)


def main(args=None):
    rclpy.init(args=args)
    node = RammpCuroboNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
