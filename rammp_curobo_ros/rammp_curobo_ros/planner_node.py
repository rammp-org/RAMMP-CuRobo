"""The rammp_curobo ROS 2 node: a pure planning service.

    ros2 launch rammp_curobo_ros planner.launch.py

Actions (node name rammp_curobo):
    /rammp_curobo/plan_to_pose        rammp_curobo_interfaces/PlanToPose
    /rammp_curobo/plan_to_joints      rammp_curobo_interfaces/PlanToJoints
Services:
    /rammp_curobo/set_world           rammp_curobo_interfaces/srv/SetWorld
    /rammp_curobo/check_pose_lock     rammp_curobo_interfaces/srv/CheckPoseLock

World in, plan out. This node NEVER interacts with the arm: it holds no
driver, no controller client, no gripper client, and no /joint_states
subscription. The configuration to plan from arrives in the goal as
`start_joints`, supplied by whoever owns the robot (kinova_arm_ros2), and
the returned trajectory is theirs to execute under their own gates.

That boundary is the point — see issue #6. A planner that also drove the
arm meant two safety authorities with different rules, plans made from a
joint state up to 2 s stale, and a dependency arrow pointing backwards
from the planner to the driver's IDL. Do not reintroduce any of it.
"""

import math
import re
import threading
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rammp_curobo import PoseConstraint, ViaPoint
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.msg import PoseAxisLock
from rammp_curobo_interfaces.srv import CheckPoseLock, SetWorld
from rammp_curobo_ros.conversions import trajectory_to_msg


def check_start_joints(start_joints, joint_names):
    """Why this start state is unusable, or None if it is fine.

    REQUIRED, not optional. The planner does not observe the arm, so an
    absent start state has no fallback to fall back TO — the old "empty
    means read /joint_states" behaviour is exactly the coupling this node
    shed. Empty and wrong-length report distinctly, and neither is ever
    described as a missing topic (issue #5: a 6-long vector surfaced as
    "no fresh joint state on /joint_states" and sent callers off to debug
    a bringup that was running fine).
    """
    n = len(joint_names)
    if not len(start_joints):
        return (
            "start_joints is required: %d joint positions expected, got "
            "none — the caller owns the arm's state and must supply the "
            "configuration to plan from" % n
        )
    if len(start_joints) != n:
        return "start_joints has %d values, expected %d (%s)" % (
            len(start_joints),
            n,
            ", ".join(joint_names),
        )
    if not all(math.isfinite(float(v)) for v in start_joints):
        return "start_joints contains a non-finite value"
    return None


def constraint_from_goal(request):
    """(PoseConstraint, ViaPoint, why_rejected) from a PlanToPose goal, or
    from a CheckPoseLock request.

    A goal that sets neither message maps to "unconstrained", which is also
    what an older client's absent fields deserialise to. CheckPoseLock has
    no `approach_via` field at all (it only asks about the axis lock), so
    that field is read defensively and defaults to an inert ViaPoint — one
    mapper serves both the action goal and the service request.
    """
    lock = request.axis_lock
    if lock.reference_frame not in (PoseAxisLock.FRAME_BASE, PoseAxisLock.FRAME_GOAL):
        return (
            PoseConstraint(),
            ViaPoint(),
            "unknown reference_frame %d (expected FRAME_BASE=%d or FRAME_GOAL=%d)"
            % (lock.reference_frame, PoseAxisLock.FRAME_BASE, PoseAxisLock.FRAME_GOAL),
        )
    constraint = PoseConstraint(
        hold_roll=bool(lock.lock_roll),
        hold_pitch=bool(lock.lock_pitch),
        hold_yaw=bool(lock.lock_yaw),
        hold_x=bool(lock.lock_x),
        hold_y=bool(lock.lock_y),
        hold_z=bool(lock.lock_z),
        in_base_frame=(lock.reference_frame == PoseAxisLock.FRAME_BASE),
    )
    approach = getattr(request, "approach_via", None)
    if approach is None:
        via = ViaPoint()
    else:
        via = ViaPoint(
            offset_m=float(approach.offset),
            linear_axis=int(approach.axis),
            tstep_fraction=float(approach.at_fraction),
        )
    try:
        constraint.validate()
        via.validate()
    except ValueError as exc:
        return constraint, via, str(exc)
    return constraint, via, None


_AXIS_RE = re.compile(r"'(roll|pitch|yaw|x|y|z)'")
_NUM_RE = re.compile(r"([-+]?\d*\.?\d+)")


def check_reply(ok, reason):
    """(ok, reason) from the core -> the CheckPoseLock reply fields."""
    out = {
        "satisfied": bool(ok),
        "message": reason or "",
        "worst_axis": "",
        "worst_error": 0.0,
        "limit": 0.0,
        "marginal": bool(ok) and bool(reason) and reason.startswith("marginal"),
    }
    if not reason:
        return out
    m = _AXIS_RE.search(reason)
    if m:
        out["worst_axis"] = m.group(1)
    nums = _NUM_RE.findall(reason)
    if nums:
        out["worst_error"] = float(nums[0])
    if len(nums) > 1:
        out["limit"] = float(nums[1])
    return out


class RammpCuroboNode(Node):
    def __init__(self):
        super().__init__("rammp_curobo")
        self._cb = ReentrantCallbackGroup()

        self.config = self.declare_parameter("config", "gen3.yaml").value
        self.world = self.declare_parameter("world", "").value

        # Refuse to double-serve. Two planner nodes on the same action names
        # answer every goal twice and race each other — on hardware this
        # produced phantom TRACKING FAILUREs and an aborted scan
        # (2026-08-13). Fail loudly instead.
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

        self._plan_lock = threading.Lock()

        # Heavy GPU init BEFORE the servers exist: a goal sent during the
        # (possibly minutes-long) warmup must not silently queue and then
        # be answered long after the client gave up (RAMMP-Kinova lesson).
        self.get_logger().info("Loading cuRobo planner (%s)..." % self.config)
        from rammp_curobo import CuRoboPlanner

        self.planner = CuRoboPlanner.from_config(self.config)
        if self.world:
            self.planner.update_world(self.world)
            self.get_logger().info("World overridden: %s" % self.world)

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
        self.create_service(
            SetWorld, "~/set_world", self._set_world_cb, callback_group=self._cb
        )
        self.create_service(
            CheckPoseLock,
            "~/check_pose_lock",
            self._check_pose_lock_cb,
            callback_group=self._cb,
        )

        # Real-arm misconfiguration tripwire: without sim time this node is
        # presumably planning for the REAL Gen3, and the sim kitchen's
        # obstacles do not exist on the real bench — cuRobo would dodge
        # phantom furniture and happily route through real objects.
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        world_name = self.world or self.planner.world_name
        if not use_sim_time and "sim" in world_name:
            self.get_logger().warning(
                "=== Planning against the SIM world (%s) WITHOUT sim time — "
                "if these plans are for the real arm, relaunch with "
                "world:=world_real_bench.yaml (MEASURED first; see "
                "docs/HARDWARE_BRINGUP.md). ===" % world_name
            )

        self.get_logger().info(
            "rammp_curobo ready — planning only, %d joints, world %s. "
            "This node never moves the arm; execution is the caller's."
            % (len(self.planner.joint_names), world_name)
        )

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
        constraint, via, why = constraint_from_goal(goal_handle.request)
        if why is not None:
            result.success = False
            result.message = "invalid constraint: %s" % why
            goal_handle.abort()
            return result
        res = self._plan(
            lambda q: self.planner.plan_to_pose(
                pos, quat, start=q, constraint=constraint, via=via
            ),
            goal_handle.request.start_joints,
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
            goal_handle.request.start_joints,
        )
        out = self._finish_plan(goal_handle, result, res)
        # _plan returns None (busy) or a string (bad start) as sentinels —
        # only a real PlanResult carries goal_mismatch_rad.
        mismatch = getattr(res, "goal_mismatch_rad", None)
        if mismatch is not None:
            out.goal_mismatch_rad = float(mismatch)
        return out

    def _plan(self, plan_fn, start_joints):
        """One library planning call from the caller-supplied start state.

        Returns the PlanResult, None if another plan is in flight, or a
        string explaining why start_joints was unusable.
        """
        why = check_start_joints(start_joints, self.planner.joint_names)
        if why is not None:
            return why
        if not self._plan_lock.acquire(blocking=False):
            return None
        try:
            self.get_logger().info("Planning...")
            return plan_fn([float(v) for v in start_joints])
        finally:
            self._plan_lock.release()

    def _finish_plan(self, goal_handle, result, res):
        if res is None:
            result.success = False
            result.message = "planner busy — one plan at a time"
            goal_handle.abort()
            return result
        if isinstance(res, str):
            result.success = False
            result.message = res
            self.get_logger().error("Goal rejected — " + res)
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

    def _check_pose_lock_cb(self, request, response):
        """FK and arithmetic only — no plan lock, so this never blocks (or
        is blocked by) a plan in flight, and a UI may poll it freely."""
        why = check_start_joints(request.start_joints, self.planner.joint_names)
        if why is not None:
            response.satisfied = False
            response.message = why
            return response
        pos = [
            request.target.position.x,
            request.target.position.y,
            request.target.position.z,
        ]
        quat = [
            request.target.orientation.x,
            request.target.orientation.y,
            request.target.orientation.z,
            request.target.orientation.w,
        ]
        constraint, _, bad = constraint_from_goal(request)
        if bad is not None:
            response.satisfied = False
            response.message = bad
            return response
        ok, reason = self.planner.constraint_satisfied_at_start(
            [float(v) for v in request.start_joints], pos, quat, constraint
        )
        for k, v in check_reply(ok, reason).items():
            setattr(response, k, v)
        return response


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
