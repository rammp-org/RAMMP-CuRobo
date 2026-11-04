"""The rammp_curobo ROS 2 node: a pure planning service.

    ros2 launch rammp_curobo_ros planner.launch.py

Actions (node name rammp_curobo):
    /rammp_curobo/plan_to_pose        rammp_curobo_interfaces/PlanToPose
    /rammp_curobo/plan_to_joints      rammp_curobo_interfaces/PlanToJoints
    /rammp_curobo/blend               rammp_curobo_interfaces/Blend
Services:
    /rammp_curobo/set_world           rammp_curobo_interfaces/srv/SetWorld
Subscriptions:
    /reset                            std_msgs/Empty — back to the world
                                      loaded at boot, layers included

World in, plan out. This node NEVER interacts with the arm: it holds no
driver, no controller client, no gripper client, and no /joint_states
subscription. The configuration to plan from arrives in the goal as
`start_joints`, supplied by whoever owns the robot (kinova_arm_ros2), and
the returned trajectory is theirs to execute under their own gates.

That boundary is the point — see issue #6. A planner that also drove the
arm meant two safety authorities with different rules, plans made from a
joint state up to 2 s stale, and a dependency arrow pointing backwards
from the planner to the driver's IDL. Do not reintroduce any of it.

The robot model is the arm's own: the URDF arm_state_publisher latches on
/arm/robot_description. The node waits for it before building the planner,
so it plans in that model's root frame (arm_mount) with its joint names and
limits, and every returned trajectory says arm_mount in its header. The
model is read once; a different one published later is reported as an
ERROR, every time, and never used — restart the planner to take it.
"""

import logging
import math
import tempfile
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from rammp_curobo import PoseConstraint, ViaPoint
from rammp_curobo.config import is_example, load_planner_config, resolve_config
from rammp_curobo.contract import check_robot_dir
from rammp_curobo.urdf_model import write_urdf
from rammp_curobo_interfaces.action import Blend, PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
from rammp_curobo_ros.conversions import trajectory_from_msg, trajectory_to_msg


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


ROBOT_DESCRIPTION_TOPIC = "/arm/robot_description"
# No `config` = the example robot shipped in the image, planned standalone.
EXAMPLE_CONFIG = "example:gen3.yaml"
EXAMPLE_URDF = "example:arm_level.urdf"
EXAMPLE_MARKER = "(example robot)"
# What robot_state_publisher latches with: a later subscriber still gets it.
MODEL_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def planner_config(config, world):
    """(planner config dict, config dir) with the `world` parameter applied.

    The world goes into the config BEFORE the planner is built: update_world
    refuses an empty world (cuRobo silently keeps the old one there), so a
    deployment's world_empty.yaml can only take effect at construction. An
    empty `world` keeps the config's own.
    """
    cfg, cfg_dir = load_planner_config(config)
    if world:
        cfg["world"] = world
    return cfg, cfg_dir


def _where(path):
    return "%s (example, image)" % path if is_example(path) else str(path)


def robot_data_summary(config, cfg, cfg_dir, model_source):
    """One line naming where every piece of robot data came from."""
    layers = [
        _where(resolve_config(r, relative_to=cfg_dir, search_cwd=False))
        for r in cfg.get("layers", [])
    ]
    return "robot data: config %s, robot file %s, world %s, layers [%s], model %s" % (
        _where(resolve_config(config)),
        _where(resolve_config(cfg["robot"], relative_to=cfg_dir, search_cwd=False)),
        _where(resolve_config(cfg["world"], relative_to=cfg_dir, search_cwd=False)),
        ", ".join(layers),
        model_source,
    )


_HOLD_MODES = (
    PlanToPose.Goal.HOLD_NONE,
    PlanToPose.Goal.HOLD_LEVEL,
    PlanToPose.Goal.HOLD_FIXED,
)


def constraint_from_goal(goal):
    """(PoseConstraint, ViaPoint, why_rejected) from a PlanToPose goal.

    A goal that sets neither field maps to "unconstrained", which is also what
    an older client's absent fields deserialise to.

    An out-of-range mode is REFUSED rather than coerced to HOLD_NONE: silently
    treating an unknown value as "no constraint" would hand back an
    unconstrained trajectory to a caller who asked for a held one.
    """
    if goal.hold not in _HOLD_MODES:
        return (
            PoseConstraint(),
            ViaPoint(),
            "unknown hold %d (expected HOLD_NONE=%d, HOLD_LEVEL=%d or "
            "HOLD_FIXED=%d)" % ((goal.hold,) + _HOLD_MODES),
        )
    constraint = PoseConstraint(hold=int(goal.hold))
    approach = goal.approach_via
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


def legs_from_goal(goal, joint_names):
    """(legs, speed_scales, holds, hold_orientations, why_rejected) from a
    Blend goal: every leg a trajectory in `joint_names` (trajectory_from_msg),
    one speed scale, hold and hold orientation per leg or none, the holds
    HOLD_* values; REFUSED otherwise. holds are PoseConstraints, None where a
    leg holds nothing; hold_orientations xyzw, None where none applies."""
    n = len(goal.legs)
    if not n:
        return [], None, None, None, "no legs"
    legs = []
    for k, msg in enumerate(goal.legs):
        leg, why = trajectory_from_msg(msg, joint_names)
        if why is not None:
            return [], None, None, None, "leg %d/%d: %s" % (k + 1, n, why)
        legs.append(leg)
    per_leg = (
        ("speed_scale", goal.speed_scale),
        ("hold", goal.hold),
        ("hold_orientation", goal.hold_orientation),
    )
    for name, values in per_leg:
        if len(values) not in (0, n):
            why = "%d legs: %s needs %d values or none, got %d"
            return [], None, None, None, why % (n, name, n, len(values))
    holds = [int(h) for h in goal.hold] or [PlanToPose.Goal.HOLD_NONE] * n
    for k, h in enumerate(holds):
        if h not in _HOLD_MODES:
            why = "leg %d/%d: hold must be one of %s, got %d"
            return [], None, None, None, why % (k + 1, n, _HOLD_MODES, h)
    refs = [None] * n
    for k, o in enumerate(goal.hold_orientation):
        if holds[k]:
            refs[k] = (o.x, o.y, o.z, o.w)
    return (
        legs,
        [float(s) for s in goal.speed_scale] or None,
        [PoseConstraint(hold=h) if h else None for h in holds],
        refs,
        None,
    )


class RammpCuroboNode(Node):
    def __init__(self):
        super().__init__("rammp_curobo")
        self._cb = ReentrantCallbackGroup()

        self.config = self.declare_parameter("config", "").value
        self.example_mode = not self.config
        if self.example_mode:
            self.config = EXAMPLE_CONFIG
            self.get_logger().warning(
                "=== NO CONFIG GIVEN: running the EXAMPLE robot from the image "
                "(Gen3 + 2F-85, level mount, world_sim_kitchen). Not your robot: "
                "pass config:=/config/robot/curobo/planner.yaml. ==="
            )
        self.world = self.declare_parameter("world", "").value
        # dynamic_typing: `-p robot_description_timeout_s:=3` is an INTEGER,
        # which a DOUBLE-typed declaration refuses with a traceback.
        timeout = self.declare_parameter(
            "robot_description_timeout_s",
            30.0,
            ParameterDescriptor(dynamic_typing=True),
        ).value
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            timeout_s = math.nan
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            self.get_logger().fatal(
                "robot_description_timeout_s must be a positive number of "
                "seconds, got %r" % (timeout,)
            )
            raise SystemExit(1)

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

        try:
            planner_cfg = planner_config(self.config, self.world)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().fatal("planner config %s: %s" % (self.config, exc))
            raise SystemExit(1)
        # The model first: no planner and no servers until a valid one is in.
        # The example robot brings its own, and waits for nothing.
        if self.example_mode:
            model = resolve_config(EXAMPLE_URDF).read_text(encoding="utf-8")
        else:
            model = self._wait_for_model(timeout_s)
        try:
            xml = check_robot_dir(*planner_cfg, model)
            summary = robot_data_summary(
                self.config,
                *planner_cfg,
                "example" if self.example_mode else "topic",
            )
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().fatal(
                "robot data or the model on %s: %s" % (ROBOT_DESCRIPTION_TOPIC, exc)
            )
            raise SystemExit(1)
        self.get_logger().info(summary)
        # Removed in destroy_node; inside the container it dies with it anyway.
        self._model_dir = tempfile.TemporaryDirectory(prefix="rammp_curobo_")
        urdf_path = write_urdf(xml, self._model_dir.name)

        # Heavy GPU init BEFORE the servers exist: a goal sent during the
        # (possibly minutes-long) warmup must not silently queue and then
        # be answered long after the client gave up (RAMMP-Kinova lesson).
        self.get_logger().info("Loading cuRobo planner (%s)..." % self.config)
        from rammp_curobo import CuRoboPlanner

        self.planner = CuRoboPlanner(*planner_cfg, urdf_path=urdf_path)
        self.get_logger().info(
            "model from %s: base %s, ee %s, joints [%s]"
            % (
                EXAMPLE_URDF if self.example_mode else ROBOT_DESCRIPTION_TOPIC,
                self.planner.base_frame,
                self.planner.ee_link,
                ", ".join(self.planner.joint_names),
            )
        )
        # The core library's INFO logs do not reach the ROS logger, and this
        # line is the operator's proof of which layers are in the world.
        self.get_logger().info("Collision world: %s" % self.planner.world_summary())
        self._model_sub = self._watch_model(model)

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
            Blend,
            "~/blend",
            self._blend_cb,
            callback_group=self._cb,
        )
        self.create_service(
            SetWorld, "~/set_world", self._set_world_cb, callback_group=self._cb
        )
        # The ROS-wide reset convention: a global topic, not ~/reset.
        self.create_subscription(
            Empty, "/reset", self._reset_cb, 10, callback_group=self._cb
        )

        # Real-arm misconfiguration tripwire: without sim time this node is
        # presumably planning for the REAL Gen3, and the sim kitchen's
        # obstacles do not exist on the real bench — cuRobo would dodge
        # phantom furniture and happily route through real objects.
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        world_name = self.planner.world_name
        if not use_sim_time and "sim" in world_name:
            self.get_logger().warning(
                "=== Planning against the SIM world (%s) WITHOUT sim time — "
                "if these plans are for the real arm, relaunch with "
                "your robot's config (config:=/config/robot/curobo/planner.yaml; "
                "its world MEASURED first, see docs/HARDWARE_BRINGUP.md). ==="
                % world_name
            )

        self.get_logger().info(
            "rammp_curobo ready — planning only, %d joints, world %s (%s). "
            "This node never moves the arm; execution is the caller's."
            % (
                len(self.planner.joint_names),
                world_name,
                self.planner.world_summary(),
            )
        )

    def _wait_for_model(self, timeout_s):
        """The URDF text latched on /arm/robot_description.

        Exits 1 with a FATAL naming its publisher after `timeout_s`. A
        shutdown (SIGINT/SIGTERM) during the wait propagates as
        KeyboardInterrupt or ExternalShutdownException, which main() treats
        as a clean exit.
        """
        got = []
        sub = self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, lambda m: got.append(m.data), MODEL_QOS
        )
        self.get_logger().info(
            "Waiting up to %.0f s for the model on %s..."
            % (timeout_s, ROBOT_DESCRIPTION_TOPIC)
        )
        deadline = time.monotonic() + timeout_s
        while not got and time.monotonic() < deadline:
            if not rclpy.ok():
                raise ExternalShutdownException()
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        if not got:
            self.get_logger().fatal(
                "no model on %s after %.0f s. It is published by "
                "arm_state_publisher: ros2 launch kinova_gen3_description "
                "description.launch.py mount_file:=<robot>/mount.yaml"
                % (ROBOT_DESCRIPTION_TOPIC, timeout_s)
            )
            raise SystemExit(1)
        return got[0]

    def _watch_model(self, model):
        """Keep listening on /arm/robot_description, and log an ERROR each
        time a model other than `model` (the one in use) arrives. In example
        mode every published model is ignored: one WARN, the first time.

        The model in use never changes: the planner, its spheres' links and
        every plan in flight are built on it. A re-measured mount means a
        restart, and this makes a missed restart visible in the planner's
        own log. The latched copy of `model` itself is not reported.
        """

        warned = []

        def check(msg):
            if self.example_mode:
                if not warned:
                    warned.append(True)
                    self.get_logger().warning(
                        "a robot model is being published on %s, and this "
                        "planner is IGNORING it: no config was given, so it "
                        "plans the example robot." % ROBOT_DESCRIPTION_TOPIC
                    )
            elif msg.data != model:
                self.get_logger().error(
                    "a DIFFERENT model was published on %s; this planner is "
                    "still using the one it started with -- restart the "
                    "planner" % ROBOT_DESCRIPTION_TOPIC
                )

        return self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, check, MODEL_QOS
        )

    def _tag(self, message):
        """`message` with the example marker on the end in example mode."""
        return message + " " + EXAMPLE_MARKER if self.example_mode else message

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
            result.message = self._tag("invalid constraint: %s" % why)
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
            result.message = self._tag(
                "%d target joints for %d-joint arm"
                % (len(target), len(self.planner.joint_names))
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

    def _blend_cb(self, goal_handle):
        result = Blend.Result()
        goal = goal_handle.request
        legs, scales, holds, refs, why = legs_from_goal(goal, self.planner.joint_names)
        if why is not None:
            result.success = False
            result.message = self._tag("invalid legs: %s" % why)
            goal_handle.abort()
            return result
        # the first leg's start is the plan's: _plan checks it like any start
        res = self._plan(
            lambda q: self.planner.blend_legs(
                legs, list(goal.blend_radius), scales, holds, refs
            ),
            list(legs[0].positions[0]),
        )
        out = self._finish_plan(
            goal_handle, result, res, speed="at the legs' speed scales"
        )
        if getattr(res, "corners_m", None) is not None:
            out.corner_m = [float(c) for c in res.corners_m]
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

    def _finish_plan(self, goal_handle, result, res, speed="at full speed"):
        if res is None:
            result.success = False
            result.message = self._tag("planner busy — one plan at a time")
            goal_handle.abort()
            return result
        if isinstance(res, str):
            result.success = False
            result.message = self._tag(res)
            self.get_logger().error("Goal rejected — " + res)
            goal_handle.abort()
            return result
        result.success = bool(res.success)
        result.planning_time = float(res.timing)
        if res.success:
            result.message = "planned %d points, %.2f s %s" % (
                res.joint_traj.n_points,
                res.joint_traj.duration,
                speed,
            )
            if res.error:
                result.message += " — warning: " + res.error
            result.message = self._tag(result.message)
            result.trajectory = trajectory_to_msg(res.joint_traj)
            self.get_logger().info(result.message)
            goal_handle.succeed()
        else:
            result.message = self._tag("%s: %s" % (res.status, res.error))
            self.get_logger().error("Plan failed — " + result.message)
            goal_handle.abort()
        return result

    def destroy_node(self):
        super().destroy_node()
        self._model_dir.cleanup()

    # ---------------------------------------------------------------- services
    def _set_world_cb(self, request, response):
        with self._plan_lock:
            try:
                # resolved as update_world does: next to the config, or example:
                path = resolve_config(
                    request.world,
                    relative_to=self.planner._config_dir,
                    search_cwd=False,
                )
                self.planner.update_world(path)
                response.success = True
                # the layers the config adds to every world are in the reply
                response.message = self._tag(
                    "world set to %s%s (%s)"
                    % (
                        request.world,
                        " (example, image)" if is_example(path) else "",
                        self.planner.world_summary(),
                    )
                )
            except Exception as exc:
                response.success = False
                response.message = self._tag(str(exc))
        return response

    def _reset_cb(self, _msg):
        # Waits out an in-flight plan rather than resetting under it.
        with self._plan_lock:
            try:
                self.planner.reset()
            except Exception as exc:
                self.get_logger().error("Reset failed — %s" % exc)
                return
        self.get_logger().info(
            "Planner reset — collision world: %s" % self.planner.world_summary()
        )


def main(args=None):
    # The core logs through Python logging (the world it built, warm-up);
    # without a handler its INFO never reaches the console.
    core_log = logging.getLogger("rammp_curobo")
    core_log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s]: %(message)s"))
    core_log.addHandler(handler)
    rclpy.init(args=args)
    try:
        node = RammpCuroboNode()
    except (KeyboardInterrupt, ExternalShutdownException):
        return  # stopped while waiting for the model: a clean exit
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
