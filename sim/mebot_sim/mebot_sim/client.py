"""A test client of the sim stack, using it as a consumer does: the planner
nodes plan, the driver interface flies, and the sim's own topics report.

    tester = Tester("my_test")
    tester.connect()
    tester.teleport(start)
    report = tester.move(
        stack.CHAIR, stack.CHAIR_WORLD, start, pos=[0.5, 0.4, 0.3], rpy_deg=[90, 0, 90]
    )

Before each flight it reads the ROS graph, and flies nothing unless
/mebot_sim alone publishes /joint_states and serves /execute_joint_trajectory
on the domain. The sim looks for another driver only when it starts, and an
action goal reaches every server of its name.
"""

import json
import os
import sys
import threading
import time

from mebot_sim import stack

# How long a flown plan is given to come to rest before its contacts are read,
# and how long, after that, a report of it sampled once it ended is waited for:
# the sim reports every 0.1 s.
SETTLE_S = 0.5
REPORT_S = 5.0
WHO = {"standin": "the person", "mebot": "the wheelchair", "floor": "the floor"}


def dilate(traj, speed):
    """Slow the trajectory_msgs/JointTrajectory `traj` to `speed` (0 < speed
    <= 1), in place: times over speed, velocities times speed, accelerations
    times speed squared. A plan is made at full speed; flying it slower is the
    executor's call, and here the client is that."""
    for p in traj.points:
        t = (p.time_from_start.sec + p.time_from_start.nanosec * 1e-9) / speed
        # in whole nanoseconds, so a time just under a second rounds up to
        # the next second, never to nanosec 1e9, which C++ consumers refuse
        p.time_from_start.sec, p.time_from_start.nanosec = divmod(
            round(t * 1e9), 1_000_000_000
        )
        p.velocities = [v * speed for v in p.velocities]
        p.accelerations = [a * speed * speed for a in p.accelerations]


class Tester:
    """One node, spun in the background, that reads /joint_states and
    /mebot_sim/contacts, moves the sim arm, and plans and flies through the
    planner nodes and the driver interface."""

    def __init__(self, name="mebot_sim_tester"):
        stack.isolate()
        import rclpy
        from rammp_arm_interfaces.action import ExecuteJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String

        self._rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node(name)
        self.q = None  # the arm's joints, as /joint_states last said
        self.sim = None  # the last /mebot_sim/contacts message, decoded
        self.node.create_subscription(
            JointState, "/joint_states", self._joints, qos_profile_sensor_data
        )
        self.node.create_subscription(String, "/mebot_sim/contacts", self._contacts, 10)
        self._teleport = self.node.create_publisher(
            Float64MultiArray, "/mebot_sim/teleport", 10
        )
        self._posture = self.node.create_publisher(String, "/mebot_sim/posture", 10)
        self._Joints, self._String = Float64MultiArray, String
        self._Goal = ExecuteJointTrajectory.Goal
        self.driver = ActionClient(
            self.node, ExecuteJointTrajectory, "/execute_joint_trajectory"
        )
        self.planners = {
            planner: stack.Planner(self.node, planner)
            for planner in (stack.CHAIR, stack.CONTROL)
        }
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

    def _joints(self, msg):
        names = list(msg.name)
        if all(n in names for n in stack.JOINTS):
            self.q = [msg.position[names.index(n)] for n in stack.JOINTS]

    def _contacts(self, msg):
        self.sim = json.loads(msg.data)

    def connect(self, timeout=300.0):
        """Wait until the whole stack answers: both planners, the driver
        interface, /joint_states and the sim's own /mebot_sim/contacts."""
        domain = os.environ["ROS_DOMAIN_ID"]
        print(
            "waiting for the sim arm and both planners on ROS_DOMAIN_ID=%s..." % domain,
            file=sys.stderr,
            flush=True,
        )
        deadline = time.monotonic() + timeout
        while True:
            missing = [name for name, p in self.planners.items() if not p.ready()]
            if not self.driver.server_is_ready():
                missing.append("/execute_joint_trajectory")
            if self.q is None:
                missing.append("/joint_states")
            if self.sim is None:
                missing.append("/mebot_sim/contacts")
            if not missing:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "the sim stack is not up on ROS_DOMAIN_ID=%s: nothing from %s. "
                    "Start it with: ros2 launch mebot_sim sim.launch.py"
                    % (domain, ", ".join(missing))
                )
            time.sleep(0.2)

    def teleport(self, q, timeout=5.0):
        """Put the sim arm at `q`, at rest, once /joint_states says it is there."""
        q = [float(v) for v in q]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._teleport.publish(self._Joints(data=q))
            time.sleep(0.25)
            if self.q is not None and max(abs(a - b) for a, b in zip(self.q, q)) < 0.03:
                time.sleep(0.3)  # for its actuators to take the load
                return
        raise RuntimeError("the sim arm did not go to %s" % q)

    def posture(self, name, timeout=30.0):
        """Seat the stand-in in posture `name` (none: an empty chair)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.sim is not None and self.sim["posture"] == name:
                return
            self._posture.publish(self._String(data=name))
            time.sleep(0.5)
        raise RuntimeError("the sim did not seat the stand-in %s" % name)

    def move(
        self,
        planner,
        world,
        start,
        pos=None,
        rpy_deg=None,
        joints=None,
        speed=0.5,
        execute=True,
    ):
        """Plan on `planner` (stack.CHAIR or stack.CONTROL), in `world` (pushed
        first; None plans in the world it holds), from `start`, to a tool pose
        (`pos` and `rpy_deg`) or a joint goal (`joints`); then fly the plan on
        the sim at `speed`.

        Returns {planned, message, plan_s, world} (the planner's answers; world
        is its reply to the push) and, once flown, {executed, exec_error,
        contacts, closest, tracking_rad, end_error_rad}: the worst contact per
        group during the flight, the closest approach per group, and how far
        the arm was from the plan's last point on any joint when the flight
        ended, and once it came to rest.
        """
        if not 0 < speed <= 1:
            raise ValueError("speed must be in (0, 1], got %r" % speed)
        if (joints is None) == (pos is None) or (pos is None) != (rpy_deg is None):
            raise ValueError("a goal is a tool pose (pos and rpy_deg) or joints")
        client = self.planners[planner]
        out = {"planned": False, "message": "", "plan_s": 0.0, "world": None}
        if world is not None:
            pushed, out["world"] = client.set_world(world)
            if not pushed:
                out["message"] = "set_world refused: %s" % out["world"]
                return out
        result = client.plan(start, pos=pos, rpy_deg=rpy_deg, joints=joints)
        if result is None:
            out["message"] = "%s rejected the goal" % planner
            return out
        out.update(
            planned=bool(result.success),
            message=result.message,
            plan_s=float(result.planning_time),
        )
        if result.success and execute:
            out.update(self.fly(result.trajectory, speed))
        return out

    def check_state(self, planner, q):
        """`planner`'s verdict on the arm at `q`, in the world it holds. Its
        interfaces have no check of a state, so it is asked as a consumer
        can ask it: a plan from `q` to `q`, which the planner refuses with an
        INVALID_START status, naming the boxes `q` is in, when `q` is not a
        valid state. Returns move's report of it; nothing is pushed or flown."""
        return self.move(planner, None, q, joints=q, execute=False)

    def fly(self, traj, speed):
        """Fly a plan on the sim at `speed`, from where the arm is; what the
        sim saw of it (see move)."""
        times, rows = stack.trajectory_arrays(traj)
        off = max(abs(a - b) for a, b in zip(self.q, rows[0]))
        if off > 0.05:
            return {
                "executed": False,
                "exec_error": "the arm is %.2f rad from the plan's start; "
                "teleport it there first" % off,
            }
        before = self.sim["move"]
        dilate(traj, speed)
        goal = self._Goal(trajectory=traj, sender_id=self.node.get_name())
        self._only_the_sim()
        handle = stack.wait(self.driver.send_goal_async(goal), 30.0, "flying")
        if not handle.accepted:
            return {"executed": False, "exec_error": "the driver interface rejected it"}
        result = stack.wait(
            handle.get_result_async(), times[-1] / speed + 60.0, "flying"
        ).result
        time.sleep(SETTLE_S)
        report = self._report_after(before)
        return {
            "executed": result.error_code == 0,
            "exec_error": result.error_string,
            "contacts": report["worst"],
            "closest": report["closest"],
            # the driver's final error: measured minus commanded as it ended
            "tracking_rad": max(
                (abs(v) for v in result.final_error.positions), default=None
            ),
            "end_error_rad": max(abs(a - b) for a, b in zip(self.q, rows[-1])),
        }

    def _report_after(self, move):
        """The sim's first report of a move after `move` sampled once no flight
        was under way: its worst contacts and closest approaches cover the
        whole flight. One sampled during the flight misses what it touched
        after, and one of `move` or before is not of this flight at all."""
        deadline = time.monotonic() + REPORT_S
        while True:
            report = self.sim  # one message: the spinning thread swaps it whole
            if report["move"] > move and not report["moving"]:
                return report
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "the sim reported no move for the plan it flew"
                    if report["move"] <= move
                    else "the sim has not reported the flight since it ended: what "
                    "it touched at its end is unknown"
                )
            time.sleep(0.05)

    def _only_the_sim(self):
        """Refuse to fly unless /mebot_sim is the one publisher of
        /joint_states and of the driver action's status, which every server
        of /execute_joint_trajectory publishes."""
        for topic in ("/joint_states", "/execute_joint_trajectory/_action/status"):
            nodes = [
                "%s/%s" % (p.node_namespace.rstrip("/"), p.node_name)
                for p in self.node.get_publishers_info_by_topic(topic)
            ]
            if nodes != ["/mebot_sim"]:
                raise RuntimeError(
                    "%s is published by %s, not by the sim arm alone: nothing is "
                    "flown" % (topic, ", ".join(nodes) or "nothing")
                )

    def close(self):
        # the spinning thread must be out of rcl before the node and the
        # context go, or the process can abort on its way out
        self._executor.shutdown()
        self._spin.join()
        self.node.destroy_node()
        self._rclpy.try_shutdown()


def contact_text(contacts):
    """The worst contacts of a flight, in words."""
    if not contacts:
        return "no contact"
    return "; ".join(
        "%s hit %s (%s), %.0f N" % (c["link"], WHO[c["group"]], c["part"], c["force_n"])
        for c in contacts
    )


def closest_text(closest):
    """The closest approach per group, in words."""
    return ", ".join(
        "%s %.1f mm" % (WHO[group].split()[-1], c["mm"])
        for group, c in sorted(closest.items())
    )
