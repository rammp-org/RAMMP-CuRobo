"""/mebot_sim: a MuJoCo Gen3 standing in for the arm driver, beside the
MEBot, a seated stand-in and the floor (scene.py).

    /joint_states              sensor_msgs/JointState, best effort, 50 Hz:
                               joint_1..joint_7 and the 2F-85's knuckle, the
                               names the driver publishes
    /execute_joint_trajectory  rammp_arm_interfaces/ExecuteJointTrajectory:
                               flown on the trajectory's own time base. As
                               the driver does, a goal of preemption
                               LATEST_WINS ends the one flying, and one of
                               QUEUE (a goal's default) waits for it; a
                               cancel ends both PREEMPTED, the arm holding
                               where it is
    /mebot_sim/contacts        std_msgs/String, JSON, 10 Hz: the contacts
                               now, and since the last move began the worst
                               contact and the closest approach per group
    /mebot_sim/teleport        std_msgs/Float64MultiArray: seven joints to
                               put the arm at, at rest
    /mebot_sim/posture         std_msgs/String: upright_lap, lean_lap,
                               forearms_forward or none (an empty chair)
    /mebot_sim/reset           std_srvs/Trigger: the arm back at its start
    http://127.0.0.1:8765      the live view (view.py)

Physics steps at 2 ms in real time, and menagerie's position actuators
track the reference: the trajectory being flown, or else the last state
commanded. As the real driver does, it flies what it is sent at the speed
it is sent; slowing a plan down is the client's job.

A contact is any MuJoCo contact between an arm or gripper geom and a
group's shape (scene.GROUPS) pressing with more than 1 N, sampled every
10 ms of sim time. Penetration depth decides nothing: in a prototype of this
sim the arm pressed into the stand-in for 4.8 s at 744 N while soft
contacts held it within a millimetre of the surface. The closest approach
of the moving arm to each group is reported beside, never gated.

No node here speaks to an arm: it serves the driver's names only on a
domain of its own (stack.isolate), and refuses to start beside anything
already serving them.
"""

import json
import math
import os
import queue
import signal
import threading
import time

import mujoco
import numpy as np

from mebot_sim import scene, stack

TIMESTEP = 0.002
SAMPLE_STEPS = 5  # contacts and closest approaches every 10 ms of sim time
CONTACT_N = 1.0
# Closest approaches beyond 1 m are not looked for.
DISTMAX = 1.0
# A loop further behind real time than this (a rebuild, a slow frame) drops
# the backlog instead of rushing the physics to catch up.
BEHIND_S = 0.1
# The driver publishes the 2F-85's one degree of freedom under this name; it
# is menagerie's left driver joint.
KNUCKLE = "robotiq_85_left_knuckle_joint"
GRIPPER_JOINT = "g_left_driver_joint"
PREEMPTED = -6  # ExecuteJointTrajectory's result codes
SUCCESSFUL = 0
LATEST_WINS = 1  # its goal's preemption; anything else is QUEUE to the driver
STOPPING = "the sim stopping"


class Flight:
    """A trajectory to fly: seconds from its start, and rows of joint_1..joint_7
    positions. `queue`: it waits for a flight under way to be done, as the
    driver queues a goal of preemption QUEUE, instead of ending that flight.
    The sim sets `ended` when it is done with it: `outcome` is "done", or
    "preempted" with a `reason`, and `final_error` the measured joints minus
    the reference at that moment (None if it never flew)."""

    def __init__(self, times, rows, queue=False):
        self.times = np.asarray(times, dtype=float)
        self.rows = np.asarray(rows, dtype=float)
        self.queue = queue
        self.steps = 0
        self.outcome = self.reason = self.final_error = None
        self.ended = threading.Event()

    def end(self, outcome, reason, final_error=None):
        self.outcome, self.reason, self.final_error = outcome, reason, final_error
        self.ended.set()

    @property
    def elapsed(self):
        return self.steps * TIMESTEP

    def reference(self):
        """Where the arm is told to be now: linear between the points."""
        t = self.elapsed
        i = int(np.searchsorted(self.times, t))
        if i >= len(self.times):
            return self.rows[-1].copy()
        if i == 0:
            return self.rows[0].copy()
        a = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        return self.rows[i - 1] + a * (self.rows[i] - self.rows[i - 1])


def trajectory_problem(traj):
    """Why the driver interface refuses the trajectory_msgs/JointTrajectory
    `traj`, or None.

    The real driver refuses a trajectory with no points or a point without
    seven positions, and takes those as joint_1..joint_7 whatever the names
    say. The sim refuses those, and also what the driver might fly other
    than as its sender meant, so that no test passes on it: names other
    than joint_1..joint_7 in that order, a position that is not finite,
    times that do not rise from 0."""
    if not traj.points:
        return "the trajectory has no points"
    try:
        times, rows = stack.trajectory_arrays(traj)
    except ValueError as exc:
        return str(exc)
    last = -math.inf
    for k, (t, row) in enumerate(zip(times, rows)):
        if len(row) != len(stack.JOINTS):
            return "point %d has %d positions, not %d" % (
                k,
                len(row),
                len(stack.JOINTS),
            )
        if not all(math.isfinite(v) for v in row):
            return "point %d has a position that is not finite" % k
        if t < 0 or t <= last:
            return "point %d is at %.3f s, not after the one before it" % (k, t)
        last = t
    return None


def _hull(m, g):
    """The points of arm geom `g` that bound its distance to anything, in the
    geom's frame: its mesh's convex hull vertices (what MuJoCo collides), or
    a box's corners."""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        return corners * m.geom_size[g]
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(
            "arm geom %r is neither a mesh nor a box; the distance guard has no "
            "points for it" % m.geom(g).name
        )
    mesh = m.geom_dataid[g]
    start = m.mesh_vertadr[mesh]
    vertices = m.mesh_vert[start : start + m.mesh_vertnum[mesh]]
    graph = m.mesh_graphadr[mesh]
    if graph >= 0:  # [numvert, numface, vert_edgeadr, vert_globalid, ...]
        n = m.mesh_graph[graph]
        vertices = vertices[m.mesh_graph[graph + 2 + n : graph + 2 + 2 * n]]
    return np.asarray(vertices, dtype=float)


def _signed_distance(m, d, g, points):
    """The signed distance from each of `points` (N, 3) to shape `g`: negative
    inside. The shapes a contact group holds: boxes and cylinders (the
    MEBot), capsules and spheres (the stand-in), and a plane (the floor)."""
    local = (points - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)
    size, kind = m.geom_size[g], m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(local) - size
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial = np.linalg.norm(local[:, :2], axis=1) - size[0]
        axial = np.abs(local[:, 2]) - size[1]
        outside = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0))
        return outside + np.minimum(np.maximum(radial, axial), 0)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = np.zeros_like(local)
        axis[:, 2] = np.clip(local[:, 2], -size[1], size[1])
        return np.linalg.norm(local - axis, axis=1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(local, axis=1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return local[:, 2]
    raise ValueError("no signed distance for geom %r" % m.geom(g).name)


class Sim:
    """The arm and its scene, stepped by whoever owns it: the real-time loop
    of run(), or a test. Other threads post() commands, which the owner
    carries out in poll(), and read snapshot(). `flight` is the flight under
    way, and `queued` the one waiting for it; `stopping` is set once nothing
    will step the sim again."""

    def __init__(self, assets, posture, start, offscreen=(960, 720)):
        self.assets = assets
        self.posture = posture
        self.start = np.array(start, dtype=float)
        self.ref = self.start.copy()
        self.offscreen = offscreen
        self.flight = self.queued = None
        self.stopping = threading.Event()
        self.steps = 0
        self.moves = 0
        self.worst, self.closest = {}, {}
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.state = {}
        self._build(self.start)

    # ------------------------------------------------------------- the model
    def _build(self, q):
        m = scene.build(self.assets, self.posture, self.offscreen)
        m.opt.timestep = TIMESTEP
        self.model, self.data = m, mujoco.MjData(m)
        joints = [m.joint(name).id for name in stack.JOINTS]
        self.qpos = m.jnt_qposadr[joints]
        self.dof = m.jnt_dofadr[joints]
        self.actuators = [m.actuator(name).id for name in stack.JOINTS]
        self.knuckle = m.jnt_qposadr[m.joint(GRIPPER_JOINT).id]
        self.arm, self.groups = scene.contact_groups(m)
        self.armset = set(self.arm.tolist())
        self.group_of = {int(g): k for k, gs in self.groups.items() for g in gs}
        # base_link is welded to the world: MuJoCo never collides it with the
        # chair, the stand-in or the floor, which are too, and its distance to
        # them is the same in every state
        base = m.body("base_link").id
        self.moving = [int(g) for g in self.arm if m.geom_bodyid[g] != base]
        self.hulls = {g: _hull(m, g) for g in self.moving}
        self._place(q)

    def _place(self, q):
        """Put the arm at `q`, at rest, with the actuators holding the
        reference."""
        d = self.data
        d.qpos[self.qpos] = q
        d.qvel[:] = 0.0
        d.ctrl[self.actuators] = self.ref
        mujoco.mj_forward(self.model, d)
        self._sample()

    @property
    def q(self):
        """The arm's joints as measured: joint_1..joint_7."""
        return self.data.qpos[self.qpos].copy()

    # -------------------------------------------------------------- commands
    def post(self, command, *args):
        """Queue a command for the thread that owns the sim: "fly" (a Flight),
        "stop" (a Flight, or None for whatever flies), "teleport" (seven
        joints), "reset", "posture" (a name, None for an empty chair) or
        "jog" (a joint's index and position)."""
        self.commands.put((command, args))

    def poll(self):
        """Carry out every command queued."""
        while True:
            try:
                command, args = self.commands.get_nowait()
            except queue.Empty:
                return
            getattr(self, "_" + command)(*args)

    def _fly(self, flight):
        if flight.queue and self.flight is not None:
            # the driver keeps one goal queued, the newer
            if self.queued is not None:
                self.queued.end("preempted", "a newer trajectory")
            self.queued = flight
            return
        self._end("preempted", "a newer trajectory")
        self._take_off(flight)

    def _take_off(self, flight):
        if flight.times[0] > 0:
            # a planner stamps its first point one step ahead: start from
            # where the arm is told to be now
            flight.times = np.concatenate([[0.0], flight.times])
            flight.rows = np.vstack([self.ref, flight.rows])
        self.flight = flight
        self._new_move()

    def _stop(self, flight=None):
        if self.flight is not None and flight in (None, self.flight, self.queued):
            self._end("preempted", "a stop")
            self.ref = self.q

    def _teleport(self, q, reason="a teleport"):
        self._end("preempted", reason)
        self.ref = np.array(q, dtype=float)
        self._new_move()
        self._place(self.ref)

    def _reset(self):
        self._teleport(self.start, "a reset")

    def _posture(self, name):
        self._end("preempted", "a posture switch")
        self.ref = self.q
        self.posture = name
        self._new_move()
        self._build(self.ref)

    def _jog(self, joint, position):
        if self.flight is None:
            self.ref[joint] = position

    def _end(self, outcome, reason):
        """End the flight under way. Unless it is done, the one queued behind
        it ends too, as the driver ends its whole chain."""
        flight, self.flight = self.flight, None
        if flight is not None:
            flight.end(outcome, reason, self.q - self.ref)
        if outcome != "done" and self.queued is not None:
            queued, self.queued = self.queued, None
            queued.end(outcome, reason)

    def _new_move(self):
        self.moves += 1
        self.worst, self.closest = {}, {}

    # --------------------------------------------------------------- physics
    def step(self, n):
        """`n` physics steps, the reference following the flight."""
        m, d = self.model, self.data
        for _ in range(n):
            if self.flight is not None:
                self.flight.steps += 1
                self.ref = self.flight.reference()
                if self.flight.elapsed >= self.flight.times[-1]:
                    self._end("done", None)
                    if self.queued is not None:
                        # from where this one ended, with no pause
                        flight, self.queued = self.queued, None
                        self._take_off(flight)
            d.ctrl[self.actuators] = self.ref
            mujoco.mj_step(m, d)
            self.steps += 1
            if self.steps % SAMPLE_STEPS == 0:
                self._sample()

    def close(self):
        """Stop for good: nothing will step the sim again. The flight under
        way and the one queued end; so does any goal whose flight the sim
        never took up (Driver._execute watches `stopping`)."""
        self.stopping.set()
        self._end("preempted", STOPPING)

    def run(self, view, ok):
        """Step in real time, carry out commands and render for `view` at
        its frame rate, while `ok()`."""
        clock = next_frame = time.monotonic()
        while ok():
            self.poll()
            now = time.monotonic()
            due = int((now - clock) / TIMESTEP)
            if due * TIMESTEP > BEHIND_S:
                due = int(BEHIND_S / TIMESTEP)
                clock = now - due * TIMESTEP
            self.step(due)
            clock += due * TIMESTEP
            if now >= next_frame:
                view.render(self)
                next_frame = now + 1.0 / view.FPS
            time.sleep(0.001)

    # ------------------------------------------------- contacts and distance
    def _name(self, g):
        """A shape's name in a report: the stand-in's part or the MEBot's link."""
        m = self.model
        name = m.geom(g).name or m.body(m.geom_bodyid[g]).name
        return name.split("/", 1)[-1]

    def _sample(self):
        m, d = self.model, self.data
        now = {}
        force = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            a, g = int(contact.geom1), int(contact.geom2)
            if g not in self.group_of:
                a, g = g, a
            if g not in self.group_of or a not in self.armset:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            group = self.group_of[g]
            if force[0] > CONTACT_N and (
                group not in now or force[0] > now[group]["force_n"]
            ):
                now[group] = {
                    "group": group,
                    "link": m.body(m.geom_bodyid[a]).name,
                    "part": self._name(g),
                    "force_n": round(float(force[0]), 1),
                    "depth_mm": round(-1e3 * float(contact.dist), 1),
                }
        for group, contact in now.items():
            if (
                group not in self.worst
                or contact["force_n"] > self.worst[group]["force_n"]
            ):
                self.worst[group] = contact
        for group in scene.GROUPS:
            closest = self._closest(group)
            if closest and (
                group not in self.closest or closest["mm"] < self.closest[group]["mm"]
            ):
                self.closest[group] = closest
        flight = self.flight
        state = {
            "t": round(float(d.time), 3),
            "q": self.q.tolist(),
            "v": d.qvel[self.dof].tolist(),
            "effort": d.actuator_force[self.actuators].tolist(),
            "ref": self.ref.tolist(),
            "knuckle": float(d.qpos[self.knuckle]),
            "moving": flight is not None,
            "progress": None
            if flight is None
            else min(1.0, flight.elapsed / max(flight.times[-1], TIMESTEP)),
            "posture": self.posture or "none",
            "move": self.moves,
            "now": list(now.values()),
            "worst": list(self.worst.values()),
            "closest": dict(self.closest),
        }
        with self.lock:
            self.state = state

    def snapshot(self):
        """The sim's state at its last 10 ms sample, for other threads."""
        with self.lock:
            return dict(self.state)

    def vertex_distance(self, a, g):
        """The least signed distance from a hull vertex of arm geom `a` to shape
        `g`: an upper bound on their distance, the hull's edges and faces
        being left out."""
        d = self.data
        points = self.hulls[a] @ d.geom_xmat[a].reshape(3, 3).T + d.geom_xpos[a]
        return float(_signed_distance(self.model, d, g, points).min())

    def _touching(self, a, g):
        d = self.data
        return any(
            {int(d.contact[i].geom1), int(d.contact[i].geom2)} == {a, g}
            for i in range(d.ncon)
        )

    def distance(self, a, g, distmax=DISTMAX):
        """mj_geomDistance between arm geom `a` and shape `g`, guarded.

        In MuJoCo 3.10 it returns false zeros and flipped signs for some pairs
        of an arm mesh and a MEBot primitive: a prototype of this sim saw 237
        to 584 false zeros per trajectory, and -346 mm for a pair 346 mm
        apart. So a value at or below zero counts only when the pair is in
        contact or a vertex of the arm's hull is inside the shape. Otherwise
        the query is made again with a distmax just past the hull's vertex
        bound, and the bound stands in if that is not positive either.
        """
        m, d = self.model, self.data
        value = mujoco.mj_geomDistance(m, d, a, g, distmax, None)
        if value > 0:
            return value
        bound = self.vertex_distance(a, g)
        if bound <= 0 or self._touching(a, g):
            return value
        again = mujoco.mj_geomDistance(m, d, a, g, min(distmax, bound + 0.005), None)
        return min(again, bound) if again > 0 else bound

    def _closest(self, group):
        """The group's closest approach to the moving arm: {mm, link, part},
        or None when it has no shape within DISTMAX."""
        m, d = self.model, self.data
        shapes = self.groups[group]
        if not len(shapes):
            return None
        moving = self.moving
        if group == "floor":  # a plane has no bounding sphere to sort by
            gap = np.zeros((len(moving), 1))
        else:
            centres = np.linalg.norm(
                d.geom_xpos[moving][:, None] - d.geom_xpos[shapes][None], axis=2
            )
            gap = centres - m.geom_rbound[moving][:, None] - m.geom_rbound[shapes][None]
        best, pair = DISTMAX, None
        for index in np.argsort(gap, axis=None):
            i, j = divmod(int(index), gap.shape[1])
            if gap[i, j] >= best:
                break
            a, g = moving[i], int(shapes[j])
            value = self.distance(a, g, best)
            if value < best:
                best, pair = value, (a, g)
        if pair is None:
            return None
        return {
            "mm": round(best * 1e3, 1),
            "link": m.body(m.geom_bodyid[pair[0]]).name,
            "part": self._name(pair[1]),
        }


class Driver:
    """The ROS side of /mebot_sim: the arm driver's names and the sim's own."""

    def __init__(self, node, sim):
        from rammp_arm_interfaces.action import ExecuteJointTrajectory
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger

        self.node, self.sim = node, sim
        self.action = ExecuteJointTrajectory
        self.answer = GoalResponse
        group = ReentrantCallbackGroup()
        self.joint_states = node.create_publisher(
            JointState, "/joint_states", qos_profile_sensor_data
        )
        self.contacts = node.create_publisher(String, "/mebot_sim/contacts", 10)
        self.JointState, self.String = JointState, String
        node.create_timer(0.02, self._publish_joints, callback_group=group)
        node.create_timer(0.1, self._publish_contacts, callback_group=group)
        node.create_subscription(
            Float64MultiArray,
            "/mebot_sim/teleport",
            self._teleport,
            10,
            callback_group=group,
        )
        node.create_subscription(
            String, "/mebot_sim/posture", self._posture, 10, callback_group=group
        )
        node.create_service(
            Trigger, "/mebot_sim/reset", self._reset, callback_group=group
        )
        ActionServer(
            node,
            ExecuteJointTrajectory,
            "/execute_joint_trajectory",
            self._execute,
            goal_callback=self._goal,
            cancel_callback=lambda _handle: CancelResponse.ACCEPT,
            callback_group=group,
        )

    def _publish_joints(self):
        state = self.sim.snapshot()
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        # NaN for what is not measured, as the driver publishes the gripper
        msg.name = stack.JOINTS + [KNUCKLE]
        msg.position = state["q"] + [state["knuckle"]]
        msg.velocity = state["v"] + [math.nan]
        msg.effort = state["effort"] + [math.nan]
        self.joint_states.publish(msg)

    def _publish_contacts(self):
        state = self.sim.snapshot()
        keys = ("t", "posture", "moving", "move", "now", "worst", "closest")
        self.contacts.publish(self.String(data=json.dumps({k: state[k] for k in keys})))

    def _teleport(self, msg):
        q = list(msg.data)
        if len(q) != len(stack.JOINTS) or not all(math.isfinite(v) for v in q):
            self.node.get_logger().error(
                "teleport ignored: seven finite joint positions expected, got %s" % q
            )
            return
        self.sim.post("teleport", q)

    def _posture(self, msg):
        name = msg.data
        if name != "none" and name not in self.sim.assets.postures["postures"]:
            self.node.get_logger().error(
                "posture %r ignored: one of %s or none expected"
                % (name, ", ".join(self.sim.assets.postures["postures"]))
            )
            return
        self.sim.post("posture", None if name == "none" else name)

    def _reset(self, _request, response):
        self.sim.post("reset")
        response.success, response.message = True, "the arm is back at its start"
        return response

    def _goal(self, goal):
        why = trajectory_problem(goal.trajectory)
        if why is not None:
            self.node.get_logger().warning("rejecting goal: %s" % why)
            return self.answer.REJECT
        return self.answer.ACCEPT

    def _execute(self, handle):
        request = handle.request
        times, rows = stack.trajectory_arrays(request.trajectory)
        flight = Flight(times, rows, queue=request.preemption != LATEST_WINS)
        self.sim.post("fly", flight)
        result = self.action.Result()
        feedback = self.action.Feedback(joint_names=list(stack.JOINTS))
        while not flight.ended.wait(0.05):
            if handle.is_cancel_requested:
                self.sim.post("stop", flight)
                flight.ended.wait(1.0)
                handle.canceled()
                result.error_code = PREEMPTED
                result.error_string = (
                    "PREEMPTED: cancelled; the arm holds where it stopped"
                )
                return self._final(result, flight)
            if self.sim.stopping.is_set():
                # a flight the sim had not taken up would never end, and the
                # node's shutdown waits for this goal
                break
            if not flight.steps:
                continue  # queued: the driver gives feedback on the goal it flies
            state = self.sim.snapshot()
            feedback.header.stamp = self.node.get_clock().now().to_msg()
            feedback.desired.positions = state["ref"]
            feedback.actual.positions = state["q"]
            feedback.actual.velocities = state["v"]
            feedback.error.positions = [a - r for a, r in zip(state["q"], state["ref"])]
            feedback.fraction_complete = float(state["progress"] or 0.0)
            handle.publish_feedback(feedback)
        if flight.outcome == "done":
            handle.succeed()
            result.error_code = SUCCESSFUL
        else:
            handle.abort()
            result.error_code = PREEMPTED
            # a flight the sim never took up was never ended, so has no reason
            result.error_string = "PREEMPTED by %s" % (flight.reason or STOPPING)
        return self._final(result, flight)

    @staticmethod
    def _final(result, flight):
        if flight.final_error is not None:
            result.final_error.positions = flight.final_error.tolist()
        return result


def _refuse_a_second_driver(node):
    """Stop if anything on this domain already serves the driver's names: it
    could be a real arm's driver, on a domain someone chose by hand."""
    time.sleep(1.0)  # let discovery find what is already there
    served = {name for name, _types in node.get_service_names_and_types()}
    if node.count_publishers("/joint_states") or (
        "/execute_joint_trajectory/_action/send_goal" in served
    ):
        raise SystemExit(
            "something on ROS_DOMAIN_ID=%s already serves /joint_states or "
            "/execute_joint_trajectory, perhaps a real arm's driver: the sim arm "
            "will not serve the same names beside it" % os.environ.get("ROS_DOMAIN_ID")
        )


def main(args=None):
    stack.isolate()
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    from mebot_sim import view

    # Not rclpy's signal handlers: they shut the context down at Ctrl-C, under
    # a goal being flown, which could then neither end nor tell its client.
    # SIGINT is a KeyboardInterrupt in the sim's loop instead, even where the
    # sim was started with it ignored, as a script's background jobs are.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = Node("mebot_sim")
    log = node.get_logger()
    posture = node.declare_parameter("posture", "upright_lap").value
    port = int(node.declare_parameter("port", 8765).value)
    hosts = [
        h
        for h in str(node.declare_parameter("host", "127.0.0.1").value).split(",")
        if h
    ]
    menagerie = node.declare_parameter("menagerie", scene.MENAGERIE).value
    humanoid = node.declare_parameter("humanoid", scene.HUMANOID).value
    _refuse_a_second_driver(node)

    assets = scene.Assets(menagerie, humanoid)
    if posture != "none" and posture not in assets.postures["postures"]:
        raise SystemExit(
            "posture:=%s: one of %s or none"
            % (posture, ", ".join(assets.postures["postures"]))
        )
    log.info("MEBot URDF from %s" % assets.urdf_origin)
    sim = Sim(
        assets,
        None if posture == "none" else posture,
        stack.scenarios()["start"],
        offscreen=view.SIZE,
    )
    Driver(node, sim)
    planners = {
        name: stack.Planner(node, name) for name in (stack.CHAIR, stack.CONTROL)
    }
    live = view.View(sim, planners)
    # the ports are taken before the executor spins: an exit while a thread
    # is inside rcl aborts the process
    urls = []
    for host in hosts:
        try:
            urls += live.serve([host], port)
        except OSError as exc:
            raise SystemExit(
                "the live view cannot listen on %s:%d (%s): pass another port:= "
                "or host:=" % (host, port, exc.strerror or exc)
            )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        for url in urls:
            log.info("live view: %s" % url)
        log.info(
            "sim arm up on ROS_DOMAIN_ID=%s, the stand-in %s"
            % (os.environ["ROS_DOMAIN_ID"], posture)
        )
        sim.run(live, rclpy.ok)
    except KeyboardInterrupt:
        pass
    finally:
        # Every goal ends first, PREEMPTED within 50 ms: those flown or
        # queued, one whose flight was posted but never taken up, and one
        # accepted from now on. rclpy sends their results from the executor,
        # which is given half a second before it and the context go; its
        # shutdown waits for every goal's callback to return.
        sim.close()
        time.sleep(0.5)
        live.close()
        executor.shutdown()
        # out of rcl before the node and the context go
        spin.join()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
