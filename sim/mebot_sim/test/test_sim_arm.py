"""The simulated arm (mebot_sim/sim_arm.py): physics, flights and contacts,
and the driver's ROS side.

CPU MuJoCo only, stepped by the test instead of the real-time loop: no ROS
graph and no rendering. The driver's side runs on a stub node, with the
messages of ROS and of the interfaces overlay. It needs what test_scene.py
needs, and skips without it.
"""

import math
import threading
import time
import types

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mebot_sim import scene, sim_arm, stack  # noqa: E402

START = np.array(stack.scenarios()["start"])
# A state inside the upright stand-in, found by a seeded search: its gripper
# is 25 mm inside the left hand, and nothing else is touched.
IN_STANDIN = [-0.472, 0.15, 2.264, -2.502, -0.712, 0.39, 2.172]


@pytest.fixture
def sim(assets):
    return sim_arm.Sim(assets, "upright_lap", START)


def run(sim, seconds):
    sim.poll()
    sim.step(int(round(seconds / sim_arm.TIMESTEP)))


def straight(to, seconds, dt=0.02, start=START):
    """A trajectory from `start` to `start` + `to`, as the planner stamps one:
    point k at (k + 1) * dt."""
    n = int(round(seconds / dt))
    times = [(k + 1) * dt for k in range(n)]
    rows = [start + np.asarray(to) * (k + 1) / n for k in range(n)]
    return times, rows


def trajectory(times, rows, names=stack.JOINTS):
    """A trajectory_msgs/JointTrajectory: `rows` of positions at `times`."""
    msgs = pytest.importorskip("trajectory_msgs.msg")
    traj = msgs.JointTrajectory(joint_names=list(names))
    for t, row in zip(times, rows):
        point = msgs.JointTrajectoryPoint(positions=[float(v) for v in row])
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        traj.points.append(point)
    return traj


AWAY = [0.3, -0.2, 0.2, 0.2, 0.3, -0.2, 0.3]
# from where AWAY ends back to START
BACK = straight(np.negative(AWAY), 1.0, start=START + AWAY)


def test_a_trajectory_is_flown_on_its_own_time_base(sim):
    flight = sim_arm.Flight(*straight(AWAY, 1.0))
    sim.post("fly", flight)
    run(sim, 0.98)
    assert not flight.ended.is_set()
    run(sim, 0.03)
    assert flight.ended.is_set() and flight.outcome == "done"
    # the arm lags its reference a little, then settles on the last point
    assert np.abs(flight.final_error).max() < 0.1
    run(sim, 0.5)
    assert np.abs(sim.q - (START + AWAY)).max() < 0.02


def test_the_latest_trajectory_wins(sim):
    first = sim_arm.Flight(*straight(AWAY, 2.0))
    queued = sim_arm.Flight(*BACK, queue=True)
    sim.post("fly", first)
    sim.post("fly", queued)
    run(sim, 0.2)
    second = sim_arm.Flight(*straight([0.1] * 7, 0.2))
    sim.post("fly", second)
    run(sim, 0.0)
    # and the one queued behind it, as the driver drops it
    for flight in (first, queued):
        assert flight.outcome == "preempted" and "newer" in flight.reason
    run(sim, 0.3)
    assert second.outcome == "done"


def test_a_queued_trajectory_waits_for_the_one_under_way(sim):
    first = sim_arm.Flight(*straight(AWAY, 1.0))
    second, third = (sim_arm.Flight(*BACK, queue=True) for _ in range(2))
    sim.post("fly", first)
    run(sim, 0.2)
    sim.post("fly", second)
    sim.post("fly", third)
    run(sim, 0.0)
    # the driver keeps one trajectory queued: the newer
    assert second.outcome == "preempted" and "newer" in second.reason
    assert not first.ended.is_set() and not third.ended.is_set()
    run(sim, 0.85)
    # flown when the first is done, from where it ended
    assert first.outcome == "done" and sim.flight is third
    run(sim, 1.05)
    assert third.outcome == "done"
    run(sim, 0.5)
    assert np.abs(sim.q - START).max() < 0.02


def test_a_stop_ends_the_flight_and_the_one_queued_behind_it(sim):
    # as a cancel ends the driver's whole chain, whichever goal it names
    first = sim_arm.Flight(*straight(AWAY, 2.0))
    queued = sim_arm.Flight(*BACK, queue=True)
    sim.post("fly", first)
    sim.post("fly", queued)
    run(sim, 0.3)
    sim.post("stop", queued)
    run(sim, 0.0)
    for flight in (first, queued):
        assert flight.outcome == "preempted" and "stop" in flight.reason
    assert sim.flight is None and sim.queued is None


def test_a_stopped_arm_holds_where_it_is(sim):
    flight = sim_arm.Flight(*straight(AWAY, 2.0))
    sim.post("fly", flight)
    run(sim, 0.5)
    sim.post("stop", flight)
    run(sim, 0.0)
    assert flight.outcome == "preempted"
    held = sim.ref.copy()
    run(sim, 0.5)
    assert np.array_equal(sim.ref, held)
    assert np.abs(sim.q - held).max() < 0.02
    # nowhere near where the flight was going
    assert np.abs(sim.q - (START + AWAY)).max() > 0.1


def test_a_flight_under_way_ends_when_the_sim_stops(sim):
    # its goal is waiting on it, and the node's shutdown on the goal
    under_way = sim_arm.Flight(*straight(AWAY, 2.0))
    queued = sim_arm.Flight(*BACK, queue=True)
    sim.post("fly", under_way)
    sim.post("fly", queued)
    run(sim, 0.1)
    sim.close()
    for flight in (under_way, queued):
        assert flight.ended.is_set() and flight.outcome == "preempted"
        assert "stopping" in flight.reason


def test_teleport_and_reset(sim):
    flight = sim_arm.Flight(*straight(AWAY, 2.0))
    sim.post("fly", flight)
    run(sim, 0.1)
    there = START + [0.2] * 7
    sim.post("teleport", there)
    run(sim, 0.0)
    assert flight.outcome == "preempted"
    assert sim.q == pytest.approx(there) and sim.ref == pytest.approx(there)
    sim.post("reset")
    run(sim, 0.0)
    assert sim.q == pytest.approx(START)


def test_contacts_over_a_newton_are_reported_by_group(sim):
    run(sim, 0.05)
    state = sim.snapshot()
    assert state["now"] == [] and state["worst"] == []
    moves = state["move"]

    # flown into the stand-in, the arm's wrist meets its right upper arm
    flight = sim_arm.Flight(*straight(np.subtract(IN_STANDIN, START), 3.0))
    sim.post("fly", flight)
    run(sim, 4.0)
    state = sim.snapshot()
    assert flight.outcome == "done" and state["move"] == moves + 1
    (worst,) = state["worst"]
    assert worst["group"] == "standin" and worst["force_n"] > 1.0
    arm = {sim.model.body(sim.model.geom_bodyid[g]).name for g in sim.arm}
    assert worst["link"] in arm
    standin = scene.standin(sim.assets, "upright_lap")[0]
    assert worst["part"] in {g["name"] for g in standin}
    # pressed into it, the closest approach is a real penetration: below zero
    assert state["closest"]["standin"]["mm"] < 0

    # a new move starts a new record
    sim.post("teleport", START)
    run(sim, 0.05)
    state = sim.snapshot()
    assert state["now"] == [] and state["worst"] == []
    assert state["closest"]["standin"]["mm"] > 0


def test_closest_approach_is_reported_per_group(sim):
    run(sim, 0.05)
    closest = sim.snapshot()["closest"]
    assert set(closest) == {"mebot", "standin", "floor"}
    for group, entry in closest.items():
        assert 0 < entry["mm"] < 1000, group
        # an upper bound: the arm's hull vertices against that group's shapes
        # (the report is rounded to 0.1 mm)
        bound = min(
            sim.vertex_distance(a, g) for a in sim.moving for g in sim.groups[group]
        )
        assert entry["mm"] <= bound * 1e3 + 0.05 + 1e-9, group


def test_a_distance_at_or_below_zero_counts_only_for_a_real_touch(sim, monkeypatch):
    run(sim, 0.05)
    # a pair well apart: the gripper's base and the MEBot's left drive wheel
    a = next(
        g
        for g in sim.moving
        if sim.model.body(sim.model.geom_bodyid[g]).name == "g_base"
    )
    g = next(
        g
        for g in sim.groups["mebot"]
        if sim.model.body(sim.model.geom_bodyid[g]).name == "mebot/drive_wheel_l"
    )
    bound = sim.vertex_distance(a, g)
    real = sim.distance(a, g)
    assert 0.001 < real <= bound + 1e-9

    answers = []
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: answers.pop(0))
    # MuJoCo 3.10's false zero, twice: the hull's vertex bound stands in
    answers[:] = [0.0, 0.0]
    assert sim.distance(a, g) == bound
    # its sign flip: a second, tighter query that is positive is believed
    answers[:] = [-0.346, 0.346]
    assert sim.distance(a, g) == 0.346
    assert answers == []
    # (a real penetration keeping its sign is in the contact test above)


def test_the_driver_interface_refuses_a_malformed_trajectory():
    good = [[0.0] * 7, [0.1] * 7]
    assert sim_arm.trajectory_problem(trajectory([0.02, 0.04], good)) is None
    eight = [[0.0] * 8, [0.1] * 8]
    for traj, why in (
        (trajectory([], []), "no points"),
        # the driver takes positions as joint_1..joint_7, never by name
        (trajectory([0.02, 0.04], good, names=stack.JOINTS[::-1]), "in order"),
        (trajectory([0.02, 0.04], good, names=stack.JOINTS[:6] + ["x"]), "in order"),
        (trajectory([0.02, 0.04], eight, names=stack.JOINTS + ["joint_8"]), "in order"),
        # and seven of them in every point
        (trajectory([0.02, 0.04], [[0.0] * 6, [0.1] * 7]), "point 0"),
        (trajectory([0.02, 0.04], [[0.0] * 7, [0.1] * 8]), "point 1"),
        (trajectory([0.02, 0.04], [[0.0] * 7, [float("nan")] * 7]), "point 1"),
        (trajectory([0.04, 0.04], good), "point 1"),
    ):
        assert why in sim_arm.trajectory_problem(traj)


def test_the_posture_switch_rebuilds_the_standin_and_keeps_the_arm(sim):
    flight = sim_arm.Flight(*straight(AWAY, 2.0))
    sim.post("fly", flight)
    run(sim, 0.3)
    q = sim.q.copy()
    sim.post("posture", None)
    run(sim, 0.0)
    assert flight.outcome == "preempted"
    assert len(sim.groups["standin"]) == 0 and sim.snapshot()["posture"] == "none"
    assert sim.q == pytest.approx(q)
    sim.post("posture", "lean_lap")
    run(sim, 0.0)
    assert len(sim.groups["standin"]) > 0 and sim.snapshot()["posture"] == "lean_lap"


# ------------------------------------------------------ the driver's ROS side


class Node:
    """What Driver and _refuse_a_second_driver ask of an rclpy node, answered
    without a ROS graph. What the driver makes is kept: `made` by topic or
    service name, `timers` as (period, callback). The graph holds
    `publishers` of /joint_states and the services `served`."""

    def __init__(self, publishers=0, served=()):
        self.publishers, self.served = publishers, served
        self.made, self.timers, self.logged = {}, [], []

    def create_publisher(self, kind, topic, qos):
        publisher = types.SimpleNamespace(qos=qos, sent=[])
        publisher.publish = publisher.sent.append
        self.made[topic] = publisher
        return publisher

    def create_timer(self, period, callback, callback_group=None):
        self.timers.append((period, callback))

    def create_subscription(self, kind, topic, callback, qos, callback_group=None):
        self.made[topic] = callback

    def create_service(self, kind, name, callback, callback_group=None):
        self.made[name] = callback

    def get_logger(self):
        return types.SimpleNamespace(
            warning=self.logged.append, error=self.logged.append
        )

    def get_clock(self):
        from rclpy.clock import Clock

        return Clock()

    def count_publishers(self, topic):
        return self.publishers if topic == "/joint_states" else 0

    def get_service_names_and_types(self):
        return [(name, []) for name in self.served]


@pytest.fixture
def driver(sim, monkeypatch):
    """A Driver of `sim` on a stub Node. The action server it would serve is
    kept in `served`: its name, execute callback and keyword arguments."""
    action = pytest.importorskip("rclpy.action")
    pytest.importorskip("rammp_arm_interfaces.action")
    served = {}

    def server(node, kind, name, execute, **options):
        served.update(options, name=name, execute=execute)

    monkeypatch.setattr(action, "ActionServer", server)
    node = Node()
    return types.SimpleNamespace(
        driver=sim_arm.Driver(node, sim), node=node, served=served, sim=sim
    )


class Goal:
    """A goal being executed, as Driver._execute sees rclpy's goal handle:
    its request (with `preemption` 0, QUEUE, as a client leaves it, or 1,
    LATEST_WINS), a cancel its client may ask for, and the state and the
    feedback the driver gives it."""

    def __init__(self, times, rows, preemption=0):
        from rammp_arm_interfaces.action import ExecuteJointTrajectory

        self.request = ExecuteJointTrajectory.Goal(
            trajectory=trajectory(times, rows), preemption=preemption
        )
        self.cancel = threading.Event()
        self.state, self.feedback, self.result = "executing", [], None

    @property
    def is_cancel_requested(self):
        return self.cancel.is_set()

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.state = "succeeded"

    def abort(self):
        self.state = "aborted"

    def canceled(self):
        self.state = "canceled"


def execute(driver, goal):
    """Driver._execute(goal) on a thread of its own, as rclpy's executor runs
    it; the result lands in goal.result."""

    def run():
        goal.result = driver._execute(goal)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def step_until(sim, done, timeout=30.0):
    """Step the sim, carrying out what is posted, until done(): the real-time
    loop's part, at about real time, since a goal looks at its flight every
    50 ms."""
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "not done after %.0f s" % timeout
        sim.poll()
        sim.step(sim_arm.SAMPLE_STEPS)
        time.sleep(0.005)


def progress(sim):
    return sim.snapshot()["progress"] or 0.0


def test_joint_states_are_published_as_the_driver_publishes_them(driver):
    from rclpy.qos import ReliabilityPolicy

    node = driver.node
    (publish,) = [cb for period, cb in node.timers if period == pytest.approx(0.02)]
    publish()  # at 50 Hz
    joint_states = node.made["/joint_states"]
    assert joint_states.qos.reliability == ReliabilityPolicy.BEST_EFFORT
    (msg,) = joint_states.sent
    assert list(msg.name) == stack.JOINTS + [sim_arm.KNUCKLE]
    assert list(msg.position[:7]) == pytest.approx(driver.sim.q)
    # the gripper's speed and effort are not measured: NaN, as the driver
    # publishes them
    assert all(math.isfinite(v) for v in list(msg.velocity[:7]) + list(msg.effort[:7]))
    assert math.isnan(msg.velocity[7]) and math.isnan(msg.effort[7])


def test_a_malformed_goal_is_rejected_and_a_cancel_always_taken(driver):
    from rclpy.action import CancelResponse, GoalResponse

    served = driver.served
    assert served["name"] == "/execute_joint_trajectory"
    good = types.SimpleNamespace(trajectory=trajectory(*straight(AWAY, 1.0)))
    bad = types.SimpleNamespace(trajectory=trajectory([0.02], [[0.0] * 6]))
    assert served["goal_callback"](good) == GoalResponse.ACCEPT
    assert served["goal_callback"](bad) == GoalResponse.REJECT
    assert "point 0" in driver.node.logged[-1]
    assert served["cancel_callback"](None) == CancelResponse.ACCEPT


def test_a_goal_flown_to_its_end_succeeds(driver):
    goal = Goal(*straight(AWAY, 1.0))
    thread = execute(driver.driver, goal)
    step_until(driver.sim, lambda: not thread.is_alive())
    assert goal.state == "succeeded"
    assert goal.result.error_code == sim_arm.SUCCESSFUL
    assert np.abs(goal.result.final_error.positions).max() < 0.1


def test_a_cancelled_goal_is_preempted_and_the_arm_holds(driver):
    sim = driver.sim
    goal = Goal(*straight(AWAY, 4.0))
    thread = execute(driver.driver, goal)
    step_until(sim, lambda: progress(sim) > 0.2)
    goal.cancel.set()
    step_until(sim, lambda: not thread.is_alive())
    assert goal.state == "canceled"
    assert goal.result.error_code == sim_arm.PREEMPTED
    assert "PREEMPTED" in goal.result.error_string
    # feedback while it flew: the joints, where they are told to be and are
    last = goal.feedback[-1]
    assert list(last.joint_names) == stack.JOINTS
    assert len(last.desired.positions) == len(last.actual.positions) == 7
    assert 0 < last.fraction_complete < 1
    held = sim.ref.copy()
    run(sim, 0.5)
    assert np.array_equal(sim.ref, held)
    assert np.abs(sim.q - held).max() < 0.02
    assert np.abs(sim.q - (START + AWAY)).max() > 0.1


def test_a_latest_wins_goal_preempts_the_one_flying(driver):
    sim = driver.sim
    first = Goal(*straight(AWAY, 4.0))
    second = Goal(*straight([0.1] * 7, 0.5), preemption=sim_arm.LATEST_WINS)
    one = execute(driver.driver, first)
    step_until(sim, lambda: progress(sim) > 0.1)
    two = execute(driver.driver, second)
    step_until(sim, lambda: not one.is_alive() and not two.is_alive())
    assert first.state == "aborted"
    assert first.result.error_code == sim_arm.PREEMPTED
    assert "newer" in first.result.error_string
    assert second.state == "succeeded"
    assert second.result.error_code == sim_arm.SUCCESSFUL


def test_a_queued_goal_is_flown_once_the_one_flying_is_done(driver):
    # QUEUE is what a goal says unless it says LATEST_WINS
    sim = driver.sim
    first, second = Goal(*straight(AWAY, 1.0)), Goal(*BACK)
    one = execute(driver.driver, first)
    step_until(sim, lambda: progress(sim) > 0.1)
    two = execute(driver.driver, second)
    step_until(sim, lambda: progress(sim) > 0.5)
    # it waits, given no feedback: the driver gives that to the goal it flies
    assert first.state == second.state == "executing" and not second.feedback
    step_until(sim, lambda: not one.is_alive() and not two.is_alive())
    assert first.state == second.state == "succeeded"
    assert second.feedback
    run(sim, 0.5)
    assert np.abs(sim.q - START).max() < 0.02


def test_every_goal_ends_when_the_sim_stops(driver):
    # A goal whose flight the sim never takes up, posted before its loop
    # stopped or after: it ends all the same, or the node's shutdown waits
    # on it for ever.
    sim = driver.sim
    before, after = Goal(*straight(AWAY, 1.0)), Goal(*straight(AWAY, 1.0))
    threads = [execute(driver.driver, before)]
    deadline = time.monotonic() + 5.0
    while sim.commands.empty():  # posted, and never carried out
        assert time.monotonic() < deadline
        time.sleep(0.01)
    sim.close()
    threads.append(execute(driver.driver, after))
    for goal, thread in zip((before, after), threads):
        thread.join(5.0)
        assert not thread.is_alive()
        assert goal.state == "aborted"
        assert goal.result.error_code == sim_arm.PREEMPTED
        assert "stopping" in goal.result.error_string


class Refused(Exception):
    """_refuse_a_second_driver was called."""


def test_the_sim_arm_looks_for_another_driver_before_it_serves_anything(
    monkeypatch,
):
    rclpy = pytest.importorskip("rclpy")
    import rclpy.node

    class Started:
        """The node main() starts, as far as it uses it before it serves."""

        def __init__(self, name):
            pass

        def get_logger(self):
            return types.SimpleNamespace(info=lambda message: None)

        def declare_parameter(self, name, value):
            return types.SimpleNamespace(value=value)

    def refuse(node):
        raise Refused()

    def serve(node, sim):
        raise AssertionError("the driver's names served before the check")

    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr(rclpy, "init", lambda **options: None)
    monkeypatch.setattr(rclpy.node, "Node", Started)
    monkeypatch.setattr(sim_arm, "_refuse_a_second_driver", refuse)
    monkeypatch.setattr(sim_arm, "Driver", serve)
    with pytest.raises(Refused):
        sim_arm.main()


def test_the_sim_arm_will_not_start_beside_another_driver(monkeypatch):
    # the second it gives discovery is not needed: this graph is a stub's
    monkeypatch.setattr(sim_arm.time, "sleep", lambda seconds: None)
    sim_arm._refuse_a_second_driver(Node())
    for node in (
        Node(publishers=1),
        Node(served=["/execute_joint_trajectory/_action/send_goal"]),
    ):
        with pytest.raises(SystemExit, match="already serves"):
            sim_arm._refuse_a_second_driver(node)
