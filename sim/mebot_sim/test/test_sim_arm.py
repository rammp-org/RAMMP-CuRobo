"""The simulated arm (mebot_sim/sim_arm.py): physics, flights and contacts.

CPU MuJoCo only, stepped by the test instead of the real-time loop: no ROS
graph and no rendering. It needs what test_scene.py needs, and skips
without it.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mebot_sim import scene, sim_arm, stack  # noqa: E402

try:
    ASSETS = scene.Assets()
except (FileNotFoundError, OSError) as exc:  # pragma: no cover
    pytest.skip("scene inputs unavailable: %s" % exc, allow_module_level=True)

START = np.array(stack.scenarios()["start"])
# A state inside the upright stand-in, found by a seeded search: its gripper
# is 25 mm inside the left hand, and nothing else is touched.
IN_STANDIN = [-0.472, 0.15, 2.264, -2.502, -0.712, 0.39, 2.172]


@pytest.fixture
def sim():
    return sim_arm.Sim(ASSETS, "upright_lap", START)


def run(sim, seconds):
    sim.poll()
    sim.step(int(round(seconds / sim_arm.TIMESTEP)))


def straight(to, seconds, dt=0.02):
    """A trajectory from START to START + `to`, as the planner stamps one:
    point k at (k + 1) * dt."""
    n = int(round(seconds / dt))
    times = [(k + 1) * dt for k in range(n)]
    rows = [START + np.asarray(to) * (k + 1) / n for k in range(n)]
    return times, rows


AWAY = [0.3, -0.2, 0.2, 0.2, 0.3, -0.2, 0.3]


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
    sim.post("fly", first)
    run(sim, 0.2)
    second = sim_arm.Flight(*straight([0.1] * 7, 0.2))
    sim.post("fly", second)
    run(sim, 0.0)
    assert first.outcome == "preempted" and "newer" in first.reason
    run(sim, 0.3)
    assert second.outcome == "done"


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
    flight = sim_arm.Flight(*straight(AWAY, 2.0))
    sim.post("fly", flight)
    run(sim, 0.1)
    sim.close()
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
    assert worst["part"] in {g["name"] for g in scene.standin(ASSETS, "upright_lap")[0]}
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
    msgs = pytest.importorskip("trajectory_msgs.msg")

    def trajectory(rows, times, names=stack.JOINTS):
        traj = msgs.JointTrajectory(joint_names=list(names))
        for row, t in zip(rows, times):
            point = msgs.JointTrajectoryPoint(positions=list(row))
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            traj.points.append(point)
        return traj

    good = [[0.0] * 7, [0.1] * 7]
    assert sim_arm.trajectory_problem(trajectory(good, [0.02, 0.04])) is None
    for traj, why in (
        (trajectory([], []), "no points"),
        (trajectory(good, [0.02, 0.04], names=stack.JOINTS[:6] + ["x"]), "joint_7"),
        (trajectory([[0.0] * 6, [0.1] * 7], [0.02, 0.04]), "point 0"),
        (trajectory([[0.0] * 7, [float("nan")] * 7], [0.02, 0.04]), "point 1"),
        (trajectory(good, [0.04, 0.04]), "point 1"),
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
