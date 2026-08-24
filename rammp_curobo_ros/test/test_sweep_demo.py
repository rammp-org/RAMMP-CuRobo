"""Sweep demo decision logic — offline, no ROS graph, no GPU.

These functions decide when a moving arm gets interrupted, so they are
worth pinning down away from the hardware.
"""

import types

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo_ros.sweep_demo import (
    BLIND_CHECKS,
    SweepDemo,
    blind_update,
    progress_index,
    span_excess_deg,
    traj_index,
    watchdog_index,
)


def verdict(collision_free=True, min_clearance=float("inf")):
    return types.SimpleNamespace(
        collision_free=collision_free, min_clearance=min_clearance, message=""
    )


def demo(margin=0.0):
    return types.SimpleNamespace(margin=margin)


def traj_msg(rows):
    """JointTrajectory from a list of position rows (radians), 1 s apart."""
    msg = JointTrajectory()
    msg.joint_names = ["j%d" % (i + 1) for i in range(len(rows[0]))]
    for k, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.time_from_start.sec = k + 1
        msg.points.append(p)
    return msg


def test_clear_path_does_not_trip():
    assert SweepDemo.tripped(demo(), verdict()) is False


def test_collision_trips():
    assert SweepDemo.tripped(demo(), verdict(collision_free=False)) is True


def test_a_dropped_check_is_not_evidence():
    # a timed-out service call must NOT stop the arm: the demo would then
    # halt on DDS hiccups rather than on obstacles
    assert SweepDemo.tripped(demo(), None) is False


def test_margin_trips_before_contact():
    assert SweepDemo.tripped(demo(0.05), verdict(min_clearance=0.04)) is True
    assert SweepDemo.tripped(demo(0.05), verdict(min_clearance=0.06)) is False


def test_zero_margin_defers_to_curobo():
    # cuRobo's own verdict flips at world_padding (measured 0.020 m); the
    # margin is opt-in on top of that, and 0 means "trust cuRobo"
    assert SweepDemo.tripped(demo(0.0), verdict(min_clearance=0.001)) is False


def test_margin_can_be_suppressed_for_one_stroke():
    # a margin wider than a legitimate plan's own clearance would trip on
    # every fresh plan; run() disables it for that stroke rather than
    # livelocking, but a real collision must still trip
    assert SweepDemo.tripped(demo(0.05), verdict(min_clearance=0.04),
                             use_margin=False) is False
    assert SweepDemo.tripped(demo(0.05), verdict(collision_free=False),
                             use_margin=False) is True


# ------------------------------------------------------- contortion guard

def test_direct_long_travel_has_no_excess():
    # sim starts at q=0, far from the pinned endpoints: the first stroke
    # travels far but DIRECTLY, and must pass — an absolute span cap
    # refused it and retried the identical plan forever (livelock)
    t = traj_msg([[0.0], [1.0], [2.0], [3.0]])         # 172 deg, monotonic
    assert span_excess_deg(t).max() < 1.0              # passes any sane limit


def test_net_zero_contortion_is_all_excess():
    # the bench contortion: j2 arcing over the top and coming back
    t = traj_msg([[0.0], [np.radians(193.0)], [0.0]])
    assert span_excess_deg(t).max() > 120.0            # refused at the default


def test_excess_is_per_joint():
    # j1 travels far but directly; j2 winds out and back — only j2 shows
    t = traj_msg([[0.0, 0.0], [0.5, 2.0], [1.0, 0.1]])
    excess = span_excess_deg(t)
    assert excess[0] < 1.0
    assert excess[1] > 100.0


# ------------------------------------------------------- watchdog indexing

def test_traj_index_tracks_dilated_time():
    times = np.arange(1, 101) * 0.02          # 100 points, 2 s at full speed
    assert traj_index(times, 0.0, 1.0) == 0
    # at half speed the arm is only half way through the trajectory after
    # a full trajectory-duration of wall clock
    assert traj_index(times, 2.0, 0.5) == traj_index(times, 1.0, 1.0)
    assert traj_index(times, 1.0, 0.5) < traj_index(times, 1.0, 1.0)
    # past the end clamps to len(times), i.e. an empty remaining span
    assert traj_index(times, 100.0, 1.0) == len(times)


def test_progress_index_maps_fraction_to_trajectory():
    times = np.arange(1.0, 101.0)             # 1..100 s, exact floats
    assert progress_index(times, 0.0) == 0
    assert progress_index(times, 0.5) == 49   # times[49] == 50.0 == 0.5 * end
    assert progress_index(times, 1.0) == 99   # last point: empty remaining span


def test_watchdog_index_prefers_progress_feedback():
    # the wall clock starts at goal ACCEPTANCE, before the arm moves; once
    # feedback arrives it is ground truth no matter what the clock says
    times = np.arange(1.0, 101.0)
    assert watchdog_index(times, 0.5, 999.0, 1.0) == progress_index(times, 0.5)


def test_watchdog_index_clamps_first_second_without_feedback():
    times = np.arange(1, 101) * 0.02
    # handshake lag: the unchecked gap is exactly where the arm is, so
    # the first second must check from the very start
    assert traj_index(times, 0.5, 1.0) > 0    # the clock alone would skip ahead
    assert watchdog_index(times, None, 0.5, 1.0) == 0
    # after the first second the wall-clock estimate takes over
    assert watchdog_index(times, None, 1.5, 1.0) == traj_index(times, 1.5, 1.0)


# ------------------------------------------------- watchdog fail-open guard

def test_blind_counter_resets_on_any_real_verdict():
    assert blind_update(0, verdict()) == (0, None)
    # a blocked verdict is still a REAL verdict — tripped() handles it
    assert blind_update(2, verdict(collision_free=False)) == (0, None)


def test_first_dropped_check_warns_only():
    assert blind_update(0, None) == (1, "warn")


def test_consecutive_dropped_checks_block():
    count, seen = 0, []
    for _ in range(BLIND_CHECKS):
        count, action = blind_update(count, None)
        seen.append(action)
    assert seen == ["warn", None, "blocked"]


def test_broken_streak_starts_over():
    count, _ = blind_update(0, None)
    count, _ = blind_update(count, None)
    count, action = blind_update(count, verdict())
    assert (count, action) == (0, None)
    assert blind_update(count, None) == (1, "warn")


def test_report_can_change_severity(monkeypatch):
    """rclpy caches a logger context per CALL SITE with the first call's
    severity; logging warning then error from one line raises. The demo
    died on the bench exactly that way: HOLD (warning), then a plan
    failure (error)."""
    import rclpy

    from rammp_curobo_ros.sweep_demo import SweepDemo

    rclpy.init()
    try:
        node = rclpy.create_node("report_probe")
        d = types.SimpleNamespace(_last_report=("", 0.0), get_logger=node.get_logger)
        SweepDemo.report(d, "warning", "HOLD")
        SweepDemo.report(d, "error", "plan failed")      # must not raise
        SweepDemo.report(d, "warning", "HOLD again")
        node.destroy_node()
    finally:
        rclpy.shutdown()
