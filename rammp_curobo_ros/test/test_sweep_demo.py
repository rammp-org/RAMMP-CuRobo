"""Sweep demo decision logic — offline, no ROS graph, no GPU.

These two functions decide when a moving arm gets interrupted, so they
are worth pinning down away from the hardware.
"""

import types

import numpy as np

from rammp_curobo_ros.sweep_demo import SweepDemo, traj_index


def verdict(collision_free=True, min_clearance=float("inf")):
    return types.SimpleNamespace(
        collision_free=collision_free, min_clearance=min_clearance, message=""
    )


def demo(margin=0.0):
    return types.SimpleNamespace(margin=margin)


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
    # cuRobo's collision_activation_distance (0.03) already fires
    # collision_free=False at 3 cm; the margin is opt-in on top of that
    assert SweepDemo.tripped(demo(0.0), verdict(min_clearance=0.001)) is False


def test_traj_index_tracks_dilated_time():
    times = np.arange(1, 101) * 0.02          # 100 points, 2 s at full speed
    assert traj_index(times, 0.0, 1.0) == 0
    # at half speed the arm is only half way through the trajectory after
    # a full trajectory-duration of wall clock
    assert traj_index(times, 2.0, 0.5) == traj_index(times, 1.0, 1.0)
    assert traj_index(times, 1.0, 0.5) < traj_index(times, 1.0, 1.0)
    # past the end clamps to len(times), i.e. an empty remaining span
    assert traj_index(times, 100.0, 1.0) == len(times)
