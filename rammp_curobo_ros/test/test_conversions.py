"""trajectory_to_msg carries every derivative the planner produced — no ROS
runtime, no hardware.

Issue #15: accelerations were dropped at this boundary, so a consumer that
picks its interpolation order from what the points carry (as
joint_trajectory_controller and the kinova driver do) could never reach
quintic, and cubic's acceleration steps were audible on the arm.
"""

import numpy as np
import pytest

try:
    from rammp_curobo import Trajectory
    from rammp_curobo_ros.conversions import trajectory_from_msg, trajectory_to_msg
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )

JOINTS = ["arm_joint_%d" % i for i in range(1, 8)]


def _traj(velocities=True, accelerations=True, n=4, dt=0.02):
    rng = np.random.default_rng(0)
    return Trajectory(
        joint_names=JOINTS,
        positions=rng.normal(size=(n, 7)),
        velocities=rng.normal(size=(n, 7)) if velocities else None,
        accelerations=rng.normal(size=(n, 7)) if accelerations else None,
        dt=dt,
        frame_id="arm_mount",
    )


def test_all_derivatives_copied_per_point():
    traj = _traj()
    msg = trajectory_to_msg(traj)
    assert msg.header.frame_id == "arm_mount"
    assert list(msg.joint_names) == JOINTS
    assert len(msg.points) == traj.n_points
    for k, p in enumerate(msg.points):
        np.testing.assert_allclose(p.positions, traj.positions[k])
        np.testing.assert_allclose(p.velocities, traj.velocities[k])
        np.testing.assert_allclose(p.accelerations, traj.accelerations[k])


def test_absent_accelerations_stay_empty():
    """Empty, not zeros: a zero profile would claim quintic data that the
    planner never produced."""
    msg = trajectory_to_msg(_traj(accelerations=False))
    assert all(len(p.velocities) == 7 for p in msg.points)
    assert all(len(p.accelerations) == 0 for p in msg.points)


def test_absent_velocities_stay_empty():
    msg = trajectory_to_msg(_traj(velocities=False, accelerations=False))
    assert all(len(p.velocities) == 0 for p in msg.points)


def test_stamping_unchanged():
    """Point k at (k + 1) * dt — see the issue's consumer-side note."""
    msg = trajectory_to_msg(_traj(dt=0.02))
    t = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    np.testing.assert_allclose(t, [0.02, 0.04, 0.06, 0.08])


def test_from_msg_round_trips_in_the_planner_joint_order():
    traj = _traj()
    msg = trajectory_to_msg(traj)
    back, why = trajectory_from_msg(msg, JOINTS, "arm_mount")
    assert why is None and back.dt == pytest.approx(0.02)
    np.testing.assert_allclose(back.positions, traj.positions)
    np.testing.assert_allclose(back.velocities, traj.velocities)
    np.testing.assert_allclose(back.accelerations, traj.accelerations)
    # the same trajectory, its joints listed in another order
    shuffled, why = trajectory_from_msg(msg, JOINTS[::-1])
    assert why is None
    np.testing.assert_allclose(shuffled.positions, traj.positions[:, ::-1])


def test_from_msg_refuses_what_the_planner_could_not_have_produced():
    msg = trajectory_to_msg(_traj(accelerations=False))
    assert trajectory_from_msg(msg, JOINTS)[0].accelerations is None
    assert "joints" in trajectory_from_msg(msg, JOINTS[:6])[1]
    msg.points[2].time_from_start.nanosec += 5_000_000
    assert trajectory_from_msg(msg, JOINTS)[1] == "points are not evenly spaced in time"
    no_v = trajectory_to_msg(_traj(velocities=False, accelerations=False))
    assert "velocities" in trajectory_from_msg(no_v, JOINTS)[1]
    nan = trajectory_to_msg(_traj())
    nan.points[1].velocities[3] = float("nan")
    assert "non-finite" in trajectory_from_msg(nan, JOINTS)[1]
    twice = trajectory_to_msg(_traj())
    twice.joint_names = JOINTS[:6] + JOINTS[:1]
    assert "joints" in trajectory_from_msg(twice, JOINTS)[1]
