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
    from rammp_curobo_ros.conversions import trajectory_to_msg
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )

JOINTS = ["joint_%d" % i for i in range(1, 8)]


def _traj(velocities=True, accelerations=True, n=4, dt=0.02):
    rng = np.random.default_rng(0)
    return Trajectory(
        joint_names=JOINTS,
        positions=rng.normal(size=(n, 7)),
        velocities=rng.normal(size=(n, 7)) if velocities else None,
        accelerations=rng.normal(size=(n, 7)) if accelerations else None,
        dt=dt,
    )


def test_all_derivatives_copied_per_point():
    traj = _traj()
    msg = trajectory_to_msg(traj)
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
