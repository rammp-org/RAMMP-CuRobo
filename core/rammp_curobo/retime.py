"""Execution-side speed scaling by exact time dilation.

The planner always plans at the robot's full velocity/acceleration limits
(cuRobo velocity_scale stays 1.0 — plan-time scaling destabilized finetune
retiming in the field and the sim tuning assumes it). Slow execution is a
pure reparameterization instead: play the same geometric path over a longer
time. With q'(t) = q(t * s) for s in (0, 1]:

    dt' = dt / s        v' = v * s        a' = a * s^2

The path (and therefore its collision-freedom) is untouched, and every
scaled velocity/acceleration is strictly below the planned one, so a
trajectory that satisfied the limits still does.
"""

from dataclasses import replace

from rammp_curobo.types import Trajectory


def scale_trajectory(traj: Trajectory, speed_scale: float) -> Trajectory:
    """Return `traj` retimed to `speed_scale` of planned speed (0 < s <= 1).

    Scaling up (> 1.0) is refused: faster-than-planned execution exceeds
    the joint limits the plan was computed against.
    """
    s = float(speed_scale)
    if not 0.0 < s <= 1.0:
        raise ValueError(
            "speed_scale must be in (0, 1], got %g — execution may only "
            "slow a plan down, never speed it up" % s
        )
    if s == 1.0:
        return traj
    return replace(
        traj,
        dt=traj.dt / s,
        velocities=None if traj.velocities is None else traj.velocities * s,
        accelerations=(
            None if traj.accelerations is None else traj.accelerations * (s * s)
        ),
        speed_scale=traj.speed_scale * s,
    )
