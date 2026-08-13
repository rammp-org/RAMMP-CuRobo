"""Post-plan validation: never trust a trajectory you didn't check.

cuRobo already enforces limits and collision inside the optimizer, but its
result buffers are PADDED and a stale tail once reached a controller in the
field with violent consequences (RAMMP-Kinova incident log). These checks
are the independent tripwire between "cuRobo said success" and "we hand
this to a real arm": finiteness, position limits, velocity limits, and
step-to-step continuity consistent with the trajectory's own timing.
"""

import numpy as np

from rammp_curobo.geometry import ang_diff

# Allowed step between consecutive points: slack * v_limit * dt. Shared
# with the ROS executor's message-level gate (executor.py imports it) —
# one constant, one tripwire.
CONTINUITY_SLACK = 3.0


def validate_trajectory(
    traj,
    position_limits,
    velocity_limits,
    position_margin=1e-3,
    continuity_slack=CONTINUITY_SLACK,
):
    """Return a list of violation strings — empty means the trajectory passed.

    position_limits: (2, dof) array [lower; upper], controller joint order.
    velocity_limits: (dof,) array of absolute velocity limits, same order.
    continuity_slack: a step between consecutive points may use at most
        slack * v_limit * dt — anything bigger is a discontinuity (stale
        buffer tail, wrong joint order, corrupted plan), not motion.
    """
    problems = []
    pos = np.asarray(traj.positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 1:
        return ["trajectory has no points"]
    if pos.shape[1] != len(traj.joint_names):
        return [
            "%d position columns != %d joint names"
            % (pos.shape[1], len(traj.joint_names))
        ]

    if not np.isfinite(pos).all():
        problems.append("non-finite positions")
    lower = np.asarray(position_limits[0], dtype=float) - position_margin
    upper = np.asarray(position_limits[1], dtype=float) + position_margin
    below, above = pos < lower, pos > upper
    for j, name in enumerate(traj.joint_names):
        if below[:, j].any() or above[:, j].any():
            problems.append(
                "%s exceeds position limits [%.3f, %.3f]: range [%.3f, %.3f]"
                % (name, lower[j], upper[j], pos[:, j].min(), pos[:, j].max())
            )

    vmax = np.asarray(velocity_limits, dtype=float)
    if traj.velocities is not None:
        vel = np.asarray(traj.velocities, dtype=float)
        if not np.isfinite(vel).all():
            problems.append("non-finite velocities")
        for j, name in enumerate(traj.joint_names):
            v = np.abs(vel[:, j]).max()
            if v > vmax[j] * 1.01:
                problems.append(
                    "%s velocity %.3f > limit %.3f rad/s" % (name, v, vmax[j])
                )

    if pos.shape[0] > 1:
        steps = np.abs(np.diff(pos, axis=0))
        allowed = continuity_slack * vmax * traj.dt
        worst = steps / allowed
        k, j = np.unravel_index(np.argmax(worst), worst.shape)
        if worst[k, j] > 1.0:
            problems.append(
                "discontinuity: %s jumps %.3f rad between points %d->%d "
                "(allowed %.3f at dt=%.3fs) — refusing; this is the padded-"
                "tail failure mode"
                % (traj.joint_names[j], steps[k, j], k, k + 1, allowed[j], traj.dt)
            )
    return problems


def start_state_matches(traj, current_positions, tol_rad=0.05):
    """Is the arm (current_positions, controller order) close enough to the
    trajectory's first point to execute it? The catch-up sweep to a stale
    plan is UNPLANNED motion — free to pass through obstacles.

    Wrap-aware: reported angles of continuous joints flip by 2*pi at the
    +/-pi boundary (geometry.ang_diff)."""
    q = np.asarray(current_positions, dtype=float)
    first = np.asarray(traj.positions[0], dtype=float)
    err = float(max(abs(ang_diff(f, c)) for f, c in zip(first, q)))
    return err <= float(tol_rad), err
