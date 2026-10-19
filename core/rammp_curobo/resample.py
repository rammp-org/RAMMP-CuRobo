"""Dense resampling of cuRobo's optimized knots, consistent in p, v and a.

cuRobo v0.7.8 resamples its ~32 optimized knots to interpolation_dt with
LINEAR_CUDA, which lerps position, velocity and acceleration INDEPENDENTLY.
Whenever the optimized spacing is coarser than interpolation_dt (most
longer plans), the dense positions are chords with a kink at every knot
while the dense accelerations claim a smooth ramp — measured 95-365% apart
on real plans (#15). A consumer interpolating from those derivatives has to
turn each kink with several times the planned acceleration.

This fits one quintic spline through the same knots, on the same time grid
cuRobo used, and samples p, v and a from it, so they agree by construction.
"""

import numpy as np


def knot_times(n_knots, n_dense, dt):
    """cuRobo's grid (warp linear_interpolate_trajectory_kernel): the knots
    are spread so the LAST knot lands one step past the final dense point.
    Pinned to real cuRobo output by core/tests/data's golden fixture."""
    return np.arange(n_knots) * n_dense * dt / (n_knots - 1)


def resample_quintic(knots, linear, dt, tol=1e-4):
    """Resample `knots` (H, dof) onto `linear`'s grid (N points at `dt`).

    `linear` is cuRobo's own dense positions. Returns (p, v, a), each
    (N, dof), or None if `linear` is not the linear resample of `knots` on
    that grid — then the caller keeps cuRobo's output rather than smooth a
    trajectory it cannot identify.

    cuRobo stops by repeating the goal over its last knots, arriving at the
    first of them still moving. A spline through those repeats passes the
    goal and comes back (up to 2.6 mrad over 22 Jetson plans); one that must
    stop AT the first of them costs ~5x the jerk. So the repeats are
    replaced by a single goal knot at the last dense point: the spline runs
    through every knot before them and lands on the goal at rest there. It
    can still pass the goal slightly on a fast arrival (0.52 mrad worst of
    the same 22; cuRobo's chords never do). Plans start from rest too
    (start_joints carries no velocity).
    """
    from scipy.interpolate import make_interp_spline

    knots = np.asarray(knots, dtype=float)
    linear = np.asarray(linear, dtype=float)
    if linear.shape[1:] != knots.shape[1:]:
        return None
    t = knot_times(len(knots), len(linear), dt)
    tq = np.arange(len(linear)) * dt
    lerp = np.stack(
        [np.interp(tq, t, knots[:, j]) for j in range(knots.shape[1])], axis=1
    )
    if not np.abs(lerp - linear).max() <= tol:  # NaN fails too
        return None

    goal = knots[-1]
    first_goal = np.flatnonzero(np.abs(knots - goal).max(axis=1) > 0)
    if first_goal.size == 0:  # start == goal
        return linear.copy(), np.zeros_like(linear), np.zeros_like(linear)
    first_goal = first_goal[-1] + 1
    if t[first_goal - 1] >= tq[-1]:
        return None
    rest = [(1, np.zeros(knots.shape[1])), (2, np.zeros(knots.shape[1]))]
    spline = make_interp_spline(
        np.r_[t[:first_goal], tq[-1]],
        np.vstack([knots[:first_goal], goal]),
        k=5,
        bc_type=(rest, rest),
        axis=0,
    )
    return tuple(spline(tq, nu) for nu in (0, 1, 2))
