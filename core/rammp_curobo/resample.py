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
    The last three knots are equal, so the plan still ends on the goal."""
    return np.arange(n_knots) * n_dense * dt / (n_knots - 1)


def resample_quintic(knots, linear, dt, v0, a0, tol=1e-4):
    """Resample `knots` (H, dof) onto `linear`'s grid (N points at `dt`).

    `linear` is cuRobo's own dense positions. Returns (p, v, a), each
    (N, dof), or None if `linear` is not the linear resample of `knots` on
    that grid — then the caller keeps cuRobo's output rather than smooth a
    trajectory it cannot identify. `v0`/`a0` are the start state's.
    """
    from scipy.interpolate import make_interp_spline

    knots = np.asarray(knots, dtype=float)
    linear = np.asarray(linear, dtype=float)
    t = knot_times(len(knots), len(linear), dt)
    tq = np.arange(len(linear)) * dt
    lerp = np.stack(
        [np.interp(tq, t, knots[:, j]) for j in range(knots.shape[1])], axis=1
    )
    if np.abs(lerp - linear).max() > tol:
        return None
    # cuRobo holds the last three knots equal: the goal is reached at rest.
    rest = np.zeros(knots.shape[1])
    spline = make_interp_spline(
        t,
        knots,
        k=5,
        bc_type=([(1, v0), (2, a0)], [(1, rest), (2, rest)]),
        axis=0,
    )
    return spline(tq), spline(tq, 1), spline(tq, 2)
