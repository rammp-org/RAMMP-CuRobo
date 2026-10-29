"""One trajectory through several waypoints, without stopping at the ones between.

plan_through plans a leg to each waypoint (cuRobo's plans, or a straight joint
segment), every leg from rest to rest, and this module OVERLAPS them in time:
leg k+1 starts m steps before leg k has finished, and while both run the arm
moves by the sum of their displacements,

    q(t) = q_start + sum_k (p_k(t - t_k) - p_k(0))

so velocities and accelerations add too. Every leg starts and ends at rest
(zero velocity and acceleration), so the sum is exactly as continuous as the
legs, and its last point is exactly the last leg's goal. At an overlapped
waypoint the path cuts the corner: the arm passes NEAR the waypoint, still
moving, instead of stopping on it.

Nothing here knows the robot. Whether an overlap is acceptable (joint limits,
collision, how far the corner may be cut) is the caller's `accept`, and
overlap 0 is always acceptable: it is the legs as planned, end to end, which
already passed every check.
"""

import numpy as np


def offsets(lengths, overlaps):
    """Each leg's first index on the shared grid: leg k+1 starts overlaps[k]
    steps before leg k's last point (0: on it, the legs end to end)."""
    out = [0]
    for n, m in zip(lengths[:-1], overlaps):
        out.append(out[-1] + n - 1 - m)
    return out


def max_overlaps(lengths, blended):
    """The longest overlap each junction may have, given which junctions
    blend (`blended`, one bool each): a leg overlaps its neighbour by up to
    its whole length, or half of it where it blends at its other end too,
    so no two windows ever meet and each junction can be decided alone."""
    last = len(lengths) - 1
    caps = []
    for k, on in enumerate(blended):
        tail = lengths[k] - 1 if k == 0 or not blended[k - 1] else (lengths[k] - 1) // 2
        nxt = k + 1 < last and blended[k + 1]
        head = (lengths[k + 1] - 1) // 2 if nxt else lengths[k + 1] - 1
        caps.append(min(tail, head) if on else 0)
    return caps


def window(lengths, overlaps, k):
    """The indices junction k's overlap covers: leg k+1's start to leg k's
    end, both legs at rest at either edge."""
    o = offsets(lengths, overlaps)
    return slice(o[k + 1], o[k] + lengths[k])


def blend(legs, overlaps):
    """(p, v, a) of `legs` overlapped by `overlaps`.

    legs: [(p, v, a)], each (N_k, dof) on one time grid, rest to rest, leg
        k+1 starting where leg k ends; a may be None, and then so is the
        result's.
    overlaps: steps per junction, len(legs) - 1 of them, no two windows
        meeting (max_overlaps keeps to that).
    """
    lengths = [len(p) for p, _, _ in legs]
    if len(overlaps) != len(legs) - 1:
        raise ValueError("%d overlaps for %d legs" % (len(overlaps), len(legs)))
    room = [n - 1 for n in lengths]  # each leg's steps not yet overlapped
    for k, m in enumerate(overlaps):
        if not 0 <= m <= min(room[k], room[k + 1]):
            raise ValueError(
                "overlap %d at junction %d is outside 0..%d"
                % (m, k, min(room[k], room[k + 1]))
            )
        room[k + 1] -= m
    o = offsets(lengths, overlaps)
    n = o[-1] + lengths[-1]
    p = np.repeat(np.asarray(legs[0][0][:1], dtype=float), n, axis=0)
    v = np.zeros_like(p)
    a = None if any(leg[2] is None for leg in legs) else np.zeros_like(p)
    for (pk, vk, ak), start in zip(legs, o):
        pk = np.asarray(pk, dtype=float)
        end = start + len(pk)
        p[start:end] += pk - pk[0]
        p[end:] += pk[-1] - pk[0]
        v[start:end] += vk
        if a is not None:
            a[start:end] += ak
    return p, v, a


def choose_overlaps(caps, accept):
    """The longest overlap at each junction, up to caps[k] (max_overlaps),
    that `accept(k, overlaps)` allows.

    accept is asked only about junction k's window, with every other junction
    at its own decided (or still 0) overlap; windows never meet, so the order
    does not matter. Bisection between 0 (always accepted, never asked) and
    the cap: every overlap returned was accepted, but a longer one than
    bisection finds may exist where acceptance is not monotonic (a corner
    that clears an obstacle only when cut further)."""
    overlaps = [0] * len(caps)
    for k, hi in enumerate(caps):
        overlaps[k] = hi
        if hi > 0 and not accept(k, overlaps):
            lo = 0
            while hi - lo > 1:
                overlaps[k] = (lo + hi) // 2
                if accept(k, overlaps):
                    lo = overlaps[k]
                else:
                    hi = overlaps[k]
            overlaps[k] = lo
    return overlaps


def resample(p, v, dt_in, dt):
    """(p, v, a) of a leg sampled every `dt_in` (positions p, velocities v),
    resampled every `dt` by cubic Hermite interpolation, as the driver
    interpolates between points. The duration is rounded up to whole steps
    (slower by less than one step); the first and last points are kept."""
    p, v = np.asarray(p, dtype=float), np.asarray(v, dtype=float)
    if len(p) < 2:
        return p.copy(), np.zeros_like(p), np.zeros_like(p)
    duration = (len(p) - 1) * dt_in
    n = max(1, int(np.ceil(duration / dt - 1e-9)))
    stretch = duration / (n * dt)  # <= 1: the leg is slowed by this much
    s = np.arange(n + 1) * (len(p) - 1) / n  # where each sample falls, in input steps
    i = np.minimum(s.astype(int), len(p) - 2)
    u = (s - i)[:, None]
    p0, p1, m0, m1 = p[i], p[i + 1], v[i] * dt_in, v[i + 1] * dt_in
    pos = (
        (2 * u**3 - 3 * u**2 + 1) * p0
        + (u**3 - 2 * u**2 + u) * m0
        + (-2 * u**3 + 3 * u**2) * p1
        + (u**3 - u**2) * m1
    )
    vel = (
        (6 * u**2 - 6 * u) * p0
        + (3 * u**2 - 4 * u + 1) * m0
        + (-6 * u**2 + 6 * u) * p1
        + (3 * u**2 - 2 * u) * m1
    )
    acc = (12 * u - 6) * p0 + (6 * u - 4) * m0 + (-12 * u + 6) * p1 + (6 * u - 2) * m1
    pos[-1] = p[-1]
    return pos, vel * stretch / dt_in, acc * (stretch / dt_in) ** 2


def straight_leg(q0, q1, vmax, amax, dt):
    """(p, v, a) of a straight joint-space move q0 -> q1, from rest to rest,
    on a minimum-jerk profile timed so no joint passes `vmax` or `amax`
    (per joint). Its first point is q0 and its last q1, `dt` apart."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    d = q1 - q0
    # minimum jerk peaks at 1.875 x the mean speed and 5.774 d / T^2
    duration = max(
        np.max(1.875 * np.abs(d) / vmax),
        np.max(np.sqrt(5.7735 * np.abs(d) / amax)),
        dt,
    )
    n = int(np.ceil(duration / dt))
    u = np.arange(n + 1)[:, None] / n
    T = n * dt
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    sdd = (60 * u - 180 * u**2 + 120 * u**3) / T**2
    return q0 + s * d, sd * d, sdd * d
