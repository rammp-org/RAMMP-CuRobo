"""Dense resampling of cuRobo's knots (#15) — no torch, no cuRobo, no GPU.

The golden fixture is cuRobo v0.7.8's raw optimized knots and its
LINEAR_CUDA dense output for two real plans on the Jetson (cuRobo joint
order): "long" (32 knots -> 71 points, optimized_dt 45 ms) and "short"
(32 -> 31, so the kernel skips a knot). Regenerate it only from a GPU plan,
never by hand.
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest

from rammp_curobo.resample import knot_times, resample_quintic
from rammp_curobo.types import Trajectory
from rammp_curobo.validate import validate_trajectory

pytest.importorskip("scipy")

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "curobo_linear_resample.json").read_text()
)
JOINTS = ["arm_joint_%d" % i for i in range(1, 8)]


def _golden(name):
    g = {
        k: np.asarray(GOLDEN[name][k], dtype=np.float32)
        for k in ("knots", "dense", "dense_v", "dense_a")
    }
    g["dt"] = GOLDEN[name]["dt"]
    return g


def _lerp(knots, t, n, dt):
    tq = np.arange(n) * dt
    return np.stack([np.interp(tq, t, knots[:, j]) for j in range(knots.shape[1])], 1)


@pytest.mark.parametrize("name", ["long", "short"])
def test_knot_grid_reproduces_curobos_kernel(name):
    """The grid is the fix's core assumption, so it is pinned to cuRobo's
    real output rather than to itself: the off-by-one grid must NOT fit."""
    g = _golden(name)
    knots, dense, dt = g["knots"], g["dense"], g["dt"]
    n, h = len(dense), len(knots)
    assert np.abs(_lerp(knots, knot_times(h, n, dt), n, dt) - dense).max() < 1e-5
    off_by_one = np.arange(h) * (n - 1) * dt / (h - 1)
    assert np.abs(_lerp(knots, off_by_one, n, dt) - dense).max() > 1e-3


# Central differences carry a dt^2 * p''''/12 error, large on the short
# plan's high jerk; cuRobo's own LINEAR_CUDA misses the long plan by ~95%.
@pytest.mark.parametrize("name, a_tol", [("long", 0.1), ("short", 0.3)])
def test_resample_is_consistent_and_ends_at_rest(name, a_tol):
    g = _golden(name)
    knots, dt = g["knots"].astype(float), g["dt"]
    p, v, a = resample_quintic(knots, g["dense"], dt)
    assert p.shape == v.shape == a.shape == g["dense"].shape

    # Starts where cuRobo does and ends on the goal, both at rest.
    for i, q in ((0, knots[0]), (-1, knots[-1])):
        np.testing.assert_allclose(p[i], q, atol=1e-12)
        np.testing.assert_allclose(v[i], 0.0, atol=1e-9)
        np.testing.assert_allclose(a[i], 0.0, atol=1e-9)

    v_fd = (p[2:] - p[:-2]) / (2 * dt)
    a_fd = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2
    assert np.abs(v_fd - v[1:-1]).max() < 0.03 * np.abs(v).max()
    assert np.abs(a_fd - a[1:-1]).max() < a_tol * np.abs(a).max()

    # Passing the goal and coming back, from the last knot that is not yet
    # the goal: under 0.05 mrad on these fixtures (0.52 mrad worst of 22
    # Jetson plans; through cuRobo's repeated goal knots it was 2.6).
    goal = knots[-1]
    last = np.flatnonzero(np.abs(knots - goal).max(axis=1) > 0)[-1]
    tail = np.arange(len(p)) * dt >= knot_times(len(knots), len(p), dt)[last]
    side = np.sign(knots[last] - goal)
    assert ((p[tail] - goal) * side).min() > -0.05e-3


def _broken(knots, dense):
    mid = len(knots) // 2
    moved = knots.copy()
    moved[mid] += 1e-3
    holed = dense.copy()
    holed[5, 2] = np.nan
    return {
        "dense lost a point": (knots, dense[:-1]),
        "dense shifted a step": (knots, np.roll(dense, 1, axis=0)),
        "one knot moved 1 mrad": (moved, dense),
        "non-finite dense": (knots, holed),
        "knots reversed": (knots[::-1], dense),
    }


@pytest.mark.parametrize(
    "case",
    [
        "dense lost a point",
        "dense shifted a step",
        "one knot moved 1 mrad",
        "non-finite dense",
        "knots reversed",
    ],
)
def test_resample_refuses_a_plan_it_cannot_identify(case):
    """Keep cuRobo's output rather than smooth a trajectory other than the
    one cuRobo produced and collision-checked."""
    g = _golden("long")
    knots, dense = _broken(g["knots"].astype(float), g["dense"])[case]
    assert resample_quintic(knots, dense, g["dt"]) is None


def test_resample_start_equal_to_goal_holds_still():
    knots = np.tile(_golden("long")["knots"][-1], (32, 1)).astype(float)
    p, v, a = resample_quintic(knots, knots[:10], 0.02)
    np.testing.assert_array_equal(p, knots[:10])
    assert not v.any() and not a.any()


def test_resample_refuses_a_plan_too_short_to_land_on_the_goal():
    """The last knot before cuRobo's repeated goal must fall before the last
    dense point, or there is no time left to land there at rest."""
    knots = _golden("long")["knots"].astype(float)
    n, dt = 8, 0.02
    t, tq = knot_times(len(knots), n, dt), np.arange(n) * dt
    dense = np.stack([np.interp(tq, t, knots[:, j]) for j in range(7)], 1)
    assert resample_quintic(knots, dense, dt) is None


# --- CuRoboPlanner._resample / _finish, with torch stand-ins -------------------


class _Tensor:
    """Just enough of a torch tensor for _resample and _finish."""

    device = "cpu"

    def __init__(self, a):
        self.a = np.array(a, dtype=np.float32)

    shape = property(lambda self: self.a.shape)
    dtype = property(lambda self: self.a.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def item(self):
        return self.a.item()

    def copy_(self, src):
        self.a[...] = src
        return self


class _Plan:
    def __init__(self, p, v, a):
        self.position, self.velocity, self.acceleration = (
            _Tensor(p),
            _Tensor(v),
            _Tensor(a),
        )

    def clone(self):
        return _Plan(self.position.a, self.velocity.a, self.acceleration.a)

    def get_ordered_joint_state(self, names):
        return self


_TORCH = types.SimpleNamespace(
    as_tensor=lambda x, dtype, device: np.asarray(x, dtype=dtype)
)


def _resample(plan, result):
    from rammp_curobo.planner import CuRoboPlanner

    return CuRoboPlanner._resample(types.SimpleNamespace(_torch=_TORCH), plan, result)


def _golden_plan_and_result(knots=None):
    g = _golden("long")
    plan = _Plan(g["dense"], g["dense_v"], g["dense_a"])
    knots = g["knots"] if knots is None else knots
    result = types.SimpleNamespace(
        optimized_plan=types.SimpleNamespace(position=_Tensor(knots)),
        interpolation_dt=g["dt"],
    )
    return g, plan, result


def test_resample_swaps_in_the_spline_on_a_copy():
    g, plan, result = _golden_plan_and_result()
    out, consistent = _resample(plan, result)
    assert consistent and out is not plan
    # cuRobo's buffers are untouched; the copy carries the spline.
    np.testing.assert_array_equal(plan.position.a, g["dense"])
    np.testing.assert_array_equal(plan.acceleration.a, g["dense_a"])
    p, v, a = resample_quintic(g["knots"], g["dense"], g["dt"])
    for got, want in ((out.position, p), (out.velocity, v), (out.acceleration, a)):
        np.testing.assert_allclose(got.a, want, atol=1e-6)


@pytest.mark.parametrize(
    "result",
    [
        types.SimpleNamespace(interpolation_dt=0.02),  # no knots at all
        types.SimpleNamespace(optimized_plan=None, interpolation_dt=0.02),
    ],
    ids=["no attribute", "None"],
)
def test_resample_falls_back_without_knots(result):
    _, plan, _ = _golden_plan_and_result()
    out, consistent = _resample(plan, result)
    assert out is plan and not consistent


def test_resample_falls_back_when_knots_disagree():
    g = _golden("long")
    _, plan, result = _golden_plan_and_result(knots=g["knots"] + 1e-3)
    out, consistent = _resample(plan, result)
    assert out is plan and not consistent


def _finish(consistent, acc_limit=25.0, scale_a=1.0):
    from rammp_curobo.planner import CuRoboPlanner

    g = _golden("long")
    plan = _Plan(g["dense"], g["dense_v"], g["dense_a"] * scale_a)
    stub = types.SimpleNamespace(
        joint_names=JOINTS,
        base_frame="arm_mount",  # _finish stamps it once #38 lands
        _resample=lambda plan, result: (plan, consistent),
        _recheck_constraints=lambda plan: [],
        joint_limits=lambda: {
            "position": np.array([[-7.0] * 7, [7.0] * 7]),
            "velocity": np.full(7, 10.0),
            "acceleration": np.full(7, acc_limit),
        },
    )
    result = types.SimpleNamespace(
        success=_Tensor(True),
        interpolation_dt=g["dt"],
        get_interpolated_plan=lambda: plan,
    )
    return CuRoboPlanner._finish(stub, result, 0.0)


def test_finish_ships_no_accelerations_when_they_are_not_consistent():
    """LINEAR_CUDA's accelerations disagree with its positions (#15): send
    none, so a consumer drops to cubic rather than trusting them."""
    res = _finish(consistent=False)
    assert res.success, res.error
    assert res.joint_traj.accelerations is None
    assert res.joint_traj.velocities is not None
    assert _finish(consistent=True).joint_traj.accelerations is not None


def test_finish_rejects_accelerations_over_the_limit():
    res = _finish(consistent=True, acc_limit=1.0)
    assert not res.success
    assert "acceleration" in res.error


def test_validate_checks_accelerations_only_when_given_limits():
    n = 10
    traj = Trajectory(
        joint_names=JOINTS,
        positions=np.zeros((n, 7)),
        velocities=np.zeros((n, 7)),
        accelerations=np.zeros((n, 7)),
        dt=0.02,
    )
    traj.accelerations[4, 2] = 30.0
    pos_lim, vel_lim = np.array([[-1.0] * 7, [1.0] * 7]), np.full(7, 1.0)
    assert validate_trajectory(traj, pos_lim, vel_lim) == []
    problems = validate_trajectory(
        traj, pos_lim, vel_lim, acceleration_limits=np.full(7, 25.0)
    )
    assert problems and "arm_joint_3 acceleration 30.000 > limit 25.000" in problems[0]
    traj.accelerations[4, 2] = 25.2  # inside the same 1% slack velocities get
    assert (
        validate_trajectory(
            traj, pos_lim, vel_lim, acceleration_limits=np.full(7, 25.0)
        )
        == []
    )
