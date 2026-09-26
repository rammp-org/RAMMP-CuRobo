# EE Pose Constraints and Blended Via Point — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a caller plan a trajectory that holds chosen end-effector axes fixed (keep a spoon level) and/or approaches the goal through one blended via point that the arm does not stop at.

**Architecture:** Both features are the same cuRobo object — `MotionGenPlanConfig.pose_cost_metric`. A pure-Python value layer (`constraints.py`) describes what the caller wants and owns the axis-ordering conversion; `CuRoboPlanner` translates it into a `PoseCostMetric` at the per-plan config that is already rebuilt on every call; the ROS node maps appended action fields onto it. No torch or cuRobo imports leave `planner.py`.

**Tech Stack:** Python 3.10, cuRobo v0.7.8 (pinned), numpy, pytest, ROS 2 Humble + rosidl, Cyclone DDS.

**Spec:** GitHub issues `rammp-org/RAMMP-CuRobo#16` (EE constraints) and `#17` (blended via point). Both are in the `v1.1.0 Release` milestone. Read both before starting — they carry the evidence for the design decisions this plan assumes, especially why the graph planner is not a blocker and why chained segments were rejected.

## Global Constraints

- **cuRobo stays pinned at v0.7.8.** `docker/Dockerfile:86-93` asserts the commit hash. Nothing in this plan may require a newer version.
- **No ROS imports anywhere in `core/`** — package rule, `core/rammp_curobo/__init__.py:1-6`.
- **`constraints.py` must import neither torch nor cuRobo**, at module level or inside functions. It has to be importable on a laptop with no GPU stack, like every other core module except `planner.py`.
- **The graph planner stays disabled.** `enable_graph=False` and `enable_graph_attempt=None` (`planner.py:452-460`) are load-bearing: the Jetson torch wheel dies in `torch.svd`. No task may set either.
- **Only the trimmed interpolated plan leaves `CuRoboPlanner`** (`planner.py:483`). Unchanged by this plan.
- **New planner config keys must be added in both `core/rammp_curobo/config.py` `PLANNER_DEFAULTS` and `core/rammp_curobo/configs/gen3.yaml`**, or `test_gen3_yaml_matches_planner_defaults` fails. **This plan adds none** — every new value is per-call.
- **IDL changes are append-only, at the end of the block, with a default, and zero must mean "off".** That is what makes them a minor bump under `rammp-interfaces-ros2`'s rules: on Humble a subscriber matches on type name and an absent trailing field reads as its declared default. Inserting a field anywhere else corrupts every field after it. Holds only while every module runs Cyclone DDS.
- **Tests run from the repo root:** `pytest core/tests -q`. GPU tests live in `core/tests/test_smoke.py` behind the CUDA gate at `test_smoke.py:12-15` and only run on the Jetson.
- **`pre-commit run --all-files` must pass** (ruff, hadolint, actionlint).

## Review Focus

Five things the spec implies that no task's happy path exercises. Each has a test assigned to the task that owns the code.

1. **A constraint and a via point requested together** — this is cuRobo's own grasp-approach combination, and the two write to overlapping fields of one `PoseCostMetric`. If the second silently overwrites the first, a caller carrying a level plate through an approach gets an unconstrained plate. → Task 2, Step 9.
1. **`via_tstep_fraction` outside `(0, 1)`** — a client sending `1.5` or `-0.2` must be refused with a reason, not handed to cuRobo where it becomes an unhelpful status. → Task 1, Step 5 (validation) and Task 4, Step 7 (rejection at the ROS boundary).
1. **Non-finite values** in the via offset or a constraint — NaN from a client's uninitialised struct must be refused before it reaches the GPU. → Task 1, Step 5.
1. **All six axes held** — this asks for a goal pose identical to the start pose. cuRobo rejects it with a bare enum; we must say something a person can act on. → Task 3, Step 7.
1. **A start that satisfies the held axes only marginally** — cuRobo's pre-check tolerances are 0.05 rad and 0.005 m, so a start 0.049 rad off passes and then behaves unpredictably. The caller deserves a distinguishable warning rather than a mystery. → Task 3, Step 5.

______________________________________________________________________

### Task 1: The value layer — `PoseConstraint` and `ViaPoint`

Pure Python. This is where the axis ordering lives, tested, once — it is the single easiest thing in this feature to get backwards.

**Files:**

- Create: `core/rammp_curobo/constraints.py`
- Modify: `core/rammp_curobo/__init__.py:8-23` (exports)
- Test: `core/tests/test_offline.py` (append; it is the GPU-free suite)

**Interfaces:**

- Consumes: nothing.

- Produces:

  - `PoseConstraint(hold_roll: bool = False, hold_pitch: bool = False, hold_yaw: bool = False, hold_x: bool = False, hold_y: bool = False, hold_z: bool = False, in_base_frame: bool = True)`, frozen dataclass, with `is_active() -> bool`, `hold_vec_weight() -> list[float]` (6 floats), `validate() -> None` (raises `ValueError`).
  - `ViaPoint(offset_m: float = 0.0, linear_axis: int = 2, tstep_fraction: float = 0.8)`, frozen dataclass, with `is_active() -> bool` and `validate() -> None`.

- [ ] **Step 1: Write the failing tests for the value layer**

Append to `core/tests/test_offline.py`:

```python
from rammp_curobo.constraints import PoseConstraint, ViaPoint


def test_pose_constraint_inactive_by_default():
    c = PoseConstraint()
    assert not c.is_active()
    assert c.hold_vec_weight() == [0.0] * 6


def test_hold_vec_weight_is_orientation_first():
    """cuRobo's vec_weight is [rx, ry, rz, x, y, z] — orientation FIRST.
    Getting this backwards silently constrains position instead of
    orientation, which still plans, so no test but this one would catch it."""
    c = PoseConstraint(hold_roll=True, hold_pitch=True)
    assert c.hold_vec_weight() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    c = PoseConstraint(hold_z=True)
    assert c.hold_vec_weight() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_pose_constraint_is_active_when_any_axis_held():
    assert PoseConstraint(hold_yaw=True).is_active()
    assert PoseConstraint(hold_x=True).is_active()


def test_via_point_inactive_at_zero_offset():
    assert not ViaPoint().is_active()
    assert not ViaPoint(offset_m=0.0, linear_axis=2).is_active()
    assert ViaPoint(offset_m=0.1).is_active()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest core/tests/test_offline.py -q -k "pose_constraint or via_point or hold_vec"`
Expected: FAIL — `ModuleNotFoundError: No module named 'rammp_curobo.constraints'`

- [ ] **Step 3: Write `constraints.py`**

```python
"""What a caller wants held or approached — no cuRobo, no torch, no ROS.

The axis ordering of cuRobo's vec_weight is [rx, ry, rz, x, y, z]:
ORIENTATION FIRST, position last, 1.0 = hold that axis. That conversion
lives in exactly one place — PoseConstraint.hold_vec_weight() — because
getting it backwards produces a plan that is constrained in the wrong
three axes and still succeeds.

Which of rx/ry/rz is roll vs pitch vs yaw follows cuRobo's base-frame
axis order (x, y, z). Verified on hardware in the smoke suite, not here.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PoseConstraint:
    """Axes of the tool pose to hold fixed for the whole trajectory."""

    hold_roll: bool = False
    hold_pitch: bool = False
    hold_yaw: bool = False
    hold_x: bool = False
    hold_y: bool = False
    hold_z: bool = False
    # True: axes are the robot BASE frame, which is what "level with
    # gravity" means. False: cuRobo's default projection into the GOAL
    # frame.
    in_base_frame: bool = True

    def is_active(self) -> bool:
        return any(self.hold_vec_weight())

    def hold_vec_weight(self):
        """cuRobo's 6-vector: [rx, ry, rz, x, y, z]. 1.0 holds that axis."""
        return [
            1.0 if self.hold_roll else 0.0,
            1.0 if self.hold_pitch else 0.0,
            1.0 if self.hold_yaw else 0.0,
            1.0 if self.hold_x else 0.0,
            1.0 if self.hold_y else 0.0,
            1.0 if self.hold_z else 0.0,
        ]

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class ViaPoint:
    """One blended intermediate target, offset from the GOAL along a tool axis.

    This is cuRobo's only genuine via-point mechanism and it is a cost, not
    a constraint: the trajectory passes NEAR the offset without stopping,
    it does not hit it exactly. offset_m == 0.0 means no via point.
    """

    offset_m: float = 0.0
    linear_axis: int = 2  # 2 = the tool approach axis
    tstep_fraction: float = 0.8  # activate from 80% of the horizon onward

    def is_active(self) -> bool:
        return self.offset_m != 0.0

    def validate(self) -> None:
        if not math.isfinite(self.offset_m):
            raise ValueError("via offset_m must be finite, got %r" % (self.offset_m,))
        if not self.is_active():
            return None
        if self.linear_axis not in (0, 1, 2):
            raise ValueError(
                "via linear_axis must be 0, 1 or 2, got %r" % (self.linear_axis,)
            )
        if not math.isfinite(self.tstep_fraction) or not (
            0.0 < self.tstep_fraction < 1.0
        ):
            raise ValueError(
                "via tstep_fraction must be in (0, 1), got %r"
                % (self.tstep_fraction,)
            )
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest core/tests/test_offline.py -q -k "pose_constraint or via_point or hold_vec"`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the validation tests (Review Focus 2 and 3)**

Append to `core/tests/test_offline.py`:

```python
def test_via_point_rejects_out_of_range_fraction():
    for bad in (1.5, -0.2, 0.0, 1.0):
        with pytest.raises(ValueError, match="tstep_fraction"):
            ViaPoint(offset_m=0.1, tstep_fraction=bad).validate()


def test_via_point_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        ViaPoint(offset_m=float("nan")).validate()
    with pytest.raises(ValueError, match="finite"):
        ViaPoint(offset_m=0.1, tstep_fraction=float("inf")).validate()


def test_via_point_rejects_bad_axis():
    with pytest.raises(ValueError, match="linear_axis"):
        ViaPoint(offset_m=0.1, linear_axis=7).validate()


def test_inactive_via_point_validates_clean():
    """An all-defaults ViaPoint arrives from every caller that wants
    nothing; it must never raise."""
    ViaPoint().validate()
    PoseConstraint().validate()
```

- [ ] **Step 6: Run them and confirm they pass**

Run: `pytest core/tests/test_offline.py -q -k "via_point or pose_constraint"`
Expected: PASS. If `test_via_point_rejects_out_of_range_fraction` fails on `0.0` or `1.0`, the bounds in `validate()` are inclusive — they must be exclusive.

- [ ] **Step 7: Export the new types**

In `core/rammp_curobo/__init__.py`, add the import alongside the others and extend `__all__`:

```python
from rammp_curobo.constraints import PoseConstraint, ViaPoint
```

```python
__all__ = [
    "CuRoboPlanner",
    "PlanResult",
    "Trajectory",
    "PoseConstraint",
    "ViaPoint",
    "Scene",
    "load_scene",
    "scale_trajectory",
    "validate_trajectory",
    "start_state_matches",
]
```

Note for the reviewer: `__all__` is the declared public surface (`CONTRIBUTING.md:119-124`), so this is the additive change that makes the feature a MINOR bump.

- [ ] **Step 8: Run the whole offline suite and pre-commit**

Run: `pytest core/tests/test_offline.py -q && pre-commit run --all-files`
Expected: all pass. The offline suite must stay importable without torch.

- [ ] **Step 9: Commit**

```bash
git add core/rammp_curobo/constraints.py core/rammp_curobo/__init__.py core/tests/test_offline.py
git commit -m "feat(core): PoseConstraint and ViaPoint value types

The cuRobo vec_weight ordering ([rx, ry, rz, x, y, z], orientation first)
is converted in exactly one tested place."
```

______________________________________________________________________

### Task 2: Translate into a `PoseCostMetric` and thread it through `plan_to_pose`

**Files:**

- Modify: `core/rammp_curobo/planner.py:452-467` (`_plan_config`), `:147-206` (`plan_to_pose`)
- Test: `core/tests/test_offline.py`

**Interfaces:**

- Consumes: `PoseConstraint`, `ViaPoint` from Task 1.
- Produces:
  - `CuRoboPlanner._pose_cost_kwargs(constraint, via) -> dict | None` — a **plain-value** dict (no tensors), or `None` when neither is active. Keys when active: `hold_partial_pose: bool`, `hold_vec_weight: list[float]`, `project_to_goal_frame: bool`, and when a via point is active `offset_position: float`, `linear_axis: int`, `offset_tstep_fraction: float`.
  - `CuRoboPlanner._plan_config(constraint=None, via=None) -> MotionGenPlanConfig`
  - `CuRoboPlanner.plan_to_pose(position, quaternion, start, quat_order="xyzw", apply_tool_correction=None, constraint=None, via=None) -> PlanResult`

Keeping `_pose_cost_kwargs` free of tensors is what lets CI test the mapping on a machine with no GPU — the tensor conversion is one line in `_plan_config`, which is GPU-only territory anyway.

- [ ] **Step 1: Write the failing test for the kwargs builder**

Append to `core/tests/test_offline.py`. Note this constructs no planner — it calls the method on the class, unbound, so no CUDA is touched:

```python
def test_pose_cost_kwargs_none_when_nothing_requested():
    from rammp_curobo.planner import CuRoboPlanner

    assert CuRoboPlanner._pose_cost_kwargs(None, None) is None
    assert CuRoboPlanner._pose_cost_kwargs(PoseConstraint(), ViaPoint()) is None


def test_pose_cost_kwargs_for_a_held_constraint():
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(
        PoseConstraint(hold_roll=True, hold_pitch=True), None
    )
    assert kw["hold_partial_pose"] is True
    assert kw["hold_vec_weight"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # in_base_frame=True means DON'T project into the goal frame
    assert kw["project_to_goal_frame"] is False
    assert "offset_position" not in kw
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest core/tests/test_offline.py -q -k pose_cost_kwargs`
Expected: FAIL — `AttributeError: type object 'CuRoboPlanner' has no attribute '_pose_cost_kwargs'`

- [ ] **Step 3: Implement `_pose_cost_kwargs` as a staticmethod**

Add to `core/rammp_curobo/planner.py`, next to `_plan_config`:

```python
    @staticmethod
    def _pose_cost_kwargs(constraint, via):
        """Plain-value kwargs for a PoseCostMetric, or None if unconstrained.

        Deliberately tensor-free so the mapping is testable without a GPU;
        _plan_config does the one-line tensor conversion.
        """
        constraint = constraint or PoseConstraint()
        via = via or ViaPoint()
        constraint.validate()
        via.validate()
        if not constraint.is_active() and not via.is_active():
            return None

        kw = {
            "hold_partial_pose": True,
            "hold_vec_weight": constraint.hold_vec_weight(),
            "project_to_goal_frame": not constraint.in_base_frame,
        }
        if via.is_active():
            # cuRobo frees the linear axis the approach travels along, so
            # the trajectory can move ALONG it while the rest stays held.
            kw["hold_vec_weight"][3 + via.linear_axis] = 0.0
            kw["offset_position"] = float(via.offset_m)
            kw["linear_axis"] = int(via.linear_axis)
            kw["offset_tstep_fraction"] = float(via.tstep_fraction)
        return kw
```

Add the import at the top of `planner.py`, with the other first-party imports:

```python
from rammp_curobo.constraints import PoseConstraint, ViaPoint
```

It is a `staticmethod` with exactly two parameters, so the tests call it through the class as `CuRoboPlanner._pose_cost_kwargs(constraint, via)` with no `self`.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest core/tests/test_offline.py -q -k pose_cost_kwargs`
Expected: PASS

- [ ] **Step 5: Thread it into `_plan_config`**

Replace `_plan_config` (`planner.py:452-467`):

```python
    def _plan_config(self, constraint=None, via=None):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        kw = {}
        if not self.enable_graph:
            # cuRobo silently ENABLES the graph planner after 3 failed
            # attempts unless this is None — and the graph planner is the
            # exact thing the Jetson wheel cannot run. Never let it engage.
            kw["enable_graph_attempt"] = None

        metric_kw = self._pose_cost_kwargs(constraint, via)
        if metric_kw is not None:
            from curobo.rollout.cost.pose_cost import PoseCostMetric

            metric_kw = dict(metric_kw)
            metric_kw["hold_vec_weight"] = self._tensor(
                metric_kw["hold_vec_weight"]
            )
            kw["pose_cost_metric"] = PoseCostMetric(**metric_kw)

        return MotionGenPlanConfig(
            max_attempts=self.max_attempts,
            enable_graph=self.enable_graph,
            enable_finetune_trajopt=self.enable_finetune,
            finetune_attempts=self.finetune_attempts,
            **kw,
        )
```

If `PoseCostMetric` rejects `offset_position` as a float (it may want a tensor or may pair only with `create_grasp_approach_metric`), fall back to building the metric with `PoseCostMetric.create_grasp_approach_metric(offset_position=..., linear_axis=..., tstep_fraction=...)` when a via point is active and no axes are held, and record which form worked in the docstring. **This is the one construction detail in the plan that was read from source but never executed** — settle it in Task 5 on the Jetson, not by guessing here.

- [ ] **Step 6: Thread it through `plan_to_pose`**

In `plan_to_pose` (`planner.py:147-206`), add the two keyword arguments to the signature after `apply_tool_correction`:

```python
        constraint=None,
        via=None,
```

extend the docstring:

```
        constraint: PoseConstraint — tool axes to hold fixed for the whole
            trajectory (keeping a carried object level). None = free.
        via: ViaPoint — one blended intermediate target offset from the
            goal along a tool axis. The path passes NEAR it without
            stopping; it is a cost, not a waypoint to hit. None = none.
```

and change the plan call:

```python
            result = self._motion_gen.plan_single(
                start_state, goal, self._plan_config(constraint, via)
            )
```

Wrap the validation so a bad request fails as a `PlanResult`, not an exception, matching how this method already handles a zero-length quaternion. Immediately after the `t0 = time.monotonic()` line:

```python
        try:
            self._pose_cost_kwargs(constraint, via)
        except ValueError as exc:
            return PlanResult.failure("BAD_CONSTRAINT", str(exc))
```

- [ ] **Step 7: Confirm nothing regressed offline**

Run: `pytest core/tests/test_offline.py -q`
Expected: PASS. The GPU suite cannot run here; Task 5 covers it.

- [ ] **Step 8: Commit**

```bash
git add core/rammp_curobo/planner.py core/tests/test_offline.py
git commit -m "feat(core): pose constraints and a via point on plan_to_pose

Both map onto one PoseCostMetric. The mapping is tensor-free so it is
covered by the GPU-free suite."
```

- [ ] **Step 9: Write the combination test (Review Focus 1)**

Append to `core/tests/test_offline.py`:

```python
def test_constraint_and_via_point_compose():
    """cuRobo's own grasp-approach metric is exactly this combination: hold
    the orientation, free the axis the approach travels along. If one
    overwrote the other, a carried plate would tilt during the approach."""
    from rammp_curobo.planner import CuRoboPlanner

    kw = CuRoboPlanner._pose_cost_kwargs(
        PoseConstraint(hold_roll=True, hold_pitch=True, hold_x=True, hold_y=True),
        ViaPoint(offset_m=0.10, linear_axis=2, tstep_fraction=0.8),
    )
    # orientation still held
    assert kw["hold_vec_weight"][:3] == [1.0, 1.0, 0.0]
    # x, y still held; z (3 + 2) freed for the approach to travel along
    assert kw["hold_vec_weight"][3:] == [1.0, 1.0, 0.0]
    assert kw["offset_position"] == 0.10
    assert kw["offset_tstep_fraction"] == 0.8
```

- [ ] **Step 10: Run it, then commit**

Run: `pytest core/tests/test_offline.py -q -k compose`
Expected: PASS

```bash
git add core/tests/test_offline.py
git commit -m "test(core): constraint and via point compose without clobbering"
```

______________________________________________________________________

### Task 3: The start-state pre-check and an honest failure message

cuRobo refuses a held-axis plan when the start pose does not already satisfy the held axes — 0.05 rad angular, 0.005 m linear — and returns `INVALID_PARTIAL_POSE_COST_METRIC`, for which `_explain` has no branch. So "keep the tool level" is inherently two-phase, and the caller needs to be told that in words.

**Files:**

- Modify: `core/rammp_curobo/geometry.py` (add `rotvec_between`), `core/rammp_curobo/planner.py` (`_explain`, new public method)
- Test: `core/tests/test_offline.py`

**Interfaces:**

- Consumes: `PoseConstraint` (Task 1), `CuRoboPlanner.fk` (existing, `planner.py:318-326`).

- Produces:

  - `geometry.rotvec_between(q_a_xyzw, q_b_xyzw) -> list[float]` — the rotation from a to b as an axis×angle 3-vector in the base frame.
  - `CuRoboPlanner.constraint_satisfied_at_start(start, position, quaternion, constraint, quat_order="xyzw") -> tuple[bool, str | None]` — `(True, None)`, `(True, "marginal: ...")` or `(False, reason)`.

- [ ] **Step 1: Write the failing test for `rotvec_between`**

Append to `core/tests/test_offline.py`:

```python
def test_rotvec_between_is_zero_for_equal_quaternions():
    from rammp_curobo.geometry import rotvec_between

    q = euler_deg_to_quat_xyzw([10.0, -20.0, 35.0])
    assert max(abs(v) for v in rotvec_between(q, q)) < 1e-9


def test_rotvec_between_recovers_a_single_axis_rotation():
    from rammp_curobo.geometry import rotvec_between

    a = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    b = euler_deg_to_quat_xyzw([12.0, 0.0, 0.0])
    v = rotvec_between(a, b)
    assert abs(v[0] - math.radians(12.0)) < 1e-6
    assert abs(v[1]) < 1e-6 and abs(v[2]) < 1e-6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest core/tests/test_offline.py -q -k rotvec`
Expected: FAIL — `ImportError: cannot import name 'rotvec_between'`

- [ ] **Step 3: Implement `rotvec_between`**

Append to `core/rammp_curobo/geometry.py`:

```python
def rotvec_between(a_xyzw, b_xyzw):
    """Rotation taking quaternion a to b, as an axis*angle 3-vector (rad).

    Components are about the base frame's x, y, z — the same ordering
    PoseConstraint.hold_vec_weight() uses for its first three entries.
    """
    ax, ay, az, aw = [float(v) for v in a_xyzw]
    bx, by, bz, bw = [float(v) for v in b_xyzw]
    # r = b * conj(a)
    rw = bw * aw + bx * ax + by * ay + bz * az
    rx = bx * aw - bw * ax - by * az + bz * ay
    ry = by * aw - bw * ay - bz * ax + bx * az
    rz = bz * aw - bw * az - bx * ay + by * ax
    n = math.sqrt(rx * rx + ry * ry + rz * rz)
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    if rw < 0.0:  # shortest arc
        rw, rx, ry, rz = -rw, -rx, -ry, -rz
    angle = 2.0 * math.atan2(n, rw)
    return [angle * rx / n, angle * ry / n, angle * rz / n]
```

`geometry.py` already imports `math` (used by `ang_diff`); confirm before adding a duplicate import.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest core/tests/test_offline.py -q -k rotvec`
Expected: PASS

- [ ] **Step 5: Write the pre-check tests, including the marginal band (Review Focus 5)**

These call the method unbound with a stub `fk`, so no GPU is needed. Append to `core/tests/test_offline.py`:

```python
class _StubFk:
    """Minimal stand-in for CuRoboPlanner: only fk() is exercised."""

    def __init__(self, pos, quat):
        self._pos, self._quat = pos, quat

    def fk(self, q, quat_order="xyzw"):
        return self._pos, self._quat


def _check(stub, goal_pos, goal_quat, constraint):
    from rammp_curobo.planner import CuRoboPlanner

    return CuRoboPlanner.constraint_satisfied_at_start(
        stub, [0.0] * 7, goal_pos, goal_quat, constraint
    )


def test_pre_check_passes_when_held_axes_already_match():
    q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], q)
    ok, why = _check(stub, [0.6, 0.2, 0.4], q, PoseConstraint(hold_roll=True))
    assert ok and why is None


def test_pre_check_rejects_a_tilted_start():
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([12.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold_roll=True))
    assert not ok
    assert "roll" in why
    assert "0.05" in why  # the tolerance is named, so the caller can act


def test_pre_check_passes_a_tilted_start_when_the_goal_is_equally_tilted():
    """A held axis is held at the GOAL'S value, not at zero. A tool at 45
    degrees planning to a 45-degree goal is perfectly legal and stays at 45
    the whole way — 'held' means unchanged, not level."""
    tilted = euler_deg_to_quat_xyzw([45.0, 0.0, 0.0])
    stub = _StubFk([0.3, 0.0, 0.4], tilted)
    ok, why = _check(
        stub, [0.6, 0.2, 0.4], tilted, PoseConstraint(hold_roll=True, hold_pitch=True)
    )
    assert ok and why is None


def test_pre_check_flags_a_marginal_start():
    """0.045 rad passes cuRobo's 0.05 rad gate but only just. Silence here
    turns into an unreproducible planning failure later."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([2.6, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, PoseConstraint(hold_roll=True))
    assert ok
    assert why is not None and "marginal" in why
```

- [ ] **Step 6: Implement the pre-check**

Add to `core/rammp_curobo/planner.py` as a public method:

```python
    # cuRobo's own gate inside update_pose_cost_metric. Mirrored here so a
    # caller gets a sentence instead of INVALID_PARTIAL_POSE_COST_METRIC.
    HOLD_TOL_RAD = 0.05
    HOLD_TOL_M = 0.005

    def constraint_satisfied_at_start(
        self, start, position, quaternion, constraint, quat_order="xyzw"
    ):
        """Can this constrained plan even be attempted from `start`?

        cuRobo requires the HELD components of the start pose to already
        match the goal, so 'keep the tool level' is two-phase: level it
        with an unconstrained move, then transport under the constraint.

        `quat_order` describes only the caller's own `quaternion` argument
        (the goal) — the start pose is always read from `fk` in xyzw, so a
        `quat_order="wxyz"` caller never mixes conventions between the two
        operands fed to `rotvec_between`.

        This check is scoped to `in_base_frame=True` (cuRobo's default):
        with `in_base_frame=False` cuRobo gates the constraint in the GOAL
        frame instead, which this check does not reproduce, so it declines
        to judge rather than name the wrong axis.

        Returns (ok, reason). reason is None when clean, a 'marginal: ...'
        string when inside tolerance but close to it, an explanation when
        not, or a 'not pre-checked: ...' string when the mode is out of
        scope (goal-frame locking).
        """
        if constraint is None or not constraint.is_active():
            return True, None
        if not constraint.in_base_frame:
            return True, (
                "not pre-checked: goal-frame locking is gated in cuRobo's "
                "goal frame, which this check does not reproduce — expect "
                "cuRobo to accept or refuse it"
            )

        cur_pos, cur_quat = self.fk(start, quat_order="xyzw")
        if quat_order == "wxyz":
            goal_quat = geometry.wxyz_to_xyzw(quaternion)
        else:
            goal_quat = [float(v) for v in quaternion]

        rot = geometry.rotvec_between(cur_quat, goal_quat)
        lin = [float(position[i]) - float(cur_pos[i]) for i in range(3)]
        held = constraint.hold_vec_weight()
        names = ("roll", "pitch", "yaw", "x", "y", "z")

        worst_ratio, worst = 0.0, None
        for i in range(6):
            if held[i] == 0.0:
                continue
            err = abs(rot[i]) if i < 3 else abs(lin[i - 3])
            tol = CuRoboPlanner.HOLD_TOL_RAD if i < 3 else CuRoboPlanner.HOLD_TOL_M
            ratio = err / tol
            if ratio > worst_ratio:
                worst_ratio, worst = ratio, (names[i], err, tol)

        if worst is None:
            return True, None
        name, err, tol = worst
        if worst_ratio > 1.0:
            return False, (
                "held axis '%s' differs by %.4f between start and goal "
                "(limit %.3f): a held axis is held AT THE GOAL'S VALUE, so "
                "the start must already match it — move there with an "
                "unconstrained plan first, then plan the constrained one"
                % (name, err, tol)
            )
        if worst_ratio > 0.8:
            return True, (
                "marginal: held axis '%s' is %.4f from the goal, %.0f%% of "
                "the %.3f limit" % (name, err, 100.0 * worst_ratio, tol)
            )
        return True, None
```

Confirm `planner.py` imports `geometry` at module level (it uses `geometry.xyzw_to_wxyz` in `plan_to_pose`); if it imports specific names instead, match that style.

- [ ] **Step 7: Add the all-axes-held test (Review Focus 4)**

```python
def test_pre_check_explains_when_every_axis_is_held():
    """Holding all six axes asks for a goal identical to the start. The
    caller must be told that, not handed a status enum."""
    stub = _StubFk([0.3, 0.0, 0.4], euler_deg_to_quat_xyzw([0.0, 0.0, 0.0]))
    goal_q = euler_deg_to_quat_xyzw([0.0, 0.0, 0.0])
    all_held = PoseConstraint(
        hold_roll=True, hold_pitch=True, hold_yaw=True,
        hold_x=True, hold_y=True, hold_z=True,
    )
    ok, why = _check(stub, [0.6, 0.0, 0.4], goal_q, all_held)
    assert not ok
    assert "'x'" in why
```

- [ ] **Step 8: Add the missing `_explain` branch**

In `_explain` (`planner.py:546-574`), insert **before** the `"IK" in s` test, since the substring tests are order-sensitive and a later status could collide:

```python
        if "PARTIAL_POSE" in s:
            return (
                "the constrained plan was refused because the START pose "
                "does not already match the GOAL on the held axes — a held "
                "axis is held at the goal's value, so an unconstrained move "
                "has to bring those axes there first (see "
                "constraint_satisfied_at_start)."
            )
```

- [ ] **Step 9: Run the full offline suite, then commit**

Run: `pytest core/tests -q && pre-commit run --all-files`
Expected: PASS (the GPU suite skips without CUDA)

```bash
git add core/rammp_curobo/geometry.py core/rammp_curobo/planner.py core/tests/test_offline.py
git commit -m "feat(core): pre-check a constrained start, and explain the refusal

cuRobo returns a bare INVALID_PARTIAL_POSE_COST_METRIC; this turns it
into a sentence that names the axis and what to do about it."
```

______________________________________________________________________

### Task 4: The ROS interface — named message types, not a bool array

ROS-native shape: two small message types with **named fields and named constants**, embedded in the goal. No positional arrays — the whole point of Task 1 was to stop the axis ordering being something a human has to remember, and a `bool[6]` on the wire would hand that trap straight back to every client.

**Files:**

- Create: `rammp_curobo_interfaces/msg/PoseAxisLock.msg`
- Create: `rammp_curobo_interfaces/msg/ApproachVia.msg`
- Modify: `rammp_curobo_interfaces/CMakeLists.txt:13-18` (register both)
- Modify: `rammp_curobo_interfaces/action/PlanToPose.action`
- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py:144-157` (`_plan_to_pose_cb`)
- Create: `rammp_curobo_ros/test/test_constraint_mapping.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: `PoseConstraint`, `ViaPoint` (Task 1), `plan_to_pose(..., constraint=, via=)` (Task 2).

- Produces: `planner_node.constraint_from_goal(request) -> tuple[PoseConstraint, ViaPoint, str | None]` — a module-level function (not a method), so it is testable without a ROS runtime, exactly like `check_start_joints`.

- [ ] **Step 1: Write the two message types**

Create `rammp_curobo_interfaces/msg/PoseAxisLock.msg`:

```
# Tool-pose components to hold fixed for the WHOLE trajectory.
#
# A locked component is held AT THE GOAL'S VALUE, so the start must already
# match the goal on that component (0.05 rad / 0.005 m). "Locked" means
# unchanged, not level: a tool at 45 degrees planning to a 45-degree goal
# stays at 45 the whole way. Level is simply the case where the goal is level.
#
# All fields false (the default) means unconstrained.

# Which frame the locked axes are measured in.
uint8 FRAME_BASE=0    # the robot base frame — gravity-relative, what "level" means
uint8 FRAME_GOAL=1    # projected into the goal frame (cuRobo's own default)
uint8 reference_frame 0

bool lock_roll
bool lock_pitch
bool lock_yaw
bool lock_x
bool lock_y
bool lock_z
```

Create `rammp_curobo_interfaces/msg/ApproachVia.msg`:

```
# One blended via point for the final approach to the goal.
#
# It is a COST, not a waypoint: the path passes NEAR the offset without
# stopping there and without hitting it exactly. If an exact intermediate
# pose is required, send two goals and accept the stop between them.
#
# offset of 0.0 (the default) means no via point.

uint8 AXIS_X=0
uint8 AXIS_Y=1
uint8 AXIS_Z=2        # the tool approach axis — the usual choice

float64 offset        # metres back along `axis` from the goal
uint8 axis 2
float64 at_fraction 0.8   # fraction of the motion at which it engages, in (0, 1)
```

Naming note for the reviewer: these are `PoseAxisLock` and `ApproachVia` rather than `PoseConstraint`/`ViaPoint` so the ROS types never collide with the core Python types of the same purpose in a reader's head — the core types are what the planner consumes, these are what the wire carries.

- [ ] **Step 2: Register them**

In `rammp_curobo_interfaces/CMakeLists.txt:13-18`, add both to the `rosidl_generate_interfaces` call alongside the existing actions and service, keeping the existing dependency list.

- [ ] **Step 3: Append the fields to the action**

At the **end** of the goal block in `rammp_curobo_interfaces/action/PlanToPose.action`, immediately before the first `---`:

```
# --- appended 2026-09-26. An older client sends neither, and an absent
# trailing field deserialises to a default-constructed message: no locks,
# no via point, i.e. exactly today's behaviour. Appended-at-the-end is a
# MINOR bump under rammp-interfaces-ros2's rules; inserting either of
# these earlier would corrupt every field after it.
PoseAxisLock axis_lock
ApproachVia approach_via
```

Note for the reviewer: a nested message field cannot carry a default in the `.action`, so "off" has to be what a default-constructed `PoseAxisLock`/`ApproachVia` already means. Both messages are designed that way — all-false locks, zero offset — which is why this still satisfies the interfaces repo's "zero must mean off" rule.

- [ ] **Step 4: Write the failing mapping tests**

Create `rammp_curobo_ros/test/test_constraint_mapping.py`:

```python
"""Mapping from PlanToPose goal fields to the core's constraint types.

No ROS runtime and no GPU — same shape as test_start_joints_contract.py.
"""

import pytest

try:
    from rammp_curobo_interfaces.action import PlanToPose

    from rammp_curobo_ros.planner_node import constraint_from_goal
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )


def test_default_goal_is_unconstrained():
    """An older client, or any client that sets nothing, must plan exactly
    as it did before these fields existed."""
    c, v, why = constraint_from_goal(PlanToPose.Goal())
    assert why is None
    assert not c.is_active()
    assert not v.is_active()


def test_named_locks_map_to_the_right_axes():
    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.lock_pitch = True
    c, _, why = constraint_from_goal(g)
    assert why is None
    assert c.hold_roll and c.hold_pitch and not c.hold_yaw
    assert c.hold_vec_weight() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_reference_frame_constant_maps_to_base_frame():
    from rammp_curobo_interfaces.msg import PoseAxisLock

    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.reference_frame = PoseAxisLock.FRAME_BASE
    c, _, _ = constraint_from_goal(g)
    assert c.in_base_frame is True

    g.axis_lock.reference_frame = PoseAxisLock.FRAME_GOAL
    c, _, _ = constraint_from_goal(g)
    assert c.in_base_frame is False


def test_approach_via_maps():
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    c, v, why = constraint_from_goal(g)
    assert why is None and v.is_active()
    assert v.offset_m == pytest.approx(0.1)
    assert v.linear_axis == 2  # the message's own default


def test_bad_fraction_is_reported_not_raised():
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    g.approach_via.at_fraction = 1.5
    _, _, why = constraint_from_goal(g)
    assert why is not None and "tstep_fraction" in why


def test_unknown_reference_frame_is_refused():
    g = PlanToPose.Goal()
    g.axis_lock.lock_roll = True
    g.axis_lock.reference_frame = 7
    _, _, why = constraint_from_goal(g)
    assert why is not None and "reference_frame" in why
```

- [ ] **Step 5: Run them to verify they fail**

Run: `colcon build --packages-select rammp_curobo_interfaces rammp_curobo_ros && pytest rammp_curobo_ros/test/test_constraint_mapping.py -q -p no:anyio`
Expected: FAIL — `ImportError: cannot import name 'constraint_from_goal'` (after the build succeeds; if the build fails, the `.action` edit is malformed — check the default-value syntax)

- [ ] **Step 6: Implement the mapping**

Add to `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`, module level, beside `check_start_joints`:

```python
def constraint_from_goal(request):
    """(PoseConstraint, ViaPoint, why_rejected) from a PlanToPose goal.

    A goal that sets neither message maps to "unconstrained", which is also
    what an older client's absent fields deserialise to.
    """
    lock = request.axis_lock
    if lock.reference_frame not in (PoseAxisLock.FRAME_BASE, PoseAxisLock.FRAME_GOAL):
        return (
            PoseConstraint(),
            ViaPoint(),
            "unknown reference_frame %d (expected FRAME_BASE=%d or FRAME_GOAL=%d)"
            % (lock.reference_frame, PoseAxisLock.FRAME_BASE, PoseAxisLock.FRAME_GOAL),
        )
    constraint = PoseConstraint(
        hold_roll=bool(lock.lock_roll),
        hold_pitch=bool(lock.lock_pitch),
        hold_yaw=bool(lock.lock_yaw),
        hold_x=bool(lock.lock_x),
        hold_y=bool(lock.lock_y),
        hold_z=bool(lock.lock_z),
        in_base_frame=(lock.reference_frame == PoseAxisLock.FRAME_BASE),
    )
    approach = request.approach_via
    via = ViaPoint(
        offset_m=float(approach.offset),
        linear_axis=int(approach.axis),
        tstep_fraction=float(approach.at_fraction),
    )
    try:
        constraint.validate()
        via.validate()
    except ValueError as exc:
        return constraint, via, str(exc)
    return constraint, via, None
```

with the imports at the top of the file:

```python
from rammp_curobo import PoseConstraint, ViaPoint
from rammp_curobo_interfaces.msg import PoseAxisLock
```

- [ ] **Step 7: Run the mapping tests and confirm they pass**

Run: `pytest rammp_curobo_ros/test/test_constraint_mapping.py -q -p no:anyio`
Expected: PASS (4 tests)

- [ ] **Step 8: Wire it into the callback**

Replace the body of `_plan_to_pose_cb` (`planner_node.py:144-157`) between the quaternion unpack and the `self._plan(...)` call:

```python
        constraint, via, why = constraint_from_goal(goal_handle.request)
        if why is not None:
            result.success = False
            result.message = "invalid constraint: %s" % why
            goal_handle.abort()
            return result
        res = self._plan(
            lambda q: self.planner.plan_to_pose(
                pos, quat, start=q, constraint=constraint, via=via
            ),
            goal_handle.request.start_joints,
        )
```

**Why abort with a message rather than reject the goal.** Rejecting is the more ROS-idiomatic response to a malformed goal, but a rejected goal gives the client a null handle and *no result message* — so the explanation of what was wrong never reaches them. This node already aborts-with-a-reason for a wrong-length `target_joints` (`planner_node.py:163-170`); match that. The diagnostic is worth more here than the idiom.

- [ ] **Step 9: Add the rejection test (Review Focus 2, ROS boundary)**

Append to `rammp_curobo_ros/test/test_constraint_mapping.py`:

```python
def test_out_of_range_fraction_is_refused_before_planning():
    """A bad fraction must be refused at the boundary — not handed to the
    GPU to come back as an opaque status."""
    g = PlanToPose.Goal()
    g.approach_via.offset = 0.1
    for bad in (1.5, -0.2, 0.0):
        g.approach_via.at_fraction = bad
        _, _, why = constraint_from_goal(g)
        assert why is not None, bad


def test_non_finite_offset_is_refused():
    g = PlanToPose.Goal()
    g.approach_via.offset = float("nan")
    _, _, why = constraint_from_goal(g)
    assert why is not None and "finite" in why
```

- [ ] **Step 10: Add the changelog entry**

In `CHANGELOG.md`, under `Unreleased` → `Added`:

```markdown
- Plan while holding tool-pose axes fixed (carry an object at a fixed
  orientation, #16) and approach a goal through one blended via point the arm
  does not stop at (#17). New core types `PoseConstraint` and `ViaPoint`; new
  messages `PoseAxisLock` and `ApproachVia`, carried by `PlanToPose` as
  `axis_lock` and `approach_via`. Both are appended at the end and default to
  "off", so a client that sets neither plans exactly as before.
```

- [ ] **Step 11: Run everything reachable without a GPU, then commit**

Run: `pytest core/tests -q && pytest rammp_curobo_ros/test -q -p no:anyio && pre-commit run --all-files`
Expected: PASS; the GPU suite skips.

```bash
git add rammp_curobo_interfaces/action/PlanToPose.action rammp_curobo_ros/ CHANGELOG.md
git commit -m "feat(ros): carry pose constraints and a via point on PlanToPose

Fields are appended with defaults, so an older client is unconstrained —
a minor bump under the interfaces repo's Humble/Cyclone rules."
```

______________________________________________________________________

### Task 5: A service to ask before you plan

The two-phase workflow — bring the locked axes to the goal's values, then plan the constrained move — only works if a caller can *ask* whether the constrained plan is attemptable from where the arm is. Making that a service is the ROS-native way to expose a question that has an answer but no side effects, and it is the same shape as the reachability query the old driver had.

A **new** service is a minor bump (`CONTRIBUTING.md:114`); nothing existing changes.

**Files:**

- Create: `rammp_curobo_interfaces/srv/CheckPoseLock.srv`
- Modify: `rammp_curobo_interfaces/CMakeLists.txt` (register it)
- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py` (server)
- Test: `rammp_curobo_ros/test/test_constraint_mapping.py`

**Interfaces:**

- Consumes: `constraint_from_goal` (Task 4), `CuRoboPlanner.constraint_satisfied_at_start` (Task 3).

- Produces: the service `~/check_pose_lock`.

- [ ] **Step 1: Write the service definition**

Create `rammp_curobo_interfaces/srv/CheckPoseLock.srv`:

```
# Can a locked-axis plan be attempted from this start, to this goal?
#
# A locked axis is held AT THE GOAL'S VALUE, so the start must already match
# the goal on that axis. This answers that question without planning and
# without moving anything.
geometry_msgs/Pose target
float64[] start_joints        # REQUIRED, same contract as the plan actions
PoseAxisLock axis_lock
---
bool satisfied                # true: the constrained plan can be attempted
string message                # empty when clean; names the axis when not
string worst_axis             # "roll" | "pitch" | "yaw" | "x" | "y" | "z" | ""
float64 worst_error           # rad or m, the amount that axis is off by
float64 limit                 # the tolerance that error is measured against
bool marginal                 # satisfied, but close enough to the limit to warn
```

- [ ] **Step 2: Write the failing test for the reply mapping**

Append to `rammp_curobo_ros/test/test_constraint_mapping.py`:

```python
def test_check_reply_reports_the_offending_axis():
    from rammp_curobo_ros.planner_node import check_reply

    rep = check_reply(False, "held axis 'roll' differs by 0.1234 (limit 0.050)")
    assert rep["satisfied"] is False
    assert rep["worst_axis"] == "roll"
    assert rep["worst_error"] == pytest.approx(0.1234)
    assert rep["limit"] == pytest.approx(0.05)
    assert rep["marginal"] is False


def test_check_reply_clean():
    from rammp_curobo_ros.planner_node import check_reply

    rep = check_reply(True, None)
    assert rep["satisfied"] is True
    assert rep["message"] == ""
    assert rep["worst_axis"] == ""
    assert rep["marginal"] is False


def test_check_reply_marginal():
    from rammp_curobo_ros.planner_node import check_reply

    rep = check_reply(True, "marginal: held axis 'pitch' is 0.0450 from the goal")
    assert rep["satisfied"] is True and rep["marginal"] is True
    assert rep["worst_axis"] == "pitch"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pytest rammp_curobo_ros/test/test_constraint_mapping.py -q -p no:anyio -k check_reply`
Expected: FAIL — `ImportError: cannot import name 'check_reply'`

- [ ] **Step 4: Implement the reply mapping**

The core returns `(ok, reason)` where reason is prose. Parse it once, here, so the structured fields and the sentence can never disagree. Add to `planner_node.py`, module level:

```python
_AXIS_RE = re.compile(r"'(roll|pitch|yaw|x|y|z)'")
_NUM_RE = re.compile(r"([-+]?\d*\.?\d+)")


def check_reply(ok, reason):
    """(ok, reason) from the core -> the CheckPoseLock reply fields."""
    out = {
        "satisfied": bool(ok),
        "message": reason or "",
        "worst_axis": "",
        "worst_error": 0.0,
        "limit": 0.0,
        "marginal": bool(ok) and bool(reason) and reason.startswith("marginal"),
    }
    if not reason:
        return out
    m = _AXIS_RE.search(reason)
    if m:
        out["worst_axis"] = m.group(1)
    nums = _NUM_RE.findall(reason)
    if nums:
        out["worst_error"] = float(nums[0])
    if len(nums) > 1:
        out["limit"] = float(nums[1])
    return out
```

Add `import re` at the top of the module if it is not already imported.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `pytest rammp_curobo_ros/test/test_constraint_mapping.py -q -p no:anyio -k check_reply`
Expected: PASS (3 tests)

If `test_check_reply_reports_the_offending_axis` reads `0.050` as the error and `0.1234` as the limit, the pre-check message in Task 3 states them in the other order — fix the order in *one* place and note which, rather than adding a special case here.

- [ ] **Step 6: Register and serve it**

Register `srv/CheckPoseLock.srv` in `rammp_curobo_interfaces/CMakeLists.txt`, then add the server beside the existing `set_world` service (`planner_node.py:119-121`):

```python
        self.create_service(
            CheckPoseLock, "~/check_pose_lock", self._check_pose_lock_cb,
            callback_group=self._cb_group,
        )
```

```python
    def _check_pose_lock_cb(self, request, response):
        why = check_start_joints(request.start_joints, self.planner.joint_names)
        if why is not None:
            response.satisfied = False
            response.message = why
            return response
        pos = [request.target.position.x, request.target.position.y,
               request.target.position.z]
        quat = [request.target.orientation.x, request.target.orientation.y,
                request.target.orientation.z, request.target.orientation.w]
        constraint, _, bad = constraint_from_goal(request)
        if bad is not None:
            response.satisfied = False
            response.message = bad
            return response
        ok, reason = self.planner.constraint_satisfied_at_start(
            [float(v) for v in request.start_joints], pos, quat, constraint
        )
        for k, v in check_reply(ok, reason).items():
            setattr(response, k, v)
        return response
```

This is FK and arithmetic only — it takes no plan lock and cannot block a plan in flight, which is what makes it safe to call repeatedly from a UI.

- [ ] **Step 7: Note it in the changelog and commit**

Add to the same `Unreleased` → `Added` entry: "and `~/check_pose_lock`, which answers whether a locked-axis plan can be attempted from a given start without planning or moving."

```bash
git add rammp_curobo_interfaces/ rammp_curobo_ros/ CHANGELOG.md
git commit -m "feat(ros): check_pose_lock service — ask before planning

A locked axis is held at the goal's value, so the start must already
match it. This answers that without planning and without the plan lock."
```

______________________________________________________________________

### Task 6: Prove it on the Jetson

Everything above is mapping and validation. Nothing so far has run cuRobo. This task is where the feature is either real or not, and it is also where the two unverified details get settled: whether `PoseCostMetric` takes the via-point fields as constructed in Task 2, and whether `in_base_frame` actually means level with gravity.

**Files:**

- Modify: `core/tests/test_smoke.py`
- Modify: `docs/HARDWARE_BRINGUP.md`

**Interfaces:**

- Consumes: everything from Tasks 1-5.

- Produces: no code interface; a verified answer recorded in the docs.

- [ ] **Step 1: Write the constraint smoke test**

Append to `core/tests/test_smoke.py`:

```python
def test_constrained_plan_holds_orientation(planner, start):
    """A held-axis plan keeps roll and pitch fixed for the WHOLE path, not
    just at the endpoints — checked by FK over every point."""
    from rammp_curobo import PoseConstraint
    from rammp_curobo.geometry import rotvec_between

    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    pos, quat = planner.fk(q_target)

    ok, why = planner.constraint_satisfied_at_start(
        start, pos, quat, PoseConstraint(hold_roll=True, hold_pitch=True)
    )
    if not ok:
        pytest.skip("retract start is not level for this goal: %s" % why)

    res = planner.plan_to_pose(
        pos, quat, start, constraint=PoseConstraint(hold_roll=True, hold_pitch=True)
    )
    assert res.success, res.error

    _, quat0 = planner.fk(res.joint_traj.positions[0])
    worst = 0.0
    for q in res.joint_traj.positions:
        _, qk = planner.fk(q)
        rot = rotvec_between(quat0, qk)
        worst = max(worst, abs(rot[0]), abs(rot[1]))
    assert worst < 0.05, "roll/pitch drifted %.4f rad along the path" % worst
```

- [ ] **Step 2: Write the via-point smoke test**

```python
def test_via_point_does_not_stop_the_arm(planner, start):
    """The whole point of the via point: a blended approach with no
    zero-velocity dip. A chained two-segment plan would show one."""
    from rammp_curobo import ViaPoint

    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    q_target[5] -= 0.3
    pos, quat = planner.fk(q_target)

    res = planner.plan_to_pose(pos, quat, start, via=ViaPoint(offset_m=0.10))
    assert res.success, res.error
    vel = res.joint_traj.velocities
    assert vel is not None

    speed = np.abs(vel).max(axis=1)
    interior = speed[2:-2]  # ends are legitimately at rest
    assert interior.min() > 1e-3, (
        "commanded motion stalls mid-path (min |qd| = %.5f) — the via point "
        "is behaving like a stop, not a blend" % interior.min()
    )
```

- [ ] **Step 3: Write the unchanged-when-unconstrained test**

```python
def test_unconstrained_plan_is_unchanged(planner, start):
    """The feature must be inert when nobody asks for it."""
    q_target = list(planner.retract_pose)
    q_target[0] += 0.4
    pos, quat = planner.fk(q_target)

    a = planner.plan_to_pose(pos, quat, start)
    b = planner.plan_to_pose(pos, quat, start, constraint=None, via=None)
    assert a.success and b.success
    assert a.joint_traj.dof == b.joint_traj.dof
    assert abs(a.joint_traj.duration - b.joint_traj.duration) < 0.5
```

- [ ] **Step 4: Sync to the Jetson and run the GPU suite**

Run (from the repo root, per `CONTRIBUTING.md:88-103`): `python3 hil.py sync && python3 hil.py exec -- pytest core/tests -q`
Expected: the three new tests pass alongside the existing 23.

**If `PoseCostMetric(**metric_kw)` raises a TypeError or the via-point fields are ignored**, switch `_plan_config` to `PoseCostMetric.create_grasp_approach_metric(offset_position=..., linear_axis=..., tstep_fraction=...)` for the via-point case, re-run, and record which construction worked in the `_pose_cost_kwargs` docstring. This is the expected place for that to surface.

- [ ] **Step 5: Settle the frame question on the bench**

With the arm level and a spirit level (or a phone level app) on the gripper, plan and execute a constrained transport with `in_base_frame=True`, then with `False`. Record which one keeps the tool level with respect to **gravity**.

If `True` is not the gravity-level one, the mapping in `_pose_cost_kwargs` (`project_to_goal_frame = not in_base_frame`) is inverted — fix it there, not at the call sites, and add a test asserting the mapping.

- [ ] **Step 6: Record what was learned**

In `docs/HARDWARE_BRINGUP.md`, add a short section under the existing planning notes:

```markdown
## Constrained plans and via points

- A held axis is held **at the goal's value**, so the start must already
  match the goal on that axis (0.05 rad, 0.005 m). "Held" means
  unchanged, not level: a tool at 45 degrees planning to a 45-degree
  goal stays at 45 the whole way. Level is just the case where the goal
  is level. When the start does not match, an unconstrained move has to
  bring those axes there first — `constraint_satisfied_at_start` tells
  you before you plan.
- `in_base_frame=True` is the gravity-relative sense — verified on the
  bench <DATE>, with a level on the gripper.
- The via point is a COST, not a waypoint: the path passes near the
  offset without stopping and does not hit it exactly. If an exact
  intermediate pose is required, send two queued goals and accept the
  stop between them.
```

Replace `<DATE>` with the date of the bench run, and correct the second bullet if Step 5 found the opposite.

- [ ] **Step 7: Commit**

```bash
git add core/tests/test_smoke.py docs/HARDWARE_BRINGUP.md
git commit -m "test(smoke): constrained plans hold orientation; via points do not stop

Verified on the Jetson against the real planner, including which frame
sense keeps the tool level with gravity."
```

______________________________________________________________________

## Notes for the reviewer

- **The graph planner is not a blocker here**, despite constrained planning looking like it should need it. cuRobo applies the pose metric only to IK, trajopt and finetune-trajopt rollouts and explicitly excludes the graph planner from that list, and NVIDIA's own constrained example runs with `enable_graph=False`. Issue #16 carries the three source citations. No task in this plan touches `enable_graph`.
- **Two details were read from cuRobo's source but never executed**: the exact `PoseCostMetric` construction for the via point (Task 2, Step 5) and which frame sense is gravity-relative (Task 6, Step 5). Both are settled on the Jetson in Task 6, and both have a named fallback. They are the parts most likely to need a second pass.
- **Tasks 1-5 are all testable on a laptop.** Only Task 6 needs hardware. If the Jetson is unavailable, Tasks 1-5 still merge safely: the feature is inert unless a caller asks for it, which Task 6, Step 3 pins.
