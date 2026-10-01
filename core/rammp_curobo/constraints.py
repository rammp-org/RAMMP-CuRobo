"""What a caller wants held or approached — no cuRobo, no torch, no ROS.

The axis ordering of cuRobo's vec_weight is [rx, ry, rz, x, y, z]:
ORIENTATION FIRST, position last, 1.0 = hold that axis. That conversion
lives in exactly one place — PoseConstraint.hold_vec_weight() — because
getting it backwards produces a plan that is constrained in the wrong
three axes and still succeeds.

Which of rx/ry/rz is roll vs pitch vs yaw follows cuRobo's base-frame
axis order (x, y, z). VERIFIED on the Jetson against real cuRobo
(2026-09-26): the smoke suite's `test_constrained_plan_holds_orientation`
passed — holding roll/pitch genuinely holds them across a real plan. That
is a pass/fail, not a bound, which is why the planner now measures the
worst deviation and refuses a plan that exceeds the tolerance.
"""

import math
from dataclasses import dataclass


# Orientation hold modes. Two, deliberately — see RAMMP-CuRobo#16.
#
# There is no frame parameter and no position hold. Per-axis holds expressed a
# MECHANISM ("hold rx and ry in the base frame") rather than a request, in a
# frame whose axes mean nothing to a caller. Position holds were worse: they
# could only express "travel along one base axis", and the general case — a
# straight line in an arbitrary direction — is not reachable through
# PoseCostMetric at all, because hold_vec_weight is a DIAGONAL weight and
# freeing an arbitrary direction v needs the non-diagonal form I - vv^T. A
# Cartesian linear move is a different operation and belongs elsewhere (#17).
HOLD_NONE = 0
HOLD_LEVEL = 1
HOLD_FIXED = 2
_HOLD_NAMES = {HOLD_NONE: "none", HOLD_LEVEL: "level", HOLD_FIXED: "fixed"}


@dataclass(frozen=True)
class PoseConstraint:
    """How much of the tool's ORIENTATION to keep for the whole trajectory.

    HOLD_LEVEL keeps roll and pitch and leaves yaw free, so the only motion the
    tool may add is a spin about base Z. A pure rotation about Z has rotation
    vector (0, 0, theta) for any theta, so that freedom is exact at any
    magnitude — and a spin about the vertical cannot change how far something
    is tipped.

    Three things this does NOT do, all easy to assume:

    It PRESERVES tilt rather than creating level. roll/pitch are pinned at the
    held values, so a gripper that starts 20 degrees off stays 20 degrees off
    for the whole motion. "Level" means "as level as you already are", which is
    why the planner derives the held orientation from the START.

    It reads "level with the world" as "level with base Z". True while the arm
    is mounted level, and silently false the moment it is not: on a tilted
    mount the freed rotation is about a tilted axis, so it tips the very object
    the hold was asked to protect. The fix is declaring the mount transform so
    base_link IS gravity-aligned (#20), not another field here — an undeclared
    tilt breaks the driver's gravity compensation too.

    It is a COST, not a clamp. cuRobo penalises deviation, it does not forbid
    it. The planner therefore VERIFIES the returned trajectory and refuses a
    plan whose worst deviation exceeds the configured tolerance, rather than
    trusting the optimiser to have honoured the request.
    """

    hold: int = HOLD_NONE

    def is_active(self) -> bool:
        return self.hold != HOLD_NONE

    def name(self) -> str:
        return _HOLD_NAMES.get(self.hold, str(self.hold))

    def hold_vec_weight(self):
        """cuRobo's 6-vector: [rx, ry, rz, x, y, z]. 1.0 holds that axis.

        Position entries are always 0.0 — this interface does not hold
        position. See the module note above for why.
        """
        if self.hold == HOLD_LEVEL:
            return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if self.hold == HOLD_FIXED:
            return [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def validate(self) -> None:
        if self.hold not in _HOLD_NAMES:
            raise ValueError(
                "hold must be HOLD_NONE (0), HOLD_LEVEL (1) or HOLD_FIXED (2), "
                "got %r" % (self.hold,)
            )
        return None


@dataclass(frozen=True)
class ViaPoint:
    """One blended intermediate target, offset from the GOAL along a tool axis.

    This is cuRobo's grasp-approach metric
    (`PoseCostMetric.create_grasp_approach_metric`, read by introspection
    against real cuRobo v0.7.8 on the Jetson) and it is a cost, not a hard
    waypoint: the trajectory passes NEAR the offset without stopping, it
    does not hit it exactly. offset_m == 0.0 means no via point.

    SHELVED — not exposed on the arm interface. See kinova-gen3-ros2#40 for
    everything measured. The capability stays here; nothing drives it.

    A via point holds the other FIVE pose components (all three rotations plus
    the two linear axes other than `linear_axis`) at the GOAL's values — but
    only from `tstep_fraction` ONWARD, not for the whole trajectory. An earlier
    version of this docstring claimed the whole trajectory, and that error
    propagated into the arm interface and two demo scripts. Measured on the arm
    with one fixed start and goal: a PoseConstraint on a travelled axis is
    refused with INVALID_PARTIAL_POSE_COST_METRIC, while the same goal carrying
    a via plans normally — because the via's hold is not active at the start.

    `linear_axis` is also resolved in the GOAL frame, always:
    create_grasp_approach_metric accepts project_to_goal_frame and never uses
    it, so the constructed metric inherits PoseCostConfig's projected default.
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
        if not math.isfinite(self.tstep_fraction):
            raise ValueError(
                "via tstep_fraction must be finite, got %r" % (self.tstep_fraction,)
            )
        if self.linear_axis not in (0, 1, 2):
            raise ValueError(
                "via linear_axis must be 0, 1 or 2, got %r" % (self.linear_axis,)
            )
        if not (0.0 < self.tstep_fraction < 1.0):
            raise ValueError(
                "via tstep_fraction must be in (0, 1), got %r" % (self.tstep_fraction,)
            )
        return None
