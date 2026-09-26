"""What a caller wants held or approached — no cuRobo, no torch, no ROS.

The axis ordering of cuRobo's vec_weight is [rx, ry, rz, x, y, z]:
ORIENTATION FIRST, position last, 1.0 = hold that axis. That conversion
lives in exactly one place — PoseConstraint.hold_vec_weight() — because
getting it backwards produces a plan that is constrained in the wrong
three axes and still succeeds.

Which of rx/ry/rz is roll vs pitch vs yaw follows cuRobo's base-frame
axis order (x, y, z). This is TO BE verified on hardware — it has never
run there — and the smoke suite's
`test_constrained_plan_holds_orientation` exists for that purpose, not
here.
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
