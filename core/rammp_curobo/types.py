"""Public result types. No cuRobo (or ROS) types cross this boundary."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Trajectory:
    """A time-parameterized joint trajectory on a uniform grid.

    Point k (0-based) is scheduled at time_from_start = (k + 1) * dt — the
    convention the ros2_control JointTrajectoryController expects when the
    start state is the arm's current pose (point 0 is one step in the
    future, not "now").
    """

    joint_names: List[str]
    positions: np.ndarray  # (N, dof) rad
    velocities: Optional[np.ndarray]  # (N, dof) rad/s, or None
    accelerations: Optional[np.ndarray]  # (N, dof) rad/s^2, or None
    dt: float  # uniform point spacing, s
    speed_scale: float = 1.0  # cumulative retiming applied

    @property
    def n_points(self) -> int:
        return int(self.positions.shape[0])

    @property
    def dof(self) -> int:
        return int(self.positions.shape[1])

    @property
    def duration(self) -> float:
        return self.n_points * self.dt

    def scaled(self, speed_scale: float) -> "Trajectory":
        """Exact time-dilation retiming — see retime.scale_trajectory."""
        from rammp_curobo.retime import scale_trajectory

        return scale_trajectory(self, speed_scale)


@dataclass
class PlanResult:
    """Outcome of one planning call.

    `success` is only True when cuRobo reported success AND the library's
    own limit/continuity validation passed. Never execute a trajectory from
    a result with success=False — there may not even be one.
    """

    success: bool
    joint_traj: Optional[Trajectory]
    timing: float  # planner wall time, s
    error: Optional[str]  # human-readable, when failed
    status: str = ""  # raw cuRobo status string
    validated: bool = False  # library validation ran clean
    final_joints: Optional[np.ndarray] = None  # last trajectory point
    # plan_to_joints via the FK-pose fallback reaches the requested EE pose
    # but not necessarily the exact joint vector — this records the gap.
    goal_mismatch_rad: Optional[float] = None

    @classmethod
    def failure(cls, status: str, error: str, timing: float = 0.0) -> "PlanResult":
        return cls(
            success=False, joint_traj=None, timing=timing, error=error, status=status
        )
