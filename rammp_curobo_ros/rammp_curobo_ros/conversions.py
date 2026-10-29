"""Message <-> library-type conversions. Pure functions, unit-testable."""

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo import Trajectory


def duration_msg(seconds):
    sec = int(seconds)
    return DurationMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def trajectory_to_msg(traj):
    """rammp_curobo Trajectory -> trajectory_msgs/JointTrajectory.

    Point k is stamped at (k + 1) * dt, the library's convention: point 0
    is one step in the future so the executing controller ramps from the
    current (matching) state instead of jumping. The header carries the
    planning frame (arm_mount); the consumer refuses a pose plan without it.
    """
    msg = JointTrajectory()
    msg.header.frame_id = traj.frame_id
    msg.joint_names = list(traj.joint_names)
    for k in range(traj.n_points):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in traj.positions[k]]
        if traj.velocities is not None:
            p.velocities = [float(v) for v in traj.velocities[k]]
        if traj.accelerations is not None:
            p.accelerations = [float(v) for v in traj.accelerations[k]]
        p.time_from_start = duration_msg((k + 1) * traj.dt)
        msg.points.append(p)
    return msg


def trajectory_from_msg(msg, joint_names, frame_id=""):
    """trajectory_msgs/JointTrajectory -> (rammp_curobo Trajectory in
    `joint_names` order, None), or (None, why) when it is not a trajectory
    the planner could have produced: other joints, unevenly spaced points, or
    a point without positions and velocities. Accelerations are kept only when
    every point has them."""
    if sorted(msg.joint_names) != sorted(joint_names):
        return None, "joints [%s], expected [%s]" % (
            ", ".join(msg.joint_names),
            ", ".join(joint_names),
        )
    if not msg.points:
        return None, "no points"
    n = len(joint_names)
    if any(len(p.positions) != n or len(p.velocities) != n for p in msg.points):
        return None, "every point needs %d positions and %d velocities" % (n, n)
    order = [list(msg.joint_names).index(j) for j in joint_names]
    t = np.array(
        [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    )
    dt = t[0] if len(t) == 1 else t[1] - t[0]
    if not dt > 0 or np.abs(np.diff(t) - dt).max(initial=0.0) > 1e-6:
        return None, "points are not evenly spaced in time"
    has_a = all(len(p.accelerations) == n for p in msg.points)
    return (
        Trajectory(
            joint_names=list(joint_names),
            positions=np.array([p.positions for p in msg.points])[:, order],
            velocities=np.array([p.velocities for p in msg.points])[:, order],
            accelerations=(
                np.array([p.accelerations for p in msg.points])[:, order]
                if has_a
                else None
            ),
            dt=float(dt),
            frame_id=frame_id,
        ),
        None,
    )
