"""Message <-> library-type conversions. Pure functions, unit-testable."""

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def duration_msg(seconds):
    sec = int(seconds)
    return DurationMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def trajectory_to_msg(traj):
    """rammp_curobo Trajectory -> trajectory_msgs/JointTrajectory.

    Point k is stamped at (k + 1) * dt, the library's convention: point 0
    is one step in the future so the controller ramps from the current
    (matching) state instead of jumping.
    """
    msg = JointTrajectory()
    msg.joint_names = list(traj.joint_names)
    for k in range(traj.n_points):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in traj.positions[k]]
        if traj.velocities is not None:
            p.velocities = [float(v) for v in traj.velocities[k]]
        p.time_from_start = duration_msg((k + 1) * traj.dt)
        msg.points.append(p)
    return msg


def msg_arrays(msg):
    """JointTrajectory msg -> (positions (N,dof), velocities|None, times (N,)).

    Works for non-uniform spacing — the executor's validation and scaling
    operate on these arrays, so trajectories from any planner are handled.
    """
    pos = np.array([p.positions for p in msg.points], dtype=float)
    times = np.array(
        [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
         for p in msg.points], dtype=float)
    if all(len(p.velocities) == len(msg.joint_names) for p in msg.points):
        vel = np.array([p.velocities for p in msg.points], dtype=float)
    else:
        vel = None
    return pos, vel, times


def scaled_msg(msg, speed_scale):
    """Exact time dilation of a JointTrajectory msg: t/s, v*s, a*s^2."""
    s = float(speed_scale)
    out = JointTrajectory()
    out.joint_names = list(msg.joint_names)
    for p in msg.points:
        q = JointTrajectoryPoint()
        q.positions = list(p.positions)
        q.velocities = [v * s for v in p.velocities]
        q.accelerations = [a * s * s for a in p.accelerations]
        t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        q.time_from_start = duration_msg(t / s)
        out.points.append(q)
    return out
