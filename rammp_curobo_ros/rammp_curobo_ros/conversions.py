"""Message <-> library-type conversions. Pure functions, unit-testable."""

from builtin_interfaces.msg import Duration as DurationMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def duration_msg(seconds):
    sec = int(seconds)
    return DurationMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def trajectory_to_msg(traj):
    """rammp_curobo Trajectory -> trajectory_msgs/JointTrajectory.

    Point k is stamped at (k + 1) * dt, the library's convention: point 0
    is one step in the future so the executing controller ramps from the
    current (matching) state instead of jumping.
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
