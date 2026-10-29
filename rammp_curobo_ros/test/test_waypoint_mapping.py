"""Mapping from PlanThrough goal fields to the core's Waypoint, and from a
Blend goal's legs to its Trajectory.

No ROS runtime and no GPU — same shape as test_constraint_mapping.py.
"""

import pytest

try:
    from rammp_curobo_interfaces.action import Blend, PlanThrough
    from rammp_curobo_interfaces.msg import Waypoint as WaypointMsg
    from trajectory_msgs.msg import JointTrajectory

    from rammp_curobo_ros.planner_node import legs_from_goal, waypoints_from_goal
except ImportError as exc:  # pragma: no cover
    # see test_constraint_mapping.py: the skip names what was missing
    pytest.skip(
        "cannot import the planner node or its messages, so NOTHING in this "
        "file ran: %s. Source install/setup.bash and put RAMMP-CuRobo/core "
        "and RAMMP-CuRobo/rammp_curobo_ros on PYTHONPATH." % exc,
        allow_module_level=True,
    )


def _goal(*waypoints):
    return PlanThrough.Goal(waypoints=list(waypoints), start_joints=[0.0] * 7)


def _pose(**fields):
    w = WaypointMsg(kind=WaypointMsg.POSE, **fields)
    w.pose.position.x, w.pose.orientation.w = 0.4, 1.0
    return w


def test_poses_and_joints_map_field_by_field():
    goal = _goal(
        _pose(hold=1, blend_radius=0.05),
        WaypointMsg(kind=WaypointMsg.JOINTS, joints=[0.1] * 7, straight=True),
    )
    (pose, joints), why = waypoints_from_goal(goal, 7)
    assert why is None
    assert pose.position == (0.4, 0.0, 0.0) and pose.quaternion == (0, 0, 0, 1.0)
    assert pose.constraint.hold == 1 and pose.blend_radius == 0.05
    assert joints.joints == (0.1,) * 7 and joints.straight
    assert joints.blend_radius == 0.0  # the default stops


@pytest.mark.parametrize(
    "waypoint, why",
    [
        (WaypointMsg(kind=7), "waypoint 1/1: unknown kind 7"),
        (WaypointMsg(kind=WaypointMsg.JOINTS, joints=[0.0] * 6), "7 joints expected"),
        (
            WaypointMsg(kind=WaypointMsg.JOINTS, joints=[float("nan")] * 7),
            "7 joints expected",
        ),
        (_pose(hold=9), "hold must be"),
        (_pose(straight=True), "only a joint waypoint can be straight"),
        (_pose(blend_radius=-0.1), "blend_radius must be finite and >= 0"),
    ],
)
def test_a_bad_waypoint_is_refused_not_coerced(waypoint, why):
    waypoints, reason = waypoints_from_goal(_goal(waypoint), 7)
    assert waypoints == [] and why in reason


def test_no_waypoints_is_refused():
    assert waypoints_from_goal(_goal(), 7) == ([], "no waypoints")


def test_a_blend_goal_names_the_leg_it_refuses():
    joints = ["arm_joint_%d" % i for i in range(1, 8)]
    assert legs_from_goal(Blend.Goal(), joints) == ([], "no legs")
    goal = Blend.Goal(legs=[JointTrajectory(joint_names=joints[:6])])
    legs, why = legs_from_goal(goal, joints)
    assert legs == [] and why.startswith("leg 1/1: joints")
