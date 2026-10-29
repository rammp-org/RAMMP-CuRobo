"""Mapping from a Blend goal to the core's legs, speed scales and holds.

No ROS runtime and no GPU — same shape as test_constraint_mapping.py.
"""

import pytest

try:
    from rammp_curobo_interfaces.action import Blend
    from trajectory_msgs.msg import JointTrajectory

    from rammp_curobo import Trajectory
    from rammp_curobo_ros.conversions import trajectory_to_msg
    from rammp_curobo_ros.planner_node import legs_from_goal
except ImportError as exc:  # pragma: no cover
    # see test_constraint_mapping.py: the skip names what was missing
    pytest.skip(
        "cannot import the planner node or its messages, so NOTHING in this "
        "file ran: %s. Source install/setup.bash and put RAMMP-CuRobo/core "
        "and RAMMP-CuRobo/rammp_curobo_ros on PYTHONPATH." % exc,
        allow_module_level=True,
    )

import numpy as np

JOINTS = ["arm_joint_%d" % i for i in range(1, 8)]


def _leg():
    return trajectory_to_msg(
        Trajectory(JOINTS, np.zeros((3, 7)), np.zeros((3, 7)), None, 0.02)
    )


def test_legs_scales_and_holds_map_one_per_leg():
    goal = Blend.Goal(legs=[_leg(), _leg()], speed_scale=[0.5, 1.0], hold=[1, 0])
    legs, scales, holds, why = legs_from_goal(goal, JOINTS)
    assert why is None and len(legs) == 2
    assert scales == [0.5, 1.0]
    assert holds[0].hold == 1 and holds[1] is None


def test_empty_scales_and_holds_mean_full_speed_and_no_hold():
    legs, scales, holds, why = legs_from_goal(Blend.Goal(legs=[_leg()]), JOINTS)
    assert why is None and scales is None and holds == [None]


@pytest.mark.parametrize(
    "goal, why",
    [
        (Blend.Goal(), "no legs"),
        (
            Blend.Goal(legs=[JointTrajectory(joint_names=JOINTS[:6])]),
            "leg 1/1: joints",
        ),
        (Blend.Goal(legs=[_leg(), _leg()], speed_scale=[0.5]), "speed_scale needs 2"),
        (Blend.Goal(legs=[_leg(), _leg()], hold=[1]), "hold needs 2"),
        (Blend.Goal(legs=[_leg()], hold=[7]), "leg 1/1: hold must be one of"),
    ],
)
def test_a_bad_goal_is_refused_naming_what(goal, why):
    legs, _, _, reason = legs_from_goal(goal, JOINTS)
    assert legs == [] and why in reason
