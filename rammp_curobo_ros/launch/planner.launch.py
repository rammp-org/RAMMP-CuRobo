"""Launch the rammp_curobo planner node.

The node is a planning service: it takes an end position (a tool pose or a
joint goal) and returns the collision-free joint trajectory. It never owns
the arm — execution goes to whatever ros2_control stack is already running
(started separately; on this lab's Jetson that is the RAMMP-Kinova
workspace's kortex bringup or MuJoCo sim).

Planning-only (the Docker/service use — nothing can move):

    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml

Plan + execute against an existing bringup (sim or real):

    ros2 launch rammp_curobo_ros planner.launch.py use_sim_time:=true \
        execute:=true                                                   # sim
    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml \
        execute:=true                                                   # real

Execution stays disabled until execute:=true is passed — the node plans
(dry-run) but refuses ExecuteTrajectory goals.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "config",
            default_value="gen3.yaml",
            description="planner YAML (packaged name or path)",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="",
            description="world YAML override (empty = config default)",
        ),
        DeclareLaunchArgument(
            "execute",
            default_value="false",
            description="allow motion (default: dry-run only)",
        ),
        DeclareLaunchArgument(
            "speed_scale",
            default_value="0.0",
            description="execution speed scale; 0 = config default (0.25)",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="true when running against the MuJoCo sim",
        ),
        DeclareLaunchArgument(
            "controller_action",
            default_value="/joint_trajectory_controller/follow_joint_trajectory",
            description="FollowJointTrajectory action of the arm's controller",
        ),
    ]

    node = Node(
        package="rammp_curobo_ros",
        executable="planner_node",
        name="rammp_curobo",
        output="screen",
        emulate_tty=True,
        # an in-flight cuRobo solve runs ~5 s and cannot be interrupted;
        # shutdown waits for it, so give launch more than its 5 s default
        # before it escalates SIGINT to SIGTERM
        sigterm_timeout="12",
        parameters=[
            {
                "config": LaunchConfiguration("config"),
                "world": LaunchConfiguration("world"),
                # typed: a bare substitution is a STRING, and execute:=1
                # or speed_scale:=1 would abort the node at declare time
                "execute": ParameterValue(
                    LaunchConfiguration("execute"), value_type=bool
                ),
                "speed_scale": ParameterValue(
                    LaunchConfiguration("speed_scale"), value_type=float
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "controller_action": LaunchConfiguration("controller_action"),
            }
        ],
    )
    return LaunchDescription(args + [node])
