"""Launch the rammp_curobo planner node.

The node is a planning service: it takes a start configuration and an end
position (a tool pose or a joint goal) and returns the collision-free joint
trajectory. It never owns the arm — it holds no driver, no controller
client, and no /joint_states subscription. Executing the returned
trajectory is the caller's job (kinova_arm_ros2 on this bench).

    ros2 launch rammp_curobo_ros planner.launch.py
    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml

There is no execute flag: nothing this node does can move the arm.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
            "use_sim_time",
            default_value="false",
            description="true when planning against the MuJoCo sim world",
        ),
    ]

    node = Node(
        package="rammp_curobo_ros",
        executable="planner_node",
        name="rammp_curobo",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "config": LaunchConfiguration("config"),
                "world": LaunchConfiguration("world"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )
    return LaunchDescription(args + [node])
