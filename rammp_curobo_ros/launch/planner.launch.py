"""Launch the rammp_curobo planner node.

The node is a planning service: it takes an end position (a tool pose or a
joint goal) and returns the collision-free joint trajectory. It never owns
the arm — execution goes to the kinova_arm_ros2 driver's
/execute_joint_trajectory action (start `kinova_arm_node` separately:
`--sim` for the simulator, `--ip 192.168.1.10` for the real Gen3).

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
            description="true only under a /clock-publishing simulator "
            "(kinova_arm_node --sim runs on wall clock: leave false)",
        ),
        DeclareLaunchArgument(
            "arm_action",
            default_value="/execute_joint_trajectory",
            description="the driver's ExecuteJointTrajectory action name",
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
                "execute": LaunchConfiguration("execute"),
                "speed_scale": LaunchConfiguration("speed_scale"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "arm_action": LaunchConfiguration("arm_action"),
            }
        ],
    )
    return LaunchDescription(args + [node])
