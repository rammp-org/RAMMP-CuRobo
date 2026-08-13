"""Launch the rammp_curobo planner node — optionally with the arm driver.

One-terminal hardware bringup (driver + planner together; needs the
RAMMP-Kinova workspace sourced for kortex_bringup):

    ros2 launch rammp_curobo_ros planner.launch.py \
        config:=gen3_real.yaml execute:=true launch_arm:=true

launch_arm defaults FALSE because exactly one controller manager may own
the arm: leave it off when a kortex bringup or the MuJoCo sim is already
running (both claim /controller_manager — a second one fights the first).

Planner-only against an existing bringup (sim or real):

    ros2 launch rammp_curobo_ros planner.launch.py use_sim_time:=true   # sim
    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml \
        execute:=true                                                   # real

Execution stays disabled until execute:=true is passed — the node plans
(dry-run) but refuses ExecuteTrajectory goals.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
        ),
        DeclareLaunchArgument(
            "launch_arm",
            default_value="false",
            description="also start the real-arm kortex bringup (driver + "
            "controllers). NEVER with a sim or second bringup running — "
            "one /controller_manager per arm.",
        ),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.10"),
    ]

    kortex_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("kortex_bringup"), "launch", "gen3.launch.py"]
            )
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "dof": "7",
            "gripper": "robotiq_2f_85",
            "use_fake_hardware": "false",
            "launch_rviz": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_arm")),
    )

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
                "controller_action": LaunchConfiguration("controller_action"),
            }
        ],
    )
    return LaunchDescription(args + [kortex_bringup, node])
