"""Launch the rammp_curobo planner node.

This launch does NOT own any robot bringup — start the arm side first, then
this node (identical either way, only the world/use_sim_time change):

  sim (RAMMP-Kinova workspace):
    ros2 launch mujoco_sim mujoco_bringup.launch.py
    ros2 launch rammp_curobo_ros planner.launch.py use_sim_time:=true

  real Gen3 (ros2_kortex; human on the physical e-stop):
    ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 \
        dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false
    ros2 launch rammp_curobo_ros planner.launch.py world:=world_real_bench.yaml

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
            description="true when running against the MuJoCo sim",
        ),
        DeclareLaunchArgument(
            "controller_action",
            default_value="/joint_trajectory_controller/follow_joint_trajectory",
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
                "controller_action": LaunchConfiguration("controller_action"),
            }
        ],
    )
    return LaunchDescription(args + [node])
