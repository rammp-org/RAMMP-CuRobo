"""The MEBot sim stack: the sim arm beside this checkout's planner node, run twice.

    ros2 launch mebot_sim sim.launch.py
    ros2 launch mebot_sim sim.launch.py posture:=none host:=127.0.0.1,<address>

    /mebot_sim             the MuJoCo arm with the MEBot, the seated stand-in
                           and the floor, serving the arm driver's names, and
                           the live view (http://127.0.0.1:8765)
    /rammp_curobo          config:=gen3_mebot.yaml: what a consumer on the
                           chair gets
    /rammp_curobo_control  config:=gen3.yaml, which composes no layers: it
                           plans in whatever scene a test pushes it, so the
                           same request can be planned without a layer

/mebot_sim serves the driver's names, so the stack never runs on the bench's
ROS domain: when ROS_DOMAIN_ID is 0 or unset, everything launched here runs
on 88 instead, and the launch says so. Another non-zero domain is kept. The
stack runs whole or not at all: when any of its nodes ends, /mebot_sim
refusing to start beside another driver included, the launch ends the rest.
The planner nodes run this checkout's rammp_curobo, its core/ first on their
PYTHONPATH (this host has an editable install of another checkout); the
launch refuses to start them if they would import any other.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mebot_sim import scene, stack

PLANNERS = (("rammp_curobo", "gen3_mebot.yaml"), ("rammp_curobo_control", "gen3.yaml"))


def generate_launch_description():
    actions = [
        DeclareLaunchArgument(
            "posture",
            default_value="upright_lap",
            description="the stand-in's posture: upright_lap, lean_lap, "
            "forearms_forward, or none for an empty chair",
        ),
        DeclareLaunchArgument(
            "port", default_value="8765", description="the live view's port"
        ),
        DeclareLaunchArgument(
            "host",
            default_value="127.0.0.1",
            description="the live view's addresses, comma-separated",
        ),
        DeclareLaunchArgument(
            "menagerie",
            default_value=scene.MENAGERIE,
            description="a mujoco_menagerie checkout: kinova_gen3 and robotiq_2f85",
        ),
        DeclareLaunchArgument(
            "humanoid",
            default_value=scene.HUMANOID,
            description="the DeepMind humanoid the stand-in is posed from",
        ),
    ]
    if stack.on_bench_domain(os.environ):
        domain = stack.SIM_DOMAIN
        actions.append(
            LogInfo(
                msg="ROS_DOMAIN_ID is 0 or unset, the bench's domain: the sim "
                "stack runs on ROS_DOMAIN_ID=%s instead" % domain
            )
        )
    else:
        domain = os.environ["ROS_DOMAIN_ID"]
    # Each node is given its domain and middleware itself: a process starts
    # with the launch's environment as it stands when the launch reaches it,
    # so an action that changed that would miss the nodes listed before it.
    ros = {
        "ROS_DOMAIN_ID": domain,
        "RMW_IMPLEMENTATION": os.environ.get(
            "RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"
        ),
    }

    environment = stack.planner_environment(os.environ)
    stack.check_planner_imports(environment)
    actions.append(
        Node(
            package="mebot_sim",
            executable="sim_arm",
            name="mebot_sim",
            output="screen",
            parameters=[
                {
                    name: LaunchConfiguration(name)
                    for name in ("posture", "port", "host", "menagerie", "humanoid")
                }
            ],
            additional_env=ros,
            on_exit=[Shutdown(reason="/mebot_sim ended, and the sim stack with it")],
        )
    )
    for name, config in PLANNERS:
        actions.append(
            Node(
                package="rammp_curobo_ros",
                executable="planner_node",
                name=name,
                output="screen",
                emulate_tty=True,
                parameters=[{"config": config}],
                additional_env=dict(ros, **environment),
                on_exit=[
                    Shutdown(reason="/%s ended, and the sim stack with it" % name)
                ],
            )
        )
    return LaunchDescription(actions)
