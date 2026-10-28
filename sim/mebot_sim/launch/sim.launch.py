"""The MEBot sim stack: the sim arm beside this checkout's planner node, run twice.

    ros2 launch mebot_sim sim.launch.py
    ros2 launch mebot_sim sim.launch.py posture:=none host:=127.0.0.1,<address>

    /mebot_sim             the MuJoCo arm with the MEBot, the seated stand-in
                           and the floor, serving the arm driver's names, and
                           the live view (http://127.0.0.1:8765); it latches
                           the arm's model on /arm/robot_description, which
                           the planner nodes wait for
    /rammp_curobo          config:=<checkout>/core/tests/data/mebot/planner.yaml:
                           what a consumer on the chair gets. By path, not by
                           name: the chair's files are one robot's geometry and
                           ship with that robot, not in the package (issue
                           #19), so the sim runs the checkout's copy
    /rammp_curobo_control  config:=gen3.yaml, which composes no layers: it
                           plans in whatever scene a test pushes it, so the
                           same request can be planned without a layer

/mebot_sim serves the driver's names, so the stack never runs on the bench's
ROS domain, 0: when ROS_DOMAIN_ID is anything but a domain from 1 to 232
(stack.isolate), the launch itself and everything launched here run on 88
instead, and the launch says so. Such a domain is kept. The stack runs
whole or not at all: when any of its nodes ends, /mebot_sim refusing to
start beside another driver included, the launch ends the rest.
The planner nodes run this checkout's rammp_curobo, its core/ first on their
PYTHONPATH (this host has an editable install of another checkout); the
launch refuses to start them if they would import any other.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mebot_sim import scene, stack


def planners():
    """The two planner nodes and the `config` each is started with.

    The chair's config is not packaged (issue #19): it is one robot's geometry,
    so it ships with the robot and the sim runs this checkout's copy, by path.
    The control planner's gen3.yaml is module config and still ships in the
    image, so it stays a name.
    """
    return (
        ("rammp_curobo", str(stack.chair_config())),
        ("rammp_curobo_control", stack.CONTROL_CONFIG),
    )


def generate_launch_description():
    # The launch's own process first: a launch_ros action such as a RosTimer
    # starts a ROS node in it, and a process launched without a domain of its
    # own inherits its environment.
    stack.isolate()
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
    # Each node is given its domain and middleware itself as well: a process
    # starts with the launch's environment as it stands when the launch
    # reaches it, so an action that changed that would miss the nodes listed
    # before it.
    ros = {name: os.environ[name] for name in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION")}

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
    for name, config in planners():
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
