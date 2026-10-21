"""The MEBot sim suite: the stack of sim.launch.py, and run_suite beside it.

    ros2 launch mebot_sim suite.launch.py            # what `make sim` runs
    ros2 launch mebot_sim suite.launch.py port:=8766

run_suite waits for the stack, runs S0 to S5 and ends; the launch then ends
the stack and exits with run_suite's verdict: 0 if it passed, 1 otherwise (a
launch exits 0 or 1, nothing else). Should any node of the stack end first,
the stack ends whole and run_suite with it, and that fails too. The
arguments are sim.launch.py's; run_suite's S0 is given the same menagerie
and humanoid as the sim.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import ExecutableInPackage

from mebot_sim import stack


def _verdict(event, _context):
    # A launch exits 1 when an action raises, and 0 otherwise.
    if event.returncode:
        raise RuntimeError(
            "run_suite failed (exit status %d): its report is above" % event.returncode
        )
    return [EmitEvent(event=Shutdown(reason="run_suite passed"))]


def ends_with(suite):
    """The launch ends when the process `suite` ends, with its verdict."""
    return RegisterEventHandler(OnProcessExit(target_action=suite, on_exit=_verdict))


def generate_launch_description():
    stack.isolate()
    here = os.path.dirname(os.path.abspath(__file__))
    # run_suite is no node of the launch's: a Node would hand it ROS arguments
    suite = ExecuteProcess(
        cmd=[
            ExecutableInPackage("run_suite", "mebot_sim"),
            "--menagerie",
            LaunchConfiguration("menagerie"),
            "--humanoid",
            LaunchConfiguration("humanoid"),
        ],
        name="run_suite",
        output="screen",
        additional_env={
            name: os.environ[name] for name in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION")
        },
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(here, "sim.launch.py"))
            ),
            ends_with(suite),
            suite,
        ]
    )
