"""sim.launch.py and suite.launch.py, loaded but not launched: the domain the
launch and every process it starts run on, what it starts, and that the
stack ends whole. The node names and their parameters resolve only at
launch, so those are left to running it. How the suite's launch ends is run,
with stand-ins for the stack and the suite.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("launch_ros")
from launch import LaunchContext, LaunchDescription, LaunchService  # noqa: E402
from launch.actions import (  # noqa: E402
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    Shutdown,
)
from launch.events.process import ProcessExited  # noqa: E402
from launch.utilities import perform_substitutions  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LAUNCH = REPO / "sim" / "mebot_sim" / "launch"


def _module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), LAUNCH / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _description(monkeypatch, domain, launch="sim.launch.py"):
    if domain is None:
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", domain)
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    # the planners' own package from this checkout, as .sim_ws provides it
    path = os.pathsep.join(
        [str(REPO / "rammp_curobo_ros"), os.environ.get("PYTHONPATH", "")]
    )
    monkeypatch.setenv("PYTHONPATH", path)
    return _module(launch).generate_launch_description().entities


def _processes(entities):
    """Each process the launch starts, with the environment it adds to the
    launch's own. A process gets the launch's environment as it stands when
    the launch reaches it, so only what is added here reaches every process
    whatever the order of the actions."""
    context = LaunchContext()
    return [
        (
            action,
            {
                perform_substitutions(context, key): perform_substitutions(
                    context, value
                )
                for key, value in action.additional_env or ()
            },
        )
        for action in entities
        if isinstance(action, ExecuteProcess)
    ]


@pytest.mark.parametrize("domain", [None, "0", "-1"])
def test_the_bench_domain_is_left_for_88_and_said(monkeypatch, capsys, domain):
    entities = _description(monkeypatch, domain)
    # the launch's own process too: a launch_ros action such as a RosTimer
    # starts a ROS node in it, and a process launched without a domain of
    # its own inherits the launch's
    assert os.environ["ROS_DOMAIN_ID"] == "88"
    processes = _processes(entities)
    assert len(processes) == 3
    for _process, env in processes:
        assert env["ROS_DOMAIN_ID"] == "88"
        assert env["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert "ROS_DOMAIN_ID=88" in capsys.readouterr().err


def test_another_domain_is_kept(monkeypatch, capsys):
    entities = _description(monkeypatch, "89")
    assert os.environ["ROS_DOMAIN_ID"] == "89"
    for _process, env in _processes(entities):
        assert env["ROS_DOMAIN_ID"] == "89"
    assert "ROS_DOMAIN_ID" not in capsys.readouterr().err


def test_the_sim_arm_and_this_checkouts_planner_twice(monkeypatch):
    processes = _processes(_description(monkeypatch, "89"))
    assert all(isinstance(process, Node) for process, _env in processes)
    assert [(p.node_package, p.node_executable) for p, _env in processes] == [
        ("mebot_sim", "sim_arm"),
        ("rammp_curobo_ros", "planner_node"),
        ("rammp_curobo_ros", "planner_node"),
    ]
    for _planner, env in processes[1:]:
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "core")


def test_the_stack_ends_when_any_of_its_nodes_ends(monkeypatch):
    # /mebot_sim refusing to start beside another driver included: the
    # planners must not stay up beside whatever serves the driver's names
    for process, _env in _processes(_description(monkeypatch, "89")):
        assert any(isinstance(a, Shutdown) for a in process.get_sub_entities())


def test_the_suite_is_the_stack_and_run_suite_on_its_domain(monkeypatch, capsys):
    entities = _description(monkeypatch, None, "suite.launch.py")
    assert os.environ["ROS_DOMAIN_ID"] == "88"
    assert "ROS_DOMAIN_ID=88" in capsys.readouterr().err
    (include,) = [e for e in entities if isinstance(e, IncludeLaunchDescription)]
    context = LaunchContext()
    source = include.launch_description_source
    source.get_launch_description(context)
    assert source.location == str(LAUNCH / "sim.launch.py")
    # run_suite, with the stack's domain and middleware, as sim.launch.py
    # gives them to its nodes
    ((suite, env),) = _processes(entities)
    assert perform_substitutions(context, suite.cmd[0][0].executable) == "run_suite"
    assert env == {"ROS_DOMAIN_ID": "88", "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
    # and the launch ends when it does
    (handler,) = [e for e in entities if isinstance(e, RegisterEventHandler)]
    ended = ProcessExited(
        action=suite, name="run_suite", cmd=[], cwd=None, env=None, pid=1, returncode=0
    )
    assert handler.event_handler.matches(ended)


def _exits(stack, suite):
    """The exit status of a launch of `stack` and `suite`, stand-ins for the
    stack and run_suite, that ends as suite.launch.py ends."""
    ends_with = _module("suite.launch.py").ends_with
    service = LaunchService(argv=[])
    service.include_launch_description(
        LaunchDescription([stack, ends_with(suite), suite])
    )
    return service.run()


def _python(code, **options):
    return ExecuteProcess(cmd=[sys.executable, "-c", code], **options)


@pytest.mark.parametrize("status", [0, 1, 3])
def test_the_suites_launch_exits_with_run_suites_verdict(status):
    # a launch can exit 0 or 1 only: 1 for any failure
    stack = _python("import time; time.sleep(60)")
    suite = _python("import sys; sys.exit(%d)" % status)
    assert _exits(stack, suite) == (1 if status else 0)


def test_a_stack_that_ends_before_run_suite_fails_the_suite():
    # sim.launch.py ends the stack whole when a node ends, run_suite with it
    stack = _python("pass", on_exit=[Shutdown(reason="a node ended")])
    suite = _python("import time; time.sleep(60)")
    assert _exits(stack, suite) == 1
