"""sim.launch.py, loaded but not launched: the domain rule it applies to
everything it starts, and what it starts. The node names and their
parameters resolve only at launch, so those are left to running it.
"""

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("launch_ros")
from launch import LaunchContext  # noqa: E402
from launch.actions import LogInfo, SetEnvironmentVariable  # noqa: E402
from launch.utilities import perform_substitutions  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _description(monkeypatch, domain):
    if domain is None:
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", domain)
    # the planners' own package from this checkout, as .sim_ws provides it
    path = os.pathsep.join(
        [str(REPO / "rammp_curobo_ros"), os.environ.get("PYTHONPATH", "")]
    )
    monkeypatch.setenv("PYTHONPATH", path)
    spec = importlib.util.spec_from_file_location(
        "sim_launch", REPO / "sim" / "mebot_sim" / "launch" / "sim.launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description().entities


def _environment(entities):
    context = LaunchContext()
    return {
        perform_substitutions(context, a.name): perform_substitutions(context, a.value)
        for a in entities
        if isinstance(a, SetEnvironmentVariable)
    }


@pytest.mark.parametrize("domain", [None, "0"])
def test_the_bench_domain_is_left_for_88_and_said(monkeypatch, domain):
    entities = _description(monkeypatch, domain)
    assert _environment(entities)["ROS_DOMAIN_ID"] == "88"
    context = LaunchContext()
    said = [
        perform_substitutions(context, a.msg)
        for a in entities
        if isinstance(a, LogInfo)
    ]
    assert any("ROS_DOMAIN_ID" in s and "88" in s for s in said)


def test_another_domain_is_kept(monkeypatch):
    assert "ROS_DOMAIN_ID" not in _environment(_description(monkeypatch, "89"))


def test_the_sim_arm_and_this_checkouts_planner_twice(monkeypatch):
    nodes = [a for a in _description(monkeypatch, "89") if isinstance(a, Node)]
    assert [(n.node_package, n.node_executable) for n in nodes] == [
        ("mebot_sim", "sim_arm"),
        ("rammp_curobo_ros", "planner_node"),
        ("rammp_curobo_ros", "planner_node"),
    ]
    for planner in nodes[1:]:
        path = planner.additional_env[0][1]
        first = perform_substitutions(LaunchContext(), path).split(os.pathsep)[0]
        assert first == str(REPO / "core")
