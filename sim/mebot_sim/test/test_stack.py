"""What every process of the sim stack shares (mebot_sim/stack.py).

No ROS graph, no GPU. /mebot_sim serves the real arm driver's names, so the
domain rule is the stack's safety: none of its processes may join the
bench's ROS domain (0, which an unset ROS_DOMAIN_ID also means). The import
guard keeps the planners on this checkout's rammp_curobo, though this host
has an editable install of another checkout.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mebot_sim import stack

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "sim" / "mebot_sim"


@pytest.mark.parametrize("value", [None, "", "0", " 0", "00", "0x0"])
def test_a_process_never_joins_the_bench_domain(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", value)
    stack.isolate()
    assert os.environ["ROS_DOMAIN_ID"] == "88"
    # and it says so
    assert "ROS_DOMAIN_ID" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["17", "89"])
def test_another_domain_is_kept(monkeypatch, capsys, value):
    monkeypatch.setenv("ROS_DOMAIN_ID", value)
    stack.isolate()
    assert os.environ["ROS_DOMAIN_ID"] == value
    assert capsys.readouterr().err == ""


def test_the_checkout_is_the_one_holding_the_package():
    assert stack.checkout() == REPO


def _fake_package(root, name):
    """A package `name` under `root` that is not this checkout's."""
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text("")
    return root / name / "__init__.py"


def _python(code, pythonpath):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in pythonpath))
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_core_comes_from_this_checkout_whatever_is_first_on_the_path(tmp_path):
    _fake_package(tmp_path / "core", "rammp_curobo")
    run = _python(
        "from mebot_sim import stack; print(stack.core().__file__)",
        [tmp_path / "core", PACKAGE],
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == str(REPO / "core" / "rammp_curobo" / "__init__.py")


def test_a_core_imported_from_elsewhere_is_refused(tmp_path):
    fake = _fake_package(tmp_path / "core", "rammp_curobo")
    run = _python(
        "import rammp_curobo; from mebot_sim import stack; stack.core()",
        [tmp_path / "core", PACKAGE],
    )
    assert run.returncode != 0
    assert "RuntimeError" in run.stderr
    # both core/ directories
    assert str(fake.parents[1]) in run.stderr and str(REPO / "core") in run.stderr


def test_the_planners_get_this_checkouts_core_first():
    env = stack.planner_environment({"PYTHONPATH": "/opt/ros/humble/lib"})
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(REPO / "core"),
        "/opt/ros/humble/lib",
    ]
    assert stack.planner_environment({})["PYTHONPATH"] == str(REPO / "core")


def test_planners_that_would_import_another_checkout_are_refused(tmp_path):
    ours = os.pathsep.join([str(REPO / "core"), str(REPO / "rammp_curobo_ros")])
    stack.check_planner_imports({"PYTHONPATH": ours})

    fake = _fake_package(tmp_path / "core", "rammp_curobo")
    shadowed = os.pathsep.join([str(tmp_path / "core"), ours])
    with pytest.raises(RuntimeError) as exc:
        stack.check_planner_imports({"PYTHONPATH": shadowed})
    assert "rammp_curobo" in str(exc.value) and str(fake) in str(exc.value)

    fake = _fake_package(tmp_path / "ros", "rammp_curobo_ros")
    shadowed = os.pathsep.join([str(REPO / "core"), str(tmp_path / "ros")])
    with pytest.raises(RuntimeError) as exc:
        stack.check_planner_imports({"PYTHONPATH": shadowed})
    assert "rammp_curobo_ros" in str(exc.value) and str(fake) in str(exc.value)


def test_each_world_holds_the_layers_it_is_said_to(tmp_path, monkeypatch):
    stack.core()
    from rammp_curobo.config import load_planner_config, resolve_config
    from rammp_curobo.scene import load_layers, load_scene
    from rammp_curobo.world import compose_cuboids

    # layer files in the current directory must change nothing
    monkeypatch.chdir(tmp_path)
    (tmp_path / "layer_mebot_platform.yaml").write_text("not: a layer\n")
    configs = {stack.CHAIR: "gen3_mebot.yaml", stack.CONTROL: "gen3.yaml"}
    for planner, world, sees in stack.worlds().values():
        cfg, where = load_planner_config(configs[planner])
        layers = load_layers(cfg["layers"], relative_to=where, listed_in="test")
        scene = load_scene(resolve_config(world, relative_to=where))
        _, sources = compose_cuboids(scene, layers=layers)
        assert sorted(set(sources.values())) == sorted(sees.split(", ")), world
        assert "scene" not in sources.values()


def test_trajectory_rows_follow_the_joint_names():
    msgs = pytest.importorskip("trajectory_msgs.msg")
    traj = msgs.JointTrajectory(joint_names=list(reversed(stack.JOINTS)))
    for k in range(2):
        point = msgs.JointTrajectoryPoint(positions=[k + j / 10 for j in range(7)])
        point.time_from_start.sec, point.time_from_start.nanosec = k, 500000000
        traj.points.append(point)
    times, rows = stack.trajectory_arrays(traj)
    assert times == [0.5, 1.5]
    # columns joint_1..joint_7: the message listed them backwards
    assert rows[1] == pytest.approx([1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])

    traj.joint_names[0] = "joint_9"
    with pytest.raises(ValueError, match="joint_1"):
        stack.trajectory_arrays(traj)
