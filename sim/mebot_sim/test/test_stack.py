"""What every process of the sim stack shares (mebot_sim/stack.py).

No ROS graph, no GPU. /mebot_sim serves the real arm driver's names, so the
domain rule is the stack's safety: none of its processes may join the
bench's ROS domain (0, which an unset ROS_DOMAIN_ID also means). The import
guard keeps the planners on this checkout's rammp_curobo, though this host
has an editable install of another checkout.
"""

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mebot_sim import stack

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "sim" / "mebot_sim"


# rcl reads "00", "0x0" and " 0" as 0, and "-1" and 2**64 - 1 as its default
# domain, which Cyclone DDS makes 0; "017" is octal to it, and Cyclone has
# no domain above 232
@pytest.mark.parametrize(
    "value",
    [None, "", "0", " 0", "00", "0x0", "-1", "18446744073709551615", "017", "233"],
)
def test_a_process_never_joins_the_bench_domain(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", value)
    stack.isolate()
    assert os.environ["ROS_DOMAIN_ID"] == "88"
    # and it says so
    assert "ROS_DOMAIN_ID" in capsys.readouterr().err


class Started(Exception):
    """rclpy.init was called."""


@pytest.mark.parametrize("entry", ["sim_arm", "try_goal"])
def test_a_process_started_by_hand_leaves_the_bench_domain_before_ros(
    monkeypatch, entry
):
    # `ros2 run mebot_sim sim_arm` and `ros2 run mebot_sim try_goal` (a
    # client.Tester) start outside the launch, with the shell's domain
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("mujoco")  # the sim's
    pytest.importorskip("rammp_arm_interfaces.action")  # the Tester's
    from mebot_sim import client, sim_arm

    seen = []

    def init(*args, **options):
        seen.append(os.environ.get("ROS_DOMAIN_ID"))
        raise Started()

    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr(rclpy, "init", init)
    with pytest.raises(Started):
        {"sim_arm": sim_arm.main, "try_goal": client.Tester}[entry]()
    assert seen == ["88"]


@pytest.mark.parametrize("value", ["1", "17", "89", "232"])
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
    # the chair's config is not packaged (issue #19): it is named by path
    configs = {stack.CHAIR: stack.chair_config(), stack.CONTROL: stack.CONTROL_CONFIG}
    for planner, world, sees in stack.worlds().values():
        cfg, where = load_planner_config(configs[planner])
        layers = load_layers(cfg["layers"], relative_to=where, listed_in="test")
        scene = load_scene(resolve_config(world, relative_to=where))
        _, sources = compose_cuboids(scene, layers=layers)
        assert sorted(set(sources.values())) == sorted(sees.split(", ")), world
        assert "scene" not in sources.values()


def test_trajectory_rows_are_read_as_the_driver_reads_them():
    msgs = pytest.importorskip("trajectory_msgs.msg")
    traj = msgs.JointTrajectory(joint_names=list(stack.JOINTS))
    for k in range(2):
        point = msgs.JointTrajectoryPoint(positions=[k + j / 10 for j in range(7)])
        point.time_from_start.sec, point.time_from_start.nanosec = k, 500000000
        traj.points.append(point)
    times, rows = stack.trajectory_arrays(traj)
    assert times == [0.5, 1.5]
    assert rows[1] == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

    # The driver takes each point's positions as arm_joint_1..arm_joint_7 and
    # never reads the names: read by name, these would fly here as the arm
    # would not fly them. Menagerie's names are the MuJoCo model's, not the
    # driver's.
    for names in (
        list(reversed(stack.JOINTS)),
        list(stack.MUJOCO_JOINTS),
        stack.JOINTS[:6] + ["joint_9"],
        stack.JOINTS + ["joint_8"],
        [],
    ):
        traj.joint_names = names
        with pytest.raises(ValueError, match="in order"):
            stack.trajectory_arrays(traj)


def test_the_requests_are_tied_to_the_generated_layer_files(monkeypatch):
    # and not to what a planner config lists: a chair config that drops a
    # layer is a regression for the suite's gates to catch, not a reason to
    # find the requests again
    outputs = stack.generator().load_data()["outputs"].values()
    expected = {
        Path(path).name: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        for path in outputs
    }
    assert len(expected) == 3
    monkeypatch.setattr(stack, "CHAIR_CONFIG", "gen3.yaml")
    assert stack.layer_digests() == expected


def test_the_arm_model_is_the_one_the_planner_takes_with_the_drivers_names():
    # what /mebot_sim latches on /arm/robot_description and find_pairs plans
    # with: the planner node refuses any other model before cuRobo sees it
    stack.core()
    import xml.etree.ElementTree as ET

    from rammp_curobo.urdf_model import ROOT_LINK, prepare_urdf

    root = ET.fromstring(prepare_urdf(stack.arm_model()))
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert all(name in joints for name in stack.JOINTS)
    # the arm is tilted on arm_mount, the layers' frame, by the tilt the
    # layers were generated for and the MuJoCo arm is turned by
    mount = joints["arm_base_joint"]
    assert mount.find("parent").get("link") == ROOT_LINK
    origin = mount.find("origin")
    tilt = stack.generator().load_data()["mount"]["arm_base_joint"]["rpy_deg"]
    rpy = [float(v) for v in origin.get("rpy").split()]
    assert rpy == pytest.approx([math.radians(v) for v in tilt], abs=1e-12)
    assert [float(v) for v in origin.get("xyz").split()] == [0.0] * 3
    # the driver publishes the knuckle under the model's name for it
    from mebot_sim import sim_arm

    assert sim_arm.KNUCKLE in joints
