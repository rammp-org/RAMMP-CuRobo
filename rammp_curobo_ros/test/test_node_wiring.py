"""The node's wiring — ROS runtime, no GPU.

The node is built for real with the planner and the model wait stubbed, so
CI pins what the GPU suite never sees: the model goes through prepare_urdf
to the planner as a file, the `world` parameter reaches the config, a
refused model exits 1 before any planner exists, and main() lets that exit
code out of the process.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

import rammp_curobo  # noqa: E402

from rammp_curobo_ros import planner_node  # noqa: E402

# from the checkout: in the image, rammp_curobo is installed in site-packages
FIXTURE = (
    Path(__file__).resolve().parents[2] / "core" / "tests" / "data" / "arm_level.urdf"
)


class _FakePlanner:
    made = []

    def __init__(self, cfg, cfg_dir, *, urdf_path=None):
        _FakePlanner.made.append((cfg, cfg_dir, urdf_path))
        self.base_frame = "arm_mount"
        self.ee_link = "arm_tool_frame"
        self.joint_names = list(cfg.get("joint_names", []))
        self.world_name = cfg["world"]

    def world_summary(self):
        return "0 boxes"


@pytest.fixture
def named_config(tmp_path_factory):
    """A named config (no config at all would be example mode)."""
    d = tmp_path_factory.mktemp("named")
    (d / "planner.yaml").write_text(
        "robot: example:robot_gen3_2f85.yaml\nworld: example:world_sim_kitchen.yaml\n"
    )
    return "config:=%s" % (d / "planner.yaml")


@pytest.fixture
def make_node(monkeypatch, domain_id, named_config):
    monkeypatch.setattr(rammp_curobo, "CuRoboPlanner", _FakePlanner)
    _FakePlanner.made.clear()
    nodes = []

    def make(model, *params):
        monkeypatch.setattr(
            planner_node.RammpCuroboNode, "_wait_for_model", lambda self, t: model
        )
        if not any(p.startswith("config:=") for p in params):
            params = (named_config,) + params
        args = ["--ros-args"] + [a for p in params for a in ("-p", p)]
        rclpy.init(args=args, domain_id=domain_id)
        nodes.append(planner_node.RammpCuroboNode())
        return nodes[-1]

    yield make
    for n in nodes:
        n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_bad_model_exits_1_before_any_planner(make_node):
    with pytest.raises(SystemExit) as exc:
        make_node("<robot name='x'><link name='base_link'/></robot>")
    assert exc.value.code == 1 and not _FakePlanner.made


def test_the_sanitised_model_and_the_world_param_reach_the_planner(make_node):
    node = make_node(FIXTURE.read_text(), "world:=example:world_empty.yaml")
    ((cfg, _cfg_dir, urdf_path),) = _FakePlanner.made
    assert cfg["world"] == "example:world_empty.yaml"
    text = Path(urdf_path).read_text()
    assert "<robot" in text
    assert "<visual" not in text and "<ros2_control" not in text
    # and it keeps watching the topic for a different model
    assert node.count_subscribers(planner_node.ROBOT_DESCRIPTION_TOPIC) == 1


def test_a_missing_robot_file_is_a_fatal_exit_1_naming_it(make_node, tmp_path, capfd):
    (tmp_path / "planner.yaml").write_text("robot: nope.yaml\nworld: world.yaml\n")
    with pytest.raises(SystemExit) as exc:
        make_node(FIXTURE.read_text(), "config:=%s" % (tmp_path / "planner.yaml"))
    assert exc.value.code == 1 and not _FakePlanner.made
    err = capfd.readouterr().err
    assert "FATAL" in err and "nope.yaml" in err


def test_a_malformed_planner_yaml_is_a_fatal_exit_1(make_node, tmp_path, capfd):
    (tmp_path / "planner.yaml").write_text("robot: [unclosed\n")
    with pytest.raises(SystemExit) as exc:
        make_node(FIXTURE.read_text(), "config:=%s" % (tmp_path / "planner.yaml"))
    assert exc.value.code == 1 and not _FakePlanner.made
    err = capfd.readouterr().err
    assert "FATAL" in err and "planner.yaml" in err


@pytest.mark.parametrize("value", ["0.0", "-1", "abc", ".nan"])
def test_a_bad_timeout_exits_1_before_the_wait(make_node, value):
    with pytest.raises(SystemExit) as exc:
        make_node(FIXTURE.read_text(), "robot_description_timeout_s:=" + value)
    assert exc.value.code == 1 and not _FakePlanner.made


def test_the_models_temp_dir_goes_with_the_node(make_node):
    node = make_node(FIXTURE.read_text())
    ((_cfg, _cfg_dir, urdf_path),) = _FakePlanner.made
    assert Path(urdf_path).is_file()
    node.destroy_node()
    assert not Path(urdf_path).parent.exists()


def test_main_exits_1_when_no_model_arrives(domain_id, named_config):
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain_id))
    t0 = time.monotonic()
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rammp_curobo_ros.planner_node import main; main()",
            "--ros-args",
            "-p",
            named_config,
            "-p",
            "robot_description_timeout_s:=1",  # an INT on purpose
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 1, r.stderr[-2000:]
    assert "arm_state_publisher" in r.stderr + r.stdout
    # 1 s wait + 1 s peer discovery + start-up; the 30 s default would fail
    assert time.monotonic() - t0 < 15
