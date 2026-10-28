"""Example mode — no `config` runs the image's example robot, and says so.

ROS runtime, no GPU: the planner is stubbed as in test_node_wiring.py.
"""

import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

from std_msgs.msg import String  # noqa: E402

import rammp_curobo  # noqa: E402
from rammp_curobo_interfaces.srv import SetWorld  # noqa: E402

from rammp_curobo_ros import planner_node  # noqa: E402
from rammp_curobo_ros.planner_node import EXAMPLE_MARKER  # noqa: E402

BANNER = "=== NO CONFIG GIVEN: running the EXAMPLE robot from the image"
IGNORING = "this planner is IGNORING it"


FIXTURE = (
    Path(__file__).resolve().parents[2] / "core" / "tests" / "data" / "arm_level.urdf"
)
waits = []


class _FakePlanner:
    made = []

    def __init__(self, cfg, cfg_dir, *, urdf_path=None):
        _FakePlanner.made.append((cfg, cfg_dir, urdf_path))
        self.base_frame = "arm_mount"
        self.ee_link = "arm_tool_frame"
        self.joint_names = list(cfg.get("joint_names", []))
        self.world_name = cfg["world"]
        self._config_dir = cfg_dir

    def world_summary(self):
        return "0 boxes"

    def update_world(self, world):
        pass


@pytest.fixture
def make_node(monkeypatch, domain_id):
    monkeypatch.setattr(rammp_curobo, "CuRoboPlanner", _FakePlanner)
    _FakePlanner.made.clear()

    waits.clear()

    def wait(self, t):
        waits.append(t)
        return FIXTURE.read_text()

    monkeypatch.setattr(planner_node.RammpCuroboNode, "_wait_for_model", wait)
    nodes = []

    def make(*params):
        args = ["--ros-args"] + [a for p in params for a in ("-p", p)]
        rclpy.init(args=args, domain_id=domain_id)
        nodes.append(planner_node.RammpCuroboNode())
        return nodes[-1]

    yield make
    for n in nodes:
        n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _spin(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def test_no_config_builds_the_example_robot_with_the_banner_first(make_node, capfd):
    node = make_node()
    assert node.example_mode and not waits
    ((cfg, _dir, urdf_path),) = _FakePlanner.made
    assert (
        cfg["world"] == "world_sim_kitchen.yaml"
        and "<robot" in Path(urdf_path).read_text()
    )
    err = capfd.readouterr().err
    assert err.index(BANNER) < err.index("Loading cuRobo planner")
    assert "pass config:=/config/robot/curobo/planner.yaml. ===" in err
    assert "robot data: config " in err and "(example, image)" in err
    assert "model example" in err


def test_a_published_model_is_ignored_with_one_warn(make_node, capfd):
    node = make_node()
    pub = node.create_publisher(
        String, planner_node.ROBOT_DESCRIPTION_TOPIC, planner_node.MODEL_QOS
    )
    for _ in range(3):
        pub.publish(String(data="<robot name='other'/>"))
        _spin(node, 0.5)
    assert capfd.readouterr().err.count(IGNORING) == 1


def test_results_and_set_world_replies_carry_the_marker(make_node):
    node = make_node()
    assert node._tag("planned 5 points").endswith(" " + EXAMPLE_MARKER)
    resp = node._set_world_cb(
        SetWorld.Request(world="example:world_empty.yaml"), SetWorld.Response()
    )
    assert resp.success and resp.message.endswith(" " + EXAMPLE_MARKER)
    assert "(example, image)" in resp.message

    def refuse(world):
        raise ValueError("refused")

    node.planner.update_world = refuse
    bad = node._set_world_cb(
        SetWorld.Request(world="example:world_empty.yaml"), SetWorld.Response()
    )
    assert not bad.success and bad.message == "refused " + EXAMPLE_MARKER


def test_set_world_never_reads_the_cwd(make_node, tmp_path, monkeypatch):
    node = make_node()
    (tmp_path / "stray.yaml").write_text("base_frame: arm_mount\nobstacles: []\n")
    monkeypatch.chdir(tmp_path)
    bad = node._set_world_cb(SetWorld.Request(world="stray.yaml"), SetWorld.Response())
    assert not bad.success and "stray.yaml" in bad.message


def test_a_named_config_has_no_marker_and_no_banner(make_node, tmp_path, capfd):
    (tmp_path / "p.yaml").write_text(
        "robot: example:robot_gen3_2f85.yaml\nworld: example:world_empty.yaml\n"
    )
    node = make_node("config:=%s" % (tmp_path / "p.yaml"))
    assert not node.example_mode and len(waits) == 1
    assert node._tag("planned") == "planned"
    err = capfd.readouterr().err
    assert BANNER not in err and "model topic" in err
    assert "(example, image)" in err  # its robot file and world are the image's


def test_an_unresolvable_config_is_fatal_naming_it_and_never_example_mode(
    make_node, capfd
):
    with pytest.raises(SystemExit) as exc:
        make_node("config:=/nonexistent/planner.yaml")
    assert exc.value.code == 1 and not _FakePlanner.made
    err = capfd.readouterr().err
    assert "FATAL" in err and "/nonexistent/planner.yaml" in err
    assert BANNER not in err
