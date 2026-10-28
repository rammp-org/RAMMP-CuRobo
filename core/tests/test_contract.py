"""check_robot_dir: every problem in a robot directory at once."""

import shutil
from pathlib import Path

import pytest
import yaml

from rammp_curobo.config import EXAMPLE_DIR, load_planner_config
from rammp_curobo.contract import RobotDataError, check_robot_dir

DATA = Path(__file__).parent / "data"
URDF = (DATA / "arm_level.urdf").read_text()


def _dir(tmp_path, planner="robot: robot.yaml\nworld: world.yaml\n"):
    (tmp_path / "planner.yaml").write_text(planner)
    shutil.copy(EXAMPLE_DIR / "robot_gen3_2f85.yaml", tmp_path / "robot.yaml")
    shutil.copy(EXAMPLE_DIR / "arm_kinematics.json", tmp_path / "arm_kinematics.json")
    shutil.copy(EXAMPLE_DIR / "world_empty.yaml", tmp_path / "world.yaml")
    return tmp_path


def _check(tmp_path):
    cfg, d = load_planner_config(tmp_path / "planner.yaml")
    return check_robot_dir(cfg, d, URDF)


def test_every_problem_in_one_error(tmp_path):
    (tmp_path / "planner.yaml").write_text(
        "robot: robot.yaml\nworld: world.yaml\nlayers: [missing_layer.yaml]\n"
    )
    shutil.copy(EXAMPLE_DIR / "robot_gen3_2f85.yaml", tmp_path / "robot.yaml")
    (tmp_path / "world.yaml").write_text(
        "base_frame: base\nobstacles: [{name: b, position: [0,0,0], dims: [1,1,1]}]\n"
    )
    with pytest.raises(RobotDataError) as e:
        _check(tmp_path)
    msg = str(e.value)
    assert len(e.value.problems) == 3, msg
    assert "arm_kinematics.json" in msg
    assert "missing_layer.yaml" in msg and "'base'" in msg
    assert msg.startswith("robot data in %s: 3 problems:\n  - " % tmp_path)


def test_a_robot_file_without_model_check_is_named(tmp_path):
    d = _dir(tmp_path)
    data = yaml.safe_load((d / "robot.yaml").read_text())
    del data["model_check"]
    (d / "robot.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(RobotDataError) as e:
        _check(d)
    assert len(e.value.problems) == 1 and "model_check" in str(e.value)


def test_a_missing_robot_file_is_one_problem(tmp_path):
    d = _dir(tmp_path)
    (d / "robot.yaml").unlink()
    with pytest.raises(RobotDataError) as e:
        _check(d)
    assert len(e.value.problems) == 1 and "robot.yaml" in str(e.value)


def test_a_model_the_robot_file_does_not_describe_is_listed_with_a_bad_world(tmp_path):
    d = _dir(tmp_path)
    (d / "world.yaml").write_text("obstacles: []\n")
    cfg, cdir = load_planner_config(d / "planner.yaml")
    with pytest.raises(RobotDataError) as e:
        check_robot_dir(cfg, cdir, "<robot name='x'><link name='base_link'/></robot>")
    assert len(e.value.problems) == 2
    assert "base_frame" in str(e.value) and "root link" in str(e.value)


def test_a_layer_composed_with_less_padding_than_it_states_is_refused(tmp_path):
    d = _dir(tmp_path, "robot: robot.yaml\nworld: world.yaml\nlayers: [l.yaml]\n")
    (d / "l.yaml").write_text(
        "layer: chair\nexcludable: false\nmeta: {padding_assumed_m: 0.05}\n"
        "obstacles: [{name: a, position: [0.5,0,0.5], dims: [0.1,0.1,0.1]}]\n"
    )
    with pytest.raises(RobotDataError) as e:
        _check(d)
    assert len(e.value.problems) == 1 and "padding" in str(e.value)


def test_the_example_passes():
    cfg, d = load_planner_config("example:gen3.yaml")
    assert "<ros2_control" not in check_robot_dir(cfg, d, URDF)


def test_the_mebot_chair_fixture_passes_with_the_tilted_urdf():
    cfg, d = load_planner_config(DATA / "mebot" / "planner.yaml")
    assert "<ros2_control" not in check_robot_dir(
        cfg, d, (DATA / "arm_tilt30.urdf").read_text()
    )


def test_a_cache_overflow_is_listed(tmp_path):
    d = _dir(tmp_path, "robot: robot.yaml\nworld: world.yaml\nplanner: {collision_cache_obb: 1}\n")
    (d / "world.yaml").write_text(
        "base_frame: arm_mount\nobstacles:\n"
        "  - {name: a, position: [1,0,0], dims: [.1,.1,.1]}\n"
        "  - {name: b, position: [2,0,0], dims: [.1,.1,.1]}\n"
    )
    with pytest.raises(RobotDataError) as e:
        _check(d)
    assert len(e.value.problems) == 1 and "collision_cache_obb=1" in str(e.value)


def test_layers_are_still_composed_when_the_world_fails(tmp_path):
    d = _dir(tmp_path, "robot: robot.yaml\nworld: world.yaml\nlayers: [l.yaml]\n")
    (d / "world.yaml").write_text("obstacles: []\n")  # no base_frame
    (d / "l.yaml").write_text(
        "layer: chair\nexcludable: false\nmeta: {padding_assumed_m: 0.05}\n"
        "obstacles: [{name: a, position: [0.5,0,0.5], dims: [0.1,0.1,0.1]}]\n"
    )
    with pytest.raises(RobotDataError) as e:
        _check(d)
    assert len(e.value.problems) == 2
    assert "base_frame" in str(e.value) and "padding" in str(e.value)


def test_a_config_dict_without_planner_is_a_listed_problem(tmp_path):
    d = _dir(tmp_path)
    cfg, cdir = load_planner_config(d / "planner.yaml")
    del cfg["planner"]
    with pytest.raises(RobotDataError) as e:
        check_robot_dir(cfg, cdir, URDF)
    assert e.value.problems == ["planner config has no 'planner' entry"]


@pytest.mark.parametrize("text", ["robot: [unclosed\n", "- a\n- b\n"])
def test_a_planner_config_that_is_not_a_mapping_or_yaml_is_a_value_error(tmp_path, text):
    (tmp_path / "planner.yaml").write_text(text)
    with pytest.raises(ValueError, match="planner.yaml"):
        load_planner_config(tmp_path / "planner.yaml")
