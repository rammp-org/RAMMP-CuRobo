"""Helpers the core tests share. A plain module, not conftest: `import conftest`
names whichever conftest.py pytest imported last (the ROS and sim suites have
their own)."""

from rammp_curobo.config import resolve_config
from rammp_curobo.robot_config import load_robot_spec
from rammp_curobo.scene import load_scene as _load_scene
from rammp_curobo.scene import scene_from_obstacles as _scene_from_obstacles

FRAME = "arm_mount"  # the example Gen3's planning frame


def gen3_spec():
    """The example Gen3 robot file's names: what prepare_urdf checks a model against."""
    return load_robot_spec(resolve_config("example:robot_gen3_2f85.yaml"))


def load_scene(path):
    return _load_scene(path, FRAME)


def scene_from_obstacles(entries):
    return _scene_from_obstacles(entries, FRAME)
