"""The startup check of a robot directory and the arm's model, as one.

A deployment's robot data (planner config, robot file, kinematics reference,
world, layers) and the URDF the arm publishes must agree. `check_robot_dir`
reads all of it and raises ONE `RobotDataError` listing every problem it
finds, so a bad deployment is fixed in one pass, not one restart per fault.

Pure Python: nothing here imports cuRobo, torch or ROS.
"""

import xml.etree.ElementTree as ET

import yaml

from rammp_curobo.config import resolve_config
from rammp_curobo.robot_config import load_robot_spec
from rammp_curobo.scene import Scene, load_layers, load_scene
from rammp_curobo.urdf_model import prepare_urdf
from rammp_curobo.world import compose_cuboids

# What a loader raises for a file that is missing, unreadable or wrong.
_LOAD_ERRORS = (OSError, ValueError, KeyError, TypeError, yaml.YAMLError)


class RobotDataError(ValueError):
    """Robot data that cannot be used; `.problems` lists every one found."""

    def __init__(self, config_dir, problems):
        self.problems = list(problems)
        super().__init__(
            "robot data in %s: %d problem%s:\n%s"
            % (
                config_dir,
                len(self.problems),
                "" if len(self.problems) == 1 else "s",
                "\n".join("  - " + p for p in self.problems),
            )
        )


def _root_link(robot_path):
    """The robot file's kinematics.base_link, or None when it cannot be read:
    the world is checked against it even when the rest of the file is bad."""
    try:
        with open(robot_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return str(data["robot_cfg"]["kinematics"]["base_link"])
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return None


def _check_robot_file_fields(cfg, robot_path, spec):
    """What CuRoboPlanner reads from the robot file beyond RobotSpec: the IK
    seed, the collision spheres, and the config's joint_names against the
    file's joints. Each failure here would otherwise raise at construction."""
    with open(robot_path, "r", encoding="utf-8") as f:
        kin = yaml.safe_load(f)["robot_cfg"]["kinematics"]
    missing = [
        k
        for k, v in (
            ("cspace.retract_config", kin["cspace"].get("retract_config")),
            ("collision_spheres", kin.get("collision_spheres")),
        )
        if not v
    ]
    if missing:
        raise ValueError(
            "robot file %s is missing kinematics.%s"
            % (robot_path, ", kinematics.".join(missing))
        )
    names = cfg.get("joint_names")
    if names is not None and sorted(names) != sorted(spec.joint_names):
        raise ValueError(
            "planner config joint_names %s are not the robot file's joints %s"
            % (list(names), list(spec.joint_names))
        )


def _check_collision_links(robot_path, urdf_xml):
    """The links the robot file puts collision spheres on, against the model's
    links: a link the model lacks would otherwise fail inside cuRobo."""
    with open(robot_path, "r", encoding="utf-8") as f:
        kin = yaml.safe_load(f)["robot_cfg"]["kinematics"]
    spheres = kin.get("collision_spheres")
    named = set(kin.get("collision_link_names") or [])
    named |= set(spheres) if isinstance(spheres, dict) else set()
    links = {link.get("name") for link in ET.fromstring(urdf_xml).findall("link")}
    missing = sorted(named - links)
    if missing:
        raise ValueError(
            "robot file %s puts collision spheres on links the robot model "
            "does not have: %s" % (robot_path, ", ".join(missing))
        )


def check_robot_dir(cfg, cfg_dir, urdf_xml):
    """Planner config `cfg` (from `cfg_dir`) and the model text `urdf_xml` ->
    the sanitised URDF text, or RobotDataError listing every problem."""
    problems = []

    def attempt(what):
        try:
            return what()
        except KeyError as exc:  # a config dict short of a key
            problems.append("planner config has no %s entry" % exc)
        except _LOAD_ERRORS as exc:
            problems.append(str(exc))
            return None

    robot_path = attempt(
        lambda: resolve_config(cfg["robot"], relative_to=cfg_dir, search_cwd=False)
    )
    spec = attempt(lambda: load_robot_spec(robot_path)) if robot_path else None
    if spec:
        attempt(lambda: _check_robot_file_fields(cfg, robot_path, spec))
    frame = spec.root_link if spec else (robot_path and _root_link(robot_path))

    world_path = attempt(
        lambda: resolve_config(cfg["world"], relative_to=cfg_dir, search_cwd=False)
    )
    scene = (
        attempt(lambda: load_scene(world_path, frame)) if world_path and frame else None
    )
    layers = attempt(
        lambda: load_layers(
            cfg.get("layers", []),
            relative_to=cfg_dir,
            listed_in="planner config in %s" % cfg_dir,
        )
    )
    if layers is not None and (scene is not None or frame):
        # padding below what a layer states, an invalid or duplicate box, a
        # world over the cache: refused when composed, so say so now
        attempt(
            lambda: compose_cuboids(
                scene or Scene(frame, [], []),
                padding=cfg["planner"]["world_padding"],
                no_pad_names=frozenset(cfg["planner"]["no_pad_names"] or []),
                cache_obb=cfg["planner"]["collision_cache_obb"],
                layers=layers,
                allow_empty=True,
            )
        )

    xml = None
    if spec:
        xml = attempt(lambda: prepare_urdf(urdf_xml, spec))
    if xml:
        attempt(lambda: _check_collision_links(robot_path, xml))
    if problems:
        raise RobotDataError(cfg_dir, problems)
    return xml
