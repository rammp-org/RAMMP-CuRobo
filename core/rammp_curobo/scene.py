"""Scene/world model: obstacles + props + named targets from one YAML.

Ported from RAMMP-Kinova's curobo_planner.scene (the format the sim kitchen
world is authored in) and kept deliberately dependency-light (stdlib + PyYAML
— NO ROS, NO cuRobo, NO numpy) so any RAMMP module can import it without the
GPU stack. The planner converts a Scene into cuRobo's collision world (see
world.py); poses are authored human-friendly as position [x, y, z] (metres,
base frame) plus roll/pitch/yaw in DEGREES.
"""

import yaml

from rammp_curobo.geometry import euler_deg_to_quat_xyzw


class Obstacle:
    """Static furniture: an oriented box the arm must route around."""

    __slots__ = ("name", "position", "rpy_deg", "dims", "color")

    def __init__(self, d):
        self.name = d["name"]
        self.position = [float(v) for v in d["position"]]
        self.rpy_deg = [float(v) for v in d.get("rpy_deg", [0, 0, 0])]
        self.dims = [float(v) for v in d["dims"]]
        self.color = [float(v) for v in d.get("color", [0.55, 0.4, 0.3, 0.85])]


class SceneObject:
    """A prop (bottle, mug...). Collision-avoided as its bounding box —
    except when a caller ignores it (you reach FOR the bottle, you can't
    also dodge it). Types: box (dims=full extents), cylinder
    (radius+height), sphere (radius)."""

    __slots__ = (
        "name",
        "type",
        "position",
        "rpy_deg",
        "dims",
        "radius",
        "height",
        "color",
        "free",
        "density",
    )

    def __init__(self, d):
        self.name = d["name"]
        self.type = d.get("type", "box")
        self.position = [float(v) for v in d["position"]]
        self.rpy_deg = [float(v) for v in d.get("rpy_deg", [0, 0, 0])]
        self.dims = [float(v) for v in d.get("dims", [0.05, 0.05, 0.05])]
        self.radius = float(d.get("radius", 0.03))
        self.height = float(d.get("height", 0.1))
        self.color = [float(v) for v in d.get("color", [0.8, 0.8, 0.8, 1.0])]
        self.free = bool(d.get("free", False))
        self.density = float(d.get("density", 400.0))

    def bounding_dims(self):
        """Axis-aligned bounding box (full extents) — the collision proxy."""
        if self.type == "cylinder":
            return [2 * self.radius, 2 * self.radius, self.height]
        if self.type == "sphere":
            return [2 * self.radius] * 3
        return list(self.dims)


class Target:
    """A named goal pose for the end effector (fingertip midpoint)."""

    __slots__ = (
        "name",
        "position",
        "rpy_deg",
        "keywords",
        "description",
        "ignore_objects",
        "standoff",
        "standoff_position",
        "standoff_rpy_deg",
    )

    def __init__(self, d):
        self.name = d["name"]
        self.position = [float(v) for v in d["position"]]
        self.rpy_deg = [float(v) for v in d.get("rpy_deg", [180, 0, 0])]
        self.keywords = [str(k).lower() for k in d.get("keywords", [])]
        self.description = str(d.get("description", ""))
        self.ignore_objects = [str(n) for n in d.get("ignore_objects", [])]
        self.standoff = float(d.get("standoff", 0.10))
        sp = d.get("standoff_position")
        self.standoff_position = [float(v) for v in sp] if sp else None
        sr = d.get("standoff_rpy_deg")
        self.standoff_rpy_deg = [float(v) for v in sr] if sr else None

    def quat_xyzw(self):
        return euler_deg_to_quat_xyzw(self.rpy_deg)


class Scene:
    def __init__(self, base_frame, obstacles, targets, objects=()):
        self.base_frame = base_frame
        self.obstacles = obstacles
        self.targets = targets
        self.objects = list(objects)

    @property
    def target_names(self):
        return [t.name for t in self.targets]

    def target(self, name):
        for t in self.targets:
            if t.name == name:
                return t
        return None


def load_scene(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Scene(
        base_frame=data.get("base_frame", "base_link"),
        obstacles=[Obstacle(o) for o in data.get("obstacles", [])],
        targets=[Target(t) for t in data.get("targets", [])],
        objects=[SceneObject(o) for o in data.get("objects", [])],
    )


def scene_from_obstacles(entries, base_frame="base_link"):
    """Build a Scene from plain dicts (the `update_world(obstacles)` path).

    Each entry follows the YAML object schema: name + position required,
    plus either dims (box) or type/radius/height (cylinder, sphere).
    """
    return Scene(
        base_frame=base_frame,
        obstacles=[],
        targets=[],
        objects=[SceneObject(dict(e)) for e in entries],
    )
