"""Scene/world model: obstacles + props + named targets from one YAML.

Ported from RAMMP-Kinova's curobo_planner.scene (the format the sim kitchen
world is authored in) and kept deliberately dependency-light. Some parsed
fields exist for format parity rather than local use: color/free/density
feed RAMMP-Kinova's MuJoCo styling and physics, and the targets section
(Target, Scene.target, target_names) serves its NL command layer — this
repo only plans against obstacles/objects, but keeps the schema whole so
one YAML drives both stacks. Kept dependency-light (stdlib + PyYAML
— NO ROS, NO cuRobo, NO numpy) so any RAMMP module can import it without the
GPU stack. The planner converts a Scene into cuRobo's collision world (see
world.py); poses are authored human-friendly as position [x, y, z] (metres,
base frame) plus roll/pitch/yaw in DEGREES.

A LAYER is a named group of obstacle boxes in a file of its own (the
wheelchair the arm is mounted on, the person sitting in it). Its entries
follow the obstacle schema, but the file is loaded strictly (load_layer).
A scene lists the layers that belong to it under `layers:`; the planner
config lists the ones every world gets.
"""

import re
import sys
from pathlib import Path

import yaml

from rammp_curobo.config import resolve_config
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


class Layer:
    """One layer file: its name, its boxes (Obstacles) and the file they
    came from, as a resolved absolute path. `excludable` says whether a
    caller may leave the whole layer out of a world."""

    __slots__ = ("name", "excludable", "obstacles", "path")

    def __init__(self, name, excludable, obstacles, path):
        self.name = name
        self.excludable = excludable
        self.obstacles = obstacles
        self.path = path


class Scene:
    def __init__(self, base_frame, obstacles, targets, objects=(), layers=()):
        self.base_frame = base_frame
        self.obstacles = obstacles
        self.targets = targets
        self.objects = list(objects)
        self.layers = list(layers)

    @property
    def target_names(self):
        return [t.name for t in self.targets]

    def target(self, name):
        for t in self.targets:
            if t.name == name:
                return t
        return None


class _MergeKey:
    """Stands for `<<` among the keys of a mapping, and reads as '<<' in a
    refusal. No key PyYAML builds equals it: a quoted "<<" is a plain key,
    which a merge key does not repeat."""

    def __repr__(self):
        return "'<<'"


_MERGE_KEY = _MergeKey()


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping key written twice. Plain PyYAML
    keeps the last one, so a second `obstacles:` in a layer file would
    silently drop the boxes of the first."""

    def __init__(self, stream):
        super().__init__(stream)
        self._checked = set()

    def construct_mapping(self, node, deep=False):
        self._refuse_duplicate_keys(node, deep)
        return super().construct_mapping(node, deep=deep)

    def _refuse_duplicate_keys(self, node, deep):
        """Check `node` and every mapping it merges in, each of them once.

        PyYAML resolves `<<` in place: it copies the merged keys into the
        node, after doing the same to the node of each mapping they come
        from. A node rewritten like that holds a key that overrides a merged
        one twice, though it was written once, and it may be rewritten
        before its mapping is constructed. A mapping written under `<<` and
        nowhere else is not constructed at all. So each node is checked the
        first time construction reaches it, by itself or through a merge
        key: that is before PyYAML rewrites it.
        """
        # a node that is no mapping, and a key that cannot be hashed (hence
        # a list, not a set), are PyYAML's to refuse
        if not isinstance(node, yaml.MappingNode) or node in self._checked:
            return
        # before its merge keys are followed: they may lead back to it
        self._checked.add(node)
        seen = []
        for key_node, value_node in node.value:
            # a key that overrides one it merges in is not written twice. A
            # second `<<` is: PyYAML merges both, and the keys of the last win
            if key_node.tag == "tag:yaml.org,2002:merge":
                merged = (
                    value_node.value
                    if isinstance(value_node, yaml.SequenceNode)
                    else [value_node]
                )
                for source in merged:
                    self._refuse_duplicate_keys(source, deep)
                key = _MERGE_KEY
            else:
                key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                raise ValueError(
                    "%s: duplicate key %r on line %d" % (mark.name, key, mark.line + 1)
                )
            seen.append(key)


_LAYER_KEYS = ("layer", "excludable", "meta", "obstacles")
_LAYER_ENTRY_KEYS = ("name", "position", "dims", "rpy_deg", "color")
_LAYER_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _is_number(v):
    """A YAML int or float that a float can hold: not a bool (an int to
    Python), not a string float() would parse, and not nan or inf."""
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and abs(v) <= sys.float_info.max  # False for nan too
    )


def _check_layer_entry(d, earlier):
    """What a layer file asks of an entry beyond the obstacle schema: no
    unknown key, a name that is new and has no '/', and vectors of 3 real
    numbers. Obstacle() alone would take "0.1" and true for numbers, and
    cuRobo loads a NaN position or a negative dim as an enabled box that
    can never collide."""
    for key in d:
        if key not in _LAYER_ENTRY_KEYS:
            raise ValueError(
                "unknown key %r (allowed: %s)" % (key, ", ".join(_LAYER_ENTRY_KEYS))
            )
    name = d["name"]
    if not isinstance(name, str) or not name or "/" in name:
        raise ValueError(
            "'name' must be a non-empty string without '/', got %r" % (name,)
        )
    for i, other in enumerate(earlier):
        if other.name == name:
            raise ValueError("same name as obstacles[%d]" % i)
    for key, required, positive in (
        ("position", True, False),
        ("dims", True, True),
        ("rpy_deg", False, False),
    ):
        if not required and key not in d:
            continue
        v = d[key]
        if not (
            isinstance(v, list)
            and len(v) == 3
            and all(_is_number(x) and (x > 0 or not positive) for x in v)
        ):
            raise ValueError(
                "%r must be 3 finite numbers%s, got %r"
                % (key, " greater than 0" if positive else "", v)
            )


def _entries(cls, raw, where, section, check=None):
    """`cls(d)` for each entry `d` of one section of a world file.

    A malformed entry is a ValueError naming the file and the entry, e.g.
    "<file>: obstacles[3] (table): missing 'dims'": SetWorld replies with
    the bare message, and a KeyError's is only "'dims'". A section that is
    not a list is one naming the file and the section: `obstacles:` with
    every entry commented out is None. `check(d, built)` sees each entry
    first, with the ones built before it, and refuses it the way cls would:
    a KeyError for a missing key, a ValueError otherwise.
    """
    if not isinstance(raw, list):
        raise ValueError("%s: %r must be a list, got %r" % (where, section, raw))
    built = []
    for i, d in enumerate(raw):
        try:
            if not isinstance(d, dict):
                raise TypeError("must be a mapping, got %s" % type(d).__name__)
            if check is not None:
                check(d, built)
            built.append(cls(d))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            name = d.get("name") if isinstance(d, dict) else None
            raise ValueError(
                "%s: %s[%d]%s: %s"
                % (
                    where,
                    section,
                    i,
                    " (%s)" % (name,) if name else "",
                    "missing %r" % (exc.args[0],) if isinstance(exc, KeyError) else exc,
                )
            ) from None
    return built


def load_layer(path):
    """Load one layer file: `layer` (its name), `obstacles` (a non-empty
    list of obstacle entries), and optionally `excludable` and `meta`.
    `meta` is for people: nothing here reads it.

    Strict where load_scene is lenient, because each of these would
    otherwise pass unnoticed and leave a box out or wrong: a key written
    twice, an unknown key (`rpy:` for `rpy_deg:`, a nested `layers:`), two
    boxes of one name, a number that is not one. Every refusal is a
    ValueError naming the file, the layer and the entry. Only YAML that
    PyYAML itself cannot read stays its yaml.YAMLError, which names the
    file and the line.
    """
    path = Path(path).resolve()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_StrictLoader)
    except UnicodeDecodeError as exc:
        # its message names no file, and several layers may be listed
        raise ValueError("%s: not UTF-8: %s" % (path, exc)) from None
    if not isinstance(data, dict):
        raise ValueError(
            "%s: a layer file must be a mapping, got %s" % (path, type(data).__name__)
        )
    name = data.get("layer")
    # 'scene' is what a composed world calls the boxes that are in no layer
    if not isinstance(name, str) or not _LAYER_NAME.fullmatch(name) or name == "scene":
        raise ValueError(
            "%s: 'layer' must match %s and not be 'scene', got %r"
            % (path, _LAYER_NAME.pattern, name)
        )
    where = "%s: layer %r" % (path, name)
    for key in data:
        if key not in _LAYER_KEYS:
            raise ValueError(
                "%s: unknown key %r (allowed: %s)"
                % (where, key, ", ".join(_LAYER_KEYS))
            )
    excludable = data.get("excludable", False)
    if not isinstance(excludable, bool):
        raise ValueError(
            "%s: 'excludable' must be true or false, got %r" % (where, excludable)
        )
    entries = data.get("obstacles")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "%s: 'obstacles' must be a non-empty list, got %r" % (where, entries)
        )
    obstacles = _entries(
        Obstacle, entries, where, "obstacles", check=_check_layer_entry
    )
    return Layer(name, excludable, obstacles, path)


def load_layers(refs, relative_to, listed_in):
    """Load the layer files a scene or a planner config lists, in order.

    A ref is an absolute path, a path relative to `relative_to` (the
    directory of the listing file) or the name of a packaged layer. It is
    never looked up in the current directory: the planner can be started
    from anywhere, and a file there that shares a packaged layer's name
    must not replace that layer. For a listing file opened through a
    symlink, `relative_to` is the link's directory, not its target's: an
    install made of links (colcon's --symlink-install) then finds the
    layers a copying install would. A file listed twice loads once; two
    files that declare the same layer are refused. `listed_in` names the
    listing file in the messages.
    """
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise ValueError(
            "%s: 'layers' must be a list of layer file names, got %r"
            % (listed_in, refs)
        )
    layers = []
    for ref in refs:
        try:
            path = resolve_config(ref, relative_to=relative_to, search_cwd=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "layer %r listed in %s not found; tried: %s"
                % (ref, listed_in, str(exc).partition("tried: ")[2])
            ) from None
        layer = load_layer(path)
        if any(layer.path == other.path for other in layers):
            continue
        for other in layers:
            if other.name == layer.name:
                raise ValueError(
                    "%s: layer %r is defined by two files: %s and %s"
                    % (listed_in, layer.name, other.path, layer.path)
                )
        layers.append(layer)
    return layers


def load_scene(path):
    # UTF-8 whatever the locale: the packaged worlds have non-ASCII comments
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scene(
        base_frame=data.get("base_frame", "base_link"),
        obstacles=_entries(Obstacle, data.get("obstacles", []), path, "obstacles"),
        targets=_entries(Target, data.get("targets", []), path, "targets"),
        objects=_entries(SceneObject, data.get("objects", []), path, "objects"),
        layers=load_layers(
            data.get("layers", []), relative_to=Path(path).parent, listed_in=path
        ),
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
