"""Obstacle layers, offline: no torch, no cuRobo, no ROS, so they run in CI."""

import importlib.util
import logging
import math
import os
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

import rammp_curobo
from rammp_curobo.config import PACKAGED_CONFIG_DIR, load_planner_config, resolve_config
from rammp_curobo.geometry import euler_deg_to_quat_xyzw
from rammp_curobo.planner import CuRoboPlanner
from rammp_curobo.scene import (
    Layer,
    Obstacle,
    Scene,
    SceneObject,
    load_layer,
    load_layers,
    load_scene,
    scene_from_obstacles,
)
from rammp_curobo.types import PlanResult
from rammp_curobo.world import (
    compose_cuboids,
    make_world_config,
    point_box_distance,
    sphere_box_penetration,
    summarize,
    to_world_config,
    world_cuboids,
)

# a layer file that uses every allowed key. Each malformed case below is this
# file with one edit, so its refusal is about that edit alone
LAYER = """\
layer: block
excludable: true
meta: {source: written by hand, boxes: 2}
obstacles:
  - {name: box, position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]}
  - name: slab
    position: [0.0, 0.5, 0.1]
    rpy_deg: [0, 0, 45]
    dims: [0.4, 0.02, 0.4]
    color: [1, 0, 0, 0.5]
"""


def _edited(old, new):
    assert LAYER.count(old) == 1, old
    return LAYER.replace(old, new)


def _layer(path, name, boxes=("box",), excludable=False):
    """Write a layer file that declares `name`, with a small box for each name
    in `boxes`: the name tells one copy of a file from another. Only an
    excludable layer says so: every other layer rests on the default, so
    the tests that use one test the default too."""
    path.write_text(
        "layer: %s\n%sobstacles:\n" % (name, "excludable: true\n" if excludable else "")
        + "".join(
            "  - {name: %s, position: [0.5, %d, 0.2], dims: [0.1, 0.2, 0.3]}\n"
            % (box, i)
            for i, box in enumerate(boxes)
        )
    )
    return path


def _scene(path, *refs):
    """Write a scene whose only content is the layers it lists."""
    path.write_text("layers: [%s]\n" % ", ".join(str(ref) for ref in refs))
    return path


def _config(folder, *refs):
    """The layers a planner config in `folder` lists."""
    return load_layers(
        list(refs), relative_to=folder, listed_in="planner config in %s" % folder
    )


def _box(name, **changes):
    """A box entry, as Obstacle and scene_from_obstacles take it."""
    box = {"name": name, "position": [0.5, 0.0, 0.2], "dims": [0.1, 0.2, 0.3]}
    return dict(box, **changes)


def _no_gpu(monkeypatch):
    """Let CuRoboPlanner be built, and take worlds, without cuRobo: its
    _init_curobo only records that it ran and leaves a stand-in MotionGen,
    and cuRobo's WorldConfig.from_dict records what it is given. Returns
    the lists they record in: the warmup flag of each _init_curobo call,
    each from_dict input, and each world MotionGen.update_world loads."""
    inits, sent, loaded = [], [], []

    def from_dict(data):
        sent.append(data)
        kinds = dict(sphere=[], cylinder=[], capsule=[], mesh=[])
        return types.SimpleNamespace(data=data, cuboid=list(data["cuboid"]), **kinds)

    stand_in = types.ModuleType("curobo.geom.types")
    stand_in.WorldConfig = types.SimpleNamespace(from_dict=from_dict)
    monkeypatch.setitem(sys.modules, "curobo.geom.types", stand_in)

    def init_curobo(planner, warmup=True):
        inits.append(warmup)
        planner._motion_gen = types.SimpleNamespace(update_world=loaded.append)
        # the arm's fixed spheres come from cuRobo's FK, which the GPU tests
        # run and _curobo_stand_in stands in for; with none, no layer box is
        # refused for reaching into them
        planner._fixed_spheres = np.empty((0, 4))

    monkeypatch.setattr(CuRoboPlanner, "_init_curobo", init_curobo)
    return inits, sent, loaded


def _curobo_stand_in(monkeypatch):
    """Let the planner's own cuRobo code run, _init_curobo included, without
    cuRobo or a GPU. Stand-in modules take the places of cuRobo, torch and
    warp, and MotionGen records in `events`, in order, the world it is built
    with ("built", keys), its warm-up ("warmup",) and each world it loads
    ("loaded", keys). What only cuRobo computes, a test sets on the returned
    namespace: `feasible`, the verdict of every constraint check, and
    `spheres(q)`, FK's collision spheres at controller-order joints q,
    (n, 4). The joint limits are 3.2 rad either way."""
    spec = importlib.util.spec_from_file_location(
        "generate_layers",
        Path(__file__).resolve().parents[2] / "scripts" / "generate_layers.py",
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    keep_out = generator.keep_out_spheres(generator.load_data())
    fixed = np.array([[*centre, radius] for centre, radius, _ in keep_out])
    person = load_layer(PACKAGED_CONFIG_DIR / "layer_mebot_person.yaml")
    x, y, z = next(box.position for box in person.obstacles if box.name == "lap")

    def spheres(q):
        # the spheres no joint moves, as the layer generator finds them, and
        # one that joint_1 swings about the base's z axis: at the retract
        # (joint_1 0) it is at the centre of the person's lap, which the real
        # arm's wrist reaches into there
        c, s = math.cos(q[0]), math.sin(q[0])
        return np.vstack([fixed, [c * x - s * y, s * x + c * y, z, 0.05]])

    stand_in = types.SimpleNamespace(
        events=[], feasible=True, spheres=spheres, fixed=fixed
    )

    def from_dict(data):
        kinds = dict(sphere=[], cylinder=[], capsule=[], mesh=[])
        return types.SimpleNamespace(data=data, cuboid=list(data["cuboid"]), **kinds)

    class MotionGen:
        def __init__(self, world):
            stand_in.events.append(("built", list(world.data["cuboid"])))
            joints = ["joint_%d" % i for i in range(1, 8)]
            self.kinematics = types.SimpleNamespace(joint_names=joints)

        def warmup(self, enable_graph):
            stand_in.events.append(("warmup",))

        def update_world(self, world):
            stand_in.events.append(("loaded", list(world.data["cuboid"])))

        def check_constraints(self, state):
            return types.SimpleNamespace(feasible=np.array([stand_in.feasible]))

    def module(name, **attributes):
        stand_in_module = types.ModuleType(name)
        stand_in_module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stand_in_module)

    module("torch", tensor=lambda data, device, dtype: np.asarray(data, dtype=float))
    monkeypatch.setitem(sys.modules, "warp", None)
    module("curobo.geom.types", WorldConfig=types.SimpleNamespace(from_dict=from_dict))
    module("curobo.geom.sdf.world", CollisionCheckerType=types.SimpleNamespace(MESH=0))
    module(
        "curobo.types.base",
        TensorDeviceType=lambda: types.SimpleNamespace(device=None, dtype=None),
    )
    module("curobo.types.math", Pose=dict)
    module(
        "curobo.types.robot",
        JointState=types.SimpleNamespace(from_position=lambda position, **_: position),
    )
    module(
        "curobo.wrap.reacher.motion_gen",
        MotionGen=MotionGen,
        MotionGenConfig=types.SimpleNamespace(
            load_from_robot_config=lambda robot, world, **_: world
        ),
        MotionGenPlanConfig=dict,
    )
    limits = {"position": np.array([[-3.2] * 7, [3.2] * 7]), "velocity": np.ones(7)}
    monkeypatch.setattr(CuRoboPlanner, "joint_limits", lambda planner: limits)
    monkeypatch.setattr(
        CuRoboPlanner, "_spheres", lambda planner, q: stand_in.spheres(q)
    )
    return stand_in


def _failed(status):
    """A cuRobo plan result that failed with `status`."""
    return types.SimpleNamespace(success=np.array(False), status=status)


def _planned(positions):
    """A cuRobo plan result that succeeded with the plan `positions`, in
    controller order, 20 ms apart. Its tensors, as far as _finish reads
    them, only need to reach numpy."""
    position = types.SimpleNamespace(numpy=lambda: np.asarray(positions, dtype=float))
    position.cpu = position.detach = lambda: position
    state = types.SimpleNamespace(position=position, velocity=None, acceleration=None)
    plan = types.SimpleNamespace(get_ordered_joint_state=lambda names: state)
    return types.SimpleNamespace(
        success=np.array(True),
        interpolation_dt=0.02,
        get_interpolated_plan=lambda: plan,
    )


def test_misplaced_layers_key_is_refused(tmp_path):
    path = tmp_path / "planner.yaml"

    def load(extra):
        path.write_text(
            "robot: robot_gen3_2f85.yaml\nworld: world_sim_kitchen.yaml\n" + extra
        )
        return load_planner_config(path)[0]

    # in its place the key loads, so each refusal below is about the key alone
    assert load("")["layers"] == []
    assert load("layers: [a.yaml, b.yaml]\n")["layers"] == ["a.yaml", "b.yaml"]

    # one message for both misplacements, whatever the key holds: a bare file
    # name is what a singular `layer:` invites. Each is refused beside a valid
    # `layers:` too: a config that lists the platform and misplaces the person
    # must not load without the person
    for extra in (
        "layer: [a.yaml]\n",
        "layer: [a.yaml, b.yaml]\n",
        "layer: a.yaml\n",
        "layer:\n",
        "planner:\n  layers: [a.yaml]\n",
        "planner:\n  layers: [a.yaml, b.yaml]\n",
        "planner:\n  layers: a.yaml\n",
        "planner:\n  layers:\n",
        "layers: [a.yaml]\nlayer: [b.yaml]\n",
        "layers: [a.yaml]\nplanner:\n  layers: [b.yaml]\n",
    ):
        with pytest.raises(ValueError) as exc:
            load(extra)
        assert str(exc.value) == (
            "%s: 'layer' is not a planner-config key; "
            "did you mean the top-level 'layers:'?" % path
        ), extra
    # only a mapping can hold the misplaced key: any other `planner:` value
    # loads as it did in v1.0.0
    for extra in ("planner:\n", "planner: 5\n", "planner: [layers]\n"):
        assert load(extra)["layers"] == [], extra

    for extra, got in (
        ('layers: "a.yaml"\n', "'a.yaml'"),
        ("layers: [1]\n", "[1]"),
        ("layers: [a.yaml, 1]\n", "['a.yaml', 1]"),
        ("layers: [1, a.yaml]\n", "[1, 'a.yaml']"),
        ("layers:\n", "None"),
    ):
        with pytest.raises(ValueError) as exc:
            load(extra)
        assert str(exc.value) == (
            "%s: 'layers' must be a list of layer file names, got %s" % (path, got)
        ), extra


def test_a_planner_config_key_written_twice_is_refused(tmp_path):
    path = tmp_path / "planner.yaml"
    path.write_text(
        "robot: robot_gen3_2f85.yaml\n"
        "world: world_mebot.yaml\n"
        "layers: [layer_mebot_platform.yaml, layer_mebot_person.yaml]\n"
        "layers: []\n"
    )
    # plain PyYAML keeps the last of the two: the chair would be gone without
    # a word
    assert yaml.safe_load(path.read_text())["layers"] == []
    with pytest.raises(ValueError) as exc:
        load_planner_config(path)
    assert str(exc.value) == "%s: duplicate key 'layers' on line 4" % path


def test_resolve_config_searches_cwd_unless_told_not_to(tmp_path, monkeypatch):
    cwd, beside = tmp_path / "cwd", tmp_path / "beside"
    (cwd / "sub").mkdir(parents=True)
    beside.mkdir()
    (cwd / "stray.yaml").write_text("")
    (cwd / "sub" / "stray.yaml").write_text("")
    (cwd / "gen3.yaml").write_text("")  # the name of a packaged config
    (cwd / "near.yaml").write_text("")  # the name of the file in beside/
    (beside / "near.yaml").write_text("")
    monkeypatch.chdir(cwd)

    # the v1.0.0 order is still the default: beside the referencing file,
    # then CWD, then the package
    assert resolve_config("near.yaml", relative_to=beside) == beside / "near.yaml"
    assert resolve_config("stray.yaml") == cwd / "stray.yaml"
    assert resolve_config("sub/stray.yaml") == cwd / "sub" / "stray.yaml"
    assert resolve_config("gen3.yaml") == cwd / "gen3.yaml"

    packaged = PACKAGED_CONFIG_DIR / "gen3.yaml"
    assert resolve_config("gen3.yaml", search_cwd=False) == packaged
    near = resolve_config("near.yaml", relative_to=beside, search_cwd=False)
    assert near == beside / "near.yaml"
    # a directory part does not turn a reference into a CWD lookup
    for ref in ("stray.yaml", "sub/stray.yaml"):
        with pytest.raises(FileNotFoundError) as exc:
            resolve_config(ref, relative_to=beside, search_cwd=False)
        assert str(exc.value) == "config '%s' not found; tried: %s, %s" % (
            ref,
            beside / ref,
            PACKAGED_CONFIG_DIR / ref,
        )


TOP_KEYS = "(allowed: layer, excludable, meta, obstacles)"
ENTRY_KEYS = "(allowed: name, position, dims, rpy_deg, color)"
LAYER_NAME = "'layer' must match [a-z][a-z0-9_]* and not be 'scene', got %s"
BOX_NAME = "'name' must be a non-empty string without '/', got %s"
OBSTACLES = "layer 'block': 'obstacles' must be a non-empty list, got %s"
POSITIVE = "'dims' must be 3 finite numbers greater than 0, got %s"

# (id, file text, the refusal after "<file>: ")
MALFORMED = [
    (
        "not_a_mapping",
        "- {name: box, position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]}\n",
        "a layer file must be a mapping, got list",
    ),
    ("empty_file", "", "a layer file must be a mapping, got NoneType"),
    # plain PyYAML keeps the last of two equal keys: the boxes listed first
    # would be gone without a word
    (
        "duplicate_key",
        LAYER + "obstacles:\n  - {name: late, position: [0, 0, 1], dims: [1, 1, 1]}\n",
        "duplicate key 'obstacles' on line 11",
    ),
    (
        "duplicate_key_in_an_entry",
        _edited("dims: [0.1, 0.2, 0.3]", "dims: [0.1, 0.2, 0.3], dims: [9, 9, 9]"),
        "duplicate key 'dims' on line 5",
    ),
    # whichever earlier key it repeats, next to it or not
    (
        "duplicate_key_not_next_to_the_first",
        LAYER + "layer: other\n",
        "duplicate key 'layer' on line 11",
    ),
    (
        "duplicate_key_further_down_an_entry",
        _edited(
            "    rpy_deg: [0, 0, 45]\n",
            "    rpy_deg: [0, 0, 45]\n    position: [9, 9, 9]\n",
        ),
        "duplicate key 'position' on line 9",
    ),
    # a mapping written under `<<` is never built itself: PyYAML copies its
    # keys into the mapping that merges it, and there too the last one wins
    (
        "duplicate_key_in_a_merged_mapping",
        _edited(
            "position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]",
            "<<: {position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3], dims: [9, 9, 9]}",
        ),
        "duplicate key 'dims' on line 5",
    ),
    (
        "duplicate_key_in_a_list_of_merged_mappings",
        _edited(
            "position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]",
            "<<: [{position: [0.5, 0, 0.2]}, {dims: [0.1, 0.2, 0.3], dims: [9, 9, 9]}]",
        ),
        "duplicate key 'dims' on line 5",
    ),
    (
        "duplicate_key_in_the_first_of_a_list_of_merged_mappings",
        _edited(
            "position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]",
            "<<: [{dims: [0.1, 0.2, 0.3], dims: [9, 9, 9]}, {position: [0.5, 0, 0.2]}]",
        ),
        "duplicate key 'dims' on line 5",
    ),
    (
        "duplicate_obstacles_in_a_merged_mapping",
        "layer: block\n"
        "<<:\n"
        "  obstacles:\n"
        "    - {name: early, position: [0, 0, 1], dims: [1, 1, 1]}\n"
        "  obstacles:\n"
        "    - {name: late, position: [0, 0, 1], dims: [1, 1, 1]}\n",
        "duplicate key 'obstacles' on line 5",
    ),
    # a merge key neither ends nor restarts the check of the mapping's own keys
    (
        "duplicate_key_after_a_merge_key",
        _edited(
            "position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]",
            "<<: {position: [0.5, 0, 0.2]}, dims: [0.1, 0.2, 0.3], dims: [9, 9, 9]",
        ),
        "duplicate key 'dims' on line 5",
    ),
    (
        "duplicate_key_on_both_sides_of_a_merge_key",
        _edited(
            "position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]",
            "dims: [0.1, 0.2, 0.3], <<: {position: [0.5, 0, 0.2]}, dims: [9, 9, 9]",
        ),
        "duplicate key 'dims' on line 5",
    ),
    # and `<<` written twice is a key written twice: PyYAML merges both
    # mappings, and the keys of the second win
    (
        "duplicate_merge_key",
        "layer: block\n"
        "<<: {obstacles: [{name: early, position: [0, 0, 1], dims: [1, 1, 1]}]}\n"
        "<<: {obstacles: [{name: late, position: [0, 0, 1], dims: [1, 1, 1]}]}\n",
        "duplicate key '<<' on line 3",
    ),
    (
        "duplicate_merge_key_in_an_entry",
        _edited(
            "dims: [0.1, 0.2, 0.3]",
            "<<: {dims: [0.1, 0.2, 0.3]}, <<: {dims: [9, 9, 9]}",
        ),
        "duplicate key '<<' on line 5",
    ),
    # a scene or a config may list several layers: the one that cannot be
    # read has to be named (the text here is bytes: a Latin-1 degree sign)
    (
        "not_utf8",
        b"# yaw 45\xb0\n" + LAYER.encode(),
        "not UTF-8: 'utf-8' codec can't decode byte 0xb0 in position 8: "
        "invalid start byte",
    ),
    (
        "unknown_key",
        LAYER + "base_frame: base_link\n",
        "layer 'block': unknown key 'base_frame' " + TOP_KEYS,
    ),
    (
        "nested_layers",
        LAYER + "layers: [other.yaml]\n",
        "layer 'block': unknown key 'layers' " + TOP_KEYS,
    ),
    ("no_layer_name", _edited("layer: block\n", ""), LAYER_NAME % "None"),
    (
        "capital_in_layer_name",
        _edited("layer: block", "layer: Block"),
        LAYER_NAME % "'Block'",
    ),
    (
        "digit_first_in_layer_name",
        _edited("layer: block", "layer: 2nd"),
        LAYER_NAME % "'2nd'",
    ),
    # composed boxes are keyed <layer>/<box>
    (
        "slash_in_layer_name",
        _edited("layer: block", "layer: a/b"),
        LAYER_NAME % "'a/b'",
    ),
    # `$` in a pattern also matches before a final newline
    (
        "newline_after_layer_name",
        _edited("layer: block", 'layer: "block\\n"'),
        LAYER_NAME % "'block\\n'",
    ),
    ("layer_name_not_a_string", _edited("layer: block", "layer: 5"), LAYER_NAME % "5"),
    (
        "reserved_layer_name",
        _edited("layer: block", "layer: scene"),
        LAYER_NAME % "'scene'",
    ),
    (
        "excludable_as_a_number",
        _edited("excludable: true", "excludable: 1"),
        "layer 'block': 'excludable' must be true or false, got 1",
    ),
    (
        "excludable_as_a_string",
        _edited("excludable: true", 'excludable: "true"'),
        "layer 'block': 'excludable' must be true or false, got 'true'",
    ),
    # left empty it is not the default: the key is there, and says nothing
    (
        "excludable_left_empty",
        _edited("excludable: true", "excludable:"),
        "layer 'block': 'excludable' must be true or false, got None",
    ),
    ("no_obstacles", "layer: block\n", OBSTACLES % "None"),
    ("empty_obstacles", "layer: block\nobstacles: []\n", OBSTACLES % "[]"),
    (
        "obstacles_not_a_list",
        "layer: block\nobstacles: {box: 1}\n",
        OBSTACLES % "{'box': 1}",
    ),
    (
        "entry_not_a_mapping",
        LAYER + "  - late\n",
        "layer 'block': obstacles[2]: must be a mapping, got str",
    ),
    (
        "unknown_entry_key",
        _edited("rpy_deg: [0, 0, 45]", "rpy: [0, 0, 45]"),
        "layer 'block': obstacles[1] (slab): unknown key 'rpy' " + ENTRY_KEYS,
    ),
    (
        "no_dims",
        _edited(", dims: [0.1, 0.2, 0.3]", ""),
        "layer 'block': obstacles[0] (box): missing 'dims'",
    ),
    (
        "no_position",
        _edited("    position: [0.0, 0.5, 0.1]\n", ""),
        "layer 'block': obstacles[1] (slab): missing 'position'",
    ),
    (
        "no_box_name",
        _edited("name: box, ", ""),
        "layer 'block': obstacles[0]: missing 'name'",
    ),
    (
        "slash_in_box_name",
        _edited("name: box", "name: wheel/left"),
        "layer 'block': obstacles[0] (wheel/left): " + BOX_NAME % "'wheel/left'",
    ),
    (
        "empty_box_name",
        _edited("name: box", 'name: ""'),
        "layer 'block': obstacles[0]: " + BOX_NAME % "''",
    ),
    (
        "box_name_not_a_string",
        _edited("name: slab", "name: 7"),
        "layer 'block': obstacles[1] (7): " + BOX_NAME % "7",
    ),
    (
        "duplicate_box",
        _edited("name: slab", "name: box"),
        "layer 'block': obstacles[1] (box): same name as obstacles[0]",
    ),
    # whichever earlier box it repeats, next to it or not
    (
        "duplicate_of_a_box_two_back",
        LAYER + "  - {name: box, position: [0, 0, 1], dims: [1, 1, 1]}\n",
        "layer 'block': obstacles[2] (box): same name as obstacles[0]",
    ),
    (
        "duplicate_of_the_box_before",
        LAYER + "  - {name: slab, position: [0, 0, 1], dims: [1, 1, 1]}\n",
        "layer 'block': obstacles[2] (slab): same name as obstacles[1]",
    ),
    # cuRobo takes numbers like these without a word: a NaN position or a
    # negative dim loads as an enabled box that never collides
    (
        "nan_position",
        _edited("position: [0.5, 0, 0.2]", "position: [.nan, 0, 0.2]"),
        "layer 'block': obstacles[0] (box): "
        "'position' must be 3 finite numbers, got [nan, 0, 0.2]",
    ),
    (
        "nan_rpy",
        _edited("rpy_deg: [0, 0, 45]", "rpy_deg: [0, .nan, 45]"),
        "layer 'block': obstacles[1] (slab): "
        "'rpy_deg' must be 3 finite numbers, got [0, nan, 45]",
    ),
    (
        "negative_inf_position",
        _edited("position: [0.5, 0, 0.2]", "position: [-.inf, 0, 0.2]"),
        "layer 'block': obstacles[0] (box): "
        "'position' must be 3 finite numbers, got [-inf, 0, 0.2]",
    ),
    (
        "inf_dim",
        _edited("dims: [0.1, 0.2, 0.3]", "dims: [0.1, .inf, 0.3]"),
        "layer 'block': obstacles[0] (box): " + POSITIVE % "[0.1, inf, 0.3]",
    ),
    (
        "zero_dim",
        _edited("dims: [0.1, 0.2, 0.3]", "dims: [0.1, 0, 0.3]"),
        "layer 'block': obstacles[0] (box): " + POSITIVE % "[0.1, 0, 0.3]",
    ),
    (
        "negative_dim",
        _edited("dims: [0.1, 0.2, 0.3]", "dims: [0.1, 0.2, -0.3]"),
        "layer 'block': obstacles[0] (box): " + POSITIVE % "[0.1, 0.2, -0.3]",
    ),
    (
        "two_element_vector",
        _edited("position: [0.5, 0, 0.2]", "position: [0.5, 0]"),
        "layer 'block': obstacles[0] (box): "
        "'position' must be 3 finite numbers, got [0.5, 0]",
    ),
    (
        "four_element_vector",
        _edited("dims: [0.4, 0.02, 0.4]", "dims: [0.4, 0.02, 0.4, 0.4]"),
        "layer 'block': obstacles[1] (slab): " + POSITIVE % "[0.4, 0.02, 0.4, 0.4]",
    ),
    (
        "scalar_for_a_vector",
        _edited("dims: [0.4, 0.02, 0.4]", "dims: 0.4"),
        "layer 'block': obstacles[1] (slab): " + POSITIVE % "0.4",
    ),
    # float() takes both of these for numbers, so a scene would load them
    (
        "numbers_as_strings",
        _edited("dims: [0.1, 0.2, 0.3]", 'dims: ["0.1", "0.2", "0.3"]'),
        "layer 'block': obstacles[0] (box): " + POSITIVE % "['0.1', '0.2', '0.3']",
    ),
    (
        "boolean_as_a_number",
        _edited("position: [0.5, 0, 0.2]", "position: [true, 0, 0.2]"),
        "layer 'block': obstacles[0] (box): "
        "'position' must be 3 finite numbers, got [True, 0, 0.2]",
    ),
    # float() cannot hold it, and says so with an OverflowError
    (
        "integer_too_large_for_a_float",
        _edited("position: [0.5, 0, 0.2]", "position: [%d, 0, 0.2]" % 10**400),
        "layer 'block': obstacles[0] (box): "
        "'position' must be 3 finite numbers, got [%d, 0, 0.2]" % 10**400,
    ),
    (
        "unreadable_color",
        _edited("color: [1, 0, 0, 0.5]", "color: red"),
        "layer 'block': obstacles[1] (slab): could not convert string to float: 'r'",
    ),
    (
        "integer_too_large_for_a_float_in_color",
        _edited("color: [1, 0, 0, 0.5]", "color: [%d, 0, 0, 0.5]" % 10**400),
        "layer 'block': obstacles[1] (slab): int too large to convert to float",
    ),
]


@pytest.mark.parametrize(
    "text, refusal",
    [case[1:] for case in MALFORMED],
    ids=[case[0] for case in MALFORMED],
)
def test_malformed_layer_is_refused(tmp_path, text, refusal):
    path = tmp_path / "layer.yaml"
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    with pytest.raises(ValueError) as exc:
        load_layer(path)
    assert str(exc.value) == "%s: %s" % (path, refusal)


def test_strict_loader_changes_nothing_else_about_yaml(tmp_path):
    path = tmp_path / "layer.yaml"
    # a merge key is not a second copy of the keys it brings in, a key that
    # two of its mappings share is given once, by the first, and a key that
    # overrides one of them is not written twice
    path.write_text(
        "layer: block\n"
        "obstacles:\n"
        "  - &box {name: box, position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]}\n"
        "  - {<<: *box, name: twin, position: [0.5, 0.4, 0.2]}\n"
        "  - {<<: [{dims: [1, 1, 1]}, *box], name: cube}\n"
    )
    box, twin, cube = load_layer(path).obstacles
    assert (twin.name, twin.position, twin.dims) == ("twin", [0.5, 0.4, 0.2], box.dims)
    assert (cube.name, cube.position, cube.dims) == ("cube", box.position, [1.0] * 3)

    # nor is it once PyYAML has copied the merged keys into the mapping's
    # node: here it does so to `wide` when the box merges it, which is before
    # `wide` itself is built
    path.write_text(
        "layer: block\n"
        "meta:\n"
        "  defaults:\n"
        "    base: &base {position: [0, 0, 0], dims: [1, 1, 1]}\n"
        "    wide: &wide {<<: *base, dims: [2, 2, 2]}\n"
        "obstacles:\n"
        "  - {<<: *wide, name: box}\n"
    )
    (box,) = load_layer(path).obstacles
    assert (box.position, box.dims) == ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    # PyYAML reads a mapping that merges itself: the check follows merge keys,
    # and must not go round in one for ever
    path.write_text(
        "layer: block\n"
        "obstacles:\n"
        "  - &box {<<: *box, name: box, position: [0, 0, 1], dims: [1, 2, 3]}\n"
    )
    (box,) = load_layer(path).obstacles
    assert (box.name, box.dims) == ("box", [1.0, 2.0, 3.0])

    # a quoted "<<" is a plain key, which a merge key beside it does not repeat
    path.write_text(_edited("meta: {", 'meta: {"<<": quoted, <<: {merged: 1}, '))
    assert load_layer(path).name == "block"

    # what PyYAML refuses, it still refuses in its own words: a key no mapping
    # can hold, and a mapping tag on something else
    for text, problem in (
        ("[layer, block]: 1\n", "found unhashable key"),
        ("layer: block\nmeta: !!map none\n", "expected a mapping node"),
    ):
        path.write_text(text)
        with pytest.raises(yaml.YAMLError, match=problem):
            load_layer(path)


def test_scene_layers_resolve_next_to_the_scene_and_never_from_cwd(
    tmp_path, monkeypatch
):
    # a scene that lists no layers has none
    for name in ("world_sim_kitchen.yaml", "world_real_bench.yaml"):
        assert load_scene(PACKAGED_CONFIG_DIR / name).layers == []
    assert scene_from_obstacles([]).layers == []

    scenes, cwd, packaged, far, links = (
        tmp_path / name for name in ("scenes", "cwd", "packaged", "far", "links")
    )
    for folder in (scenes, cwd, packaged, far, links):
        folder.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr("rammp_curobo.config.PACKAGED_CONFIG_DIR", packaged)
    _layer(scenes / "near.yaml", "beside")
    _layer(cwd / "near.yaml", "stray")
    _layer(packaged / "near.yaml", "shipped")
    _layer(cwd / "shipped.yaml", "stray")
    _layer(packaged / "shipped.yaml", "shipped")
    _layer(cwd / "only_in_cwd.yaml", "stray")
    _layer(far / "elsewhere.yaml", "elsewhere")
    (scenes / "whole.yaml").write_text(LAYER)

    scene = load_scene(
        _scene(
            scenes / "scene.yaml",
            "near.yaml",
            "shipped.yaml",
            far / "elsewhere.yaml",
            "whole.yaml",
        )
    )
    assert [(layer.name, layer.path) for layer in scene.layers] == [
        ("beside", scenes / "near.yaml"),
        ("shipped", packaged / "shipped.yaml"),
        ("elsewhere", far / "elsewhere.yaml"),
        ("block", scenes / "whole.yaml"),
    ]
    # without the key a layer is not excludable: only `true` lets ignore= drop it
    assert scene.layers[0].excludable is False
    whole = scene.layers[3]
    assert whole.excludable is True
    box, slab = whole.obstacles
    assert isinstance(box, Obstacle) and isinstance(slab, Obstacle)
    assert (box.name, box.position, box.rpy_deg, box.dims) == (
        "box",
        [0.5, 0.0, 0.2],
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
    )
    assert (slab.name, slab.position, slab.rpy_deg, slab.dims, slab.color) == (
        "slab",
        [0.0, 0.5, 0.1],
        [0.0, 0.0, 45.0],
        [0.4, 0.02, 0.4],
        [1.0, 0.0, 0.0, 0.5],
    )

    # the planner may be started from any directory: a file there is not a layer
    scene = _scene(scenes / "scene.yaml", "only_in_cwd.yaml")
    with pytest.raises(FileNotFoundError) as exc:
        load_scene(scene)
    assert str(exc.value) == (
        "layer 'only_in_cwd.yaml' listed in %s not found; tried: %s, %s"
        % (scene, scenes / "only_in_cwd.yaml", packaged / "only_in_cwd.yaml")
    )

    # a scene named by a relative path still gives its layers absolute paths
    monkeypatch.chdir(scenes)
    _scene(scenes / "scene.yaml", "near.yaml")
    assert [layer.path for layer in load_scene("scene.yaml").layers] == [
        scenes / "near.yaml"
    ]

    # a scene opened through a symlink is a scene where the link is: an
    # install made of links (colcon's --symlink-install) has to find the
    # layers a copying install would, not the ones beside its source files
    _layer(links / "near.yaml", "beside_the_link")
    (links / "scene.yaml").symlink_to(scenes / "scene.yaml")
    linked = load_scene(links / "scene.yaml")
    assert [(layer.name, layer.path) for layer in linked.layers] == [
        ("beside_the_link", links / "near.yaml")
    ]


def test_missing_layer_is_named_with_every_path_tried(tmp_path, monkeypatch):
    scene = _scene(tmp_path / "scene.yaml", "gone.yaml")
    with pytest.raises(FileNotFoundError) as exc:
        load_scene(scene)
    assert str(exc.value) == (
        "layer 'gone.yaml' listed in %s not found; tried: %s, %s"
        % (scene, tmp_path / "gone.yaml", PACKAGED_CONFIG_DIR / "gone.yaml")
    )

    # an absolute path is the only place to look
    gone = tmp_path / "far" / "gone.yaml"
    scene = _scene(tmp_path / "scene.yaml", gone)
    with pytest.raises(FileNotFoundError) as exc:
        load_scene(scene)
    assert str(exc.value) == (
        "layer '%s' listed in %s not found; tried: %s" % (gone, scene, gone)
    )

    # from a planner config, whose layers resolve from its directory: the
    # planner refuses to start, before any GPU work
    inits, _, _ = _no_gpu(monkeypatch)
    config = tmp_path / "planner.yaml"
    config.write_text(
        "robot: robot_gen3_2f85.yaml\nworld: world_mebot.yaml\nlayers: [gone.yaml]\n"
    )
    with pytest.raises(FileNotFoundError) as exc:
        CuRoboPlanner.from_config(config)
    assert str(exc.value) == (
        "layer 'gone.yaml' listed in planner config in %s not found; tried: %s, %s"
        % (tmp_path, tmp_path / "gone.yaml", PACKAGED_CONFIG_DIR / "gone.yaml")
    )
    assert inits == []


def test_one_layer_name_from_two_files_is_refused_and_one_file_twice_composes_once(
    tmp_path,
):
    person = _layer(tmp_path / "person.yaml", "person")
    other = _layer(tmp_path / "other.yaml", "person")
    platform = _layer(tmp_path / "platform.yaml", "platform")
    (tmp_path / "sub").mkdir()
    (tmp_path / "link.yaml").symlink_to(person)

    # one file is one layer however it is spelled and wherever in the list it
    # comes again, and listing order is kept
    layers = load_layers(
        [
            "platform.yaml",
            "person.yaml",
            "./person.yaml",
            "sub/../person.yaml",
            str(person),
            "link.yaml",
            "platform.yaml",
        ],
        relative_to=tmp_path,
        listed_in="planner config in %s" % tmp_path,
    )
    assert [(layer.name, layer.path) for layer in layers] == [
        ("platform", platform),
        ("person", person),
    ]
    # a file that comes again is passed over, not taken for the end of the list
    scene = load_scene(
        _scene(tmp_path / "scene.yaml", "person.yaml", "person.yaml", "platform.yaml")
    )
    assert [layer.path for layer in scene.layers] == [person, platform]

    # two files that both claim a layer are two sources of truth, wherever in
    # the list the first of them is
    for refs in (
        ["person.yaml", "platform.yaml", "other.yaml"],
        ["platform.yaml", "person.yaml", "other.yaml"],
    ):
        with pytest.raises(ValueError) as exc:
            load_layers(
                refs,
                relative_to=tmp_path,
                listed_in="planner config in %s" % tmp_path,
            )
        assert str(exc.value) == (
            "planner config in %s: layer 'person' is defined by two files: %s and %s"
            % (tmp_path, person, other)
        ), refs
    scene = _scene(tmp_path / "scene.yaml", "other.yaml", "person.yaml")
    with pytest.raises(ValueError) as exc:
        load_scene(scene)
    assert str(exc.value) == (
        "%s: layer 'person' is defined by two files: %s and %s" % (scene, other, person)
    )

    # in a world, a file that both the config and the scene list is one layer,
    # in the config's place, and the layers listed after it still come; another
    # file that claims one of the config's layers is refused like a second file
    # in one list
    _layer(tmp_path / "floor.yaml", "floor")
    config = _config(tmp_path, "platform.yaml", "person.yaml")
    scene = load_scene(
        _scene(tmp_path / "scene.yaml", "link.yaml", "platform.yaml", "floor.yaml")
    )
    for layers in (config, config + config):
        _, sources = compose_cuboids(scene, layers=layers)
        assert list(sources.items()) == [
            ("platform/box", "platform"),
            ("person/box", "person"),
            ("floor/box", "floor"),
        ]
    scene = load_scene(_scene(tmp_path / "scene.yaml", "other.yaml"))
    with pytest.raises(ValueError) as exc:
        compose_cuboids(scene, layers=config)
    assert str(exc.value) == (
        "layer 'person' is defined by two files: %s and %s" % (person, other)
    )

    # the config's layers are loaded once, at start-up, and a file the config
    # lists composes as that copy until a restart: renamed since, the file is
    # still one layer, not a second one beside the first
    _layer(platform, "chair", boxes=("seat",))
    scene = load_scene(_scene(tmp_path / "scene.yaml", "platform.yaml", "floor.yaml"))
    assert [layer.name for layer in scene.layers] == ["chair", "floor"]
    _, sources = compose_cuboids(scene, layers=config)
    assert list(sources.items()) == [
        ("platform/box", "platform"),
        ("person/box", "person"),
        ("floor/box", "floor"),
    ]

    # a Layer built in code may have no path, and then it is no file's copy:
    # under its own name it is a layer of its own, and under a name already
    # taken it is refused, not passed over
    crate = scene_from_obstacles([_box("crate")])
    deck, lap = [Obstacle(_box("deck"))], [Obstacle(_box("lap"))]
    built = [Layer("platform", False, deck, None), Layer("person", True, lap, None)]
    _, sources = compose_cuboids(crate, layers=built)
    assert list(sources) == ["obj_crate", "platform/deck", "person/lap"]
    twins = [Layer("person", True, deck, None), Layer("person", True, lap, None)]
    with pytest.raises(ValueError, match="layer 'person' is defined by two files"):
        compose_cuboids(crate, layers=twins)


def test_malformed_scene_entry_names_file_and_entry(tmp_path):
    path = tmp_path / "world.yaml"
    world = """\
obstacles:
  - {name: pedestal, position: [0, 0, -0.05], dims: [0.14, 0.14, 0.04]}
  - {name: back_wall, position: [1.07, 0, 0.4], dims: [0.04, 2.4, 1.0]}
  - {name: shelf, position: [0.55, 0.42, 0.5], dims: [0.34, 0.3, 0.03]}
  - {name: table, position: [0.15, 0, -0.1], dims: [1.3, 1.4, 0.06]}
objects:
  - {name: bottle, type: cylinder, radius: 0.033, position: [0.45, 0.12, 0.04]}
  - {name: mug, type: cylinder, radius: 0.032, position: [-0.45, -0.14, 0]}
targets:
  - {name: rest, position: [0.571, 0, 0.43]}
"""

    def refusal(old, new):
        assert world.count(old) == 1, old
        path.write_text(world.replace(old, new))
        with pytest.raises(ValueError) as exc:
            load_scene(path)
        return str(exc.value)

    # unedited it loads, so each refusal below is about the edit alone
    path.write_text(world)
    scene = load_scene(path)
    assert (len(scene.obstacles), len(scene.objects), len(scene.targets)) == (4, 2, 1)

    # SetWorld replies with the bare message, and a KeyError's is just 'dims'
    assert refusal(", dims: [1.3, 1.4, 0.06]", "") == (
        "%s: obstacles[3] (table): missing 'dims'" % path
    )
    assert refusal(", position: [-0.45, -0.14, 0]", "") == (
        "%s: objects[1] (mug): missing 'position'" % path
    )
    assert refusal(", position: [0.571, 0, 0.43]", "") == (
        "%s: targets[0] (rest): missing 'position'" % path
    )
    # an entry without a usable name still has its place in the list
    assert refusal("name: shelf, ", "") == "%s: obstacles[2]: missing 'name'" % path
    assert refusal(
        "{name: back_wall, position: [1.07, 0, 0.4], dims: [0.04, 2.4, 1.0]}",
        "back_wall",
    ) == ("%s: obstacles[1]: must be a mapping, got str" % path)
    # whatever else the entry's own parsing trips on
    assert refusal("dims: [0.14, 0.14, 0.04]", "dims: [0.14, wide, 0.04]") == (
        "%s: obstacles[0] (pedestal): could not convert string to float: 'wide'" % path
    )
    assert refusal("radius: 0.033", "radius: [0.033]").startswith(
        "%s: objects[0] (bottle): float() argument must be " % path
    )
    assert refusal("radius: 0.032", "radius: %d" % 10**400) == (
        "%s: objects[1] (mug): int too large to convert to float" % path
    )
    # a section is a list of entries: with its only entry commented out it is
    # nothing at all, and keyed by name it would be read as a list of its keys
    assert refusal("  - {name: rest,", "  # - {name: rest,") == (
        "%s: 'targets' must be a list, got None" % path
    )
    assert refusal("  - {name: rest,", "  rest: {name: rest,") == (
        "%s: 'targets' must be a list, got "
        "{'rest': {'name': 'rest', 'position': [0.571, 0, 0.43]}}" % path
    )

    # `layers: near.yaml` would otherwise be looked up one letter at a time
    for layers, got in (
        ("near.yaml", "'near.yaml'"),
        ("[1]", "[1]"),
        ("[near.yaml, 1]", "['near.yaml', 1]"),
        ("", "None"),
    ):
        path.write_text(world + "layers: %s\n" % layers)
        with pytest.raises(ValueError) as exc:
            load_scene(path)
        assert str(exc.value) == (
            "%s: 'layers' must be a list of layer file names, got %s" % (path, got)
        ), layers


def _in_a_c_locale(code, *args):
    """Run `code` with `args` in a child Python whose locale is C."""
    # PYTHONUTF8=0: Python otherwise turns UTF-8 mode on for the C locale.
    # PYTHONPATH: the child must import the tree this process is testing
    env = dict(os.environ, PYTHONUTF8="0", LC_ALL="C")
    tree = str(Path(rammp_curobo.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [tree, env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_world_files_load_as_utf8_under_a_c_locale(tmp_path):
    kitchen = PACKAGED_CONFIG_DIR / "world_sim_kitchen.yaml"
    layer = tmp_path / "layer.yaml"
    layer.write_text("# 5 mm — not an ASCII comment\n" + LAYER, encoding="utf-8")
    # the child proves its own premise: there, open() without an encoding
    # cannot read either file
    code = """\
import sys

from rammp_curobo.scene import load_layer, load_scene

kitchen, layer = sys.argv[1:]
for path in (kitchen, layer):
    try:
        with open(path) as f:
            f.read()
    except UnicodeDecodeError:
        continue
    sys.exit("premise broken: this locale reads %s as it is" % path)
print(len(load_scene(kitchen).obstacles), load_layer(layer).name)
"""
    child = _in_a_c_locale(code, kitchen, layer)
    assert child.returncode == 0, child.stderr
    assert child.stdout == "11 block\n"


def test_planner_configs_load_as_utf8_under_a_c_locale():
    # what the planner reads before cuRobo: the config, its robot config, its
    # world, and the layers they list. gen3.yaml and the robot config have
    # non-ASCII comments, which the child first proves it cannot read without
    # an encoding
    code = """\
import sys
from pathlib import Path

from rammp_curobo.config import load_planner_config, resolve_config
from rammp_curobo.robot_config import load_robot_config
from rammp_curobo.scene import load_layers, load_scene


def premise(path):
    try:
        with open(path) as f:
            f.read()
    except UnicodeDecodeError:
        return
    sys.exit("premise broken: this locale reads %s as it is" % path)


premise(sys.argv[1])
for path in sys.argv[1:]:
    cfg, folder = load_planner_config(path)
    robot = resolve_config(cfg["robot"], relative_to=folder)
    premise(robot)
    ee_link = load_robot_config(robot)["kinematics"]["ee_link"]
    scene = load_scene(resolve_config(cfg["world"], relative_to=folder))
    layers = load_layers(cfg["layers"], relative_to=folder, listed_in=path)
    names = [layer.name for layer in layers + scene.layers]
    print(Path(path).name, robot.name, ee_link, len(scene.obstacles), names)
"""
    gen3, mebot = (PACKAGED_CONFIG_DIR / n for n in ("gen3.yaml", "gen3_mebot.yaml"))
    child = _in_a_c_locale(code, gen3, mebot)
    assert child.returncode == 0, child.stderr
    assert child.stdout == (
        "gen3.yaml robot_gen3_2f85.yaml tool_frame 11 []\n"
        "gen3_mebot.yaml robot_gen3_2f85.yaml tool_frame 0 "
        "['platform', 'person', 'floor']\n"
    )


V100_EMPTY = (
    "collision world is empty; refusing to build it — cuRobo would "
    "silently keep the previous world on update"
)
CHAIR_ONLY = (
    "collision world is empty apart from the configured layers (platform, "
    "person): the pushed scene has no obstacles, objects or layers — refusing it"
)
TWICE = (
    "two collision boxes are named %r: %s and %s; rename one — a world holds "
    "one box per name"
)


def test_layer_boxes_are_namespaced_padded_and_attributed(tmp_path):
    # a layer box may share a scene obstacle's name, the pedestal's too: its
    # key is its layer's, and no_pad_names exempts scene obstacles only
    _layer(tmp_path / "platform.yaml", "platform", boxes=("pedestal", "wheel"))
    _layer(tmp_path / "person.yaml", "person", boxes=("lap",), excludable=True)
    (tmp_path / "floor.yaml").write_text(
        "layer: floor\n"
        "obstacles:\n"
        "  - {name: slab, position: [0, 0, -0.5], rpy_deg: [0, 0, 30], "
        "dims: [3, 3, 0.1]}\n"
    )
    world = tmp_path / "world.yaml"
    world.write_text(
        "obstacles:\n"
        "  - {name: pedestal, position: [0, 0, -0.05], dims: [0.14, 0.14, 0.04]}\n"
        "  - {name: table, position: [0.15, 0, -0.1], rpy_deg: [0, 0, 10], "
        "dims: [1.3, 1.4, 0.06]}\n"
        "objects:\n"
        "  - {name: bottle, type: cylinder, radius: 0.033, height: 0.22, "
        "position: [0.45, 0.12, 0.04]}\n"
        "layers: [floor.yaml]\n"
    )
    scene = load_scene(world)
    config = _config(tmp_path, "platform.yaml", "person.yaml")
    pedestal = {"pedestal"}
    cuboids, sources = compose_cuboids(scene, no_pad_names=pedestal, layers=config)

    # scene obstacles, props, the config's layers in its order, the scene's
    assert list(sources.items()) == [
        ("pedestal", "scene"),
        ("table", "scene"),
        ("obj_bottle", "scene"),
        ("platform/pedestal", "platform"),
        ("platform/wheel", "platform"),
        ("person/lap", "person"),
        ("floor/slab", "floor"),
    ]
    assert list(cuboids) == list(sources)
    # the scene's own boxes are v1.0.0's, the pedestal unpadded
    v100 = world_cuboids(scene, padding=0.02, no_pad_names=pedestal)
    assert list(cuboids.items())[:3] == list(v100.items())
    assert cuboids["pedestal"]["dims"] == [0.14, 0.14, 0.04]
    # every layer box grows by the padding on each side, posed as written
    for padding in (0.02, 0.05):
        cuboids, _ = compose_cuboids(
            scene, padding=padding, no_pad_names=pedestal, layers=config
        )
        for layer in config + scene.layers:
            for o in layer.obstacles:
                x, y, z, w = euler_deg_to_quat_xyzw(o.rpy_deg)
                assert cuboids["%s/%s" % (layer.name, o.name)] == {
                    "dims": [d + 2 * padding for d in o.dims],
                    "pose": o.position + [w, x, y, z],
                }


@pytest.mark.parametrize(
    "config", ["gen3.yaml", "gen3_real.yaml"], ids=["kitchen", "bench"]
)
def test_layer_free_worlds_compose_exactly_as_v100(config):
    cfg, cfg_dir = load_planner_config(config)
    assert cfg["layers"] == []
    scene = load_scene(resolve_config(cfg["world"], relative_to=cfg_dir))
    p = cfg["planner"]
    no_pad = frozenset(p["no_pad_names"])
    # ignore holds props a task reaches for, and an obstacle's name, which
    # v1.0.0 never left out; no_pad_names the props' names, which v1.0.0
    # padded all the same: it exempts scene obstacles only
    for ignore, no_pad_names in (
        (frozenset(), no_pad),
        ({"bottle", "mug", "table"}, no_pad),
        (frozenset(), no_pad | {o.name for o in scene.objects}),
    ):
        kw = dict(padding=p["world_padding"], ignore=ignore, no_pad_names=no_pad_names)
        cuboids, sources = compose_cuboids(
            scene, cache_obb=p["collision_cache_obb"], **kw
        )
        v100 = world_cuboids(scene, **kw)
        assert list(cuboids.items()) == list(v100.items())
        assert sources == dict.fromkeys(v100, "scene")


def test_duplicate_names_are_refused_not_collapsed(tmp_path):
    platform = _layer(tmp_path / "platform.yaml", "platform", boxes=("deck", "x"))
    config = _config(tmp_path, "platform.yaml")
    table, wall, mug = _box("table"), _box("wall"), _box("mug")

    def refusal(scene, layers=config):
        with pytest.raises(ValueError) as exc:
            compose_cuboids(scene, layers=layers)
        return str(exc.value)

    # v1.0.0 kept one of two boxes of a name, without a word
    twice = Scene("base_link", [Obstacle(table), Obstacle(wall), Obstacle(table)], [])
    assert list(world_cuboids(twice)) == ["table", "wall"]
    assert refusal(twice) == TWICE % (
        "table",
        "scene obstacles[0]",
        "scene obstacles[2]",
    )
    # a scene obstacle that takes a layer box's key
    posing = Scene("base_link", [Obstacle(table), Obstacle(_box("platform/x"))], [])
    assert refusal(posing) == TWICE % (
        "platform/x",
        "scene obstacles[1]",
        "layer 'platform' obstacles[1] in %s" % platform,
    )
    # two props of a name, and an obstacle that takes a prop's key
    assert refusal(scene_from_obstacles([mug, table, mug])) == TWICE % (
        "obj_mug",
        "scene objects[0]",
        "scene objects[2]",
    )
    shadow = Scene("base_link", [Obstacle(_box("obj_mug"))], [], [SceneObject(mug)])
    assert refusal(shadow) == TWICE % (
        "obj_mug",
        "scene obstacles[0]",
        "scene objects[0]",
    )
    # two boxes of a name in one layer: possible in a Layer built in code
    path = Path("/nowhere/block.yaml")
    block = Layer("block", False, [Obstacle(_box("box")), Obstacle(_box("box"))], path)
    assert refusal(scene_from_obstacles([mug]), layers=[block]) == TWICE % (
        "block/box",
        "layer 'block' obstacles[0] in %s" % path,
        "layer 'block' obstacles[1] in %s" % path,
    )
    # an ignored prop is not in the world, so it takes no key
    cuboids, _ = compose_cuboids(
        scene_from_obstacles([mug, table, mug]), ignore={"mug"}, layers=config
    )
    assert list(cuboids) == ["obj_table", "platform/deck", "platform/x"]


def test_boxes_curobo_would_silently_ignore_are_refused():
    nan, inf = float("nan"), float("inf")
    path = Path("/nowhere/block.yaml")

    def obstacle(**changes):
        return Scene(
            "base_link",
            [Obstacle(_box("crate")), Obstacle(_box("table", **changes))],
            [],
        )

    def prop(**changes):
        return scene_from_obstacles([_box("mug"), _box("can", **changes)])

    def layer(**changes):
        boxes = [Obstacle(_box("box")), Obstacle(_box("slab", **changes))]
        return Layer("block", True, boxes, path)

    # v1.0.0 passed them on; cuRobo loads each as an enabled box that never
    # collides
    assert math.isnan(
        world_cuboids(obstacle(position=[nan, 0, 0.2]))["table"]["pose"][0]
    )

    finite = "'%s' must be 3 finite numbers, got %s"
    positive = "'dims' must be 3 finite numbers greater than 0, got %s"
    table = "collision box 'table' (scene obstacles[1]): "
    can = "collision box 'obj_can' (scene objects[1]): "
    slab = "collision box 'block/slab' (layer 'block' obstacles[1] in %s): " % path
    for scene, layers, refused in (
        (
            obstacle(position=[nan, 0, 0.2]),
            (),
            table + finite % ("position", "[nan, 0.0, 0.2]"),
        ),
        (
            obstacle(position=[0.5, -inf, 0.2]),
            (),
            table + finite % ("position", "[0.5, -inf, 0.2]"),
        ),
        (
            obstacle(rpy_deg=[0, inf, 0]),
            (),
            table + finite % ("rpy_deg", "[0.0, inf, 0.0]"),
        ),
        (
            obstacle(rpy_deg=[nan, 0, 0]),
            (),
            table + finite % ("rpy_deg", "[nan, 0.0, 0.0]"),
        ),
        (obstacle(dims=[0.1, 0, 0.3]), (), table + positive % "[0.1, 0.0, 0.3]"),
        # before padding: padded, a dim of -0.01 would be a box 3 cm wide
        (obstacle(dims=[0.1, -0.01, 0.3]), (), table + positive % "[0.1, -0.01, 0.3]"),
        (obstacle(dims=[0.1, nan, 0.3]), (), table + positive % "[0.1, nan, 0.3]"),
        (obstacle(position=[0.5, 0]), (), table + finite % ("position", "[0.5, 0.0]")),
        (obstacle(rpy_deg=[0, 0]), (), table + finite % ("rpy_deg", "[0.0, 0.0]")),
        (obstacle(dims=[0.1, 0.2]), (), table + positive % "[0.1, 0.2]"),
        (
            obstacle(dims=[0.1, 0.2, 0.3, 0.4]),
            (),
            table + positive % "[0.1, 0.2, 0.3, 0.4]",
        ),
        # a prop as the box cuRobo would get: a radius of 0 is a dim of 0
        (
            prop(type="cylinder", radius=0, height=0.12),
            (),
            can + positive % "[0.0, 0.0, 0.12]",
        ),
        (
            prop(type="cylinder", radius=0.03, height=-0.1),
            (),
            can + positive % "[0.06, 0.06, -0.1]",
        ),
        (
            prop(type="sphere", radius=-0.03),
            (),
            can + positive % "[-0.06, -0.06, -0.06]",
        ),
        (
            prop(position=[0.5, nan, 0.2]),
            (),
            can + finite % ("position", "[0.5, nan, 0.2]"),
        ),
        # a layer's boxes, whether the config lists it or the scene does
        (
            prop(),
            [layer(position=[inf, 0, 0.2])],
            slab + finite % ("position", "[inf, 0.0, 0.2]"),
        ),
        (
            Scene(
                "base_link",
                [Obstacle(_box("crate"))],
                [],
                layers=[layer(dims=[0.1, 0.2, -0.3])],
            ),
            (),
            slab + positive % "[0.1, 0.2, -0.3]",
        ),
    ):
        # unpadded as well: a box that no padding grows reaches cuRobo exactly
        # as written, as the pedestal (no_pad_names) and any box at a padding
        # of 0 do
        for kw in ({}, {"no_pad_names": {"table"}}, {"padding": 0}):
            with pytest.raises(ValueError) as exc:
                compose_cuboids(scene, layers=layers, **kw)
            assert str(exc.value) == refused, kw

    # and the padding: below 0 it would shrink a thin box past 0, and a NaN or
    # an inf would spread to every box
    for padding in (-0.06, nan, inf):
        with pytest.raises(ValueError) as exc:
            compose_cuboids(obstacle(), padding=padding)
        assert str(exc.value) == (
            "world_padding must be a finite number >= 0, got %r" % padding
        )


def test_cache_overflow_names_every_source(tmp_path):
    _layer(tmp_path / "platform.yaml", "platform", boxes=["b%d" % i for i in range(16)])
    _layer(
        tmp_path / "person.yaml",
        "person",
        boxes=["b%d" % i for i in range(6)],
        excludable=True,
    )
    _layer(tmp_path / "floor.yaml", "floor")
    config = _config(tmp_path, "platform.yaml", "person.yaml")
    (floor,) = load_layers(["floor.yaml"], relative_to=tmp_path, listed_in="a scene")

    def scene(n):
        obstacles = [Obstacle(_box("o%d" % i)) for i in range(n)]
        return Scene("base_link", obstacles, [], layers=[floor])

    # a world of exactly the cache's size keeps every box
    cuboids, _ = compose_cuboids(scene(37), layers=config)
    assert len(cuboids) == 60
    # one more is refused, never truncated
    with pytest.raises(ValueError) as exc:
        compose_cuboids(scene(41), layers=config)
    assert str(exc.value) == (
        "64 collision boxes > collision_cache_obb=60 (scene 41, platform 16, "
        "person 6, floor 1); raise the cache setting in the planner config or "
        "shrink a layer"
    )
    # a layer left out takes no room
    cuboids, _ = compose_cuboids(scene(41), ignore={"person"}, layers=config)
    assert len(cuboids) == 58
    # the limit is the planner config's; a scene with no box of its own fills
    # none of it
    with pytest.raises(ValueError) as exc:
        compose_cuboids(scene(0), cache_obb=22, layers=config)
    assert str(exc.value) == (
        "23 collision boxes > collision_cache_obb=22 (platform 16, person 6, "
        "floor 1); raise the cache setting in the planner config or shrink a layer"
    )
    # v1.0.0's overflow, which test_smoke pushes
    with pytest.raises(ValueError) as exc:
        compose_cuboids(scene_from_obstacles([_box("b%d" % i) for i in range(61)]))
    assert str(exc.value) == (
        "61 collision boxes > collision_cache_obb=60 (scene 61); raise the cache "
        "setting in the planner config or shrink a layer"
    )


def test_empty_scene_is_refused_even_with_config_layers(tmp_path):
    _layer(tmp_path / "platform.yaml", "platform")
    _layer(tmp_path / "person.yaml", "person", excludable=True)
    _layer(tmp_path / "floor.yaml", "floor")
    _layer(tmp_path / "block.yaml", "block", excludable=True)
    config = _config(tmp_path, "platform.yaml", "person.yaml")

    def refusal(scene, layers=config, ignore=()):
        with pytest.raises(ValueError) as exc:
            compose_cuboids(scene, ignore=ignore, layers=layers)
        return str(exc.value)

    nothing = Scene("base_link", [], [])
    # without config layers, v1.0.0's words: test_smoke matches 'empty'
    assert refusal(nothing, layers=()) == V100_EMPTY
    # the config's layers are not the scene's content: a push cut short must
    # not plan as the chair alone
    assert refusal(nothing) == CHAIR_ONLY
    # what ignore leaves out is not content
    mug = scene_from_obstacles([_box("mug")])
    assert refusal(mug, layers=(), ignore={"mug"}) == V100_EMPTY
    assert refusal(mug, ignore={"mug"}) == CHAIR_ONLY
    block = load_scene(_scene(tmp_path / "block_only.yaml", "block.yaml"))
    assert refusal(block, ignore={"block"}) == CHAIR_ONLY
    # a layer counts by its boxes, and one built in code may have none
    hollow = Scene("base_link", [], [], layers=[Layer("hollow", True, [], None)])
    assert refusal(hollow, layers=()) == V100_EMPTY
    assert refusal(hollow) == CHAIR_ONLY
    # and no path is no file: config layers without one are not the scene's
    pathless = [Layer(lay.name, lay.excludable, lay.obstacles, None) for lay in config]
    assert refusal(hollow, layers=pathless) == CHAIR_ONLY

    # a scene whose only content is a layer it lists is a world: the chair's
    # start-up scene lists only the floor
    start = load_scene(_scene(tmp_path / "start.yaml", "floor.yaml"))
    solid = Layer("solid", True, [Obstacle(_box("box"))], None)
    for scene in (start, block, Scene("base_link", [], [], layers=[solid])):
        for layers in ((), config):
            _, sources = compose_cuboids(scene, layers=layers)
            assert list(sources.values()) == [
                layer.name for layer in [*layers, *scene.layers]
            ]
    # so is one that lists a file the config lists too: the scene asked for it
    platform = load_scene(_scene(tmp_path / "platform_only.yaml", "platform.yaml"))
    _, sources = compose_cuboids(platform, layers=config)
    assert list(sources.values()) == ["platform", "person"]
    # and that file counts as it composed: as the copy the config loaded at
    # start-up, whatever the file says now. Left out under that copy's name,
    # it leaves the scene nothing
    _layer(tmp_path / "person.yaml", "human", excludable=True)
    renamed = load_scene(_scene(tmp_path / "person_only.yaml", "person.yaml"))
    assert [layer.name for layer in renamed.layers] == ["human"]
    assert refusal(renamed, ignore={"person"}) == CHAIR_ONLY


def test_guards_run_without_curobo(tmp_path, monkeypatch):
    # every module of cuRobo, loaded or not, fails to import
    for name in [n for n in sys.modules if n.partition(".")[0] == "curobo"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "curobo", None)
    _layer(tmp_path / "platform.yaml", "platform")
    _layer(tmp_path / "person.yaml", "person", excludable=True)
    _layer(tmp_path / "other.yaml", "person", excludable=True)
    config = _config(tmp_path, "platform.yaml", "person.yaml")
    crate = _box("crate")

    # the premise: a world that every guard lets through gets as far as
    # importing cuRobo
    with pytest.raises(ImportError):
        make_world_config(scene_from_obstacles([crate]), layers=config)
    for scene, ignore, refusal in (
        (Scene("base_link", [], []), (), "empty"),
        (
            scene_from_obstacles([_box("b%d" % i) for i in range(59)]),
            (),
            "61 collision boxes > collision_cache_obb=60",
        ),
        (scene_from_obstacles([crate, crate]), (), "named 'obj_crate'"),
        (
            scene_from_obstacles([_box("crate", dims=[0.1, 0, 0.3])]),
            (),
            "'dims' must be 3 finite numbers",
        ),
        (scene_from_obstacles([crate]), {"platform"}, "not excludable"),
        (
            load_scene(_scene(tmp_path / "scene.yaml", "other.yaml")),
            (),
            "defined by two files",
        ),
    ):
        with pytest.raises(ValueError, match=refusal):
            make_world_config(scene, ignore=ignore, layers=config)
    # to_world_config refuses an empty world itself, wherever it came from:
    # update_world would take it and keep the world loaded before
    with pytest.raises(ValueError) as exc:
        to_world_config({})
    assert str(exc.value) == V100_EMPTY


def test_ignore_drops_only_an_excludable_layer(tmp_path):
    _layer(tmp_path / "platform.yaml", "platform", boxes=("deck", "wheel"))
    _layer(tmp_path / "person.yaml", "person", boxes=("lap", "torso"), excludable=True)
    _layer(tmp_path / "floor.yaml", "floor")
    _layer(tmp_path / "block.yaml", "block", excludable=True)
    config = _config(tmp_path, "platform.yaml", "person.yaml")
    world = tmp_path / "world.yaml"
    # props named after pieces of 'person': ignore='person' must reach neither
    world.write_text(
        "obstacles:\n"
        "  - {name: table, position: [0.5, 0, -0.1], dims: [1, 1, 0.06]}\n"
        "objects:\n"
        "  - {name: son, position: [0.4, 0.1, 0.1], dims: [0.05, 0.05, 0.05]}\n"
        "  - {name: p, position: [0.4, -0.1, 0.1], dims: [0.05, 0.05, 0.05]}\n"
        "layers: [floor.yaml, block.yaml]\n"
    )
    scene = load_scene(world)
    whole, sources = compose_cuboids(scene, layers=config)

    # the person's boxes go, and every other box stays byte for byte, however
    # the name is given: a bare string is one name, not its letters
    for ignore in ({"person"}, ["person"], frozenset({"person"}), "person"):
        cuboids, kept = compose_cuboids(scene, ignore=ignore, layers=config)
        assert list(kept.items()) == [
            (key, source) for key, source in sources.items() if source != "person"
        ]
        assert cuboids == {key: whole[key] for key in kept}
    # for that world only: the next one without the ignore has the person
    assert compose_cuboids(scene, layers=config) == (whole, sources)
    # an excludable scene layer goes the same way, and props as in v1.0.0
    cuboids, kept = compose_cuboids(scene, ignore={"block", "son"}, layers=config)
    assert list(kept) == [
        key for key, source in sources.items() if source != "block" and key != "obj_son"
    ]
    # a bare string names one prop too: 1.0.0 read 'son' as its letters and
    # kept the prop
    _, kept = compose_cuboids(scene, ignore="son", layers=config)
    assert list(kept) == [key for key in sources if key != "obj_son"]
    # a name that is no layer and no prop changes nothing, though it is an
    # obstacle's or a layer box's: the obstacle stays, the safe direction
    unknown = {"nothing", "table", "deck", "person/lap"}
    assert compose_cuboids(scene, ignore=unknown, layers=config) == (whole, sources)

    # a layer that does not say it is excludable is never left out, config or
    # scene
    for name in ("platform", "floor"):
        with pytest.raises(ValueError) as exc:
            compose_cuboids(scene, ignore={name}, layers=config)
        assert str(exc.value) == "ignore=%r: layer %r is not excludable" % (name, name)
    # nor is anything, when a prop has the layer's name: which one is meant?
    for name in ("person", "platform", "block"):
        props = scene.objects + [SceneObject(_box(name))]
        clash = Scene("base_link", scene.obstacles, [], props, layers=scene.layers)
        with pytest.raises(ValueError) as exc:
            compose_cuboids(clash, ignore={name}, layers=config)
        assert str(exc.value) == (
            "ignore=%r names both a prop and a layer; rename the prop" % name
        )
        # without the ignore the prop and the layer are both in the world
        _, both = compose_cuboids(clash, layers=config)
        assert "obj_" + name in both and name in both.values()


def _rotation(axis, angle):
    """Rodrigues' formula: a rotation matrix made without a quaternion."""
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def test_sphere_box_penetration_matches_brute_force():
    rng = np.random.default_rng(19)
    boxes, frames = {}, []
    for i in range(300):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-math.pi, math.pi)
        centre = rng.uniform(-1.0, 1.0, 3)
        dims = rng.uniform(0.02, 0.2, 3)
        wxyz = [math.cos(angle / 2), *(axis * math.sin(angle / 2))]
        boxes["box%d" % i] = {"dims": dims.tolist(), "pose": centre.tolist() + wxyz}
        frames.append((centre, _rotation(axis, angle), dims / 2))
    # each sphere placed in the frame of one box: inside it, or off a face,
    # an edge or a corner, near enough to reach it or not
    spheres = []
    for i in rng.integers(0, 300, 50):
        centre, rot, half = frames[i]
        offset = rot @ (rng.uniform(-1.6, 1.6, 3) * half)
        spheres.append([*(centre + offset), rng.uniform(0.005, 0.08)])
    spheres = np.array(spheres)

    # the brute force: each box's surface, 31 x 31 points a face, edges
    # included, so no point of it is further from a sample than half the
    # diagonal of a grid cell
    g = np.linspace(-1.0, 1.0, 31)
    u, v = (a.ravel() for a in np.meshgrid(g, g))
    cube = np.concatenate(
        [
            np.insert(np.stack([u, v], 1), a, sign, axis=1)
            for a in range(3)
            for sign in (-1.0, 1.0)
        ]
    )
    signed, tol = np.empty((len(spheres), len(boxes))), np.empty(len(boxes))
    for j, (centre, rot, half) in enumerate(frames):
        surface = centre + (cube * half) @ rot.T
        # a coordinate at a time: an (N, P, 3) array is several times slower
        square = sum((spheres[:, None, k] - surface[:, k]) ** 2 for k in range(3))
        nearest = np.sqrt(square.min(axis=1))
        inside = (np.abs((spheres[:, :3] - centre) @ rot) < half).all(axis=1)
        signed[:, j] = np.where(inside, -nearest, nearest)
        tol[j] = np.linalg.norm(np.sort(half)[1:]) / 30 + 1e-9
    depth = spheres[:, 3:] - signed
    # the premise: centres inside boxes, spheres that reach a box from
    # outside, and spheres that just fall short
    assert (signed < -tol).sum() >= 10
    assert ((signed > tol) & (depth > tol)).sum() >= 10
    assert ((depth < -tol) & (depth > -0.05)).sum() >= 10

    names = list(boxes)
    each = []
    for j, sphere in enumerate(spheres):
        distance = point_box_distance(sphere[:3], boxes)
        assert list(distance) == names
        assert (np.abs(np.array(list(distance.values())) - signed[j]) <= tol).all()
        hits = sphere_box_penetration(sphere[None], boxes)
        # every box the sampling finds reached is named, and no box it
        # finds out of reach
        assert {names[i] for i in np.flatnonzero(depth[j] > tol)} <= set(hits)
        assert set(hits) <= {names[i] for i in np.flatnonzero(depth[j] > -tol)}
        for key, d in hits.items():
            i = names.index(key)
            assert abs(d - depth[j, i]) <= tol[i], key
        each.append(hits)
    # all spheres at once: each box reached, by its deepest sphere
    deepest = sphere_box_penetration(spheres, boxes)
    reached = set().union(*each)
    assert set(deepest) == reached
    for key in reached:
        assert deepest[key] == pytest.approx(
            max(hits[key] for hits in each if key in hits), abs=1e-12
        )
    # batch dimensions are only that: a trajectory's (waypoints, spheres, 4)
    # holds the same spheres
    assert sphere_box_penetration(spheres.reshape(5, 10, 4), boxes) == deepest
    # anything else is refused: centres without radii, read 4 numbers at a
    # time, would be other spheres, and a sphere that is not finite would
    # hide every hit, since no maximum taken over it is above 0
    with pytest.raises(ValueError) as exc:
        sphere_box_penetration(spheres[:4, :3], boxes)
    assert str(exc.value) == (
        "spheres must have shape (..., 4): x, y, z and radius; got (4, 3)"
    )
    hit = [*frames[0][0], 0.01]  # at a box's centre
    assert "box0" in sphere_box_penetration([hit], boxes)
    # its radius as much as its centre
    for sphere in ([math.nan, 0.0, 0.0, 0.01], [0.0, 0.0, 0.0, math.nan]):
        with pytest.raises(ValueError) as exc:
            sphere_box_penetration([hit, sphere], boxes)
        assert str(exc.value) == "spheres must be finite, got %s" % sphere
    assert sphere_box_penetration(np.empty((0, 4)), boxes) == {}
    assert sphere_box_penetration(spheres, {}) == {}
    assert point_box_distance([0.0, 0.0, 0.0], {}) == {}

    # a 0.40 x 0.02 x 0.40 slab yawed 45 degrees, composed from rpy, and a
    # point 0.15 m from its centre along its normal: inside the slab's
    # axis-aligned bounding box, and 0.14 m from the slab
    slab = _box("slab", position=[0, 0, 0], rpy_deg=[0, 0, 45], dims=[0.4, 0.02, 0.4])
    cuboids, _ = compose_cuboids(Scene("base_link", [Obstacle(slab)], []), padding=0)
    c = s = math.sqrt(0.5)
    normal = np.array([-s, c, 0.0])
    reach = 0.2 * c + 0.01 * s  # the half width of the bounding box in x and y
    aabb = {
        "aabb": {"dims": [2 * reach, 2 * reach, 0.4], "pose": [0, 0, 0, 1, 0, 0, 0]}
    }
    point = 0.15 * normal
    assert point_box_distance(point, cuboids) == {"slab": pytest.approx(0.14)}
    # a quaternion of another length is read as cuRobo reads it, normalised
    pose = cuboids["slab"]["pose"]
    scaled = {"slab": dict(cuboids["slab"], pose=pose[:3] + [3 * q for q in pose[3:]])}
    assert point_box_distance(point, scaled) == {"slab": pytest.approx(0.14)}
    assert point_box_distance(point, aabb) == {"aabb": pytest.approx(0.15 * s - reach)}
    assert sphere_box_penetration([[*point, 0.05]], cuboids) == {}
    assert sphere_box_penetration([[*point, 0.05]], aabb) == {
        "aabb": pytest.approx(0.05 + reach - 0.15 * s)
    }
    # and 0.03 m off its face, a sphere of 0.05 m reaches it by 0.02 m; one
    # 0.1 mm short of the face reaches nothing
    assert sphere_box_penetration([[*(0.04 * normal), 0.05]], cuboids) == {
        "slab": pytest.approx(0.02)
    }
    for gap, hits in ((1e-4, {}), (-1e-4, {"slab": pytest.approx(1e-4)})):
        centre = (0.01 + 0.05 + gap) * normal
        assert sphere_box_penetration([[*centre, 0.05]], cuboids) == hits, gap


def test_world_summary_distinguishes_config_and_scene_layers(tmp_path):
    _layer(tmp_path / "platform.yaml", "platform", boxes=["b%d" % i for i in range(16)])
    _layer(
        tmp_path / "person.yaml",
        "person",
        boxes=["b%d" % i for i in range(6)],
        excludable=True,
    )
    _layer(tmp_path / "floor.yaml", "floor")
    config = _config(tmp_path, "platform.yaml", "person.yaml")

    def summary(scene, layers=config, **kw):
        _, sources = compose_cuboids(scene, layers=layers, **kw)
        return summarize(sources, [layer.name for layer in layers])

    # the chair: the config's two layers, and the floor its start-up scene lists
    start = load_scene(_scene(tmp_path / "start.yaml", "floor.yaml"))
    assert summary(start) == (
        "23 boxes: scene 0 | config layers: platform 16, person 6 "
        "| scene layers: floor 1"
    )
    # the same files listed by a scene, on a planner whose config lists none
    control = load_scene(
        _scene(tmp_path / "control.yaml", "floor.yaml", "platform.yaml")
    )
    assert summary(control, layers=[]) == (
        "17 boxes: scene 0 | config layers: none | scene layers: floor 1, platform 16"
    )
    # listed by both, the platform is a config layer
    assert summary(control) == summary(start)
    # a configured layer this world leaves out is still named
    assert summary(start, ignore={"person"}) == (
        "17 boxes: scene 0 | config layers: platform 16, person 0 "
        "| scene layers: floor 1"
    )
    kitchen = load_scene(PACKAGED_CONFIG_DIR / "world_sim_kitchen.yaml")
    assert summary(kitchen, layers=[]) == (
        "26 boxes: scene 26 | config layers: none | scene layers: none"
    )
    assert summary(scene_from_obstacles([_box("crate")]), layers=[]) == (
        "1 box: scene 1 | config layers: none | scene layers: none"
    )


def test_make_world_config_hands_curobo_exactly_the_composed_boxes(monkeypatch):
    sent, built, holds = [], [], {}

    def from_dict(data):
        # like cuRobo's, it builds an entry for each box it is given, unless
        # `holds` says it builds something else
        kinds = dict(sphere=[], cylinder=[], capsule=[], mesh=[])
        kinds.update(cuboid=list(data["cuboid"]))
        kinds.update(holds)
        sent.append(data)
        built.append(types.SimpleNamespace(**kinds))
        return built[-1]

    stand_in = types.ModuleType("curobo.geom.types")
    stand_in.WorldConfig = types.SimpleNamespace(from_dict=from_dict)
    monkeypatch.setitem(sys.modules, "curobo.geom.types", stand_in)

    scene = scene_from_obstacles([_box("crate"), _box("mug")])
    cuboids, _ = compose_cuboids(scene)
    assert make_world_config(scene) is built[-1]
    assert sent[-1] == {"cuboid": cuboids}
    # a v1.0.0 positional call means what it meant: the padding
    make_world_config(scene, 0.03)
    assert sent[-1]["cuboid"]["obj_crate"]["dims"] == [
        d + 0.06 for d in _box("crate")["dims"]
    ]
    # the cache is the planner config's 60 by default
    sixty = scene_from_obstacles([_box("b%d" % i) for i in range(60)])
    assert len(make_world_config(sixty).cuboid) == 60
    with pytest.raises(ValueError, match="61 collision boxes > collision_cache_obb=60"):
        make_world_config(scene_from_obstacles([_box("b%d" % i) for i in range(61)]))
    # every argument reaches the composition, in v1.0.0's positional order:
    # the pedestal unpadded, the can left out, the layer in, the cache as given
    props = [SceneObject(_box("mug")), SceneObject(_box("can"))]
    pedestal = Scene("base_link", [Obstacle(_box("pedestal"))], [], props)
    block = Layer("block", True, [Obstacle(_box("box"))], Path("/nowhere/block.yaml"))
    args = (pedestal, 0.03, {"can"}, {"pedestal"}, 3)
    make_world_config(*args, layers=[block])
    assert sent[-1] == {"cuboid": compose_cuboids(*args, layers=[block])[0]}
    assert list(sent[-1]["cuboid"]) == ["pedestal", "obj_mug", "block/box"]
    assert sent[-1]["cuboid"]["pedestal"]["dims"] == _box("pedestal")["dims"]
    with pytest.raises(ValueError, match="3 collision boxes > collision_cache_obb=2"):
        make_world_config(*args[:4], 2, layers=[block])

    # a box short, as from a key it does not know, or anything but boxes, is
    # not the world composed
    holds["cuboid"] = ["obj_crate"]
    with pytest.raises(ValueError) as exc:
        to_world_config(cuboids)
    assert str(exc.value) == (
        "cuRobo's WorldConfig holds {'cuboid': 1, 'sphere': 0, 'cylinder': 0, "
        "'capsule': 0, 'mesh': 0} for 2 composed boxes; only those boxes may "
        "reach its collision checkers"
    )
    del holds["cuboid"]
    for kind in ("sphere", "cylinder", "capsule", "mesh"):
        holds[kind] = ["stray"]
        with pytest.raises(ValueError, match="'%s': 1" % kind):
            to_world_config(cuboids)
        del holds[kind]
    assert to_world_config(cuboids) is built[-1]


def test_chair_config_composes_within_the_cache():
    cfg, folder = load_planner_config("gen3_mebot.yaml")
    # every knob at its default, as its header says: the file sets these three
    # keys and nothing else. Read from the file, since merged, a knob it sets
    # looks no different from one it inherits
    text = (PACKAGED_CONFIG_DIR / "gen3_mebot.yaml").read_text(encoding="utf-8")
    assert set(yaml.safe_load(text)) == {"robot", "world", "layers"}
    layers = load_layers(cfg["layers"], relative_to=folder, listed_in="gen3_mebot.yaml")
    assert [(layer.name, layer.excludable, layer.path) for layer in layers] == [
        ("platform", False, PACKAGED_CONFIG_DIR / "layer_mebot_platform.yaml"),
        ("person", True, PACKAGED_CONFIG_DIR / "layer_mebot_person.yaml"),
    ]
    p = cfg["planner"]
    kw = dict(
        padding=p["world_padding"],
        no_pad_names=frozenset(p["no_pad_names"]),
        cache_obb=p["collision_cache_obb"],
        layers=layers,
    )
    names = [layer.name for layer in layers]

    # the start-up world: the chair and its occupant, and the floor that its
    # scene lists
    start = load_scene(resolve_config(cfg["world"], relative_to=folder))
    _, sources = compose_cuboids(start, **kw)
    assert summarize(sources, names) == (
        "25 boxes: scene 0 | config layers: platform 16, person 8 "
        "| scene layers: floor 1"
    )
    # the largest known world, the sim kitchen, with the chair's layers added
    # still fits the cache, which stays 60
    kitchen = load_scene(PACKAGED_CONFIG_DIR / "world_sim_kitchen.yaml")
    _, sources = compose_cuboids(kitchen, **kw)
    assert summarize(sources, names) == (
        "50 boxes: scene 26 | config layers: platform 16, person 8 "
        "| scene layers: none"
    )
    assert p["collision_cache_obb"] == 60
    # each layer was generated for the padding this config plans with
    for layer in layers + start.layers:
        meta = yaml.safe_load(layer.path.read_text(encoding="utf-8"))["meta"]
        assert meta["padding_assumed_m"] == p["world_padding"], layer.name
    # and for the robot it plans with: the keep-out gate and the person's carve
    # measure from the fixed spheres of the packaged robot config that
    # scripts/mebot_layers.yaml names, so another robot YAML voids both
    inputs = Path(__file__).resolve().parents[2] / "scripts" / "mebot_layers.yaml"
    keep_out = yaml.safe_load(inputs.read_text(encoding="utf-8"))["keep_out"]
    robot = resolve_config(cfg["robot"], relative_to=folder)
    assert robot == PACKAGED_CONFIG_DIR / keep_out["robot_config"]


def test_chair_startup_world_is_empty_to_a_v100_loader():
    path = PACKAGED_CONFIG_DIR / "world_mebot.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # v1.0.0's load_scene turned obstacles and objects into boxes and ignored
    # `layers:`; Obstacle, SceneObject and world_cuboids are still v1.0.0's
    v100 = Scene(
        data.get("base_frame", "base_link"),
        [Obstacle(o) for o in data.get("obstacles", [])],
        [],
        [SceneObject(o) for o in data.get("objects", [])],
    )
    # nothing: a 1.0.0 planner given the chair config refuses an empty world
    # instead of planning without the chair
    assert world_cuboids(v100) == {}
    # this planner's start-up world is the floor that the scene lists
    assert [layer.name for layer in load_scene(path).layers] == ["floor"]


def test_config_layers_reach_every_world_input(tmp_path, monkeypatch):
    inits, sent, loaded = _no_gpu(monkeypatch)
    planner = CuRoboPlanner.from_config("gen3_mebot.yaml")
    # the config's layers are loaded once, at start-up, before any GPU work
    assert inits == [True]
    assert [(lay.name, lay.excludable, lay.path) for lay in planner._layers] == [
        ("platform", False, PACKAGED_CONFIG_DIR / "layer_mebot_platform.yaml"),
        ("person", True, PACKAGED_CONFIG_DIR / "layer_mebot_person.yaml"),
    ]

    def installed():
        """The world the planner says it holds, by source, checked to be the
        one that reached cuRobo and that MotionGen loaded."""
        boxes = planner.world_boxes
        assert sent[-1] == {
            "cuboid": {b["name"]: {"dims": b["dims"], "pose": b["pose"]} for b in boxes}
        }
        assert loaded[-1].data is sent[-1]
        return Counter(box["layer"] for box in boxes)

    # a push replaces the scene and never the config's layers, in every form a
    # push takes: a name, a path, a Scene and a list
    kitchen = PACKAGED_CONFIG_DIR / "world_sim_kitchen.yaml"
    scene = load_scene(kitchen)
    for world in ("world_sim_kitchen.yaml", kitchen, scene):
        planner.update_world(world)
        assert installed() == {"scene": 26, "platform": 16, "person": 8}
    # the scene the planner says it holds is the one pushed last
    assert planner.scene is scene
    planner.update_world([_box("crate"), _box("mug")])
    assert installed() == {"scene": 2, "platform": 16, "person": 8}
    # the floor is the start-up scene's, and comes back with it
    planner.update_world("world_mebot.yaml")
    assert installed() == {"platform": 16, "person": 8, "floor": 1}
    assert planner.world_summary() == (
        "25 boxes: scene 0 | config layers: platform 16, person 8 "
        "| scene layers: floor 1"
    )

    # the person, an excludable layer, can be left out of one world, and only
    # its boxes go; a bare string is one name
    whole = planner.world_boxes
    for ignore in ({"person"}, "person"):
        planner.update_world("world_mebot.yaml", ignore=ignore)
        assert planner.world_boxes == [b for b in whole if b["layer"] != "person"]
    planner.update_world("world_mebot.yaml")
    assert planner.world_boxes == whole

    def held():
        """How many worlds reached cuRobo, and the world the planner says it
        holds."""
        return (
            len(sent),
            len(loaded),
            planner.scene,
            planner.world_boxes,
            planner.world_summary(),
        )

    # a push with nothing of its own is refused, a list as much as a file: the
    # chair alone is not a world anyone pushed. So is one that overflows the
    # cache beside the chair, and nothing is cut to fit. A refused push reaches
    # neither cuRobo nor what the planner says it holds
    before = held()
    for world in ([], Scene("base_link", [], [])):
        with pytest.raises(ValueError) as exc:
            planner.update_world(world)
        assert str(exc.value) == CHAIR_ONLY
    with pytest.raises(ValueError) as exc:
        planner.update_world([_box("b%d" % i) for i in range(37)])
    assert str(exc.value) == (
        "61 collision boxes > collision_cache_obb=60 (scene 37, platform 16, "
        "person 8); raise the cache setting in the planner config or shrink a layer"
    )
    with pytest.raises(ValueError, match="layer 'platform' is not excludable"):
        planner.update_world("world_mebot.yaml", ignore={"platform"})
    assert held() == before

    # a deployment mounts its own copy of the config (/config/gen3_mebot.yaml),
    # naming the layers the package ships: those are the layers, whichever
    # directory the planner starts in, and a file of the same name there is
    # not read
    mounted, cwd = tmp_path / "config", tmp_path / "cwd"
    mounted.mkdir()
    cwd.mkdir()
    text = (PACKAGED_CONFIG_DIR / "gen3_mebot.yaml").read_text(encoding="utf-8")
    (mounted / "gen3_mebot.yaml").write_text(text, encoding="utf-8")
    _layer(cwd / "layer_mebot_person.yaml", "person", boxes=("stray",), excludable=True)
    monkeypatch.chdir(cwd)
    deployed = CuRoboPlanner.from_config(mounted / "gen3_mebot.yaml")
    assert [lay.path for lay in deployed._layers] == [
        lay.path for lay in planner._layers
    ]
    deployed.update_world([_box("crate")])
    assert Counter(box["layer"] for box in deployed.world_boxes) == {
        "scene": 1,
        "platform": 16,
        "person": 8,
    }

    # and a config whose start-up world overflows the cache is refused before
    # any GPU work
    (mounted / "small.yaml").write_text(
        text + "planner: {collision_cache_obb: 24}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError) as exc:
        CuRoboPlanner.from_config(mounted / "small.yaml")
    assert str(exc.value) == (
        "25 collision boxes > collision_cache_obb=24 (platform 16, person 8, "
        "floor 1); raise the cache setting in the planner config or shrink a layer"
    )
    assert inits == [True, True]


def test_the_chair_warms_up_in_its_scene_and_then_loads_its_layers(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="rammp_curobo")
    stand_in = _curobo_stand_in(monkeypatch)
    # as on the chair, cuRobo finds the retract infeasible
    stand_in.feasible = False
    planner = CuRoboPlanner.from_config("gen3_mebot.yaml")
    whole = [box["name"] for box in planner.world_boxes]
    assert Counter(box["layer"] for box in planner.world_boxes) == {
        "platform": 16,
        "person": 8,
        "floor": 1,
    }
    # warm-up plans from the retract, which is in the person's lap: MotionGen
    # is built in the scene's own world, the floor, warms up there, and only
    # then loads the world the planner says it holds
    assert stand_in.events == [
        ("built", ["floor/floor"]),
        ("warmup",),
        ("loaded", whole),
    ]
    assert planner._warm_boxes == 1
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (
        logging.INFO,
        "warming up in the scene-only world (1 box); config layers load after "
        "warm-up",
    ) in messages
    # the spheres no joint moves are the ones the generator keeps every layer
    # clear of, and not the one joint_1 swings into the lap
    assert planner._fixed_spheres.tolist() == stand_in.fixed.tolist()
    # the retract is cuRobo's IK seed, and the start-up log says why it is no
    # valid start here: the 5 cm sphere at the lap's centre is in it by its
    # radius and the half-width to the lap's nearest padded face
    lap = next(box for box in planner.world_boxes if box["name"] == "person/lap")
    assert [m for level, m in messages if level == logging.WARNING] == [
        "the IK seed (retract_config) is not a valid state in this world: "
        "constraint violation at state; collides with person/lap (%.0f mm) — "
        "plans still work from valid starts; a chair seed is #20"
        % (1e3 * (0.05 + min(lap["dims"]) / 2))
    ]

    # without a warm-up nothing plans before the layers are in, so MotionGen is
    # built in the whole world
    text = (PACKAGED_CONFIG_DIR / "gen3_mebot.yaml").read_text(encoding="utf-8")
    (tmp_path / "cold.yaml").write_text(
        text + "planner: {warmup: false}\n", encoding="utf-8"
    )
    stand_in.events.clear()
    cold = CuRoboPlanner.from_config(tmp_path / "cold.yaml")
    assert stand_in.events == [("built", whole)]
    assert cold._warm_boxes == 25

    # a padding the layers were not generated for grows the person into the
    # arm's base, where every start state would collide: the planner refuses
    # to start, before warm-up
    (tmp_path / "wide.yaml").write_text(
        text + "planner: {world_padding: 0.03}\n", encoding="utf-8"
    )
    stand_in.events.clear()
    with pytest.raises(ValueError) as exc:
        CuRoboPlanner.from_config(tmp_path / "wide.yaml")
    assert str(exc.value) == (
        "layer box person/torso_head_3 intrudes 4.9 mm into the arm's fixed "
        "base/shoulder spheres (world_padding 0.030 m): every start state would "
        "be infeasible; the layers were generated for 0.02 m — regenerate them "
        "or restore world_padding"
    )
    assert stand_in.events == [("built", ["floor/floor"])]


def test_a_layer_box_in_the_arms_fixed_spheres_is_refused_and_the_world_kept(
    tmp_path, monkeypatch
):
    stand_in = _curobo_stand_in(monkeypatch)
    planner = CuRoboPlanner.from_config("gen3_mebot.yaml")
    (tmp_path / "block.yaml").write_text(
        "layer: block\nobstacles:\n"
        "  - {name: over_base, position: [0, 0, 0.2], dims: [0.05, 0.05, 0.05]}\n"
    )
    (tmp_path / "scene.yaml").write_text(
        "obstacles:\n"
        "  - {name: crate, position: [1.5, 1.5, 0.5], dims: [0.2, 0.2, 0.2]}\n"
        "layers: [block.yaml]\n"
    )

    # a layer box over the arm's base would put every start state in
    # collision: refused, before cuRobo or the planner's record of its world
    # changes. At the padding the layers were made for, the box is to blame
    before = (len(stand_in.events), planner.scene, planner.world_boxes)
    with pytest.raises(ValueError) as exc:
        planner.update_world(tmp_path / "scene.yaml")
    assert str(exc.value) == (
        "layer box block/over_base intrudes 116.4 mm into the arm's fixed "
        "base/shoulder spheres (world_padding 0.020 m): every start state would "
        "be infeasible; move or shrink that box, or regenerate its layer"
    )
    assert (len(stand_in.events), planner.scene, planner.world_boxes) == before
    # and at any other padding, since the box reaches the spheres at that one
    # too: below it, restoring the padding would only grow the box, and above
    # it, would not shrink the box enough. At 0.022 the chair's own boxes grow
    # 2 mm, within the 4 mm the generator keeps them from the spheres, so the
    # planner still starts
    text = (PACKAGED_CONFIG_DIR / "gen3_mebot.yaml").read_text(encoding="utf-8")
    for padding in (0.01, 0.022):
        (tmp_path / "padded.yaml").write_text(
            text + "planner: {world_padding: %g}\n" % padding, encoding="utf-8"
        )
        padded = CuRoboPlanner.from_config(tmp_path / "padded.yaml")
        with pytest.raises(ValueError) as exc:
            padded.update_world(tmp_path / "scene.yaml")
        assert str(exc.value).endswith(
            "(world_padding %.3f m): every start state would be infeasible; move "
            "or shrink that box, or regenerate its layer" % padding
        )

    # the same box as the scene's own is cuRobo's to judge, as in v1.0.0
    planner.update_world(
        [_box("over_base", position=[0, 0, 0.2], dims=[0.05, 0.05, 0.05])]
    )
    assert stand_in.events[-1] == ("loaded", [b["name"] for b in planner.world_boxes])
    assert "obj_over_base" in stand_in.events[-1][1]

    # and a world cuRobo fails to load is not the world the planner says it
    # holds
    def out_of_memory(world):
        raise RuntimeError("CUDA error: out of memory")

    planner._motion_gen.update_world = out_of_memory
    before = (planner.scene, planner.world_boxes)
    with pytest.raises(RuntimeError, match="out of memory"):
        planner.update_world("world_sim_kitchen.yaml")
    assert (planner.scene, planner.world_boxes) == before


def test_attribution_explains_an_answer_and_never_changes_or_hides_it(
    tmp_path, monkeypatch
):
    stand_in = _curobo_stand_in(monkeypatch)
    planner = CuRoboPlanner.from_config("gen3_mebot.yaml")
    # a prop and a layer box, far from the chair, so that every depth below
    # is plain arithmetic: padded, each box reaches 0.12 m from its centre
    (tmp_path / "block.yaml").write_text(
        "layer: block\nobstacles:\n"
        "  - {name: wall, position: [1, 0.27, 0.5], dims: [0.2, 0.2, 0.2]}\n"
    )
    (tmp_path / "scene.yaml").write_text(
        "objects:\n"
        "  - {name: crate, position: [1, 0, 0.5], dims: [0.2, 0.2, 0.2]}\n"
        "layers: [block.yaml]\n"
    )
    planner.update_world(tmp_path / "scene.yaml")
    world = (
        " [world: 26 boxes: scene 1 | config layers: platform 16, person 8 "
        "| scene layers: block 1]"
    )
    # a 3 cm sphere 10 cm from the crate's centre reaches 50 mm in, and one at
    # the wall's centre 150 mm: the deeper is named first, though the world
    # lists the crate first
    reaching = np.array([[1.1, 0, 0.5, 0.03], [1, 0.27, 0.5, 0.03]])
    hits = "block/wall (150 mm), obj_crate (50 mm)"
    # one 2 m above the arm's base reaches into no box
    clear = np.array([[0.0, 0.0, 2.0, 0.03]])
    # an infinite joint value gives spheres FK cannot place
    lost = np.array([[math.nan, 0.0, 0.0, 0.03]])
    unplaced = "spheres must be finite, got [nan, 0.0, 0.0, 0.03]"
    q = [0.0] * 7

    # an infeasible state is named by the boxes it reaches into
    stand_in.spheres = lambda q: reaching
    stand_in.feasible = False
    assert planner.check_state_valid(q) == (
        False,
        "constraint violation at state; collides with " + hits,
    )
    # the names only explain cuRobo's verdict: a state it finds feasible is
    # valid whatever the boxes, and one it does not stays infeasible when it
    # reaches into no box (a self-collision, a joint limit) and when its boxes
    # cannot be named, without raising
    stand_in.feasible = True
    assert planner.check_state_valid(q) == (True, "")
    stand_in.feasible = False
    stand_in.spheres = lambda q: clear
    assert planner.check_state_valid(q) == (False, "constraint violation at state")
    stand_in.spheres = lambda q: lost
    assert planner.check_state_valid(q) == (
        False,
        "constraint violation at state (attribution unavailable: %s)" % unplaced,
    )

    def note(status, **kw):
        """What a failed plan's error says after cuRobo's status explained."""
        result = _failed(status)
        res = planner._with_failure_note(planner._finish(result, 0.0), result, **kw)
        explained = planner._explain(status)
        assert (res.success, res.status) == (False, status)
        assert res.error.startswith(explained), res.error
        return res.error[len(explained) :]

    # a failed plan's error adds what the world has to do with it: the boxes
    # its start reaches into, when cuRobo refused the start, ...
    stand_in.spheres = lambda q: reaching
    refused = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
    assert note(refused, start=q) == " [start collides with %s]%s" % (hits, world)
    stand_in.spheres = lambda q: clear
    assert note(refused, start=q) == (
        " [start: no world box penetrated — self-collision or joint limits]" + world
    )
    # ... the boxes a joint goal reaches into, ...
    stand_in.spheres = lambda q: reaching
    trajopt = "MotionGenStatus.TRAJOPT_FAIL"
    assert note(trajopt, start=q, q_goal=q) == (
        " [goal collides with %s]%s" % (hits, world)
    )
    # ... the boxes around a pose goal's tool point, when no arm could hold it
    # there, ...
    ik = "MotionGenStatus.IK_FAIL"
    assert note(ik, start=q, goal_point=[1, 0.11, 0.5]) == (
        " [goal point inside obj_crate; within 5 cm of block/wall] (heuristic: "
        "tool point only)" + world
    )
    # ... and, always, where the world's boxes come from
    assert note(trajopt, start=q) == world
    # it never raises: a failure is reported, whatever its explanation, and
    # where the world's boxes come from, which needs no FK, is said all the same
    stand_in.spheres = lambda q: lost
    res = planner._with_failure_note(
        planner._finish(None, 0.0), None, start=q, q_goal=q
    )
    assert (res.success, res.status, res.error) == (
        False,
        "no result",
        "planning failed (no result) (attribution unavailable: %s)%s"
        % (unplaced, world),
    )


def test_a_pose_goal_is_attributed_at_the_tool_point_curobo_is_given(
    tmp_path, monkeypatch
):
    _curobo_stand_in(monkeypatch)
    # cuRobo's tool frame stops 10 cm short of the fingertip a pose goal
    # names, along the tool's z axis
    text = (PACKAGED_CONFIG_DIR / "gen3_mebot.yaml").read_text(encoding="utf-8")
    (tmp_path / "tipped.yaml").write_text(
        text + "tool: {tip_offset_m: 0.1}\n", encoding="utf-8"
    )
    planner = CuRoboPlanner.from_config(tmp_path / "tipped.yaml")
    planner.update_world([_box("crate", position=[1, 0, 0.5], dims=[0.2, 0.2, 0.2])])
    ik = "MotionGenStatus.IK_FAIL"
    planner._motion_gen.plan_single = lambda *_: _failed(ik)
    # a fingertip 8 cm above the padded crate, with the tool's z axis up, puts
    # the tool frame cuRobo is asked to reach 2 cm inside it
    res = planner.plan_to_pose([1, 0, 0.7], [0, 0, 0, 1], start=[0.0] * 7)
    assert (res.status, res.error) == (
        ik,
        planner._explain(ik)
        + " [goal point inside obj_crate] (heuristic: tool point only)"
        + " [world: 25 boxes: scene 1 | config layers: platform 16, person 8 "
        "| scene layers: none]",
    )


def test_auto_logs_the_joint_space_failure_it_falls_back_from(monkeypatch, caplog):
    stand_in = _curobo_stand_in(monkeypatch)
    planner = CuRoboPlanner.from_config("gen3_mebot.yaml")
    planner.update_world([_box("crate", position=[1, 0, 0.5], dims=[0.2, 0.2, 0.2])])
    world = (
        " [world: 25 boxes: scene 1 | config layers: platform 16, person 8 "
        "| scene layers: none]"
    )
    # the joint goal puts a 3 cm sphere at the crate's centre, 150 mm into it,
    # and its tool point there too: cuRobo plans neither to the joints nor to
    # their pose
    stand_in.spheres = lambda q: np.array([[1, 0, 0.5, 0.03]])
    planner.fk = lambda q, quat_order: ([1, 0, 0.5], [1, 0, 0, 0])
    trajopt, ik = "MotionGenStatus.TRAJOPT_FAIL", "MotionGenStatus.IK_FAIL"
    planner._motion_gen.plan_single_js = lambda *_: _failed(trajopt)
    planner._motion_gen.plan_single = lambda *_: _failed(ik)

    caplog.clear()
    res = planner.plan_to_joints([0.1] * 7, start=[0.0] * 7)
    # 'auto', the default, returns the fallback's failure, which knows only the
    # pose: the joint-space failure, which names the boxes the joint goal
    # reaches into, is logged
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "plan_single_js failed (%s: %s [goal collides with obj_crate (150 mm)]%s) "
        "— falling back to FK-pose planning"
        % (trajopt, planner._explain(trajopt), world)
    ]
    assert (res.status, res.error) == (
        ik,
        planner._explain(ik)
        + " [goal point inside obj_crate] (heuristic: tool point only)"
        + world,
    )

    # every state puts the sphere in the crate, the start too: when cuRobo
    # refuses the start, each failure names the box the start reaches into
    refused = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
    planner._motion_gen.plan_single_js = lambda *_: _failed(refused)
    planner._motion_gen.plan_single = lambda *_: _failed(refused)
    caplog.clear()
    res = planner.plan_to_joints([0.1] * 7, start=[0.0] * 7)
    start = " [start collides with obj_crate (150 mm)]"
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "plan_single_js failed (%s: %s%s [goal collides with obj_crate (150 mm)]%s) "
        "— falling back to FK-pose planning"
        % (refused, planner._explain(refused), start, world)
    ]
    assert (res.status, res.error) == (
        refused,
        planner._explain(refused) + start + world,
    )


def test_a_subclass_overriding_the_v100_finish_plans_and_is_explained(monkeypatch):
    stand_in = _curobo_stand_in(monkeypatch)
    calls = []

    # RAMMP-generalized subclasses the planner and overrides _finish with its
    # v1.0.0 signature, calling super() (rammp_adl/motion/curobo_stationary.py)
    class Stationary(CuRoboPlanner):
        def _finish(self, result, t0, q_goal=None):
            res = super()._finish(result, t0, q_goal=q_goal)
            calls.append((q_goal, res))
            return res

    planner = Stationary.from_config("gen3_mebot.yaml")
    planner.update_world([_box("crate", position=[1, 0, 0.5], dims=[0.2, 0.2, 0.2])])
    start, goal = [0.0] * 7, [0.1] * 7
    # the joint goal's tool point is the crate's centre
    planner.fk = lambda q, quat_order: ([1, 0, 0.5], [1, 0, 0, 0])
    plans = {
        "pose": lambda: planner.plan_to_pose([1, 0, 0.5], [0, 0, 0, 1], start=start),
        "js": lambda: planner.plan_to_joints(goal, start=start, method="js"),
        "fk_pose": lambda: planner.plan_to_joints(goal, start=start, method="fk_pose"),
    }

    def through_the_override(name):
        """Plan `name`, which must call the override once, as v1.0.0 did (a
        joint goal only from plan_single_js), and return what it returns."""
        calls.clear()
        res = plans[name]()
        handed = goal if name == "js" else None
        assert [(q, r is res) for q, r in calls] == [(handed, True)], name
        return res

    # cuRobo plans: the plan comes back, and a joint goal is measured against
    # where it ends
    planned = _planned(np.linspace(start, goal, 5))
    planner._motion_gen.plan_single = lambda *_: planned
    planner._motion_gen.plan_single_js = lambda *_: planned
    for name in plans:
        res = through_the_override(name)
        assert (res.success, res.status, res.error) == (True, "OK", None), name
        assert res.final_joints.tolist() == goal, name
        assert res.goal_mismatch_rad == (None if name == "pose" else 0.0), name

    # cuRobo fails, and every state puts a 3 cm sphere at the crate's centre,
    # 150 mm into it: each error still says what the world has to do with it
    stand_in.spheres = lambda q: np.array([[1, 0, 0.5, 0.03]])
    refused = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
    ik = "MotionGenStatus.IK_FAIL"
    planner._motion_gen.plan_single = lambda *_: _failed(ik)
    planner._motion_gen.plan_single_js = lambda *_: _failed(refused)
    world = (
        " [world: 25 boxes: scene 1 | config layers: platform 16, person 8 "
        "| scene layers: none]"
    )
    crate = "obj_crate (150 mm)"
    point = " [goal point inside obj_crate] (heuristic: tool point only)"
    reached = " [start collides with %s] [goal collides with %s]" % (crate, crate)
    failures = (("pose", ik, point), ("js", refused, reached), ("fk_pose", ik, point))
    for name, status, note in failures:
        res = through_the_override(name)
        assert (res.success, res.status, res.error) == (
            False,
            status,
            planner._explain(status) + note + world,
        ), name

    # cuRobo plans, and post-validation refuses the plan, which jumps: that
    # error says why, and the world has nothing to add to it
    jump = _planned([start, goal])
    planner._motion_gen.plan_single = lambda *_: jump
    planner._motion_gen.plan_single_js = lambda *_: jump
    for name in plans:
        res = through_the_override(name)
        assert (res.success, res.status, res.error) == (
            False,
            "LIBRARY_VALIDATION_FAILED",
            "plan rejected by post-validation: discontinuity: joint_1 jumps 0.100 "
            "rad between points 0->1 (allowed 0.060 at dt=0.020s) — refusing; this "
            "is the padded-tail failure mode",
        ), name

    # an override may make what it likes of cuRobo's failure, a plan of its
    # own too: that is the plan's result, and there is nothing to explain
    planner._motion_gen.plan_single = lambda *_: _failed(ik)
    planner._motion_gen.plan_single_js = lambda *_: _failed(refused)
    rescued = PlanResult(success=True, joint_traj=None, timing=0.0, error=None)
    monkeypatch.setattr(Stationary, "_finish", lambda *_, **__: rescued)
    for name in plans:
        assert plans[name]() is rescued, name
    assert rescued.error is None

    # and a failure it reports without an error, which PlanResult allows, is
    # still explained: what the world has to do with it is the whole error
    def unexplained(self, result, t0, q_goal=None):
        return PlanResult(
            success=False, joint_traj=None, timing=0.0, error=None, status=result.status
        )

    monkeypatch.setattr(Stationary, "_finish", unexplained)
    for name, status, note in failures:
        res = plans[name]()
        assert (res.success, res.status, res.error) == (
            False,
            status,
            note + world,
        ), name

    # and one it reports under a status of its own is explained by cuRobo's:
    # the start cuRobo refused and the goal point no arm could hold are named
    # all the same
    def relabelled(self, result, t0, q_goal=None):
        res = super(Stationary, self)._finish(result, t0, q_goal=q_goal)
        return PlanResult.failure("ADL_PLAN_FAILED", res.error)

    monkeypatch.setattr(Stationary, "_finish", relabelled)
    for name, status, note in failures:
        res = plans[name]()
        assert (res.success, res.status, res.error) == (
            False,
            "ADL_PLAN_FAILED",
            planner._explain(status) + note + world,
        ), name

    # but a plan cuRobo found and the override refuses, as RAMMP-generalized's
    # does when the trajectory it builds fails its own checks, comes back as
    # the override made it: cuRobo failed nothing the world could explain
    def refusing(self, result, t0, q_goal=None):
        assert super(Stationary, self)._finish(result, t0, q_goal=q_goal).success
        return PlanResult.failure("INTERPOLATION_VALIDATION_FAILED", "jerk")

    monkeypatch.setattr(Stationary, "_finish", refusing)
    planner._motion_gen.plan_single = lambda *_: planned
    planner._motion_gen.plan_single_js = lambda *_: planned
    for name in plans:
        res = plans[name]()
        assert (res.success, res.status, res.error) == (
            False,
            "INTERPOLATION_VALIDATION_FAILED",
            "jerk",
        ), name
