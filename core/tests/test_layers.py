"""Obstacle layers, offline: no torch, no cuRobo, no ROS, so they run in CI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import rammp_curobo
from rammp_curobo.config import PACKAGED_CONFIG_DIR, load_planner_config, resolve_config
from rammp_curobo.scene import (
    Obstacle,
    load_layer,
    load_layers,
    load_scene,
    scene_from_obstacles,
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


def _layer(path, name):
    """Write a one-box layer file that declares `name`: the name tells one
    copy of a file from another."""
    path.write_text(
        "layer: %s\n"
        "obstacles:\n"
        "  - {name: box, position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]}\n" % name
    )
    return path


def _scene(path, *refs):
    """Write a scene whose only content is the layers it lists."""
    path.write_text("layers: [%s]\n" % ", ".join(str(ref) for ref in refs))
    return path


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
        "duplicate_obstacles_in_a_merged_mapping",
        "layer: block\n"
        "<<:\n"
        "  obstacles:\n"
        "    - {name: early, position: [0, 0, 1], dims: [1, 1, 1]}\n"
        "  obstacles:\n"
        "    - {name: late, position: [0, 0, 1], dims: [1, 1, 1]}\n",
        "duplicate key 'obstacles' on line 5",
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
    # a merge key is not a second copy of the keys it brings in, and a key
    # that overrides one of them is not written twice
    path.write_text(
        "layer: block\n"
        "obstacles:\n"
        "  - &box {name: box, position: [0.5, 0, 0.2], dims: [0.1, 0.2, 0.3]}\n"
        "  - {<<: *box, name: twin, position: [0.5, 0.4, 0.2]}\n"
    )
    box, twin = load_layer(path).obstacles
    assert (twin.name, twin.position, twin.dims) == ("twin", [0.5, 0.4, 0.2], box.dims)

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


def test_missing_layer_is_named_with_every_path_tried(tmp_path):
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
    scene = load_scene(_scene(tmp_path / "scene.yaml", "person.yaml", "person.yaml"))
    assert [layer.path for layer in scene.layers] == [person]

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
    # PYTHONUTF8=0: Python otherwise turns UTF-8 mode on for the C locale.
    # PYTHONPATH: the child must import the tree this process is testing
    env = dict(os.environ, PYTHONUTF8="0", LC_ALL="C")
    tree = str(Path(rammp_curobo.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [tree, env.get("PYTHONPATH")]))
    child = subprocess.run(
        [sys.executable, "-c", code, str(kitchen), str(layer)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout == "11 block\n"
