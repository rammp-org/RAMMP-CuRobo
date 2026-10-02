"""Obstacle layers, offline: no torch, no cuRobo, no ROS, so they run in CI."""

import pytest

from rammp_curobo.config import PACKAGED_CONFIG_DIR, load_planner_config, resolve_config


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

    # one message for both misplacements. Each is refused beside a valid
    # `layers:` too: a config that lists the platform and misplaces the person
    # must not load without the person
    for extra in (
        "layer: [a.yaml]\n",
        "planner:\n  layers: [a.yaml]\n",
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
    cwd.mkdir()
    beside.mkdir()
    (cwd / "stray.yaml").write_text("")
    (cwd / "gen3.yaml").write_text("")  # the name of a packaged config
    (beside / "near.yaml").write_text("")
    monkeypatch.chdir(cwd)

    # the v1.0.0 order is still the default: CWD comes before the package
    assert resolve_config("stray.yaml") == cwd / "stray.yaml"
    assert resolve_config("gen3.yaml") == cwd / "gen3.yaml"

    packaged = PACKAGED_CONFIG_DIR / "gen3.yaml"
    assert resolve_config("gen3.yaml", search_cwd=False) == packaged
    near = resolve_config("near.yaml", relative_to=beside, search_cwd=False)
    assert near == beside / "near.yaml"
    with pytest.raises(FileNotFoundError) as exc:
        resolve_config("stray.yaml", relative_to=beside, search_cwd=False)
    assert str(exc.value) == "config 'stray.yaml' not found; tried: %s, %s" % (
        beside / "stray.yaml",
        PACKAGED_CONFIG_DIR / "stray.yaml",
    )
