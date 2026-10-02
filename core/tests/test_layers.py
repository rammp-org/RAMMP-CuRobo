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

    with pytest.raises(ValueError) as exc:
        load("layer: [a.yaml]\n")
    assert str(exc.value) == (
        "%s: 'layer' is not a planner-config key; "
        "did you mean the top-level 'layers:'?" % path
    )
    with pytest.raises(ValueError) as exc:
        load("planner:\n  layers: [a.yaml]\n")
    assert str(exc.value) == (
        "%s: 'planner.layers' is not a planner-config key; "
        "did you mean the top-level 'layers:'?" % path
    )
    for extra in ('layers: "a.yaml"\n', "layers: [1]\n", "layers:\n"):
        with pytest.raises(ValueError, match="'layers' must be a list") as exc:
            load(extra)
        assert str(path) in str(exc.value), extra


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
