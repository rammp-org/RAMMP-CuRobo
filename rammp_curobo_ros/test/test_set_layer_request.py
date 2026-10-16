"""The SetLayer request -> the core's set_layer arguments (runtime layers, #32) -- no ROS runtime, no GPU."""

from types import SimpleNamespace as NS

import pytest

try:
    from rammp_curobo_ros.planner_node import layer_from_request
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )


def req(layer, obstacles, padding_m=-1.0):
    return NS(layer=layer, obstacles=[NS(**o) for o in obstacles], padding_m=padding_m)


BOX = dict(
    name="table",
    position=[0.4, 0.0, -0.05],
    rpy_deg=[0.0, 0.0, 30.0],
    dims=[0.6, 0.8, 0.05],
)


def test_boxes_map_as_a_layer_file_writes_them():
    name, entries, padding = layer_from_request(req("room", [BOX], 0.01))
    assert (name, padding) == ("room", 0.01)
    assert entries == [BOX] and all(
        isinstance(v, float)
        for k in ("position", "rpy_deg", "dims")
        for v in entries[0][k]
    )


def test_a_negative_padding_is_the_planners_own_and_zero_is_exact():
    assert layer_from_request(req("room", [BOX]))[2] is None
    assert layer_from_request(req("room", [BOX], -5.0))[2] is None
    assert layer_from_request(req("room", [BOX], 0.0))[2] == 0.0


def test_no_obstacles_is_a_removal():
    assert layer_from_request(req("room", [])) == ("room", [], None)
