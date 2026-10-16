"""The attach_spheres request -> spheres mapping (issue #23) -- no ROS runtime, no GPU."""

from types import SimpleNamespace as NS

import pytest

try:
    from rammp_curobo_ros.planner_node import spheres_from_request
except ImportError:  # pragma: no cover
    pytest.skip(
        "ROS message packages not on PYTHONPATH (source ROS 2 first)",
        allow_module_level=True,
    )


def req(centers, radii):
    return NS(centers=[NS(x=c[0], y=c[1], z=c[2]) for c in centers], radii=list(radii))


def test_spheres_map_in_order():
    s, why = spheres_from_request(req([(0, 0, 0.1), (0, 0, 0.15)], [0.01, 0.02]))
    assert why is None and s == [[0, 0, 0.1, 0.01], [0, 0, 0.15, 0.02]]


def test_empty_request_is_a_detach():
    assert spheres_from_request(req([], [])) == ([], None)


def test_mismatched_lengths_refused():
    s, why = spheres_from_request(req([(0, 0, 0.1)], [0.01, 0.02]))
    assert s is None and "centres" in why


@pytest.mark.parametrize("radius", [0.0, -0.01, float("nan")])
def test_bad_radius_refused(radius):
    s, why = spheres_from_request(req([(0, 0, 0.1)], [radius]))
    assert s is None
