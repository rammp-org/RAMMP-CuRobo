"""Sensor-contract enforcement (field 2026-08-19: the D405's default
stereo confidence hallucinated depth on the blank bench — 20 phantom
boxes, bottle floating 25 cm up). The wrist YAML must declare the High
Accuracy preset and the enforcement plumbing must build correct
requests and fail soft when the driver is absent."""

import rclpy

from rammp_curobo_ros.cameras import (
    _sensor_param_request,
    ensure_sensor_params,
    load_camera_config,
)


def test_wrist_yaml_declares_high_accuracy_preset():
    cfg = load_camera_config("camera_d405_wrist.yaml")
    spec = cfg["sensor_params"]
    assert spec["node"] == "/d405/d405"
    assert spec["params"]["depth_module.visual_preset"] == 3


def test_request_builder_maps_python_types_to_parameter_types():
    req = _sensor_param_request(
        {"node": "/x", "params": {"i": 3, "b": True, "f": 0.5, "s": "hi"}}
    )
    by = {p.name: p.value for p in req.parameters}
    assert by["i"].type == 2 and by["i"].integer_value == 3
    assert by["b"].type == 1 and by["b"].bool_value is True  # bool BEFORE int
    assert by["f"].type == 3 and by["f"].double_value == 0.5
    assert by["s"].type == 4 and by["s"].string_value == "hi"


def test_malformed_sensor_params_fails_fast_with_the_filename(tmp_path):
    # audit 2026-08-19: a hand-edit typo ('param:' for 'params:') must
    # die at config load with a clear message, not KeyError inside the
    # node's timer and kill the perceived world
    import pytest

    bad = tmp_path / "cam.yaml"
    bad.write_text(
        "depth_topic: /x/depth/image\ninfo_topic: /x/depth/info\n"
        "parent_frame: base_link\nsensor_params:\n  node: /x\n  param: {a: 1}\n"
    )
    with pytest.raises(SystemExit) as e:
        load_camera_config(str(bad))
    assert "sensor_params" in str(e.value) and "cam.yaml" in str(e.value)


def test_ensure_is_noop_without_spec_and_soft_fails_unreachable():
    rclpy.init()
    try:
        node = rclpy.create_node("sensor_params_test")
        try:
            assert ensure_sensor_params(node, {}) is True
            missing = {"sensor_params": {"node": "/no_such_driver", "params": {"x": 1}}}
            assert ensure_sensor_params(node, missing, timeout_s=0.3) is False
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
