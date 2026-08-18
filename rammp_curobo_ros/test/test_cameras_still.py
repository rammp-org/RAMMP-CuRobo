"""The wrist-camera stillness gate: pure logic over stubbed TF poses."""

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg

from rammp_curobo_ros.cameras import CamerasNode


class _Cam:
    def __init__(self, parent_frame="end_effector_link", stamp_s=10.0, max_range=0.9):
        self.cfg = {"parent_frame": parent_frame, "max_range": max_range}
        self.ros_stamp = TimeMsg(
            sec=int(stamp_s), nanosec=int(round((stamp_s - int(stamp_s)) * 1e9))
        )


def _bare_node(poses_by_dt):
    """CamerasNode shell whose _camera_pose serves poses keyed by lookback.

    poses_by_dt maps {0.0: (R, t), 0.075: ..., 0.15: ...}; None values
    simulate a strict-TF miss.
    """
    node = object.__new__(CamerasNode)
    node.max_motion_mm = 3.0
    node._moving_skips = 0
    stamp0 = 10.0

    def fake_pose(cfg, stamp=None, strict=False):
        t = stamp.sec + stamp.nanosec * 1e-9
        return poses_by_dt.get(round(stamp0 - t, 3), None)

    node._camera_pose = fake_pose
    return node


STILL = (np.eye(3), np.zeros(3))


def test_fixed_camera_is_always_still():
    node = _bare_node({})
    assert node._camera_still(_Cam(parent_frame="base_link"))


def test_zero_or_epoch_stamp_is_distrusted_not_crashed():
    node = _bare_node({0.0: STILL, 0.075: STILL, 0.15: STILL})
    assert not node._camera_still(_Cam(stamp_s=0.0))  # would underflow
    assert not node._camera_still(_Cam(stamp_s=0.1))  # near-epoch


def test_strict_tf_miss_distrusts_the_frame():
    node = _bare_node({0.0: STILL, 0.075: None, 0.15: STILL})
    assert not node._camera_still(_Cam())


def test_still_poses_pass():
    node = _bare_node({0.0: STILL, 0.075: STILL, 0.15: STILL})
    assert node._camera_still(_Cam())


def test_translation_drift_gates():
    moved = (np.eye(3), np.array([0.01, 0.0, 0.0]))  # 10 mm > 3 mm
    node = _bare_node({0.0: moved, 0.075: STILL, 0.15: STILL})
    assert not node._camera_still(_Cam())
    assert node._moving_skips == 1


def test_pure_rotation_gates_via_point_sweep():
    a = 0.02  # rad; sweep at 0.9 m range = 18 mm >> 3 mm
    rot = np.array(
        [
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    node = _bare_node({0.0: (rot, np.zeros(3)), 0.075: STILL, 0.15: STILL})
    assert not node._camera_still(_Cam())


def test_direction_reversal_is_caught_by_the_midpoint_sample():
    # net displacement over 0.15 s is zero, but the midpoint is 10 mm out
    mid = (np.eye(3), np.array([0.0, 0.01, 0.0]))
    node = _bare_node({0.0: STILL, 0.075: mid, 0.15: STILL})
    assert not node._camera_still(_Cam())
