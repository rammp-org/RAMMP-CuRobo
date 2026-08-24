"""Fiducial pose -> standoff goal, including a synthetic detect round trip."""

import numpy as np

from rammp_curobo_ros.grasps import quat_to_mat3
from rammp_curobo_ros.tags import look_at_pose


def test_look_at_pose_stands_off_along_the_normal_and_faces_the_tag():
    tag = np.array([0.70, 0.0, 0.25])
    rot = np.stack([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]], axis=1)  # +Z = -x
    pos, quat = look_at_pose(tag, rot, 0.15)
    # standoff moves TOWARD the arm along the tag normal
    assert np.allclose(pos, [0.55, 0.0, 0.25], atol=1e-9)
    # tool z looks back at the tag
    assert np.allclose(quat_to_mat3(quat)[:, 2], [1.0, 0.0, 0.0], atol=1e-9)


def test_look_at_pose_handles_a_tag_lying_flat():
    tag = np.array([0.65, 0.0, 0.0])
    rot = np.eye(3)  # +Z straight up
    pos, quat = look_at_pose(tag, rot, 0.20)
    assert np.allclose(pos, [0.65, 0.0, 0.20], atol=1e-9)
    r = quat_to_mat3(quat)
    assert np.allclose(r[:, 2], [0.0, 0.0, -1.0], atol=1e-9)  # looks down
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_detect_recovers_a_synthetic_tag_pose():
    """Render a marker at a known pose; tags.tag_pose_from_frame recovers it."""
    cv2 = __import__("cv2")
    from rammp_curobo_ros import tags

    size, dist_m = 0.06, 0.40
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 0, 400)
    pad = 120
    img = np.full((400 + 2 * pad, 400 + 2 * pad), 255, np.uint8)
    img[pad:pad + 400, pad:pad + 400] = marker
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # a pinhole that would see a `size` marker as 400 px at dist_m
    h, w = img.shape[:2]
    fx = fy = 400.0 / size * dist_m
    k = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]])

    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    s = size / 2.0
    objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                    dtype=np.float64)
    dist = np.zeros(5)

    res = tags.tag_pose_from_frame(img, det, objp, k, dist)
    assert res is not None
    tag_id, r_tag_cam, tvec = res
    assert int(tag_id) == 0
    t = np.asarray(tvec, dtype=float).ravel()
    # Range error tracks the marker-size error linearly: the rendered
    # square spans 399 px not 400 (pixel-edge convention), a 0.25% scale
    # error -> 1 mm at 0.4 m. Same arithmetic applies to a MISMEASURED
    # printed tag, which is why make_tag.py insists you measure it.
    assert abs(float(t[2]) - dist_m) < 0.0025
    assert abs(float(t[0])) < 0.001 and abs(float(t[1])) < 0.001
    # facing the camera: tag +Z points back down the optical axis
    r = np.asarray(r_tag_cam, dtype=float)
    assert abs(abs(float(r[2, 2])) - 1.0) < 1e-3

    # id filtering: the wrong id must yield None, the right one a hit
    assert tags.tag_pose_from_frame(img, det, objp, k, dist, tag_id=7) is None
    assert tags.tag_pose_from_frame(img, det, objp, k, dist, tag_id=0) is not None
