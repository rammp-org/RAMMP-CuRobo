"""The arm's own robot model, as published on /arm/robot_description.

The planner builds its kinematics from the URDF the robot publishes, so it
plans in that model's root with the arm's own joint names and limits. The
names it expects (root link, tool link, joints, locked joints) come from the
robot file (`robot_config.RobotSpec`), not from this module. This module
checks the text is that model and sanitises it for cuRobo; the caller writes
it to a file and passes the path on. Standard library only -- no ROS, no
cuRobo.

Sanitising strips <visual>, <collision> and top-level <ros2_control>: mesh
paths may point at packages the planner image lacks, and the baked spheres
are the collision model, so meshes are never needed.

It also fills the <limit> cuRobo v0.7.8 requires on every moving joint. A
gripper's mimic joints may carry none (URDF allows that on a continuous
joint), and cuRobo's parser dereferences it even on a locked joint. Only joints
whose motion is defined by another joint (<mimic>) are filled, from the joint
they follow; a limitless joint that follows nothing is refused by name.

A model missing any joint the robot file lists (its cspace joints and the
joints it locks) is refused here by name rather than failing inside cuRobo.

The kinematics below the spec's `arm_base_link` must be the ones the spheres
were moved onto. The bake (scripts/bake_robot_config.py) records every joint
below it from the reference URDF in a JSON file the robot file names
(`model_check.kinematics_reference`), and a model whose joints differ from it
(an origin by more than 1 mm or 0.1 deg, an axis by more than 0.1 deg, a
parent, child, type or mimic, or a joint missing or extra) is refused by name:
an upstream description change would otherwise shift the baked spheres off the
links, silently. The joint above `arm_base_link` is the mount and may vary.
Limit differences are logged, not refused: the limits are the model's to set
(they are what the driver enforces).
"""

import json
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ORIGIN_TOL_M = 1e-3
ANGLE_TOL_DEG = 0.1

log = logging.getLogger("rammp_curobo")


def prepare_urdf(xml, spec):
    """URDF text -> sanitised URDF text, or ValueError naming what is wrong.

    Refuses a model that is not the robot file's before cuRobo sees it: the
    root link must be `spec.root_link`, `spec.ee_link` must exist, every one
    of `spec.joint_names` and `spec.locked_joints` must exist, and the joints
    below `spec.arm_base_link` must match `spec.kinematics_reference`.
    """
    if not xml or not xml.strip():
        raise ValueError("robot model is empty")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError("robot model does not parse as XML: %s" % e) from None

    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name") for joint in root.findall("joint")}
    children = {c.get("link") for c in root.findall("joint/child")}
    roots = sorted(links - children)
    if roots != [spec.root_link]:
        raise ValueError(
            "robot model root link is %s, expected %r" % (roots, spec.root_link)
        )
    if spec.ee_link not in links:
        raise ValueError("robot model has no %r link" % spec.ee_link)
    missing = [j for j in spec.joint_names + spec.locked_joints if j not in joints]
    if missing:
        raise ValueError("robot model is missing joints %s" % missing)
    if spec.arm_base_link not in links:
        raise ValueError(
            "robot model has no %r link (model_check.arm_base_link of %s)"
            % (spec.arm_base_link, spec.path)
        )
    try:
        ref = json.loads(spec.kinematics_reference.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "kinematics reference %s is not JSON: %s" % (spec.kinematics_reference, exc)
        ) from None
    if not ref:
        raise ValueError(
            "kinematics reference %s lists no joints; nothing would be checked"
            % spec.kinematics_reference
        )
    try:
        check_kinematics(
            kinematic_reference(root, spec.arm_base_link), ref, spec.arm_base_link
        )
    except (KeyError, TypeError, AttributeError) as exc:  # the reference's shape
        raise ValueError(
            "kinematics reference %s is malformed: %s %s"
            % (spec.kinematics_reference, type(exc).__name__, exc)
        ) from None

    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for el in root.findall("ros2_control"):
        root.remove(el)
    _fill_mimic_limits(root)
    return ET.tostring(root, encoding="unicode")


def kinematic_reference(root, base_link):
    """{joint: its kinematics} for every joint below `base_link`, from a
    parsed URDF: parent, child, type, origin xyz/rpy, axis (None on a fixed
    joint), mimic and limit, URDF defaults filled in."""
    below, out = {base_link}, {}
    grew = True
    while grew:  # the joints may come in any order
        grew = False
        for j in root.findall("joint"):
            name, parent = j.get("name"), j.find("parent").get("link")
            if parent not in below or name in out:
                continue
            child = j.find("child").get("link")
            below.add(child)
            grew = True
            origin, axis = j.find("origin"), j.find("axis")
            mimic, limit = j.find("mimic"), j.find("limit")
            fixed = j.get("type") == "fixed"
            out[name] = {
                "parent": parent,
                "child": child,
                "type": j.get("type"),
                "xyz": _floats(origin, "xyz"),
                "rpy": _floats(origin, "rpy"),
                "axis": None if fixed else _floats(axis, "xyz", "1 0 0"),
                "mimic": None
                if mimic is None
                else {
                    "joint": mimic.get("joint"),
                    "multiplier": float(mimic.get("multiplier", 1.0)),
                    "offset": float(mimic.get("offset", 0.0)),
                },
                "limit": None
                if limit is None
                else {k: float(v) for k, v in sorted(limit.attrib.items())},
            }
    return out


def check_kinematics(live, ref, base_link):
    """Refuse `live` (kinematic_reference of the model) unless it matches
    `ref` within ORIGIN_TOL_M / ANGLE_TOL_DEG, naming the joint; log limit
    differences."""
    extra, missing = sorted(set(live) - set(ref)), sorted(set(ref) - set(live))
    if extra or missing:
        raise ValueError(
            "robot model joints below %s differ from the reference the spheres "
            "were baked on: missing %s, extra %s" % (base_link, missing, extra)
        )
    for name, want in ref.items():
        got = live[name]
        for key in ("parent", "child", "type", "mimic"):
            if got[key] != want[key]:
                raise ValueError(
                    "robot model joint %r %s is %r, the reference's is %r"
                    % (name, key, got[key], want[key])
                )
        moved_m = math.dist(got["xyz"], want["xyz"])
        turned = _angle_deg(_rotation(got["rpy"]), _rotation(want["rpy"]))
        axis = 0.0 if want["axis"] is None else _axis_deg(got["axis"], want["axis"])
        if moved_m > ORIGIN_TOL_M or turned > ANGLE_TOL_DEG or axis > ANGLE_TOL_DEG:
            raise ValueError(
                "robot model joint %r is %.3f mm / %.3f deg (axis %.3f deg) from "
                "the reference the spheres were baked on (max %g mm / %g deg)"
                % (name, 1e3 * moved_m, turned, axis, 1e3 * ORIGIN_TOL_M, ANGLE_TOL_DEG)
            )
        # a mimic's limit is filled by _fill_mimic_limits, so a model that has
        # been through prepare_urdf once has one where the reference has none
        filled = want["mimic"] is not None and want["limit"] is None
        if got["limit"] != want["limit"] and not filled:
            log.warning(
                "robot model joint %r limit %s differs from the reference's %s; "
                "planning with the model's",
                name,
                got["limit"],
                want["limit"],
            )


def _floats(el, attr, default="0 0 0"):
    text = default if el is None else el.get(attr, default)
    return [float(v) for v in text.split()]


def _rotation(rpy):
    """URDF fixed-axis roll-pitch-yaw -> 3x3 rotation (Rz @ Ry @ Rx)."""
    (cr, sr), (cp, sp), (cy, sy) = ((math.cos(a), math.sin(a)) for a in rpy)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _angle_deg(a, b):
    trace = sum(a[i][k] * b[i][k] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))


def _axis_deg(a, b):
    cos = sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _fill_mimic_limits(root):
    """Give each limitless mimic joint the <limit> of the joint it follows,
    mapped through the mimic: position = multiplier * leader + offset.

    These are only the fields cuRobo requires, on joints whose motion another
    joint defines. Velocity scales by |multiplier| (cuRobo refuses a mimic
    slower than its leader); the range is the mimic's actual range, which
    must contain 0, where the gripper is locked open. A mimic of a limitless
    mimic is filled only if its leader comes first in document order, and a
    leader without lower/upper (continuous) maps to [offset, offset].
    """
    by_name = {j.get("name"): j for j in root.findall("joint")}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if joint.get("type") == "fixed" or joint.find("limit") is not None:
            continue
        mimic = joint.find("mimic")
        if mimic is None:
            raise ValueError("robot model joint %r has no <limit>" % name)
        leader = by_name.get(mimic.get("joint"))
        lead = None if leader is None else leader.find("limit")
        if lead is None:
            raise ValueError(
                "robot model joint %r mimics %r, which has no <limit>"
                % (name, mimic.get("joint"))
            )
        k = float(mimic.get("multiplier", 1.0))
        offset = float(mimic.get("offset", 0.0))
        lo, hi = sorted(
            k * float(lead.get(b, 0.0)) + offset for b in ("lower", "upper")
        )
        if not lo <= 0.0 <= hi:
            raise ValueError(
                "robot model joint %r range [%g, %g] excludes 0" % (name, lo, hi)
            )
        ET.SubElement(
            joint,
            "limit",
            effort=lead.get("effort", "0"),
            lower=repr(lo),
            upper=repr(hi),
            velocity=repr(abs(k) * float(lead.get("velocity", 0.0))),
        )


def write_urdf(xml, directory):
    """Write `xml` to <directory>/robot_description.urdf and return the path."""
    path = Path(directory) / "robot_description.urdf"
    path.write_text(xml)
    return path
