"""The arm's own robot model, as published on /arm/robot_description.

The planner builds its kinematics from the URDF the robot publishes
(kinova-gen3-ros2's arm_state_publisher), so it plans in `arm_mount` with the
arm's own joint names and limits. This module checks the text is that model
and sanitises it for cuRobo; the caller writes it to a file and passes the
path on. Standard library only — no ROS, no cuRobo.

Sanitising strips <visual>, <collision> and top-level <ros2_control>: the
rendered mesh paths point into /opt/ros/humble/share, which the planner image
does not have, and the baked spheres are the collision model, so meshes are
never needed.

The 2F-85 gripper is required: the robot config locks its joints and puts
collision spheres on its links, so a gripperless model cannot load, and it is
refused here by name rather than failing inside cuRobo.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_LINK = "arm_mount"
EE_LINK = "arm_tool_frame"
ARM_JOINTS = tuple("arm_joint_%d" % i for i in range(1, 8))
GRIPPER_JOINTS = (
    "arm_robotiq_85_left_knuckle_joint",
    "arm_robotiq_85_right_knuckle_joint",
    "arm_robotiq_85_left_inner_knuckle_joint",
    "arm_robotiq_85_right_inner_knuckle_joint",
    "arm_robotiq_85_left_finger_tip_joint",
    "arm_robotiq_85_right_finger_tip_joint",
)


def prepare_urdf(xml):
    """URDF text -> sanitised URDF text, or ValueError naming what is wrong.

    Refuses a model that is not ours before cuRobo sees it: the root link
    must be `arm_mount`, and `arm_tool_frame`, all seven `arm_joint_N` and
    the six 2F-85 `GRIPPER_JOINTS` must exist.
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
    if roots != [ROOT_LINK]:
        raise ValueError(
            "robot model root link is %s, expected %r" % (roots, ROOT_LINK)
        )
    if EE_LINK not in links:
        raise ValueError("robot model has no %r link" % EE_LINK)
    missing = [j for j in ARM_JOINTS + GRIPPER_JOINTS if j not in joints]
    if missing:
        raise ValueError("robot model is missing joints %s" % missing)

    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for el in root.findall("ros2_control"):
        root.remove(el)
    return ET.tostring(root, encoding="unicode")


def write_urdf(xml, directory):
    """Write `xml` to <directory>/robot_description.urdf and return the path."""
    path = Path(directory) / "robot_description.urdf"
    path.write_text(xml)
    return path
