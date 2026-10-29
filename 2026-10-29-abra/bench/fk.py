"""Forward kinematics from the arm URDF (no ROS): used by track.py and analyze.py."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]


class FK:
    """Forward kinematics from the URDF, in its root (arm_mount): the flange
    (the link ending in end_effector_link, what /ee_state reports) and the
    planner's tool frame (arm_tool_frame)."""

    def __init__(self, urdf, tool="arm_tool_frame"):
        root = ET.fromstring(urdf)
        self.by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        flange = [c for c in self.by_child if c.endswith("end_effector_link")]
        if tool not in self.by_child or len(flange) != 1:
            raise SystemExit(f"URDF has no {tool} or no single *end_effector_link: {sorted(self.by_child)}")
        self.root, self._flange = self._chain(flange[0])
        _, self._tool = self._chain(tool)

    def _chain(self, tip):
        chain, link = [], tip
        while link in self.by_child:
            chain.append(self.by_child[link])
            link = self.by_child[link].find("parent").get("link")
        out = []
        for j in reversed(chain):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
            name = j.get("name")
            idx = JOINTS.index(name) if name in JOINTS and j.get("type") != "fixed" else None
            out.append((xyz, R.from_euler("xyz", rpy), axis, idx))
        return link, out

    @staticmethod
    def _run(chain, q):
        q = np.atleast_2d(q)
        p = np.zeros((len(q), 3))
        rot = R.from_quat(np.tile([0.0, 0.0, 0.0, 1.0], (len(q), 1)))
        for xyz, r0, axis, idx in chain:
            p = p + rot.apply(xyz)
            rot = rot * r0
            if idx is not None:
                rot = rot * R.from_rotvec(np.outer(q[:, idx], axis))
        return p, rot

    def flange(self, q):
        """(positions (N, 3), rotations) of the flange for joints q (N, 7)."""
        return self._run(self._flange, q)

    def tool(self, q):
        return self._run(self._tool, q)
