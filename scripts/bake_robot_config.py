#!/usr/bin/env python3
"""Bake the RAMMP Gen3 + 2F-85 cuRobo robot config to a standalone YAML.

RAMMP-Kinova patched cuRobo's bundled kinova_gen3.yml IN MEMORY on every
planner start (curobo_planner/planner_node.py:329-423). This repo bakes the
same patches into core/rammp_curobo/configs/robot_gen3_2f85.yaml once (the
packaged config gen3.yaml loads), so the collision model
that plans is the one you can read and diff:

  * inner fingertip pad spheres inflated 0.01 -> 0.02 m (real pad size —
    the stock spheres let "collision-free" paths clip props),
  * an 18-sphere shell on robotiq_arg2f_base_link covering the knuckle
    housing and finger bodies (nearly invisible in the stock model),
  * the arm-link spheres wholesale replaced with the audit-tuned set
    (mesh-vs-sphere protrusion p95 ~0 vs stock's 64 mm worst case, verified
    self-collision-free at home and all scene targets),
  * cspace retract_config rewritten to RETRACT_POSE — the bundled
    retract lives in the OPPOSITE elbow family and makes IK seed flipped,
    winding reconfigurations.

The spheres were audited on NVIDIA's model, so they are authored here in
NVIDIA's link frames and then MOVED (not re-tuned) onto the upstream links
of the arm's own model, the URDF it publishes on /arm/robot_description:
each centre is re-expressed by forward kinematics of both models at q = 0
(NVIDIA's gripper is rigid; ours is locked open at 0), c_up =
T_up^-1 * T_nv * c_nv, with LINK_MAP naming the target link. Count, order
and radii are unchanged. The planner gets the URDF itself at runtime, so the
YAML carries the sentinel urdf_path `from_robot_description`.

Run in the image (needs the pinned cuRobo v0.7.8 content tree and yourdfpy):
    make bake
or, with both installed:
    python3 scripts/bake_robot_config.py [--urdf core/tests/data/arm_level.urdf]
"""

import argparse
import io
from pathlib import Path

import numpy as np
import yaml

from rammp_curobo.config import PLANNER_DEFAULTS
from rammp_curobo.robot_config import URDF_SENTINEL
from rammp_curobo.urdf_model import ARM_JOINTS, EE_LINK, ROOT_LINK

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "core" / "rammp_curobo" / "configs" / "robot_gen3_2f85.yaml"
UPSTREAM_URDF = REPO / "core" / "tests" / "data" / "arm_level.urdf"

# The IK seed baked into cspace.retract_config, and the ONLY place this
# pose is defined. The planner deliberately holds no pose of its own
# (issue #6) — it reads the seed back out of the baked file. cuRobo's
# bundled Gen3 retract sits in the OPPOSITE elbow family, which makes every
# crossing plan wind; joint_3 must stay near +pi.
RETRACT_POSE = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINT_NAMES = list(PLANNER_DEFAULTS["joint_names"])

# NVIDIA link -> the upstream link its spheres move onto. The arm links keep
# their frames and gain the prefix; NVIDIA's rigid 2F-85 links land on the
# upstream robotiq_85 links (left_outer_knuckle only carries a buffer entry).
ARM_LINKS = (
    "base_link",
    "shoulder_link",
    "half_arm_1_link",
    "half_arm_2_link",
    "forearm_link",
    "spherical_wrist_1_link",
    "spherical_wrist_2_link",
    "bracelet_link",
)
LINK_MAP = {link: "arm_" + link for link in ARM_LINKS}
LINK_MAP.update(
    {
        "robotiq_arg2f_base_link": "arm_robotiq_85_base_link",
        "left_outer_finger": "arm_robotiq_85_left_finger_link",
        "right_outer_finger": "arm_robotiq_85_right_finger_link",
        "left_inner_finger_pad": "arm_robotiq_85_left_finger_tip_link",
        "right_inner_finger_pad": "arm_robotiq_85_right_finger_tip_link",
        "left_outer_knuckle": "arm_robotiq_85_left_knuckle_link",
    }
)
PAD_SPHERE_RADIUS = 0.02

# Audit-tuned replacement arm spheres (RAMMP-Kinova planner_node._ARM_SPHERES).
# The base sphere bottom must stay ABOVE the pedestal box top or the robot
# permanently collides with its own mount and every start state is invalid.
ARM_SPHERES = {
    "base_link": [([0, 0, 0.065], 0.075), ([0, 0, 0.125], 0.065)],
    "shoulder_link": [
        ([0, 0, -0.04], 0.07),
        ([0, 0, -0.10], 0.072),
        ([0, 0, -0.16], 0.062),
    ],
    "half_arm_1_link": [
        ([0, 0, 0], 0.062),
        ([0, -0.06, 0], 0.062),
        ([0, -0.12, 0], 0.062),
        ([0, -0.17, 0], 0.06),
    ],
    "half_arm_2_link": [
        ([0, 0, 0], 0.058),
        ([0, 0, -0.07], 0.056),
        ([0, 0, -0.15], 0.056),
        ([0, 0, -0.21], 0.056),
    ],
    "forearm_link": [
        ([0, 0, 0], 0.06),
        ([0, -0.06, 0], 0.058),
        ([0, -0.12, 0], 0.058),
        ([0, -0.17, 0], 0.055),
    ],
    "spherical_wrist_1_link": [([0, 0, 0], 0.06), ([0, 0, -0.085], 0.06)],
    "spherical_wrist_2_link": [([0, 0, 0], 0.055), ([0, -0.085, 0], 0.055)],
    "bracelet_link": [
        ([0, 0, -0.045], 0.045),
        ([0, -0.05, -0.045], 0.045),
        ([0.045, 0, -0.05], 0.036),
    ],
}

# Sphere shell for the whole open 2F-85, on the gripper base link (its frame
# is the flange, z toward the fingertips; the gripper is rigid in this
# model). Rings, not finger-aligned pairs, so coverage holds regardless of
# mounting twist; the grasp gap between the fingertips stays OPEN.
GRIPPER_SHELL = [
    ([0.0, 0.0, 0.055], 0.042),
    ([0.04, 0.0, 0.06], 0.032),
    ([-0.04, 0.0, 0.06], 0.032),
    ([0.0, 0.04, 0.06], 0.032),
    ([0.0, -0.04, 0.06], 0.032),
    ([0.0, 0.0, 0.078], 0.035),
    ([0.045, 0.0, 0.10], 0.026),
    ([-0.045, 0.0, 0.10], 0.026),
    ([0.0, 0.045, 0.10], 0.026),
    ([0.0, -0.045, 0.10], 0.026),
    ([0.045, 0.0, 0.13], 0.022),
    ([-0.045, 0.0, 0.13], 0.022),
    ([0.0, 0.045, 0.13], 0.022),
    ([0.0, -0.045, 0.13], 0.022),
    ([0.045, 0.0, 0.145], 0.018),
    ([-0.045, 0.0, 0.145], 0.018),
    ([0.0, 0.045, 0.145], 0.018),
    ([0.0, -0.045, 0.145], 0.018),
]

HEADER = """\
# RAMMP Gen3 7-DoF + Robotiq 2F-85 cuRobo robot config — GENERATED FILE.
#
# Derived from cuRobo v0.7.8's bundled configs/robot/kinova_gen3.yml
# (Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, used under the cuRobo
# license) with RAMMP-Kinova's field-tuned collision-model patches baked in,
# then moved onto the arm's own (upstream) links by forward kinematics.
# Regenerate with: make bake (scripts/bake_robot_config.py) — do not hand-edit
# sphere values here without re-running the self-collision checks described
# in RAMMP-Kinova's CLAUDE.md.
#
# The robot model is the URDF the arm publishes on /arm/robot_description:
# urdf_path is the sentinel from_robot_description, replaced at load by the
# file the caller wrote it to. base_link is arm_mount; ee_link is
# arm_tool_frame, 0.120 m beyond the wrist flange, roughly the FINGERTIP
# midpoint. The 2F-85 is locked open by its actuated knuckle
# (arm_robotiq_85_left_knuckle_joint); cuRobo holds the five mimics with it.
"""


def _hom(c):
    return np.append(np.asarray(c, dtype=float), 1.0)


def _link_poses(urdf, links, ref):
    """{link: 4x4 pose in `ref`} at q = 0 (gripper open / rigid)."""
    urdf.update_cfg(np.zeros(len(urdf.actuated_joints)))
    return {link: urdf.get_transform(link, ref) for link in links}


def _rotation_deg(T):
    return np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))


def main():
    from curobo.util_file import (
        get_assets_path,
        get_robot_configs_path,
        join_path,
        load_yaml,
    )
    from yourdfpy import URDF

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(UPSTREAM_URDF), help="the arm's URDF")
    args = ap.parse_args()

    cfg = load_yaml(join_path(get_robot_configs_path(), "kinova_gen3.yml"))
    kin = cfg["robot_cfg"]["kinematics"]

    spheres = kin["collision_spheres"]
    if isinstance(spheres, str):
        loaded = load_yaml(join_path(get_robot_configs_path(), spheres))
        spheres = loaded.get("collision_spheres", loaded)
    spheres = {link: [dict(s) for s in ss] for link, ss in spheres.items()}

    for link in ("left_inner_finger_pad", "right_inner_finger_pad"):
        for s in spheres.get(link, []):
            s["radius"] = max(float(s["radius"]), PAD_SPHERE_RADIUS)
    spheres.setdefault("robotiq_arg2f_base_link", []).extend(
        {"center": list(c), "radius": r} for c, r in GRIPPER_SHELL
    )
    for link, ss in ARM_SPHERES.items():
        spheres[link] = [{"center": [float(v) for v in c], "radius": r} for c, r in ss]

    # Both models are posed relative to the arm's base (NVIDIA's base_link
    # IS arm_base_link), so the mount's tilt in --urdf never enters. The arm
    # links share frames (transforms identity to printed precision); the
    # gripper links do not: NVIDIA's gripper base is turned 1.57 rad about z
    # from the flange, the upstream one is not, so every gripper move is a
    # ~90 deg turn of the FRAMES only. The fingers sit on flange +-x in both
    # models (NVIDIA's arg2f lays them on its own base y; the 1.57
    # compensates), so the geometry is the same and a rigid move at q = 0 is
    # valid; the 4-fold-symmetric shell does not care about the turn anyway.
    # The largest move is the pads -> finger tips at 38.6 mm (the frame
    # origins differ; the sphere positions do not).
    nv_urdf = join_path(get_assets_path(), kin["urdf_path"])
    nv = _link_poses(URDF.load(nv_urdf, load_meshes=False), LINK_MAP, "base_link")
    up = _link_poses(
        URDF.load(args.urdf, load_meshes=False), LINK_MAP.values(), "arm_base_link"
    )
    moves = {n: np.linalg.inv(up[u]) @ nv[n] for n, u in LINK_MAP.items()}
    for n, T in moves.items():
        print(
            "%-24s -> %-38s %8.3f mm %8.3f deg"
            % (n, LINK_MAP[n], 1e3 * np.linalg.norm(T[:3, 3]), _rotation_deg(T))
        )

    def moved(link, ss):
        T = moves[link]
        return [
            {
                "center": [
                    round(float(v), 6) + 0.0 for v in (T @ _hom(s["center"]))[:3]
                ],
                "radius": s["radius"],
            }
            for s in ss
        ]

    def renamed(links):
        return [LINK_MAP[n] for n in links]

    kin["base_link"] = ROOT_LINK
    kin["ee_link"] = EE_LINK
    kin["collision_link_names"] = renamed(kin["collision_link_names"])
    kin["collision_spheres"] = {LINK_MAP[n]: moved(n, ss) for n, ss in spheres.items()}
    kin["self_collision_buffer"] = {
        LINK_MAP[n]: b for n, b in kin["self_collision_buffer"].items()
    }
    kin["self_collision_ignore"] = {
        LINK_MAP[n]: renamed(ns) for n, ns in kin["self_collision_ignore"].items()
    }
    kin["urdf_path"] = URDF_SENTINEL
    kin["asset_root_path"] = ""
    kin["mesh_link_names"] = []

    cspace = kin.pop("cspace")
    # The gripper is locked open by its one actuated joint: cuRobo v0.7.8
    # refuses mimic joints in lock_joints and holds the other five through
    # their <mimic>.
    kin["lock_joints"] = {"arm_robotiq_85_left_knuckle_joint": 0.0}
    assert cspace["joint_names"] == ["joint_%d" % i for i in range(1, 8)]
    assert JOINT_NAMES == list(ARM_JOINTS)
    cspace["joint_names"] = JOINT_NAMES
    cspace["retract_config"] = [float(v) for v in RETRACT_POSE]
    kin["cspace"] = cspace

    buf = io.StringIO()
    yaml.safe_dump(cfg, buf, sort_keys=False, default_flow_style=None, width=100)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + buf.getvalue())
    n = sum(len(ss) for ss in spheres.values())
    print("wrote %s (%d links, %d spheres)" % (OUT, len(spheres), n))


if __name__ == "__main__":
    main()
