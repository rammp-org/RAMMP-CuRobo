#!/usr/bin/env python3
"""Bake the RAMMP Gen3 + 2F-85 cuRobo robot config to a standalone YAML.

RAMMP-Kinova patched cuRobo's bundled kinova_gen3.yml IN MEMORY on every
planner start (curobo_planner/planner_node.py:329-423). This repo bakes the
same patches into configs/robot_gen3_2f85.yaml once, so the collision model
that plans is the one you can read and diff:

  * inner fingertip pad spheres inflated 0.01 -> 0.02 m (real pad size —
    the stock spheres let "collision-free" paths clip props),
  * an 18-sphere shell on robotiq_arg2f_base_link covering the knuckle
    housing and finger bodies (nearly invisible in the stock model),
  * the arm-link spheres wholesale replaced with the audit-tuned set
    (mesh-vs-sphere protrusion p95 ~0 vs stock's 64 mm worst case, verified
    self-collision-free at home and all scene targets),
  * cspace retract_config rewritten to the RAMMP home pose — the bundled
    retract lives in the OPPOSITE elbow family and makes IK seed flipped,
    winding reconfigurations.

urdf_path/asset_root_path stay relative: cuRobo resolves them against its
own installed content, so no URDF/meshes are vendored here.

Run (needs the pinned cuRobo v0.7.8 installed):
    python3 scripts/bake_robot_config.py
"""

import io
from pathlib import Path

import yaml

OUT = (
    Path(__file__).resolve().parent.parent
    / "core"
    / "rammp_curobo"
    / "configs"
    / "robot_gen3_2f85.yaml"
)

HOME_POSE = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
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
# license) with RAMMP-Kinova's field-tuned collision-model patches baked in.
# Regenerate with: python3 scripts/bake_robot_config.py — do not hand-edit
# sphere values here without re-running the self-collision checks described
# in RAMMP-Kinova's CLAUDE.md.
#
# ee_link is tool_frame: 0.120 m beyond the wrist flange, roughly the
# FINGERTIP midpoint. urdf_path is resolved inside the installed cuRobo
# package (content/assets/...), so this file works with any v0.7.8 install.
"""


def main():
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

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
    kin["collision_spheres"] = spheres

    cspace = kin["cspace"]
    assert cspace["joint_names"] == ["joint_%d" % i for i in range(1, 8)]
    cspace["retract_config"] = [float(v) for v in HOME_POSE]

    buf = io.StringIO()
    yaml.safe_dump(cfg, buf, sort_keys=False, default_flow_style=None, width=100)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + buf.getvalue())
    n = sum(len(ss) for ss in spheres.values())
    print("wrote %s (%d links, %d spheres)" % (OUT, len(spheres), n))


if __name__ == "__main__":
    main()
