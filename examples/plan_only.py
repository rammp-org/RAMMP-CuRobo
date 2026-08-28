#!/usr/bin/env python3
"""Plan a collision-free trajectory with cuRobo — no ROS, no hardware.

Nothing here can move an arm: this is the Layer-1 library alone. Examples:

--start is REQUIRED on every call: the planner does not know where the arm
is and will not guess one for you. Examples (start = a mild variation of the
robot config's retract pose):

    # joint-space goal (native js planning; FK-pose fallback automatic)
    python3 examples/plan_only.py --start 0.0 0.262 3.142 -2.269 0.0 0.960 1.571 \
        --joints 0.3 0.262 3.142 -2.269 0.0 0.960 1.571

    # tool_frame pose goal: position + roll/pitch/yaw in degrees
    python3 examples/plan_only.py --start 0.0 0.262 3.142 -2.269 0.0 0.960 1.571 \
        --pos 0.45 0.12 0.30 --rpy-deg 180 0 0

    # same goal, quarter-speed preview of what an executor would send
    python3 examples/plan_only.py --start 0.0 0.262 3.142 -2.269 0.0 0.960 1.571 \
        --pos 0.45 0.12 0.30 --rpy-deg 180 0 0 --speed-scale 0.25
"""

import argparse
import sys

import numpy as np

from rammp_curobo import CuRoboPlanner
from rammp_curobo.geometry import euler_deg_to_quat_xyzw


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", default="gen3.yaml", help="planner YAML (packaged name or path)"
    )
    goal = ap.add_mutually_exclusive_group(required=True)
    goal.add_argument(
        "--joints",
        type=float,
        nargs=7,
        metavar="RAD",
        help="goal joint positions, controller order",
    )
    goal.add_argument(
        "--pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="goal tool_frame position (m, base frame)",
    )
    ori = ap.add_mutually_exclusive_group()
    ori.add_argument(
        "--rpy-deg",
        type=float,
        nargs=3,
        metavar=("R", "P", "Y"),
        help="goal orientation as roll/pitch/yaw degrees",
    )
    ori.add_argument(
        "--quat",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
        help="goal orientation as xyzw quaternion",
    )
    ap.add_argument(
        "--start",
        type=float,
        nargs=7,
        metavar="RAD",
        required=True,
        help="joint configuration to plan FROM (rad, joint_1..7). Required "
        "— the planner holds no default start pose",
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=None,
        help="preview execution-side retiming (0 < s <= 1)",
    )
    args = ap.parse_args()
    if args.joints is not None and (args.quat or args.rpy_deg):
        ap.error("--quat/--rpy-deg only apply to --pos goals")

    print("Initializing planner (GPU init + warmup — first run takes a while)...")
    planner = CuRoboPlanner.from_config(args.config)

    if args.joints is not None:
        res = planner.plan_to_joints(args.joints, args.start)
        label = "joints %s" % np.round(args.joints, 3).tolist()
    else:
        if args.quat is not None:
            quat = args.quat
        elif args.rpy_deg is not None:
            quat = euler_deg_to_quat_xyzw(args.rpy_deg)
        else:
            ap.error("--pos needs --rpy-deg or --quat")
        res = planner.plan_to_pose(args.pos, quat, args.start)
        label = "pose %s" % np.round(args.pos, 3).tolist()

    if not res.success:
        print(
            "PLAN FAILED (%s) after %.2fs:\n  %s" % (res.status, res.timing, res.error)
        )
        return 1

    traj = res.joint_traj
    scale = args.speed_scale
    if scale is None:
        scale = 1.0
    if scale != 1.0:
        traj = traj.scaled(scale)
    print(
        "Planned to %s in %.2fs: %d points, %.2fs at scale %.2f (validated=%s)"
        % (
            label,
            res.timing,
            traj.n_points,
            traj.duration,
            traj.speed_scale,
            res.validated,
        )
    )
    if res.goal_mismatch_rad is not None:
        print("joint-goal mismatch (FK-pose method): %.4f rad" % res.goal_mismatch_rad)
    lo = traj.positions.min(axis=0)
    hi = traj.positions.max(axis=0)
    vmax = (
        np.abs(traj.velocities).max(axis=0)
        if traj.velocities is not None
        else np.zeros(traj.dof)
    )
    print(
        "\n%-9s %10s %10s %10s %10s" % ("joint", "start", "end", "excursion", "max |v|")
    )
    for j, name in enumerate(traj.joint_names):
        print(
            "%-9s %10.3f %10.3f %10.3f %10.3f"
            % (
                name,
                traj.positions[0, j],
                traj.positions[-1, j],
                hi[j] - lo[j],
                vmax[j],
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
