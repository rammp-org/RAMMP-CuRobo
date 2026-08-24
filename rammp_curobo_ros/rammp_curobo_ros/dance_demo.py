#!/usr/bin/env python3
"""Dance easter egg: the arm bobs, sways, circles, twists — safely.

Randomized rounds of goofy-but-gated motion: every waypoint lives in the
bench-proven safe box, the wrist stays flat except deliberate twist/nod
flourishes, and each round pre-plans as chained segments merged into ONE
trajectory (zero controller goal transitions — the no-motion-fault
lesson), executed through every ExecuteTrajectory gate.

    ros2 run rammp_curobo_ros dance_demo               # dry-run: plan + print
    ros2 run rammp_curobo_ros dance_demo --execute     # needs typed 'dance'

SAFETY: clear workspace, human on the physical e-stop, planner launched
with execute:=true. Ctrl+C cancels; the arm holds position.
"""

import argparse
import math
import random
import sys
import time

import rclpy

from rammp_curobo.geometry import ang_diff, yaw_about_world_z
from rammp_curobo_ros.tour_demo import (
    HOME,
    HOME_QUAT_XYZW,
    TourDemo,
    merge_trajectories,
    traj_end,
    traj_time,
)

# x / y / z bounds — the IK-verified volume from the calibration session
SAFE_BOX = ((0.35, 0.60), (-0.30, 0.30), (0.18, 0.60))
DANCE_CENTER = (0.45, 0.0, 0.40)


def _clamp(p):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(p, SAFE_BOX)]


def choreograph(rng, n_moves=8, center=DANCE_CENTER):
    """A random dance as ('pose', [xyz]) / ('twist', dj7) / ('nod', dj6).

    Flourishes always emit +d then -d so the chain lands back on a flat
    wrist; every pose is clamped into SAFE_BOX; the dance settles at
    `center` at the end.
    """
    cx, cy, cz = center
    moves = []
    for _ in range(n_moves):
        kind = rng.choice(["bob", "sway", "circle", "twist", "nod", "shimmy"])
        if kind == "bob":
            dz = rng.uniform(0.06, 0.12)
            moves += [
                ("pose", _clamp([cx, cy, cz - dz])),
                ("pose", _clamp([cx, cy, cz + dz])),
            ]
        elif kind == "sway":
            dy = rng.uniform(0.10, 0.20)
            moves += [
                ("pose", _clamp([cx, cy - dy, cz])),
                ("pose", _clamp([cx, cy + dy, cz])),
            ]
        elif kind == "circle":
            r = rng.uniform(0.05, 0.09)
            direction = rng.choice([-1.0, 1.0])
            for k in range(6):
                a = direction * 2.0 * math.pi * k / 6.0
                moves.append(
                    ("pose", _clamp([cx, cy + r * math.sin(a), cz + r * math.cos(a)]))
                )
        elif kind == "twist":
            d = rng.uniform(0.4, 0.7)
            moves += [("twist", d), ("twist", -d)]
        elif kind == "nod":
            d = rng.uniform(0.25, 0.4)
            moves += [("nod", d), ("nod", -d)]
        elif kind == "shimmy":
            d = rng.uniform(0.03, 0.05)
            for s in (-1, 1, -1):
                moves.append(("pose", _clamp([cx, cy + s * d, cz])))
    moves.append(("pose", list(center)))
    return moves


JOINT_IDX = {"twist": 6, "nod": 5}  # j7 / j6, 0-based in the 7-vector


def plan_move(demo, move, start):
    """One choreography move -> a chained plan (or None)."""
    kind, val = move
    if kind == "pose":
        quat = list(yaw_about_world_z(HOME_QUAT_XYZW, math.atan2(val[1], val[0])))
        return demo.plan_pose_from(val, quat, start)
    from rammp_curobo_interfaces.action import PlanToJoints

    q = [float(v) for v in start]
    q[JOINT_IDX[kind]] += float(val)
    g = PlanToJoints.Goal(target_joints=q)
    g.start_joints = [float(v) for v in start]
    return demo._call(demo.plan_joints, g)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--execute", action="store_true", help="allow motion")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.4,
        help="execution scale (dance default 0.4; 1.0 = rated)",
    )
    ap.add_argument(
        "--rounds", type=int, default=3, help="dance rounds (one merged trajectory each)"
    )
    ap.add_argument("--moves", type=int, default=8, help="moves per round")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    scale = min(max(args.speed, 0.1), 1.0)
    rng = random.Random(args.seed)

    from rclpy.signals import SignalHandlerOptions

    # own the SIGINT: rclpy's handler would tear the context down before
    # the except-branch could cancel the active goal
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("rammp_curobo_dance")
    demo = TourDemo(node)

    q_now = demo.joints()
    if max(abs(ang_diff(a, b)) for a, b in zip(q_now, HOME)) > 0.1:
        if not args.execute:
            sys.exit("arm is not at home — rerun with --execute to home it")
        if (
            input(
                "arm is away from home — type 'go' to home it at 25% "
                "(hand on e-stop): "
            ).strip()
            != "go"
        ):
            sys.exit("aborted — nothing moved")
        plan = demo.plan_home_from(None)
        if plan is None or not plan.success:
            sys.exit("cannot plan home")
        if not demo.run(plan.trajectory, 0.25):
            sys.exit("homing failed — see planner log")
        print("homed.")

    print("choreographing %d round(s) of %d moves..." % (args.rounds, args.moves))
    rounds = []
    start = None  # live state for round 1 segment 1
    for r in range(args.rounds):
        plans, skipped = [], 0
        for move in choreograph(rng, n_moves=args.moves):
            plan = plan_move(demo, move, start if start else demo.joints())
            if plan is None or not plan.success:
                skipped += 1
                continue
            plans.append(plan)
            start = traj_end(plan)
        home_plan = demo.plan_home_from(start)
        if home_plan is None or not home_plan.success:
            sys.exit("cannot plan the return home for round %d" % (r + 1))
        plans.append(home_plan)
        start = traj_end(home_plan)
        rounds.append(merge_trajectories(plans))
        total = sum(traj_time(p, scale) for p in plans)
        print(
            "  round %d: %d segments (%d unplannable skipped), %.1f s at "
            "speed %.2f" % (r + 1, len(plans), skipped, total, scale)
        )

    if not args.execute:
        print("dry-run complete — nothing moved (add --execute)")
        return

    print(
        "\n*** DANCE TIME: workspace COMPLETELY CLEAR, human on the "
        "physical e-stop. Ctrl+C stops (arm holds). ***"
    )
    if input("type 'dance' to start: ").strip() != "dance":
        print("aborted — nothing moved")
        return

    for r, merged in enumerate(rounds):
        print("round %d/%d (%d points)..." % (r + 1, len(rounds), len(merged.points)))
        ok = False
        for attempt in range(3):
            if demo.run(merged, scale):
                ok = True
                break
            moved = max(
                abs(ang_diff(a, b))
                for a, b in zip(demo.joints(), merged.points[0].positions)
            )
            if moved > 0.05 or attempt == 2:
                sys.exit(
                    "dance stopped (arm %.3f rad from round start) — arm "
                    "holds; see the planner log" % moved
                )
            print(
                "  no-motion fault at start, recovered — retrying (%d/2)"
                % (attempt + 1)
            )
            time.sleep(3.0)
        if not ok:
            sys.exit("dance failed")
        time.sleep(0.5)
    print("\nDANCE COMPLETE — take a bow (the arm already did).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndance stopped — arm holds")
