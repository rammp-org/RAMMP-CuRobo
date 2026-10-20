"""Send one goal through the sim stack: planned by a planner node, flown on
the sim arm, contacts reported.

    ros2 run mebot_sim try_goal --pos 0.50 -0.40 0.30
    ros2 run mebot_sim try_goal --pos 0.45 0.20 0.25 --control
    ros2 run mebot_sim try_goal --pos 0.35 -0.20 -0.30 --rpy-deg 180 0 0 --speed 0.25

The goal is planned on /rammp_curobo, the chair config, which sees the
floor, the platform and the person. --control then plans it again from the
same start on /rammp_curobo_control in a world of the floor and the platform
only: the same request without the person layer.

Coordinates are metres in the arm's arm_mount frame (level, at its base),
x forward, y left and z up; the occupant sits at y = +0.27, the seat pan at
z = +0.09. A tool goal is cuRobo's arm_tool_frame, the fingertip midpoint,
and --rpy-deg its orientation as ROS fixed-axis roll, pitch and yaw: the
default, 90 0 90, points the gripper forward and level.
"""

import argparse
import sys

from mebot_sim import stack
from mebot_sim.client import Tester, closest_text, contact_text


def _speed(text):
    speed = float(text)
    if not 0 < speed <= 1:
        raise argparse.ArgumentTypeError("a speed is in (0, 1], got %s" % text)
    return speed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    goal = parser.add_mutually_exclusive_group(required=True)
    goal.add_argument(
        "--pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="a tool goal, metres",
    )
    goal.add_argument(
        "--joints", nargs=7, type=float, metavar="Q", help="a joint goal, radians"
    )
    parser.add_argument(
        "--rpy-deg",
        nargs=3,
        type=float,
        default=[90.0, 0.0, 90.0],
        metavar=("R", "P", "Y"),
        help="the tool goal's orientation (default: 90 0 90)",
    )
    parser.add_argument(
        "--start",
        nargs=7,
        type=float,
        metavar="Q",
        help="start here (default: where the arm is)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="plan it again on the control planner, without the person layer",
    )
    parser.add_argument(
        "--speed",
        type=_speed,
        default=0.5,
        help="fly plans at this fraction of their speed (default: 0.5)",
    )
    parser.add_argument("--no-exec", action="store_true", help="plan only")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    worlds = stack.worlds()
    runs = [worlds["all"]] + ([worlds["no_person"]] if args.control else [])
    tester = Tester("mebot_try_goal")
    try:
        tester.connect()
        start = list(args.start or tester.q)
        what = (
            "tool at (%.2f, %.2f, %.2f), rpy %g %g %g" % (*args.pos, *args.rpy_deg)
            if args.pos
            else "joints " + " ".join("%.2f" % v for v in args.joints)
        )
        print("goal: %s; start: %s" % (what, " ".join("%.2f" % v for v in start)))
        for i, (planner, world, sees) in enumerate(runs):
            if args.start or i:
                tester.teleport(start)
            r = tester.move(
                planner,
                world,
                start,
                pos=args.pos,
                rpy_deg=args.rpy_deg if args.pos else None,
                joints=args.joints,
                speed=args.speed,
                execute=not args.no_exec,
            )
            head = "%s (%s)" % (planner, sees)
            if not r["planned"]:
                print("%s: NO PLAN in %.2f s: %s" % (head, r["plan_s"], r["message"]))
            elif args.no_exec:
                print("%s: planned in %.2f s; not flown" % (head, r["plan_s"]))
            elif not r["executed"]:
                print(
                    "%s: planned in %.2f s; not flown to its end: %s"
                    % (head, r["plan_s"], r["exec_error"])
                )
            else:
                print(
                    "%s: planned in %.2f s, flown at speed %.2f to %.3f rad of its "
                    "last point: %s; closest: %s"
                    % (
                        head,
                        r["plan_s"],
                        args.speed,
                        r["end_error_rad"],
                        contact_text(r["contacts"]),
                        closest_text(r["closest"]),
                    )
                )
    except (RuntimeError, TimeoutError) as exc:
        print("try_goal: %s" % exc, file=sys.stderr)
        return 1
    finally:
        tester.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
