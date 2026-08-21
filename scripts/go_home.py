#!/usr/bin/env python3
"""Send the arm home.

    python3 scripts/go_home.py              # plan only, nothing moves
    python3 scripts/go_home.py --execute    # the arm moves

Needs the arm bringup and a planner node with execute:=true. Plans from
the arm's LIVE state through the normal gate chain, so it is safe to run
from wherever the demo left it. Ctrl+C cancels and the arm holds.
"""

import argparse
import sys

import rclpy

from rammp_curobo_ros.tour_demo import TourDemo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speed", type=float, default=0.25,
                    help="fraction of planned speed (0, 1]")
    ap.add_argument("--execute", action="store_true", help="allow motion")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("go_home")
    demo = TourDemo(node)

    plan = demo.plan_home_from(None)          # None = from the live state
    if plan is None or not plan.success:
        sys.exit("cannot plan home: %s"
                 % (plan.message if plan is not None else "no planner response"))
    print("planned %d points, %.2f s at full speed"
          % (len(plan.trajectory.points), plan.planning_time))
    if getattr(plan, "goal_mismatch_rad", 0.0) > 0.5:
        sys.exit("goal mismatch %.2f rad — the plan reaches the home POSE but "
                 "in a different joint family; refusing" % plan.goal_mismatch_rad)
    if not args.execute:
        print("dry run — add --execute (the arm will move; e-stop in hand)")
        return

    print("homing at %.0f%% speed. Ctrl+C cancels and the arm holds."
          % (args.speed * 100))
    ok = demo.run(plan.trajectory, args.speed)
    print("home." if ok else "refused — is the planner node running with "
          "execute:=true, and does the arm match the plan's start?")
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
