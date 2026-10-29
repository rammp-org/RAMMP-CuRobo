"""Measured HOLD_LEVEL deviation (deg) during the square, HOLD_LEVEL throughout, against
the targets' orientation (HOME's), with the planner's own metric (_hold_deviation_batch,
LEVEL: yaw about arm_mount Z free): before (video10, holds not sent) and after (v2_10)."""
import sys
import numpy as np
from analyze import load
from fk import FK


def level_dev(xyzw, goal_xyzw):
    """CuRoboPlanner._hold_deviation_batch for HOLD_LEVEL, (N,) rad."""
    ax, ay, az, aw = xyzw[:, 0], xyzw[:, 1], xyzw[:, 2], xyzw[:, 3]
    bx, by, bz, bw = goal_xyzw
    ew = bw * aw + bx * ax + by * ay + bz * az
    ez = bz * aw - bw * az - bx * ay + by * ax
    return 2.0 * np.arccos(np.minimum(1.0, np.hypot(ew, ez)))


fk = FK(open(sys.argv[1]).read())
goal = np.array([0.5, 0.5, 0.5, 0.5])
for f in sys.argv[2:]:
    d = load(f)
    meas = np.degrees(level_dev(fk.tool(d["fb_act"])[1].as_quat(), goal)).max()
    plan = np.degrees(level_dev(fk.tool(d["q"])[1].as_quat(), goal)).max()
    print(f"{f.split('/')[-2]:8} {f.split('/')[-1]:24} level deviation: planned {plan:4.2f} deg, measured {meas:4.2f} deg")
