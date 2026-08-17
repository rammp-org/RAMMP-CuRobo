#!/usr/bin/env python3
"""Live sanity checks for the perceived world (planner + cameras up).

Non-destructive: talks services/topics only, never executes motion.

    export ROS_LOCALHOST_ONLY=1
    python3 scripts/cameras_checks.py

Checks: (1) update_world_boxes round-trip incl. the empty-perceived-set
case, (2) markers flowing from the cameras node, (3) a plan succeeds
while the updater hammers the world at full rate (the lock-timeout path).
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from rammp_curobo_interfaces.srv import UpdateWorldBoxes

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]


def main():
    rclpy.init()
    node = Node("cameras_checks")
    ok = True

    cli = node.create_client(UpdateWorldBoxes, "/rammp_curobo/update_world_boxes")
    if not cli.wait_for_service(timeout_sec=5.0):
        sys.exit("FAIL: planner update_world_boxes service not up")

    def send(names, centers, dims, baseline=""):
        req = UpdateWorldBoxes.Request()
        req.names, req.baseline = list(names), baseline
        req.centers = [Point(x=c[0], y=c[1], z=c[2]) for c in centers]
        req.dims = [Vector3(x=d[0], y=d[1], z=d[2]) for d in dims]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        return fut.result()

    r = send(["chk_box"], [[0.55, 0.25, 0.15]], [[0.1, 0.1, 0.3]])
    print("[1] add box:", r and r.message)
    ok &= bool(r and r.success)
    r = send([], [], [])
    print("[1] empty perceived set (baseline must survive):", r and r.message)
    ok &= bool(r and r.success)

    got = {"n": 0}
    node.create_subscription(
        MarkerArray, "/cameras/world_markers", lambda m: got.update(n=got["n"] + 1), 1
    )
    t0 = time.monotonic()
    while got["n"] == 0 and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    print(
        "[2] markers:",
        "flowing"
        if got["n"]
        else "NONE (cameras node down? OK for a planner-only test)",
    )

    try:
        from rclpy.action import ActionClient

        from rammp_curobo_interfaces.action import PlanToJoints

        ac = ActionClient(node, PlanToJoints, "/rammp_curobo/plan_to_joints")
        if not ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("plan_to_joints server not up")
        goal = PlanToJoints.Goal()
        goal.target_joints = [HOME[0] + 0.05] + HOME[1:]
        goal.start_joints = HOME
        t_end = time.monotonic() + 6.0
        sent = ac.send_goal_async(goal)

        # hammer the world while the plan runs — exercises the lock timeout
        while time.monotonic() < t_end and not sent.done():
            send(["chk_churn"], [[0.5, -0.3, 0.2]], [[0.05, 0.05, 0.2]])
            rclpy.spin_once(node, timeout_sec=0.1)
        rclpy.spin_until_future_complete(node, sent, timeout_sec=10.0)
        res_fut = sent.result().get_result_async()
        rclpy.spin_until_future_complete(node, res_fut, timeout_sec=30.0)
        res = res_fut.result().result
        print("[3] plan under churn:", res.success, res.message)
        ok &= res.success
        send([], [], [])  # leave the world clean
    except Exception as exc:
        print("[3] SKIP/FAIL:", exc)
        ok = False

    print("ALL OK" if ok else "FAILURES — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
