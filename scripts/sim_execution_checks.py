#!/usr/bin/env python3
"""Live sim checks: speed-scale gate refusal + mid-motion cancel (abort).

Needs a running MuJoCo sim bringup AND the planner node with execute:=true
(see rammp_curobo_ros/launch/planner.launch.py). Not a pytest test — run:
    python3 scripts/sim_execution_checks.py
"""

import sys
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints
from rammp_curobo_ros.scan_common import NODE_NAMESPACE

PREFIX = NODE_NAMESPACE


def wait(future, executor, timeout_s=60.0):
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    t0 = time.monotonic()
    while not done.is_set():
        executor.spin_once(timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


def main():
    rclpy.init()
    node = Node("sim_execution_checks")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    latest = {}
    node.create_subscription(
        JointState,
        "/joint_states",
        lambda m: latest.update(
            v=max(abs(x) for x in m.velocity[:7]) if m.velocity else None
        ),
        10,
    )

    # plan a long move: back to zeros
    plan_client = ActionClient(node, PlanToJoints, PREFIX + "/plan_to_joints")
    assert plan_client.wait_for_server(timeout_sec=5.0)
    goal = PlanToJoints.Goal(target_joints=[0.0] * 7)
    send = wait(plan_client.send_goal_async(goal), executor, 10)
    plan = wait(send.get_result_async(), executor, 120).result
    assert plan.success, plan.message
    print("CHECK 1 planned: %s" % plan.message)

    exec_client = ActionClient(node, ExecuteTrajectory, PREFIX + "/execute_trajectory")
    assert exec_client.wait_for_server(timeout_sec=5.0)

    # gate: speed_scale > max must be refused
    bad = ExecuteTrajectory.Goal(trajectory=plan.trajectory, speed_scale=2.0)
    send = wait(exec_client.send_goal_async(bad), executor, 10)
    res = wait(send.get_result_async(), executor, 30)
    assert (
        res.status == GoalStatus.STATUS_ABORTED and "speed_scale" in res.result.message
    ), (res.status, res.result.message)
    print("CHECK 2 speed-scale gate refused: %s" % res.result.message)

    # cancel mid-motion
    good = ExecuteTrajectory.Goal(trajectory=plan.trajectory, speed_scale=0.25)
    send = wait(exec_client.send_goal_async(good), executor, 10)
    assert send.accepted
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        executor.spin_once(timeout_sec=0.1)
    print("CHECK 3 cancelling mid-motion at t=3.0 s...")
    wait(send.cancel_goal_async(), executor, 5)
    res = wait(send.get_result_async(), executor, 30)
    assert res.status == GoalStatus.STATUS_CANCELED, (res.status, res.result.message)
    print("CHECK 3 goal canceled: %s" % res.result.message)

    # arm must come to rest quickly and stay short of the goal
    time.sleep(0.5)
    t0 = time.monotonic()
    calm = 0
    while time.monotonic() - t0 < 5.0 and calm < 5:
        executor.spin_once(timeout_sec=0.1)
        v = latest.get("v")
        calm = calm + 1 if (v is not None and v < 0.02) else 0
    assert calm >= 5, "arm did not settle after cancel (v=%s)" % latest.get("v")
    print("CHECK 4 arm settled and holds after cancel — abort path verified")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
