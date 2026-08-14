#!/usr/bin/env python3
"""Live checks of the execution gates against kinova_arm_node --sim.

The driver's sim is a STATIC transport stub: it echoes commands but the
measured q never moves. That makes real tracking unverifiable here (that
is what the attended hardware runbook is for) — but it makes the failure
paths deterministic, so this script validates exactly what sim can:

  1. planning from the live best-effort /joint_states stream (QoS wiring),
  2. the speed-scale gate refusing an over-limit request,
  3. a tiny in-tolerance move completing through the full driver pipe,
  4. mid-motion cancel -> driver holds (canceled status),
  5. the wrap-aware arrival check catching a no-motion "success" —
     the frozen sim IS a no-motion arm, the exact failure the check exists
     for.

Needs a running kinova_arm_node in SIM mode AND the planner node with
execute:=true:
    ros2 run kinova_arm_ros2 kinova_arm_node --sim --urdf models/... &
    ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml \
        execute:=true
Not a pytest test — run:  python3 scripts/sim_execution_checks.py
"""

import sys
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints
from rammp_curobo_ros.ros_util import NODE_NAMESPACE

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


def plan(plan_client, executor, target):
    send = wait(
        plan_client.send_goal_async(PlanToJoints.Goal(target_joints=target)),
        executor,
        10,
    )
    res = wait(send.get_result_async(), executor, 120).result
    assert res.success, res.message
    return res


def execute(exec_client, executor, trajectory, scale, timeout_s=120.0):
    goal = ExecuteTrajectory.Goal(trajectory=trajectory, speed_scale=scale)
    send = wait(exec_client.send_goal_async(goal), executor, 10)
    return send, wait(send.get_result_async(), executor, timeout_s)


def main():
    rclpy.init()
    node = Node("sim_execution_checks")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    seen = {}
    node.create_subscription(
        JointState,
        "/joint_states",
        lambda m: seen.update(msg=True),
        qos_profile_sensor_data,  # kinova_arm_node streams best-effort
    )

    plan_client = ActionClient(node, PlanToJoints, PREFIX + "/plan_to_joints")
    assert plan_client.wait_for_server(timeout_sec=5.0)
    exec_client = ActionClient(node, ExecuteTrajectory, PREFIX + "/execute_trajectory")
    assert exec_client.wait_for_server(timeout_sec=5.0)

    # CHECK 1: plan from the LIVE state — proves the planner receives the
    # driver's best-effort /joint_states (QoS mismatch would starve it).
    res = plan(plan_client, executor, [0.03] * 7)
    print("CHECK 1 planned from live joint state: %s" % res.message)

    # CHECK 2: the speed-scale gate refuses an over-limit request.
    _send, out = execute(exec_client, executor, res.trajectory, 2.0, 30)
    assert (
        out.status == GoalStatus.STATUS_ABORTED and "speed_scale" in out.result.message
    ), (out.status, out.result.message)
    print("CHECK 2 speed-scale gate refused: %s" % out.result.message)

    # CHECK 3: a tiny move (0.03 rad, inside the 0.08 arrival tolerance
    # even against the frozen sim) runs the full pipe to SUCCEEDED.
    _send, out = execute(exec_client, executor, res.trajectory, 0.25)
    assert out.status == GoalStatus.STATUS_SUCCEEDED, (
        out.status,
        out.result.message,
    )
    print("CHECK 3 full-pipe execution succeeded: %s" % out.result.message)

    # CHECK 4: mid-motion cancel. A base-yaw sweep keeps the bounded
    # joints (the only ones with an armed path guard) nearly still, so
    # nothing aborts before we cancel.
    res = plan(plan_client, executor, [1.2, 0.02, 0.0, -0.02, 0.0, 0.0, 0.0])
    goal = ExecuteTrajectory.Goal(trajectory=res.trajectory, speed_scale=0.25)
    send = wait(exec_client.send_goal_async(goal), executor, 10)
    assert send.accepted
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.5:
        executor.spin_once(timeout_sec=0.1)
    print("CHECK 4 cancelling mid-motion at t=2.5 s...")
    wait(send.cancel_goal_async(), executor, 5)
    out = wait(send.get_result_async(), executor, 30)
    assert out.status == GoalStatus.STATUS_CANCELED, (
        out.status,
        out.result.message,
    )
    print("CHECK 4 goal canceled, driver holds: %s" % out.result.message)

    # CHECK 5: the arrival check must catch a no-motion "success". The
    # frozen sim completes the goal on the driver's timer while q stays
    # put — exactly the blind-success failure the gate exists for.
    res = plan(plan_client, executor, [0.6, 0.02, 0.0, -0.02, 0.0, 0.0, 0.0])
    _send, out = execute(exec_client, executor, res.trajectory, 0.25)
    assert out.status == GoalStatus.STATUS_ABORTED and (
        "never left the start" in out.result.message
        or "TRACKING FAILURE" in out.result.message
    ), (out.status, out.result.message)
    print("CHECK 5 no-motion success caught: %s" % out.result.message)

    assert seen.get("msg"), "this script never received /joint_states itself"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
