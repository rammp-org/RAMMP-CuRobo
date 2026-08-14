#!/usr/bin/env python3
"""Plan a motion, preview it, and (only on explicit request) execute it.

DRY-RUN IS THE DEFAULT: without --execute this plans and prints the
trajectory, and nothing can move. Real motion needs ALL of:
  1. the arm driver (kinova_arm_node: --sim stub, or --ip on the real Gen3),
  2. the planner node launched with execute:=true,
  3. this script run with --execute,
  4. typing exactly 'yes' at the confirmation prompt.
On real hardware a human holds the physical e-stop the entire time.

Examples (planner node running — see rammp_curobo_ros/launch/planner.launch.py):

    # dry-run a small joint move near home
    python3 examples/plan_and_execute.py --joints 0.2 0.262 3.142 -2.269 0.0 0.960 1.571

    # same move, executed at 15% speed after confirmation
    python3 examples/plan_and_execute.py --joints 0.2 0.262 3.142 -2.269 0.0 0.960 1.571 \
        --execute --speed-scale 0.15

Ctrl+C during execution cancels the goal: the controller stops and holds.
"""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose

# This example is deliberately standalone — it shows what an integrating
# module needs WITHOUT importing rammp_curobo or rammp_curobo_ros, so the
# planner node's namespace and joint order are restated here (they come
# from planner_node.py's Node("rammp_curobo") and config.PLANNER_DEFAULTS).
SERVER_PREFIX = "/rammp_curobo"
JOINT_NAMES = ["joint_%d" % i for i in range(1, 8)]


def wait(future, executor, timeout_s=None):
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    t0 = time.monotonic()
    while not done.is_set():
        executor.spin_once(timeout_sec=0.1)
        if timeout_s and time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


def call_action(node, executor, action_type, name, goal, timeout_s=300.0):
    client = ActionClient(node, action_type, name)
    if not client.wait_for_server(timeout_sec=5.0):
        sys.exit(
            "Action server %s not available — is the planner node "
            "running? (ros2 launch rammp_curobo_ros planner.launch.py)" % name
        )
    send = wait(client.send_goal_async(goal), executor, 10.0)
    if send is None or not send.accepted:
        sys.exit("%s: goal not accepted" % name)
    result_future = send.get_result_async()
    try:
        wrapped = wait(result_future, executor, timeout_s)
    except KeyboardInterrupt:
        print("\n^C — cancelling goal (controller stops and holds)...")
        wait(send.cancel_goal_async(), executor, 5.0)
        wrapped = wait(result_future, executor, 10.0)
        if wrapped is not None:
            print("Cancelled: %s" % wrapped.result.message)
        sys.exit(130)
    if wrapped is None:
        sys.exit("%s: timed out" % name)
    return wrapped.result


def read_joint_state(node, executor, timeout_s=5.0):
    """One fresh /joint_states sample, mapped to joint_1..joint_7 order."""

    names = JOINT_NAMES
    slot = {}
    sub = node.create_subscription(
        JointState,
        "/joint_states",
        lambda m: slot.update(msg=m),
        qos_profile_sensor_data,  # kinova_arm_node streams best-effort
    )
    t0 = time.monotonic()
    while "msg" not in slot:
        executor.spin_once(timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            sys.exit(
                "No /joint_states within %.0f s — is the arm/sim "
                "bringup running?" % timeout_s
            )
    node.destroy_subscription(sub)
    msg = slot["msg"]
    idx = {n: i for i, n in enumerate(msg.name)}
    try:
        return [float(msg.position[idx[n]]) for n in names]
    except KeyError as exc:
        sys.exit("joint %s missing from /joint_states" % exc)


def summarize(traj, scale):
    pos = np.array([p.positions for p in traj.points])
    t_end = (
        traj.points[-1].time_from_start.sec
        + traj.points[-1].time_from_start.nanosec * 1e-9
    )
    print(
        "\nPlanned trajectory: %d points, %.2f s at full speed "
        "(%.2f s at scale %.2f)" % (len(traj.points), t_end, t_end / scale, scale)
    )
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    print("%-9s %10s %10s %10s" % ("joint", "start", "end", "excursion"))
    for j, name in enumerate(traj.joint_names):
        print(
            "%-9s %10.3f %10.3f %10.3f" % (name, pos[0, j], pos[-1, j], hi[j] - lo[j])
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    goal = ap.add_mutually_exclusive_group(required=True)
    goal.add_argument(
        "--joints",
        type=float,
        nargs=7,
        metavar="RAD",
        help="goal joint positions, controller order (joint_1..joint_7)",
    )
    goal.add_argument(
        "--joints-relative",
        type=float,
        nargs=7,
        metavar="RAD",
        help="deltas added to the CURRENT joint positions — "
        "the first-hardware-move primitive (small, known "
        "displacement from wherever the arm is)",
    )
    goal.add_argument(
        "--pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="goal tool_frame position (m, base frame); needs --quat",
    )
    ap.add_argument(
        "--quat",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
        help="orientation for --pos (xyzw)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="after previewing, offer to execute (needs the node "
        "launched with execute:=true)",
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=0.25,
        help="execution-side time dilation, 0 < s <= 1 (the node refuses "
        "anything above its max_speed_scale)",
    )
    ap.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="execute even if a joint goal was reached in a "
        "different joint family (pose-space fallback)",
    )
    args = ap.parse_args()

    rclpy.init()
    node = Node("rammp_curobo_example")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    if args.joints_relative is not None:
        q_now = read_joint_state(node, executor)
        target = [q + d for q, d in zip(q_now, args.joints_relative)]
        print("current joints: %s" % np.round(q_now, 3).tolist())
        print("target joints:  %s" % np.round(target, 3).tolist())
        args.joints = target

    if args.joints is not None:
        plan_goal = PlanToJoints.Goal(target_joints=[float(v) for v in args.joints])
        result = call_action(
            node, executor, PlanToJoints, SERVER_PREFIX + "/plan_to_joints", plan_goal
        )
    else:
        if args.quat is None:
            ap.error("--pos requires --quat")
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = args.pos
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = args.quat
        goal = PlanToPose.Goal()
        goal.target = pose
        result = call_action(
            node, executor, PlanToPose, SERVER_PREFIX + "/plan_to_pose", goal
        )

    if not result.success:
        sys.exit("PLAN FAILED: %s" % result.message)
    print(result.message)
    mismatch = getattr(result, "goal_mismatch_rad", None)
    if mismatch and mismatch > 1e-3:
        print(
            "note: joint-goal mismatch %.4f rad (pose-space fallback — "
            "the POSE is exact, the joint split may differ)" % mismatch
        )
    summarize(result.trajectory, args.speed_scale)

    if not args.execute:
        print(
            "\nDry run complete — nothing moved. Re-run with --execute "
            "to move the arm (planner node must be launched with "
            "execute:=true)."
        )
        return

    if mismatch and mismatch > 0.5 and not args.allow_mismatch:
        sys.exit(
            "REFUSING to execute: the plan reaches the requested POSE "
            "but via a different joint family (%.2f rad from the "
            "requested joints) — the motion would be much larger than "
            "asked for. Re-run with --allow-mismatch to override." % mismatch
        )

    print("\n*** ABOUT TO MOVE THE ARM at %.0f%% speed. ***" % (args.speed_scale * 100))
    print("Confirm a human is holding the physical e-stop.")
    try:
        answer = input("Type 'yes' to execute, anything else to abort: ")
    except EOFError:
        answer = ""
    if answer.strip() != "yes":
        print("Aborted — nothing moved.")
        return

    exec_goal = ExecuteTrajectory.Goal()
    exec_goal.trajectory = result.trajectory
    exec_goal.speed_scale = float(args.speed_scale)
    exec_result = call_action(
        node,
        executor,
        ExecuteTrajectory,
        SERVER_PREFIX + "/execute_trajectory",
        exec_goal,
    )
    if exec_result.success:
        print("EXECUTED: %s" % exec_result.message)
    else:
        sys.exit("EXECUTION FAILED: %s" % exec_result.message)


if __name__ == "__main__":
    main()
