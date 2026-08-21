#!/usr/bin/env python3
"""Does Ctrl+C actually STOP a moving arm? Verified against a stub controller.

    python3 scripts/abort_checks.py

Starts a stand-in FollowJointTrajectory server and a real planner node,
begins a long execution, sends SIGINT to the planner mid-stroke, and
reads back whether the controller was told to stop.

This exists because the failure it catches is silent and severe. rclpy's
default SIGINT handler shuts the context down synchronously, and the
cancel path in executor.run() is delivered by polling
`goal_handle.is_cancel_requested` — which only advances while this node's
executor is spinning. With the default handler, Ctrl+C during motion did
not stop the arm; it stopped WATCHING the arm, and the controller drove
the trajectory to its end (verified: the stub saw no cancel at all).

No arm, no camera, no GPU beyond the planner's own warmup.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.96, 1.571]

STUB = '''
import time
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

HOME = %r
NAMES = ["joint_%%d" %% i for i in range(1, 8)]


class Stub(Node):
    def __init__(self):
        super().__init__("stub_controller")
        cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(JointState, "/joint_states",
                                         qos_profile_sensor_data)
        self.create_timer(0.05, self._tick, callback_group=cb)
        ActionServer(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            execute_callback=self._exec,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=self._cancel,
            callback_group=cb,
        )
        self.q = list(HOME)

    def _tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(NAMES)
        m.position = list(self.q)
        m.velocity = [0.0] * len(NAMES)
        self.pub.publish(m)

    def _cancel(self, _goal):
        print("CANCEL RECEIVED", flush=True)
        return CancelResponse.ACCEPT

    def _exec(self, goal_handle):
        pts = goal_handle.request.trajectory.points
        dur = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
        print("goal accepted, %%d points over %%.1f s" %% (len(pts), dur), flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < dur:
            if goal_handle.is_cancel_requested:
                print("STOPPED at %%.1f s of %%.1f" %% (time.monotonic() - t0, dur),
                      flush=True)
                goal_handle.canceled()
                return FollowJointTrajectory.Result()
            time.sleep(0.02)
        print("RAN TO COMPLETION", flush=True)
        goal_handle.succeed()
        return FollowJointTrajectory.Result()


rclpy.init()
ex = MultiThreadedExecutor()
ex.add_node(Stub())
try:
    ex.spin()
except KeyboardInterrupt:
    pass
''' % (HOME,)


def kill(proc):
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def main():
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.action import ActionClient

    from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToPose

    tmp = tempfile.mkdtemp(prefix="abort_checks_")
    stub_py = os.path.join(tmp, "stub.py")
    with open(stub_py, "w") as f:
        f.write(STUB)
    stub_log = os.path.join(tmp, "stub.log")
    plan_log = os.path.join(tmp, "planner.log")

    stub = subprocess.Popen(["python3", stub_py], stdout=open(stub_log, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    planner = subprocess.Popen(
        ["ros2", "run", "rammp_curobo_ros", "planner_node", "--ros-args",
         "-r", "__node:=rammp_curobo", "-p", "config:=gen3_real.yaml",
         "-p", "execute:=true", "-p", "gripper_enabled:=false"],
        stdout=open(plan_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        print("warming up the planner (a minute or so)...")
        for _ in range(240):
            if "rammp_curobo ready" in open(plan_log).read():
                break
            if planner.poll() is not None:
                sys.exit("planner died:\n" + open(plan_log).read()[-2000:])
            time.sleep(1)
        else:
            sys.exit("planner never became ready")
        time.sleep(2)

        rclpy.init()
        node = rclpy.create_node("abort_checks")
        pc = ActionClient(node, PlanToPose, "/rammp_curobo/plan_to_pose")
        ec = ActionClient(node, ExecuteTrajectory,
                          "/rammp_curobo/execute_trajectory")
        if not (pc.wait_for_server(timeout_sec=30)
                and ec.wait_for_server(timeout_sec=30)):
            sys.exit("planner action servers never appeared")

        def spin(fut, timeout=60):
            end = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            return fut.result() if fut.done() else None

        goal = PlanToPose.Goal()
        goal.target = Pose()
        (goal.target.position.x, goal.target.position.y,
         goal.target.position.z) = 0.55, 0.30, 0.35
        (goal.target.orientation.x, goal.target.orientation.y,
         goal.target.orientation.z, goal.target.orientation.w) = 0.5, 0.5, 0.5, 0.5
        goal.start_joints = list(HOME)
        sent = spin(pc.send_goal_async(goal), 20)
        res = spin(sent.get_result_async(), 60).result
        if not res.success:
            sys.exit("plan failed: %s" % res.message)

        eg = ExecuteTrajectory.Goal()
        eg.trajectory = res.trajectory
        eg.speed_scale = 0.1                  # stretch it out
        acc = spin(ec.send_goal_async(eg), 20)
        print("execution accepted: %s" % (acc.accepted if acc else None))
        time.sleep(3.0)

        print("SIGINT to the planner, mid-stroke...")
        os.killpg(os.getpgid(planner.pid), signal.SIGINT)
        try:
            planner.wait(timeout=25)
        except subprocess.TimeoutExpired:
            print("FAIL: the planner did not exit")
        time.sleep(2)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        kill(stub)
        kill(planner)

    said = open(stub_log).read()
    print("\n--- controller saw ---\n%s" % said.strip())
    ok = "CANCEL RECEIVED" in said and "STOPPED" in said
    print("\n%s" % ("PASS — Ctrl+C stopped the arm and it holds" if ok else
                    "FAIL — the arm was left running; see docs/HARDWARE_BRINGUP.md"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
