#!/usr/bin/env python3
"""Green lights before a demo run — or the reason there won't be one.

    python3 scripts/preflight.py            # check everything, touch nothing
    python3 scripts/preflight.py --fix      # also kill stray rammp nodes

Run it BEFORE sweep_demo.launch.py. Exists because two field sessions
were derailed by invisible process state: a leftover driver holding the
Orbbec, and an orphaned cameras node feeding a whole run with the wrong
tuning while the launch's own cameras node refused to start. Checks:

  1. no stray rammp_curobo_ros processes (the launch starts its own)
  2. ROS_LOCALHOST_ONLY=1 in this shell
  3. the arm bringup: joint_trajectory_controller active, /joint_states
  4. the Orbbec: depth frames actually flowing
  5. TF: base_link -> end_effector_link resolvable
"""

import argparse
import os
import signal
import subprocess
import sys
import time

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print("  %-44s %s%s" % (name, "PASS" if ok else "FAIL",
                            ("  " + detail) if detail else ""))
    return ok


def stray_rammp():
    out = subprocess.run(["pgrep", "-af", "rammp_curobo_ros"],
                         capture_output=True, text=True).stdout
    strays = []
    for ln in out.splitlines():
        pid, _, cmd = ln.partition(" ")
        # only real node processes: an interpreter + the installed binary.
        # A plain substring match also catches SHELLS whose history
        # mentions the path (found the hard way).
        parts = cmd.split()
        if (len(parts) >= 2 and "/lib/rammp_curobo_ros/" in parts[1]
                and int(pid) != os.getpid()):
            strays.append((int(pid), parts[1].rsplit("/", 1)[-1]))
    return strays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true",
                    help="kill stray rammp nodes instead of just reporting")
    args = ap.parse_args()
    print("preflight:")
    # the ROS CLI daemon binds the ROS_DOMAIN_ID of whoever started it;
    # a test run on an isolated domain leaves `ros2 control`/`ros2 node`
    # hanging or blind for everyone after. Restart it into THIS shell's
    # domain before trusting any CLI query.
    subprocess.run(["ros2", "daemon", "stop"], capture_output=True, timeout=15)

    strays = stray_rammp()
    if strays and args.fix:
        for pid, name in strays:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(2.0)
        for pid, name in stray_rammp():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)
        strays = stray_rammp()
    check("no stray rammp nodes", not strays,
          "" if not strays else "%s — rerun with --fix"
          % ", ".join("%s(%d)" % (n, p) for p, n in strays))

    check("ROS_LOCALHOST_ONLY=1", os.environ.get("ROS_LOCALHOST_ONLY") == "1",
          "export it or nodes will not discover each other")

    try:
        ctl = subprocess.run(["ros2", "control", "list_controllers"],
                             capture_output=True, text=True, timeout=15).stdout
        check("joint_trajectory_controller active",
              "joint_trajectory_controller" in ctl and "active" in ctl,
              "" if ctl.strip() else "arm bringup not running?")
    except subprocess.TimeoutExpired:
        # the CLI can stall on a busy daemon; /joint_states below still
        # proves the bringup is alive
        check("joint_trajectory_controller active", False,
              "ros2 control CLI timed out — trusting the /joint_states check")

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    rclpy.init()
    node = rclpy.create_node("preflight")
    seen = {"depth": 0, "js": 0}
    node.create_subscription(Image, "/camera/depth/image_raw",
                             lambda _m: seen.__setitem__("depth", seen["depth"] + 1),
                             qos_profile_sensor_data)
    node.create_subscription(JointState, "/joint_states",
                             lambda _m: seen.__setitem__("js", seen["js"] + 1),
                             qos_profile_sensor_data)
    from tf2_ros import Buffer, TransformListener

    buf = Buffer()
    TransformListener(buf, node)
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    check("Orbbec depth flowing", seen["depth"] >= 5,
          "%d frames in 3 s" % seen["depth"])
    check("/joint_states flowing", seen["js"] >= 5,
          "%d messages in 3 s" % seen["js"])
    tf_ok = buf.can_transform("base_link", "end_effector_link",
                              rclpy.time.Time())
    check("TF base_link -> end_effector_link", bool(tf_ok))
    node.destroy_node()
    rclpy.shutdown()

    ok = all(RESULTS)
    print("\n%s" % ("ALL CLEAR — launch the demo."
                    if ok else "NOT READY — fix the FAILs above first."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
