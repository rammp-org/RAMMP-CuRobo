#!/usr/bin/env python3
"""Build a digital twin of the workspace: slow +/-90 deg wrist-camera sweep.

The arm holds its wrist pose and rotates its base joint from 90 deg left to
90 deg right in discrete stations (stop-and-capture: TF and depth are
exactly synchronized when stationary, which is where fused-scan error
comes from). It runs the row twice — a steep near-field view and a raised
far-field view — so both the tabletop and distant obstacles are covered,
then fuses every station's depth
into one point cloud, clusters it into boxes, and writes the world YAML.
The planner then starts against it in another terminal:

    ros2 launch rammp_curobo_ros planner.launch.py \
        world:=$HOME/.ros/rammp_curobo/scanned_world.yaml execute:=true

Usage (planner node with execute:=true and the camera driver both running;
on the REAL arm a human holds the e-stop for the whole sweep):

    ros2 run rammp_curobo_ros sweep_scan --camera camera_sim_d405.yaml --apply
    ros2 run rammp_curobo_ros sweep_scan --camera camera_d405_wrist.yaml \
        --apply                       # real D405 (mount YAML measured first!)
    ros2 run rammp_curobo_ros sweep_scan --camera ... --dry-capture --debug
                                      # no motion: one capture from here

Every station move is planned and executed through the planner node's own
safety-gated actions (collision-checked against its CURRENT world, scaled
speed, cancellable). Stations whose pose can't be reached collision-free
are skipped with a warning. What the camera never saw stays UNKNOWN — the
conservative table plane covers below, but do not treat unscanned space
behind the arm as certified free.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

from rammp_curobo.config import PLANNER_DEFAULTS
from rammp_curobo.geometry import ang_diff
from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints
from rammp_curobo_ros.scan_common import (
    NODE_NAMESPACE,
    DepthCameraGrabber,
    add_cluster_args,
    apply_world,
    cluster_boxes,
    load_camera_config,
    report_boxes,
    spin_until_done,
    write_world_yaml,
)

JOINTS = list(PLANNER_DEFAULTS["joint_names"])
PREFIX = NODE_NAMESPACE

# The "periscope" scan posture (found by validated FK search): wrist high
# (camera ~0.78 m) and pulled toward the axis (r ~0.12), view pitched ~42
# deg down. Each station covers the ground annulus r ~0.4-1.1 m, the
# fingertips stay far above the workspace (stations never get refused for
# dipping near obstacles), and the yaw sweep is a compact in-place twirl.
# Model-valid in the sim kitchen and the bare-bench world at all yaws.
SCAN_POSE = [0.0, -0.5, 3.1416, -1.4, 0.0, -1.4, 1.571]


class SweepDriver(DepthCameraGrabber):
    """Grabber + joint states + planner action clients in one node."""

    def __init__(self, camera_cfg):
        super().__init__(camera_cfg, node_name="rammp_curobo_sweep")
        self._q = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.plan_client = ActionClient(self, PlanToJoints, PREFIX + "/plan_to_joints")
        self.exec_client = ActionClient(
            self, ExecuteTrajectory, PREFIX + "/execute_trajectory"
        )

    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self._q = [float(msg.position[idx[n]]) for n in JOINTS]
        except (KeyError, IndexError):
            pass

    def joints(self, timeout_s=10.0):
        t0 = time.monotonic()
        while self._q is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - t0 > timeout_s:
                sys.exit("no /joint_states — is the arm bringup running?")
        return list(self._q)

    def _action(self, client, goal, timeout_s):
        send = spin_until_done(self, client.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(self, send.get_result_async(), timeout_s)
        return None if wrapped is None else wrapped.result

    def move_to(self, q_target, speed_scale, retries=2):
        """Plan + execute one station move via the gated planner node.

        Returns True on arrival, False to skip this station (plan failed —
        the pose is not collision-free reachable in the CURRENT world).

        Execution failures are discriminated by where the arm actually is:
        the real Gen3's controller occasionally reports success without
        moving AT ALL (JTC "Goal reached" with zero motion, goal tolerances
        disabled in the stock kortex config; observed twice on 2026-08-13,
        roughly 1 move in 12). If the arm is still exactly at the move's
        start, that is the no-motion hiccup — safe to replan and retry.
        If it stopped PARTWAY, that may be physical contact — abort the
        whole scan immediately; the arm holds.
        """
        if not self.plan_client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running (need execute:=true)")
        for attempt in range(retries + 1):
            q_before = self.joints()
            plan = self._action(
                self.plan_client,
                PlanToJoints.Goal(target_joints=[float(v) for v in q_target]),
                timeout_s=120.0,
            )
            if plan is None or not plan.success:
                self.get_logger().warning(
                    "station unreachable (%s) — skipping"
                    % ("no answer" if plan is None else plan.message)
                )
                return False
            goal = ExecuteTrajectory.Goal(
                trajectory=plan.trajectory, speed_scale=float(speed_scale)
            )
            res = self._action(self.exec_client, goal, timeout_s=180.0)
            if res is not None and res.success:
                return True
            moved = max(abs(ang_diff(a, b)) for a, b in zip(self.joints(), q_before))
            if res is not None and moved < 0.05 and attempt < retries:
                self.get_logger().warning(
                    "controller reported '%s' but the arm never moved "
                    "(%.3f rad from where it started) — the known kortex "
                    "no-motion hiccup; retrying (%d/%d)"
                    % (res.message, moved, attempt + 1, retries)
                )
                time.sleep(0.5)
                continue
            sys.exit(
                "station move FAILED (%s; arm moved %.3f rad from the move "
                "start) — aborting the scan; arm holds"
                % ("no answer" if res is None else res.message, moved)
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--camera",
        default="camera_sim_d405.yaml",
        help="camera YAML (packaged name or path)",
    )
    ap.add_argument(
        "--span-deg",
        type=float,
        default=90.0,
        help="sweep half-range around the starting base yaw",
    )
    ap.add_argument("--step-deg", type=float, default=15.0)
    ap.add_argument(
        "--pitches-deg",
        type=float,
        nargs="+",
        default=[0.0, 31.0],
        help="joint_6 offsets per row relative to the scan pose "
        "(0 = its steep near-field view; positive raises the gaze "
        "toward the far field/walls)",
    )
    ap.add_argument(
        "--scan-pose",
        type=float,
        nargs=7,
        default=SCAN_POSE,
        metavar="RAD",
        help="base joint posture for the sweep (yaw is taken from the "
        "arm's current joint_1; the arm returns to its starting pose "
        "afterwards)",
    )
    ap.add_argument("--speed-scale", type=float, default=0.25)
    add_cluster_args(ap, frames=4, min_points_per_voxel=3, min_voxels=5, max_boxes=45)
    ap.add_argument(
        "--settle",
        type=float,
        default=0.4,
        help="seconds to settle at a station before capturing",
    )
    ap.add_argument(
        "--dry-capture",
        action="store_true",
        help="NO MOTION: single capture from the current pose",
    )
    args = ap.parse_args()

    rclpy.init()
    node = SweepDriver(load_camera_config(args.camera))

    clouds = []

    def capture_here(tag):
        pts = node.capture_points(args.frames)
        if args.debug:
            R, t = node.camera_pose()
            print(
                "[%s] cam at %s, view %s, %d pts"
                % (
                    tag,
                    np.round(t, 2).tolist(),
                    np.round(R[:, 2], 2).tolist(),
                    len(pts),
                )
            )
        if len(pts):
            clouds.append(pts)

    if args.dry_capture:
        capture_here("dry")
    else:
        q0 = node.joints()
        base = list(args.scan_pose)
        base[0] = q0[0]  # sweep around the CURRENT heading
        yaw0, j6_0 = base[0], base[5]
        span = np.radians(args.span_deg)
        n_st = max(2, int(round(2 * args.span_deg / args.step_deg)) + 1)
        yaws = np.linspace(yaw0 - span, yaw0 + span, n_st)
        total = len(args.pitches_deg) * n_st
        print(
            "Sweep: %d stations (%d yaw x %d pitch rows), ~%.0f min at "
            "scale %.2f — Ctrl+C aborts, arm holds."
            % (
                total,
                n_st,
                len(args.pitches_deg),
                total * (2.5 / args.speed_scale * 0.25 + 1.2) / 60.0,
                args.speed_scale,
            )
        )
        done = 0
        for row, pitch in enumerate(args.pitches_deg):
            ordered = yaws if row % 2 == 0 else yaws[::-1]
            for yaw in ordered:
                q = list(base)
                q[0] = float(yaw)
                q[5] = float(j6_0 + np.radians(pitch))
                if not node.move_to(q, args.speed_scale):
                    continue
                time.sleep(args.settle)
                done += 1
                capture_here("st%02d y%+.0f p%+.0f" % (done, np.degrees(yaw), pitch))
        print("Returning to start pose...")
        node.move_to(q0, args.speed_scale)
        print("Captured %d/%d stations." % (done, total))

    if not clouds:
        sys.exit("no points captured — nothing to build a world from")
    pts = np.concatenate(clouds)
    print("fused cloud: %d points from %d captures" % (len(pts), len(clouds)))
    boxes, n_found = cluster_boxes(
        pts, args.voxel, args.min_points_per_voxel, args.min_voxels, args.max_boxes
    )
    np.save(args.out.replace(".yaml", "_cloud.npy"), pts)
    write_world_yaml(args.out, boxes, args.inflate, "by sweep_scan")
    report_boxes(boxes, n_found, args.out)
    print(
        "start the planner against it with:\n  ros2 launch rammp_curobo_ros "
        "planner.launch.py world:=%s execute:=true" % args.out
    )

    if args.apply:
        apply_world(node, args.out)


if __name__ == "__main__":
    main()
