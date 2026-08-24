#!/usr/bin/env python3
"""Where are the perceived boxes, is the arm seeing ITSELF, and by how
much is the camera off?

    python3 scripts/perception_debug.py                # 20 s report
    python3 scripts/perception_debug.py --apply        # ...and fix the config

Needs the arm bringup, the camera driver, and the cameras node. No
planner node: this loads its own kinematics for the arm model.

Two independent measurements:

  1. THE MAP — every box on /cameras/world_markers, how close it comes
     to the arm's own collision spheres (negative = overlaps the arm:
     that is the HOLD wedge, cuRobo cannot plan from a start state
     inside an obstacle), and what fraction of ticks it survives.

  2. THE CAMERA — the RAW depth frames registered against the arm's
     collision spheres (point-to-plane ICP, translation only). The arm
     is the one object in view whose pose is known exactly, so this is
     a calibration, not a guess. --apply writes the corrected mount_xyz
     into the camera config, keeping the old line as a comment.

     This replaces the earlier box-centroid estimate, which was biased
     by construction: the boxes it measured had already been through
     the self-filter, so only the far fringe of the displaced arm
     survived, and it measured the fringe. On the bench that
     over-corrected by ~5 cm. The registration reads the frames before
     any masking and is exact to a few mm on synthetic data with
     obstacles touching the arm.

Reads only. Never plans, never moves.
"""

import argparse
import collections
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from visualization_msgs.msg import MarkerArray

from rammp_curobo.config import resolve_config
from rammp_curobo.perception import SelfRegistrar, load_self_model, quat_to_mat
from rammp_curobo_ros.cameras import (
    REGISTER_DEFAULTS,
    cropped_points,
    load_camera_config,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--config", default="gen3_real.yaml")
    ap.add_argument("--camera-config", default="camera_orbbec_bench.yaml",
                    help="packaged name or path; its mount_xyz is what gets fixed")
    ap.add_argument("--near", type=float, default=0.05,
                    help="flag boxes closer than this to the arm (m)")
    ap.add_argument("--apply", action="store_true",
                    help="write the registered mount_xyz into the camera config")
    args = ap.parse_args()

    from rammp_curobo import CuRoboPlanner

    print("loading kinematics (%s)..." % args.config)
    planner = CuRoboPlanner.from_config(args.config, planner_overrides={"warmup": False})
    names = list(planner.joint_names)
    # depth sees the PHYSICAL surface, but link_spheres radii carry cuRobo's
    # load-time collision_sphere_buffer — register against raw radii or the
    # fit pulls the mount ~buffer toward the camera every run
    _, self_buffer = load_self_model(resolve_config("self_model_gen3_2f85.yaml"))
    cfg = load_camera_config(args.camera_config)
    if cfg.get("parent_frame") != "base_link":
        sys.exit("%s is not a fixed camera (parent_frame base_link) — registration "
                 "off the arm only makes sense for one" % args.camera_config)
    mount_xyz = np.array(cfg["mount_xyz"], dtype=float)
    q = cfg["mount_quat_xyzw"]
    rot = quat_to_mat(q[0], q[1], q[2], q[3])

    rclpy.init()
    node = rclpy.create_node("perception_debug")
    state = {"q": None, "boxes": None, "stamp": 0.0, "depth": None, "info": None}

    def js_cb(msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            state["q"] = [float(msg.position[idx[n]]) for n in names]
        except (KeyError, IndexError):
            pass

    def mk_cb(msg):
        out = []
        for m in msg.markers:
            if getattr(m, "action", 0) != 0:
                continue
            p, s = m.pose.position, m.scale
            out.append((m.ns + str(m.id), [p.x, p.y, p.z], [s.x, s.y, s.z]))
        state["boxes"] = out
        state["stamp"] = time.monotonic()

    def depth_cb(msg):
        if msg.encoding == "16UC1":
            d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
            state["depth"] = d.astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            state["depth"] = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).copy()

    def info_cb(msg):
        k = np.array(msg.k).reshape(3, 3)
        state["info"] = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    node.create_subscription(JointState, "/joint_states", js_cb, qos_profile_sensor_data)
    node.create_subscription(MarkerArray, "/cameras/world_markers", mk_cb, 1)
    node.create_subscription(Image, cfg["depth_topic"], depth_cb, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, cfg["info_topic"], info_cb, qos_profile_sensor_data)

    print("sampling %.0f s — keep the arm still\n" % args.seconds)
    seen, detail, worst_gap = collections.Counter(), {}, {}
    counts = collections.Counter()
    ticks, last = 0, -1.0
    reg = SelfRegistrar(frames=REGISTER_DEFAULTS["frames"])
    reg_out, frames_fed = None, 0
    end = time.monotonic() + args.seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if state["q"] is None:
            continue
        spheres = planner.link_spheres(state["q"])
        spheres = spheres[spheres[:, 3] > 0.0]
        # --- 2. camera registration off the raw frames
        if state["depth"] is not None and state["info"] is not None and not reg.done:
            # same crop the cameras node registers on — this tool writes the
            # config that node then uses, so it must measure the same cloud
            rd = REGISTER_DEFAULTS
            pts = cropped_points(state["depth"], state["info"], rot, mount_xyz,
                                 rd["stride"], float(cfg.get("min_range", 0.2)),
                                 float(cfg.get("max_range", 2.0)),
                                 rd["xy_extent"], rd["min_z"], rd["max_z"])
            state["depth"] = None
            frames_fed += 1
            reg_sph = spheres.copy()
            reg_sph[:, 3] -= self_buffer
            out = reg.feed(pts, reg_sph, mount_xyz)
            if out is not None:
                reg_out = out
        # --- 1. the map
        if state["boxes"] is None or state["stamp"] == last:
            continue
        last = state["stamp"]
        ticks += 1
        boxes = state["boxes"]
        counts[len(boxes)] += 1
        line = []
        for name, c, d in boxes:
            gap = planner.trajectory_clearance([state["q"]], [{"position": c, "dims": d}])
            key = tuple(np.round(np.array(c) / 0.05).astype(int))
            seen[key] += 1
            detail[key] = (c, d)
            worst_gap[key] = min(worst_gap.get(key, 1e9), gap)
            tag = "ON-ARM" if gap < 0 else "near" if gap < args.near else "ok"
            line.append("%s@%s %s(%.3f)" % (name, np.round(c, 2), tag, gap))
        if ticks % 5 == 1:
            print("  [%2d boxes] %s" % (len(boxes), "  ".join(line[:4])))

    node.destroy_node()
    rclpy.shutdown()

    if state["q"] is None:
        sys.exit("never saw /joint_states — is the arm bringup running?")

    # ------------------------------------------------------------- report 1
    if ticks:
        print("\n%d marker updates over %.0f s" % (ticks, args.seconds))
        print("box-count distribution: %s"
              % ", ".join("%d boxes x%d" % (k, v) for k, v in sorted(counts.items())))
        print("\n%-26s %-22s %-8s %-8s %s"
              % ("location (base_link)", "dims", "seen", "duty", "closest to arm"))
        on_arm = flappy = 0
        for key, n in seen.most_common():
            c, d = detail[key]
            duty, gap = n / float(ticks), worst_gap[key]
            note = ""
            if gap < 0:
                note, on_arm = "<-- OVERLAPS THE ARM", on_arm + 1
            elif gap < args.near:
                note = "<-- within %.0f cm of the arm" % (args.near * 100)
            if 0.15 < duty < 0.85:
                flappy += 1
                note += "  UNSTABLE"
            print("%-26s %-22s %-8d %-8.0f%% %.3f m %s"
                  % (np.round(c, 3), np.round(d, 3), n, duty * 100, gap, note))
        print("\nMAP VERDICT")
        print("  %d box(es) overlap the arm%s" % (on_arm, " — the arm is seeing itself; "
              "cuRobo cannot plan from inside an obstacle (HOLD)" if on_arm else ""))
        print("  %d box(es) flicker (duty 15-85%%)%s" % (flappy, "" if not flappy else
              " — raise occupied_at / min_voxels, or they are one object being re-clustered"))
    else:
        print("\nno marker updates — is the cameras node running?")

    # ------------------------------------------------------------- report 2
    print("\nCAMERA REGISTRATION (raw depth vs the arm's collision spheres)")
    if frames_fed == 0:
        print("  no depth frames on %s — is the camera driver running?" % cfg["depth_topic"])
        return
    if reg_out is None:
        print("  %d frames fed but the arm never held still for %d in a row"
              % (frames_fed, reg.frames))
        return
    if not reg_out["ok"]:
        print("  could not register: %s" % reg_out["reason"])
        return
    shift = reg_out["shift"]
    new = mount_xyz + shift
    print("  shift %s m  (|%.3f|)  from %d points, rms %.1f mm"
          % (np.round(shift, 4), float(np.linalg.norm(shift)), reg_out["used"],
             reg_out["rms"] * 1000))
    print("  mount_xyz now:       [%.5f, %.5f, %.5f]" % tuple(mount_xyz))
    print("  mount_xyz corrected: [%.5f, %.5f, %.5f]" % tuple(new))
    if np.linalg.norm(shift) < 0.01:
        print("  under 1 cm — the calibration is fine, nothing to apply")
        return
    path = _config_path(args.camera_config)
    if not args.apply:
        print("  re-run with --apply to write it into %s" % path)
        return
    with open(path) as f:
        text = f.read()
    out_lines, done = [], False
    for ln in text.splitlines():
        if ln.startswith("mount_xyz:") and not done:
            # append-only history: prior patch notes stay, new note goes on top
            out_lines.append("# registered off the arm by perception_debug --apply")
            out_lines.append("# (was %s)" % ln.strip())
            out_lines.append("mount_xyz: [%.5f, %.5f, %.5f]" % tuple(new))
            done = True
        else:
            out_lines.append(ln)
    if not done:
        sys.exit("mount_xyz: line not found in %s — nothing applied" % path)
    with open(path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print("  APPLIED to %s — restart the cameras node" % path)


def _config_path(name_or_path):
    import os

    if os.path.isfile(os.path.expanduser(name_or_path)):
        return os.path.expanduser(name_or_path)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "rammp_curobo_ros", "config", name_or_path)


if __name__ == "__main__":
    main()
