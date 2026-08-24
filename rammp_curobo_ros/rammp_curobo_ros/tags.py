#!/usr/bin/env python3
"""tag_follow — see a fiducial, plan to a point right in front of it.

    ros2 run rammp_curobo_ros tag_follow
    ros2 run rammp_curobo_ros tag_follow --ros-args -p marker_size:=0.06 -p tag_id:=7

The simple, deterministic path: an ArUco/AprilTag marker gives full
6-DoF pose from ONE colour image (marker size + camera intrinsics —
no depth, no network, no model), TF lifts it into base_link, and cuRobo
plans to a standoff pose on the tag's normal, facing it, through the
live perceived world. Move the tag, the arm follows.

Reach (measured on this arm 2026-08-20 — tool_frame is 12 cm ahead of
the flange, so standing off SHRINKS the radius into a dead zone):

    tag at 0.70-0.80 m -> 0.15 m standoff plans fine
    tag at 0.60 m      -> only 0.05-0.10 m
    tag at 0.50 m      -> nothing plans; move the tag out

The node ladders the standoff down before giving up, and says so.

Needs: planner (execute:=true), arm bringup, the D405 colour stream
(aligned depth NOT required here), and the cameras node if you want
obstacle avoidance. Print a marker with scripts/make_tag.py.
"""

import time

import numpy as np
import rclpy
from std_msgs.msg import String

from rammp_curobo.perception import mat_to_quat_xyzw
from rammp_curobo_ros.cameras import _ViewServer
from rammp_curobo_ros.seek_core import D405Grabber, joint_travel, stable_fix
from rammp_curobo_ros.tour_demo import TourDemo


def look_at_pose(tag_pos, tag_rot, standoff):
    """Standoff pose on the tag's normal, tool pointing AT the tag.

    tag_rot is the tag's base-frame rotation; its +Z is the face normal
    (out of the tag, toward whoever is looking at it)."""
    n = np.asarray(tag_rot, dtype=float)[:, 2]
    pos = np.asarray(tag_pos, dtype=float) + n * float(standoff)
    tz = -n                                  # look back at the tag
    up = np.array([0.0, 0.0, 1.0])
    tx = np.cross(up, tz)
    if np.linalg.norm(tx) < 1e-6:            # tag faces straight up/down
        tx = np.array([1.0, 0.0, 0.0])
    tx /= np.linalg.norm(tx)
    rot = np.stack([tx, np.cross(tz, tx), tz], axis=1)
    return pos, mat_to_quat_xyzw(rot)


def tag_pose_from_frame(frame, detector, objp, k, dist, tag_id=-1):
    """Pure detect->pose: (tag_id, R_tag_cam (3,3), tvec (3,)) or None.

    First marker wins unless tag_id >= 0 picks one; IPPE_SQUARE is the
    exact planar-square solver, so objp must keep the tag-frame corner
    order."""
    import cv2

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if tag_id < 0 or int(i) == tag_id:
            ok, rvec, tvec = cv2.solvePnP(
                objp, c.reshape(4, 2).astype(np.float64), k, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None
            r_tag_cam, _ = cv2.Rodrigues(rvec)
            return int(i), r_tag_cam, tvec.ravel()
    return None


class TagFollower:
    LOST_S = 3.0        # no sighting this long -> stop, wait
    DEAD_BAND = 0.03    # m; smaller tag moves don't re-plan
    WIND_RAD = 3.5      # joint travel above this = family flip, refused
    LADDER = (1.0, 0.66, 0.4)  # standoff multipliers tried in order

    def __init__(self, node):
        import cv2

        self.cv2 = cv2
        self.node = node
        p = node.declare_parameter
        self.marker_size = float(p("marker_size", 0.05).value)
        self.standoff = float(p("standoff", 0.15).value)
        self.speed = min(max(float(p("speed", 0.25).value), 0.05), 0.25)
        self.tag_id = int(p("tag_id", -1).value)      # -1 = any
        dict_name = str(p("dictionary", "DICT_4X4_50").value)
        self.demo = TourDemo(node)
        self.grab = D405Grabber(node, need_depth=False)
        self.status_pub = node.create_publisher(String, "~/status", 1)
        self._last_status = ""

        aruco = cv2.aruco
        if not hasattr(aruco, dict_name):
            raise SystemExit("unknown dictionary %r" % dict_name)
        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(getattr(aruco, dict_name)),
            aruco.DetectorParameters(),
        )
        # marker corners in the tag's own frame: +X right, +Y up, +Z out
        s = self.marker_size / 2.0
        self.objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                             dtype=np.float64)

        self.view = None
        if bool(p("view", True).value):
            try:
                self.view = _ViewServer(int(p("view_port", 8768).value))
                node.get_logger().info("tag view: http://<host>:8768/")
            except OSError as e:
                node.get_logger().warn("tag view failed (%s)" % e)

        self.recent = []          # [(pos, t)] for stability
        self.rot = None           # latest tag rotation (base frame)
        self.fix = None           # committed tag position
        self.goal = None          # last commanded standoff pose
        self.last_seen = 0.0
        self._next_try = 0.0

    def _status(self, text):
        if text != self._last_status:
            self._last_status = text
            self.node.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def _detect(self, now):
        shot = self.grab.shot(timeout_s=1.0)
        if shot is None:
            return
        frame, _depth, _intr, rot_cam, trans_cam = shot
        if self.view is not None:
            # the view wants EVERY marker's corners; the pure helper
            # returns only the picked pose — one extra detect is cheap
            vis = frame.copy()
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is not None:
                self.cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            self.cv2.putText(vis, self._last_status[:70],
                             (8, vis.shape[0] - 10),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.view.update(vis)
        hit = tag_pose_from_frame(frame, self.detector, self.objp,
                                  self.grab.k, self.grab.dist, self.tag_id)
        if hit is None:
            return
        tag_id, r_tag_cam, tvec = hit
        pos = rot_cam @ tvec + trans_cam
        self.rot = rot_cam @ r_tag_cam
        self.tag_id_seen = tag_id
        self.last_seen = now
        self.recent = [(q, t) for q, t in self.recent if now - t < 2.0]
        self.recent.append((pos, now))
        stable = stable_fix(self.recent, tol=0.03, n=3)
        if stable is not None:
            self.fix = stable

    def tick(self):
        now = time.monotonic()
        self._detect(now)
        if self.fix is None or now - self.last_seen > self.LOST_S:
            self._status("waiting for a tag%s"
                         % ("" if self.tag_id < 0 else " (id %d)" % self.tag_id))
            self.fix = None
            return
        if self.goal is not None and (
            float(np.linalg.norm(self.fix - self.goal)) < self.DEAD_BAND
        ):
            self._status("AT the tag (standoff %.2f m) — holding" % self.standoff)
            return
        if now < self._next_try:
            return
        self._next_try = now + 3.0
        for mult in self.LADDER:
            off = self.standoff * mult
            pos, quat = look_at_pose(self.fix, self.rot, off)
            plan = self.demo.plan_pose_from([float(v) for v in pos],
                                            [float(v) for v in quat], None)
            if plan is None or not plan.success:
                continue
            if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
                continue
            self._status("MOVING to %.2f m in front of tag %s"
                         % (off, getattr(self, "tag_id_seen", "?")))
            if self.demo.run(plan.trajectory, self.speed):
                self.goal = self.fix.copy()
                self._status("ARRIVED %.2f m in front of the tag" % off)
            return
        self._status(
            "tag at r=%.2f m: no standoff plans (nearer than ~0.6 m the "
            "pose folds the arm) — move the tag outward"
            % float(np.hypot(self.fix[0], self.fix[1]))
        )


def main():
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("tag_follow")
    follower = TagFollower(node)
    print("*** TAG FOLLOW — moves autonomously when a tag is seen. "
          "Ctrl+C stops. ***")
    try:
        while rclpy.ok():
            follower.tick()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
    print("\ntag_follow stopped — arm holds")


if __name__ == "__main__":
    main()
