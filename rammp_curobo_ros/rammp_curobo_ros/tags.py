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

from rammp_curobo_ros.cameras import _ViewServer
from rammp_curobo_ros.grasps import mat_to_quat_xyzw
from rammp_curobo_ros.seek_core import joint_travel, stable_fix
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


class TagGrabber:
    """Colour frame + intrinsics + camera pose. No depth needed."""

    def __init__(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener

        from rammp_curobo_ros.cameras import load_camera_config

        self.node = node
        cfg = load_camera_config("camera_d405_wrist.yaml")
        self.mount_xyz = np.asarray(cfg["mount_xyz"], dtype=float)
        self.mount_quat = list(cfg["mount_quat_xyzw"])
        self.parent = cfg["parent_frame"]
        ns = cfg["depth_topic"].rsplit("/depth/", 1)[0]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.color = self.info = None
        self.stamp = None
        node.create_subscription(Image, ns + "/color/image_raw",
                                 self._color_cb, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, ns + "/color/camera_info",
                                 self._info_cb, qos_profile_sensor_data)

    def _color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            self.color = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()
            self.stamp = msg.header.stamp

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.k = k
        self.dist = np.array(msg.d, dtype=float).ravel()
        self.info = True

    def shot(self, timeout_s=1.0):
        """(bgr, K, dist, R_base_cam, t_base_cam) or None."""
        from rammp_curobo.perception import quat_to_mat

        self.color = None
        t0 = time.monotonic()
        while self.color is None or self.info is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.monotonic() - t0 > timeout_s:
                return None
        tr = None
        for when in (rclpy.time.Time.from_msg(self.stamp), rclpy.time.Time()):
            try:
                tr = self.tf_buffer.lookup_transform("base_link", self.parent, when)
                break
            except Exception:
                continue
        if tr is None:
            return None
        q, t = tr.transform.rotation, tr.transform.translation
        r_p = quat_to_mat(q.x, q.y, q.z, q.w)
        qx, qy, qz, qw = self.mount_quat
        rot = r_p @ quat_to_mat(qx, qy, qz, qw)
        trans = r_p @ self.mount_xyz + np.array([t.x, t.y, t.z])
        return self.color, self.k, self.dist, rot, trans


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
        self.grab = TagGrabber(node)
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
        shot = self.grab.shot()
        if shot is None:
            return
        frame, k, dist, rot_cam, trans_cam = shot
        corners, ids, _ = self.detector.detectMarkers(frame)
        pick = None
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if self.tag_id < 0 or int(i) == self.tag_id:
                    pick = (c.reshape(4, 2).astype(np.float64), int(i))
                    break
        if self.view is not None:
            vis = frame.copy()
            if ids is not None:
                self.cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            self.cv2.putText(vis, self._last_status[:70],
                             (8, vis.shape[0] - 10),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.view.update(vis)
        if pick is None:
            return
        img_pts, tag_id = pick
        ok, rvec, tvec = self.cv2.solvePnP(
            self.objp, img_pts, k, dist,
            flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return
        r_tag_cam, _ = self.cv2.Rodrigues(rvec)
        pos = rot_cam @ tvec.ravel() + trans_cam
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
