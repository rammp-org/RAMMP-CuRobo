#!/usr/bin/env python3
"""seeker — continuous seek controller: perceive, decide, act.

    ros2 run rammp_curobo_ros seeker --ros-args -p target:="go to the bottle"
    ros2 service call /seeker/set_target rammp_curobo_interfaces/srv/SetTarget \
        "{text: 'go to the cup'}"          # retarget any time; "" idles

One loop, no phases: look (YOLO on the wrist D405, 3-D localize), then
either hop (one short cuRobo-planned step toward the target, wrist flat
at its height, collision-checked against the live perceived world) or
survey (not seen for a while -> one fixed vantage pose). Hops BLOCK —
the arm only looks while parked, which is when frames are best; the
worst-case reaction to a moved target is one hop (~2 s).

Autonomous once targeted (owner decision 2026-08-19; the planner's
execute param, the executor gates, and the human on the e-stop are the
safety layers). Guards, all field-earned: acquisition needs 3 agreeing
frames; localization ignores depth nearer than 16 cm (the gripper's own
fingers); a target lifted off its resting height is held, never chased;
no camera data means no motion; winding (joint-family-flip) plans are
refused. Status streams on ~/status; detections on :8767.
"""

import time

import numpy as np
import rclpy
from std_msgs.msg import String

from rammp_curobo_ros.cameras import _ViewServer, ensure_sensor_params
from rammp_curobo_ros.seek_core import (
    D405Grabber,
    box_to_center,
    detect_all,
    glance_pose,
    joint_travel,
    load_detector,
    parse_target,
    purged_count,
    roll_about_tool_z,
    stable_fix,
    track_update,
    yaw_about_world_z,
)
from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW, TourDemo


class Seeker:
    LOST_S = 4.0        # no sighting this long -> drop the fix, survey
    DEAD_BAND = 0.05    # m; smaller target moves don't re-aim the arm
    LEASH = 0.35        # m; a sighting farther from the fix is a decoy
    LIFT_HOLD = 0.15    # m above resting height -> hold, never chase
    BLIND_S = 3.0       # s without camera streams -> no motion
    HOP_MAX = 0.20      # m
    HOP_MIN = 0.05      # m
    ARRIVE_TOL = 0.04   # m; within standoff+this = arrived
    WIND_RAD = 3.5      # joint travel above this = family flip, refused
    SURVEY = (0.0, np.radians(55.0))  # bearing, pitch of the vantage pose

    def __init__(self, node):
        self.node = node
        p = node.declare_parameter
        self.conf = float(p("conf", 0.4).value)
        self.standoff = float(p("standoff", 0.18).value)
        self.speed = min(max(float(p("speed", 0.25).value), 0.05), 0.25)
        self.wrist_roll = float(p("wrist_roll", np.pi / 2.0).value)
        self.weights = str(p("weights", "~/yolo11s-seg.pt").value)
        initial = str(p("target", "").value)

        self.demo = TourDemo(node)
        self.grab = D405Grabber(node)
        ensure_sensor_params(node, self.grab.cfg)

        from rammp_curobo_interfaces.srv import SetIgnoreRegion, SetTarget

        self._SetIgnoreRegion = SetIgnoreRegion
        self.ignore_cli = node.create_client(
            SetIgnoreRegion, "/cameras/set_ignore_region"
        )
        node.create_service(SetTarget, "~/set_target", self._set_target_cb)
        self.status_pub = node.create_publisher(String, "~/status", 1)
        self._last_status = "starting"
        self._last_pub = 0.0

        self.view = None
        if bool(p("view", True).value):
            try:
                self.view = _ViewServer(int(p("view_port", 8767).value))
                node.get_logger().info("detection view: http://<host>:8767/")
            except OSError as e:
                node.get_logger().warn("detection view failed (%s)" % e)

        self.model = None
        self.target = None
        self._pending = initial or None
        self.fix = None            # dict(pos, stamp, base_z, extent)
        self.recent = []           # [(pos, t)] acquisition candidates
        self.last_seen = 0.0
        self.last_data = time.monotonic()
        self.region_pos = None
        self.region_on = True      # a predecessor may have left one
        self.at_survey = False

    # ------------------------------------------------------------ plumbing
    def _set_target_cb(self, req, res):
        text = req.text.strip()
        if not text:
            self._pending = ""
            res.success, res.message = True, "target cleared — idling"
            return res
        cls = parse_target(text)
        if cls is None:
            res.success = False
            res.message = "no COCO class in %r (bottle, cup, bowl, ...)" % text
            return res
        self._pending = text
        res.success, res.message, res.resolved_class = True, "seeking " + cls, cls
        return res

    def _status(self, text):
        now = time.monotonic()
        if text != self._last_status:
            self._last_status = text
            self.node.get_logger().info(text)
        elif now - self._last_pub < 0.5:
            return
        self.status_pub.publish(String(data=text))
        self._last_pub = now

    def _set_region(self, center, extent):
        if not self.ignore_cli.service_is_ready():
            return
        req = self._SetIgnoreRegion.Request()
        req.center.x, req.center.y, req.center.z = (float(v) for v in center)
        d = [float(max(extent[0], 0.05)) + 0.04] * 2 + [
            float(max(extent[1], 0.05)) + 0.04
        ]
        req.dims.x, req.dims.y, req.dims.z = d
        self.region_on = True  # request may land even if the reply times out
        fut = self.ignore_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        res = fut.result()
        self.region_pos = np.asarray(center, dtype=float).copy()
        if res is not None and purged_count(res.message) != 0:
            time.sleep(1.5)  # the purge reaches the planner via the 2 Hz push

    def clear_region(self):
        self.region_pos = None
        if self.region_on and self.ignore_cli.service_is_ready():
            fut = self.ignore_cli.call_async(self._SetIgnoreRegion.Request())
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
            self.region_on = False

    def _tool_pos(self):
        """Fingertip midpoint in base_link (tool_frame = ee + 0.12 z)."""
        try:
            tr = self.grab.tf_buffer.lookup_transform(
                "base_link", "end_effector_link", rclpy.time.Time()
            )
        except Exception:
            return None
        from rammp_curobo.perception import quat_to_mat

        q, t = tr.transform.rotation, tr.transform.translation
        rot = quat_to_mat(q.x, q.y, q.z, q.w)
        return np.array([t.x, t.y, t.z]) + rot[:, 2] * 0.120

    def _flat_quat(self, bearing):
        return list(
            roll_about_tool_z(
                yaw_about_world_z(HOME_QUAT_XYZW, bearing), self.wrist_roll
            )
        )

    def _move(self, pos, quat, label):
        """Plan + BLOCKING execute one motion; False on refusal."""
        plan = self.demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            self._status("%s: no plan — holding" % label)
            return False
        if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
            self._status("%s: plan winds the arm — refused" % label)
            return False
        self._status(label)
        if not self.demo.run(plan.trajectory, self.speed):
            self._status("%s: execution failed — holding" % label)
            return False
        return True

    # ------------------------------------------------------------ perceive
    def _look(self, now):
        """One frame -> detections -> fix update/acquisition."""
        shot = self.grab.shot(timeout_s=0.5)
        if shot is None:
            if not self.grab.missing():
                self.last_data = now  # streams alive, frame merely late
            return
        self.last_data = now
        frame, depth, intr, rot, trans = shot
        hits = detect_all(self.model, frame, self.target, self.conf)
        sights = []
        for xyxy, cf, mask in hits:
            loc = box_to_center(xyxy, depth, mask=mask, min_depth=0.16, **intr)
            if loc is not None:
                sights.append((0, rot @ loc[0] + trans, np.asarray(loc[1]), cf))
        if self.view is not None:
            self._render(frame, hits, intr, rot, trans, len(sights))
        if not sights:
            return
        if self.fix is not None:
            near = track_update(self.fix["pos"], sights,
                                max_jump=self.LEASH, min_move=0.0)
            if near is None:
                return
            if near[2] < self.fix["base_z"] - 0.06:
                return  # objects don't sink through their surface
            tool = self._tool_pos()
            if tool is not None and float(
                np.linalg.norm(near[:2] - tool[:2])
            ) < self.standoff - 0.03:
                return  # inside the standoff = self-sighting
            s = min(sights, key=lambda s: float(np.linalg.norm(s[1] - near)))
            self.fix["pos"] = 0.6 * self.fix["pos"] + 0.4 * near
            self.fix["extent"] = np.maximum(self.fix["extent"], s[2])
            self.last_seen = now
            return
        # acquisition: n parked frames agreeing (one frame never moves the arm)
        best = max(sights, key=lambda s: s[3])
        self.recent = [(p, t) for p, t in self.recent if now - t < 1.5]
        self.recent.append((best[1], now))
        pos = stable_fix(self.recent)
        if pos is not None:
            self.fix = dict(pos=pos, base_z=float(pos[2]),
                            extent=np.asarray(best[2]))
            self.recent = []
            self.last_seen = now
            self.at_survey = False
            self._status("ACQUIRED %s at [%.2f, %.2f, %.2f]"
                         % (self.target, pos[0], pos[1], pos[2]))

    def _render(self, frame, hits, intr, rot, trans, n_localized):
        import cv2

        img = frame.copy()
        for xyxy, cf, _ in hits:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, "%s %.2f" % (self.target, cf), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        if self.fix is not None:
            pc = (self.fix["pos"] - trans) @ rot
            if pc[2] > 0.05:
                u = int(intr["fx"] * pc[0] / pc[2] + intr["cx"])
                v = int(intr["fy"] * pc[1] / pc[2] + intr["cy"])
                cv2.drawMarker(img, (u, v), (255, 120, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, "%s | %d det / %d localized"
                    % (self._last_status[:70], len(hits), n_localized),
                    (8, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.view.update(img)

    # ---------------------------------------------------------------- tick
    def tick(self):
        now = time.monotonic()
        if self._pending is not None:
            text, self._pending = self._pending, None
            cls = parse_target(text) if text else None
            if text and cls is None:
                self._status("cannot resolve %r — still %s"
                             % (text, self.target or "IDLE"))
            elif cls != self.target:
                self.target = cls
                self.fix = None
                self.recent = []
                self.at_survey = False
                self.clear_region()
                if cls and self.model is None:
                    self._status("loading detector...")
                    self.model = load_detector(self.weights)
        if self.target is None:
            self._status("IDLE — set a target via ~/set_target")
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.1)
            return
        self._look(now)
        if now - self.last_data > self.BLIND_S:
            self._status("no camera data (%s) — motion paused"
                         % (", ".join(self.grab.missing()) or "TF"))
            return
        if self.fix is not None and now - self.last_seen > self.LOST_S:
            self._status("%s lost — surveying" % self.target)
            self.fix = None
            self.clear_region()
        if self.fix is None:
            self._status("SEARCHING for %s" % self.target)
            if not self.at_survey and now - self.last_seen > self.LOST_S:
                pos, quat = glance_pose(*self.SURVEY)
                self.at_survey = self._move(pos, quat, "moving to survey pose")
            return
        pos = self.fix["pos"]
        if pos[2] > self.fix["base_z"] + self.LIFT_HOLD:
            self._status("%s lifted — holding, not chasing a hand" % self.target)
            return
        tool = self._tool_pos()
        if tool is None:
            self._status("no TF to the arm — holding")
            return
        gap = float(np.linalg.norm(pos[:2] - tool[:2]))
        if gap <= self.standoff + self.ARRIVE_TOL:
            self._status("ARRIVED at %s (gap %.2f m)" % (self.target, gap))
            return
        if (
            self.region_pos is None
            or float(np.linalg.norm(pos - self.region_pos)) > 0.10
        ):
            self._set_region(pos, self.fix["extent"])
        hop = min(self.HOP_MAX, max(self.HOP_MIN, 0.4 * (gap - self.standoff)))
        hop = min(hop, gap - self.standoff)
        step = tool[:2] + (pos[:2] - tool[:2]) / gap * hop
        z = max(float(pos[2]) + 0.03, 0.10)
        bearing = float(np.arctan2(pos[1] - step[1], pos[0] - step[0]))
        self.at_survey = False
        self._move(
            [float(step[0]), float(step[1]), z],
            self._flat_quat(bearing),
            "SERVOING to %s [%.2f, %.2f, %.2f] (gap %.2f m)"
            % (self.target, pos[0], pos[1], pos[2], gap),
        )

    def shutdown(self):
        try:
            self.clear_region()
        except Exception:
            pass


def main():
    from rclpy.signals import SignalHandlerOptions

    # own the SIGINT: rclpy's handler would kill the context before the
    # ignore-region cleanup could run
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("seeker")
    seeker = Seeker(node)
    print("*** SEEKER up — autonomous once targeted; Ctrl+C stops. ***")
    try:
        while rclpy.ok():
            try:
                seeker.tick()
            except SystemExit as e:
                seeker._status("recoverable: %s — retrying" % e)
                time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        seeker.shutdown()
        rclpy.try_shutdown()
    print("\nseeker stopped — arm holds")


if __name__ == "__main__":
    main()
