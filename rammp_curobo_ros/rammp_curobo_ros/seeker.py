#!/usr/bin/env python3
"""seeker — continuous seek controller: perceive, decide, act.

    ros2 run rammp_curobo_ros seeker --ros-args -p target:="go to the bottle"
    ros2 service call /seeker/set_target rammp_curobo_interfaces/srv/SetTarget \
        "{text: 'go to the cup'}"          # retarget any time; "" idles

One loop, no phases: LOOK from a parked pose (YOLO on the wrist D405,
3-D localize), then act on the fix — survey when there isn't one, and
otherwise GO STRAIGHT AT IT: GraspGenX turns the masked depth into
ranked 6-DoF grasps and cuRobo plans pre-grasp -> grasp before the
gripper closes (grasps.py holds the frame contract). `grasp:=false`
approaches to a standoff instead, using the same pose geometry.

There is deliberately no "creep closer first" phase. tool_frame sits
12 cm AHEAD of the flange, so a flat wrist inside ~0.5 m radius folds
the arm into itself — every inward hop was IK_FAIL by construction
(field 2026-08-20). Poses at or near the object are reachable; poses
part-way to it are not.

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
    stable_fix,
    track_update,
)
from rammp_curobo_ros.tour_demo import TourDemo


class Seeker:
    LOST_S = 4.0        # no sighting this long -> drop the fix, survey
    AGREE_S = 2.5       # acquisition window: 3 sightings must land inside
    LEASH = 0.35        # m; a sighting farther from the fix is a decoy
    LIFT_HOLD = 0.15    # m above resting height -> hold, never chase
    BLIND_S = 3.0       # s without camera streams -> no motion
    REACH_MAX = 0.80    # m planar; beyond this the Gen3 cannot reach it
    WIND_RAD = 3.5      # joint travel above this = family flip, refused
    SURVEY = (0.0, np.radians(55.0))  # bearing, pitch of the vantage pose

    def __init__(self, node):
        self.node = node
        p = node.declare_parameter
        self.conf = float(p("conf", 0.4).value)
        self.standoff = float(p("standoff", 0.18).value)
        self.speed = min(max(float(p("speed", 0.25).value), 0.05), 0.25)
        self.weights = str(p("weights", "~/yolo11s-seg.pt").value)
        initial = str(p("target", "").value)
        # grasping: off -> stop at standoff (the old behaviour)
        self.do_grasp = bool(p("grasp", True).value)
        self.grasp_endpoint = str(p("grasp_endpoint", "tcp://127.0.0.1:5556").value)
        self.tool_offset = float(p("tool_offset", 0.120).value)
        self.pregrasp_standoff = float(p("pregrasp_standoff", 0.10).value)
        self.grasp_score_min = float(p("grasp_score_min", 0.5).value)

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

        from std_srvs.srv import Trigger

        self.open_cli = node.create_client(Trigger, "/rammp_curobo/open_gripper")
        self.close_cli = node.create_client(Trigger, "/rammp_curobo/close_gripper")
        self.grasp_cli = None       # lazy: only when we actually grasp
        self.last_frame = None      # depth+intr+pose+mask for grasp gen
        self.grasped = False        # terminal: object is in the gripper
        self._next_try_t = 0.0      # one attempt at a time (no plan spam)

        self.model = None
        self.target = None
        self._pending = initial or None
        self.fix = None            # dict(pos, stamp, base_z, extent)
        self.recent = []           # [(pos, t)] acquisition candidates
        self.last_seen = 0.0
        self.last_data = time.monotonic()
        self.region_pos = None
        self.region_on = True      # a predecessor may have left one
        self._startup_cleared = False
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
        t0 = time.monotonic()
        ok = self.demo.run(plan.trajectory, self.speed)
        # a blocking hop can outlast LOST_S while the camera is busy
        # moving — don't let its own motion time declare the target lost
        moved_for = time.monotonic() - t0
        self.last_seen += moved_for
        self.last_data += moved_for
        if not ok:
            self._status("%s: execution failed — holding" % label)
            return False
        return True

    # ------------------------------------------------------------ perceive
    def _look(self, now):
        """One frame -> detections -> fix update/acquisition."""
        shot = self.grab.shot(timeout_s=0.5)
        if shot is None:
            # streams missing OR TF unusable — both are blindness, and
            # both must be able to trip the BLIND_S motion pause
            return
        self.last_data = now
        frame, depth, intr, rot, trans = shot
        hits = detect_all(self.model, frame, self.target, self.conf)
        sights = []
        best_mask = None
        for xyxy, cf, mask in hits:
            loc = box_to_center(xyxy, depth, mask=mask, min_depth=0.16, **intr)
            if loc is not None:
                sights.append((0, rot @ loc[0] + trans, np.asarray(loc[1]), cf))
                if best_mask is None and mask is not None:
                    best_mask = mask  # hits are best-first
        if best_mask is not None:
            # keep the raw frame: GraspGenX wants depth + K + this mask
            self.last_frame = dict(depth=depth, intr=intr, rot=rot,
                                   trans=trans, mask=best_mask)
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
        # acquisition: n parked frames agreeing (one frame never moves the
        # arm). NEAREST wins, not most-confident — with two same-class
        # objects in frame, confidence jitter alternated the candidate and
        # agreement could never converge (review 2026-08-20)
        best = min(sights, key=lambda s: float(np.hypot(s[1][0], s[1][1])))
        self.recent = [(p, t) for p, t in self.recent if now - t < self.AGREE_S]
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

    # --------------------------------------------------------------- grasp
    def _trigger(self, cli, what):
        from std_srvs.srv import Trigger

        if not cli.service_is_ready():
            self._status("%s: gripper service missing" % what)
            return False
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=15.0)
        res = fut.result()
        return bool(res and res.success)

    def _grasp(self):
        """GraspGenX -> ranked 6-DoF grasps -> cuRobo pre-grasp, approach,
        close. Returns True when the object is held.

        The object's voxels stay purged (ignore region) for both motions:
        the thing being grasped must not be an obstacle. Every pose still
        goes through cuRobo against the rest of the perceived world.
        """
        from rammp_curobo_ros.grasps import GraspClient, pregrasp, quat_to_mat3, to_base

        if self.grasp_cli is None:
            self.grasp_cli = GraspClient(self.grasp_endpoint)
        f = self.last_frame
        try:
            grasps, scores = self.grasp_cli.grasps_from_mask(
                f["depth"], np.array([[f["intr"]["fx"], 0, f["intr"]["cx"]],
                                      [0, f["intr"]["fy"], f["intr"]["cy"]],
                                      [0, 0, 1]]),
                (np.asarray(f["mask"]).astype(bool)).astype(np.int32),
                threshold=self.grasp_score_min, topk=8)
        except Exception as exc:
            self._status("grasp server unreachable (%s) — holding" % exc)
            return False
        if len(grasps) == 0:
            self._status("no grasp above %.2f — holding" % self.grasp_score_min)
            return False
        self._status("%d grasp candidates (best %.2f) — planning"
                     % (len(grasps), float(scores[0])))
        self._trigger(self.open_cli, "open")
        for g, sc in zip(grasps, scores):
            pos, quat = to_base(g, f["rot"], f["trans"], self.tool_offset)
            rot3 = quat_to_mat3(quat)
            pre = pregrasp(pos, rot3, self.pregrasp_standoff)
            if pre[2] < 0.05 or pos[2] < 0.02:
                continue  # into the table
            if not self._move([float(v) for v in pre], list(quat),
                              "PRE-GRASP (score %.2f)" % float(sc)):
                continue
            if not self._move([float(v) for v in pos], list(quat),
                              "GRASPING (score %.2f)" % float(sc)):
                # backed onto an unreachable final pose — retreat and retry
                self._move([float(v) for v in pre], list(quat), "retreating")
                continue
            if not self._trigger(self.close_cli, "close"):
                self._status("gripper close failed")
                return False
            self._status("GRASPED %s (score %.2f)" % (self.target, float(sc)))
            return True
        self._status("no candidate was reachable — holding")
        return False

    # ---------------------------------------------------------------- tick
    def tick(self):
        now = time.monotonic()
        if not self._startup_cleared and self.ignore_cli.service_is_ready():
            # a crashed predecessor may have left a region purged — clear
            # it even while idle, or it blinds the shared world forever
            self._startup_cleared = True
            self.clear_region()
        if self._pending is not None:
            text = self._pending
            cls = parse_target(text) if text else None
            if text and cls is None:
                self._pending = None
                self._status("cannot resolve %r — still %s"
                             % (text, self.target or "IDLE"))
            else:
                if cls and self.model is None:
                    # load BEFORE committing the target: load_detector
                    # exits on missing weights and _pending must survive
                    # so main's degrade actually retries
                    self._status("loading detector...")
                    self.model = load_detector(self.weights)
                self._pending = None
                if cls != self.target:
                    self.target = cls
                    self.fix = None
                    self.recent = []
                    self.at_survey = False
                    self.grasped = False
                    self.last_frame = None
                    self.clear_region()
        if self.target is None:
            self._status("IDLE — set a target via ~/set_target")
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.1)
            return
        if self.grasped:
            # holding the object: done. Retarget (or "" then the class
            # again) to run another cycle.
            self._status("HOLDING %s — grasp complete" % self.target)
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.1)
            return
        self._look(now)
        if now - self.last_data > self.BLIND_S:
            self._status("no camera data (%s) — motion paused"
                         % (self.grab.last_fail or "unknown"))
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
            # release the purge too: a hand is working in that volume and
            # the world must see it while we hold
            self.clear_region()
            self._status("%s lifted — holding, not chasing a hand" % self.target)
            return
        if now < self._next_try_t:
            self._heartbeat()
            return
        self._next_try_t = now + 4.0   # one attempt at a time, no spinning
        reach = float(np.hypot(pos[0], pos[1]))
        if reach > self.REACH_MAX:
            self._status("%s is %.2f m out — beyond reach; move it closer"
                         % (self.target, reach))
            return
        self._set_region(pos, self.fix["extent"])
        if self.do_grasp:
            self.grasped = self._grasp()
        else:
            self._approach_only(pos)

    def _approach_only(self, obj):
        """grasp:=false — face the object from a standoff, don't touch it.

        Uses the SAME side-approach geometry the grasp path uses, so there
        is one pose convention in this file: a horizontal approach along
        the base->object bearing, tool at the object's height. The old
        inward-hop poses were unreachable by construction (field
        2026-08-20: tool_frame is 12 cm ahead of the flange, so a flat
        wrist inside ~0.5 m radius folds the arm into itself)."""
        from rammp_curobo_ros.grasps import mat_to_quat_xyzw

        obj = np.asarray(obj, dtype=float)
        bearing = float(np.arctan2(obj[1], obj[0]))
        approach = np.array([np.cos(bearing), np.sin(bearing), 0.0])
        x = np.cross([0.0, 0.0, 1.0], approach)
        x /= np.linalg.norm(x)
        rot = np.stack([x, np.cross(approach, x), approach], axis=1)
        quat = mat_to_quat_xyzw(rot)
        pos = obj - approach * self.standoff
        if self._move([float(v) for v in pos], [float(v) for v in quat],
                      "APPROACHING %s (standoff %.2f m)"
                      % (self.target, self.standoff)):
            self._status("ARRIVED at %s (standoff %.2f m)"
                         % (self.target, self.standoff))

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
