#!/usr/bin/env python3
"""seeker — continuous seek CONTROLLER: perceive, decide, act. No phases.

    ros2 run rammp_curobo_ros seeker                          # idle until targeted
    ros2 run rammp_curobo_ros seeker --ros-args -p target:="go to the bottle"
    ros2 service call /seeker/set_target rammp_curobo_interfaces/srv/SetTarget \
        "{text: 'go to the bottle'}"                          # retarget any time

This is the deployed form of the seek behavior (owner direction
2026-08-19: "not a sequential system — a deployed algorithm that works
in ANY circumstances"). There is no scan phase, approach phase, or
track phase — one loop runs forever:

  PERCEIVE  every frame: YOLO on the wrist D405, every instance
            3-D-localized (aligned depth + stamped TF), folded into a
            target BELIEF. A CONFIDENT sighting (conf >= sure_conf,
            two frames agreeing within 5 cm) acquires the belief AT ANY
            TIME — including mid-glance, ending the search on the spot
            (owner requirement 2026-08-19). The low-confidence
            two-viewpoint cluster path only ingests parked-arm frames
            (quantized viewpoints along one sweep are not independent;
            audit 2026-08-19), gathered during a ~1 s dwell at each
            glance pose.
  DECIDE    from the belief alone: fresh belief -> VISUAL SERVO to it
            (owner design 2026-08-19: wrist FLAT at the object's
            height — which centers the bbox vertically — camera
            re-aimed at the object every hop, short cuRobo-planned
            steps along the line to it; short hops can't wind the arm
            and 'way too high' pose ladders are gone); sight lost ->
            step BACK along the approach line to re-look (twice), then
            search glances seeded at the last known position; lifted
            target or unsafe geometry -> hold; blind camera -> hold (no
            autonomous patrol without perception); no target -> idle.
  ACT       at most one execution goal in flight; when the decision
            changes, the active goal is preempted through the
            executor's verified stop+hold and a fresh plan starts from
            wherever the arm is. Every hop goes through cuRobo against
            the live perceived world; winding (joint-family-flip) plans
            are refused; speed hard-clamped to 0.25.

Safety posture (owner decision 2026-08-19): autonomous and immediate —
no typed gates or countdowns. The planner's execute param, every
executor gate, and the human on the physical e-stop are the layers
that remain. A target lifted off its resting height is HELD, never
chased — the active goal is preempted and the ignore region cleared
(the region follows the target; chasing a hand-held object would
exclude the hand's nearest voxels from collision checking). Known
limit: a target FIRST acquired while already held in the air has that
height as its resting height — the lift guard can't recognize what it
never saw on a surface.

Needs: planner (execute:=true), cameras node, arm bringup, D405 driver
with align_depth.enable:=true. Vocabulary: 80 COCO classes + synonyms;
weights ~/yolo11s-seg.pt (never downloaded).
"""

import time

import numpy as np
import rclpy
from std_msgs.msg import String

from rammp_curobo_ros.cameras import _ViewServer, ensure_sensor_params
from rammp_curobo_ros.ros_util import spin_until_done
from rammp_curobo_ros.seek_demo import (
    GLANCES,
    _D405Grabber,
    _purged_count,
    box_to_center,
    cluster_sightings,
    detect_all,
    glance_pose,
    joint_travel,
    load_detector,
    parse_target,
    track_update,
)
from rammp_curobo.geometry import yaw_about_world_z
from rammp_curobo_ros.tour_demo import HOME_QUAT_XYZW, TourDemo


def viewpoint_key(cam_trans, grid=0.05):
    """Quantized camera position — 'distinct viewpoint' for clustering.

    Replaces the scripted scan's glance index: two sightings only
    triangulate when the camera has MOVED between them (>= one 5 cm
    grid cell); sightings from a parked camera share every systematic
    error (audit 2026-08-18) and collapse to one viewpoint here.
    Acquisition additionally only ingests parked-arm frames, so the
    grid separates PAUSES, not points along one sweep."""
    return tuple(int(round(float(v) / grid)) for v in cam_trans)


class Seeker:
    """The perceive-decide-act loop. One instance per arm."""

    FRESH_S = 1.5      # belief younger than this is actionable
    LOST_S = 4.0       # belief older than this is dropped -> search
    DEAD_BAND = 0.05   # m; smaller target moves are noise, not commands
    LEASH = 0.35       # m; a sighting farther from the belief is a decoy
    LIFT_HOLD = 0.15   # m above resting height -> held, not chased
    BUF_TTL = 10.0     # s; acquisition sighting buffer
    WIND_RAD = 3.5     # joint-travel above this = family flip, refused
    BLIND_S = 3.0      # s without camera data -> no autonomous motion
    CMD_COOLDOWN = 0.75  # s between servo hop dispatches (anti-churn)
    HOP_MAX = 0.20     # m; longest single servo hop
    HOP_MIN = 0.06     # m; shortest useful hop
    ARRIVE_TOL = 0.04  # m; within standoff+this = arrived
    Z_FLOOR = 0.10     # m; lowest fingertip height while servoing
    BACKOFF_M = 0.15   # m; step-back distance when sight is lost

    def __init__(self, node):
        self.node = node
        p = node.declare_parameter
        self.conf = float(p("conf", 0.4).value)
        self.sure_conf = float(p("sure_conf", 0.8).value)
        self.standoff = float(p("standoff", 0.18).value)
        # hard clamp — the module promises <=0.25 regardless of params
        self.speed = min(max(float(p("speed", 0.25).value), 0.05), 0.25)
        self.weights = str(p("weights", "~/yolo11s-seg.pt").value)
        initial = str(p("target", "").value)

        self.demo = TourDemo(node)
        self.grab = _D405Grabber(node)
        ensure_sensor_params(node, self.grab.cfg)

        from rammp_curobo_interfaces.srv import SetIgnoreRegion, SetTarget

        self._SetIgnoreRegion = SetIgnoreRegion
        self.ignore_cli = node.create_client(
            SetIgnoreRegion, "/cameras/set_ignore_region"
        )
        node.create_service(SetTarget, "~/set_target", self._set_target_cb)
        self.status_pub = node.create_publisher(String, "~/status", 1)
        self._last_status = "starting"
        self._last_pub_t = 0.0

        self.model = None
        self.target = None          # resolved COCO class or None
        self._pending_text = initial if initial else None

        self.belief = None          # dict(pos, stamp, conf, base_z)
        self.buf = []               # acquisition: (vkey, pos, extent, conf, t)
        self.last_extent = np.array([0.06, 0.20])
        self.pending_move = None    # (pos, t) awaiting a NEWER agreeing frame
        self.last_sure = None       # (pos, t) awaiting 2nd sure frame
        self.last_known = None      # search seed after a loss

        self.handle = None          # active execution goal
        self.result_fut = None
        self.goal_kind = None       # "approach" | "glance"
        self.goal_obj = None        # commanded target position (approach)
        self.pending_retry = False
        self.search_i = 0
        self.region_on = False
        self._region_pos = None     # where the ignore region last went
        self._just_acquired = False
        self._dwell_until = 0.0     # parked pause after each glance
        self._backoffs = 0          # step-backs tried since losing sight
        self._startup_clear_done = False
        self._last_data_t = time.monotonic()  # camera liveness
        self._next_cmd_t = 0.0      # approach dispatch cooldown

        # DETECTION view (distinct from the cameras node's :8766 WORLD
        # view, which draws perceived obstacle boxes): this one shows
        # what YOLO claims — target-class boxes with confidence, plus
        # the current belief — so "is it detecting the right thing?"
        # is answerable by eye (field question 2026-08-19)
        self.view_server = None
        if bool(p("view", True).value):
            try:
                self.view_server = _ViewServer(int(p("view_port", 8767).value))
                node.get_logger().info(
                    "detection view: http://<this-host>:8767/ "
                    "(param view:=false to disable)"
                )
            except OSError as e:
                node.get_logger().warn("detection view failed (%s)" % e)

    # ------------------------------------------------------------ plumbing
    def _set_target_cb(self, req, res):
        text = req.text.strip()
        if not text:
            self._pending_text = ""
            res.success, res.message = True, "target cleared — idling"
            return res
        cls = parse_target(text)
        if cls is None:
            res.success = False
            res.message = "no COCO class in %r (bottle, cup, bowl, ...)" % text
            return res
        self._pending_text = text
        res.success, res.message, res.resolved_class = True, "seeking " + cls, cls
        return res

    def _status(self, text):
        now = time.monotonic()
        if text != self._last_status:
            self._last_status = text
            self.node.get_logger().info(text)
            self.status_pub.publish(String(data=text))
            self._last_pub_t = now
        elif now - self._last_pub_t > 0.5:  # heartbeat, throttled
            self.status_pub.publish(String(data=text))
            self._last_pub_t = now

    def _heartbeat(self):
        # every tick republishes the standing state (throttled) so a
        # late-joining subscriber and a mid-motion dashboard both see it
        # (audit 2026-08-19: the topic went silent during every motion)
        self._status(self._last_status)

    def _set_region(self, center):
        if not self.ignore_cli.service_is_ready():
            return None
        req = self._SetIgnoreRegion.Request()
        req.center.x, req.center.y, req.center.z = (float(v) for v in center)
        d = [float(max(self.last_extent[0], 0.05)) + 0.04] * 2 + [
            float(max(self.last_extent[1], 0.05)) + 0.04
        ]
        req.dims.x, req.dims.y, req.dims.z = d
        # pessimistic: the REQUEST may land even if the response times
        # out — assume it did so a later clear is always attempted
        # (audit 2026-08-19: the optimistic flag leaked a permanent
        # blind spot on a slow cameras-node response)
        self.region_on = True
        fut = self.ignore_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        res = fut.result()
        return res.message if res is not None else None

    def clear_region(self):
        self._region_pos = None
        if self.region_on and self.ignore_cli.service_is_ready():
            fut = self.ignore_cli.call_async(self._SetIgnoreRegion.Request())
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
            self.region_on = False

    def stop_goal(self):
        """Preempt through the executor's verified stop+hold (the goal
        RESULT completes only after the arm is holding — the cancel ACK
        alone proves nothing; audit 2026-08-19)."""
        if self.handle is None:
            return True
        spin_until_done(self.node, self.handle.cancel_goal_async(), 3.0)
        if spin_until_done(self.node, self.result_fut, 8.0) is None:
            self._status("cancel unconfirmed — goal may still run; holding")
            return False
        self.handle = None
        self.result_fut = None
        self.goal_kind = None
        time.sleep(0.3)  # settle before a live-start plan
        return True

    def _drop_target_state(self):
        """Everything tied to the current pursuit — reset on target
        clear/retarget/loss so nothing stale leaks into the next one
        (audit 2026-08-19: stale goal_obj deadlocked reacquisition as a
        false HOLDING; stale pending_retry bypassed the two-frame gate;
        ratcheted last_extent blinded a small target's surroundings)."""
        self.belief = None
        self.buf = []
        self.pending_move = None
        self.last_sure = None
        self.goal_obj = None
        self.pending_retry = False
        self._just_acquired = False
        self._backoffs = 0
        self.last_extent = np.array([0.06, 0.20])

    def _harvest_result(self):
        if self.result_fut is None or not self.result_fut.done():
            return
        wrapped = self.result_fut.result()
        kind = self.goal_kind
        self.handle = None
        self.result_fut = None
        self.goal_kind = None
        if not (wrapped and wrapped.result.success):
            msg = wrapped.result.message if wrapped else "no result"
            self._status("segment FAILED (%s) — will replan" % msg)
            if kind == "approach":
                self.pending_retry = True
                self.goal_obj = None
        elif kind == "glance":
            # dwell parked: without this the next glance dispatches on
            # the very next tick, leaving ONE frame per pose — the sure
            # pair could never complete and every acquisition needed
            # multiple glances (field 2026-08-19: 'still tries all 4')
            self._dwell_until = time.monotonic() + 1.0

    def _run(self, plan, kind, obj=None):
        self.handle = self.demo.run_async(plan.trajectory, self.speed)
        if self.handle is None:
            self._status("goal not accepted — is the planner up?")
            if kind == "approach":
                self.pending_retry = True
            return False
        self.result_fut = self.handle.get_result_async()
        self.goal_kind = kind
        if kind == "approach":
            self.goal_obj = np.asarray(obj, dtype=float)
            self.pending_retry = False
        return True

    # ------------------------------------------------------------ perceive
    def _localize(self, shot):
        frame, depth, intr, rot, trans = shot
        vkey = viewpoint_key(trans)
        hits = detect_all(self.model, frame, self.target, self.conf)
        out = []
        for xyxy, cf, mask in hits:
            loc = box_to_center(xyxy, depth, mask=mask, **intr)
            if loc is None:
                continue
            center, extent = loc
            out.append((vkey, rot @ center + trans, np.asarray(extent), cf))
        if self.view_server is not None:
            self._render_view(frame, hits, intr, rot, trans, len(out))
        return out

    def _render_view(self, frame, hits, intr, rot, trans, n_localized):
        import cv2

        img = frame.copy()
        for xyxy, cf, _ in hits:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, "%s %.2f" % (self.target, cf), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        if self.belief is not None:
            pc = (self.belief["pos"] - trans) @ rot
            if pc[2] > 0.05:
                u = int(intr["fx"] * pc[0] / pc[2] + intr["cx"])
                v = int(intr["fy"] * pc[1] / pc[2] + intr["cy"])
                cv2.drawMarker(img, (u, v), (255, 120, 0),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.putText(img, "belief", (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
        cv2.putText(
            img,
            "%s | %d det / %d localized" % (self._last_status[:70],
                                            len(hits), n_localized),
            (8, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        self.view_server.update(img)

    def _perceive(self, sights, now, parked):
        if self.belief is not None:
            near = track_update(self.belief["pos"], sights,
                                max_jump=self.LEASH, min_move=0.0)
            if near is not None:
                s = min(sights, key=lambda s: np.linalg.norm(s[1] - near))
                self.belief.update(pos=near, stamp=now, conf=s[3])
                # resting height tracks DOWN (placed on a lower surface)
                # but never up — up is what the lift guard detects
                self.belief["base_z"] = min(self.belief["base_z"],
                                            float(near[2]))
                self.last_extent = np.maximum(self.last_extent, s[2])
            return
        # SURE path — runs even MID-MOTION (owner requirement 2026-08-19:
        # a confident sighting must end the search right away, not after
        # the glance tour). Strict stamped TF keeps mid-motion frames
        # honest, and two frames agreeing within 5 cm filter the smear.
        sure = [s for s in sights if s[3] >= self.sure_conf]
        if sure:
            best = max(sure, key=lambda s: s[3])
            if self.last_sure is not None and now - self.last_sure[1] < 1.0 and (
                np.linalg.norm(best[1] - self.last_sure[0]) <= 0.05
            ):
                self._acquire(best[1], best[3], best[2])
                return
            self.last_sure = (best[1], now)
        if not parked:
            return  # the CLUSTER path stays parked-only: quantized
            # viewpoints along one sweep are not independent (audit
            # 2026-08-19) — low-confidence acquisition needs real pauses
        self.buf = [b for b in self.buf if now - b[4] < self.BUF_TTL]
        self.buf.extend((v, p, e, c, now) for v, p, e, c in sights)
        clusters = cluster_sightings(
            [(v, p, e, c) for v, p, e, c, _ in self.buf]
        )
        confirmed = [c for c in clusters if len(c["glances"]) >= 2]
        if confirmed:
            confirmed.sort(key=lambda c: float(np.hypot(*c["center"][:2])))
            c = confirmed[0]  # deployed policy: nearest wins, never stall
            self._acquire(c["center"], c["conf"], c["extent"])

    def _acquire(self, pos, conf, extent):
        pos = np.asarray(pos, dtype=float)
        self.belief = dict(pos=pos, stamp=time.monotonic(), conf=float(conf),
                           base_z=float(pos[2]))
        self.last_extent = np.maximum(self.last_extent, extent)
        self.buf = []
        self.last_sure = None
        self.last_known = None
        self.pending_move = None
        self._backoffs = 0
        # acquisition is already double-confirmed — the very next decide
        # may approach without the extra pending_move frame (owner
        # requirement 2026-08-19: stop glancing the moment it's sure)
        self._just_acquired = True
        self._status("ACQUIRED %s at [%.2f, %.2f, %.2f]"
                     % (self.target, pos[0], pos[1], pos[2]))

    # -------------------------------------------------------------- decide
    def _tool_pos(self):
        """Fingertip midpoint (tool_frame) in base_link, or None.

        tool_frame exists only in the planner's kinematics — TF ends at
        end_effector_link; the tip sits 0.120 m along its z."""
        try:
            tr = self.grab.tf_buffer.lookup_transform(
                "base_link", "end_effector_link", rclpy.time.Time()
            )
        except Exception:
            return None
        q, t = tr.transform.rotation, tr.transform.translation
        from rammp_curobo.perception import quat_to_mat

        rot = quat_to_mat(q.x, q.y, q.z, q.w)
        return np.array([t.x, t.y, t.z]) + rot[:, 2] * 0.120

    def _servo(self, obj):
        """One visual-servo hop: flat wrist, at the object's height,
        camera re-aimed at the object, a short cuRobo-planned step along
        the line to it (owner design 2026-08-19: center the bbox and
        approach with the wrist flat). Short hops can't wind the arm and
        each one is collision-checked against the live world; height =
        object height (the old pose ladder relaxed UP to z 0.30 and
        'approached way too high')."""
        tool = self._tool_pos()
        if tool is None:
            self._status("no TF to the arm — cannot servo")
            return
        to_t = np.asarray(obj[:2], dtype=float) - tool[:2]
        gap = float(np.linalg.norm(to_t))
        if gap <= self.standoff + self.ARRIVE_TOL:
            self.goal_obj = np.asarray(obj, dtype=float)
            self._status("ARRIVED at %s (gap %.2f m) — holding"
                         % (self.target, gap))
            return
        if not self.stop_goal():
            return
        self._next_cmd_t = time.monotonic() + self.CMD_COOLDOWN
        if (
            self._region_pos is None
            or np.linalg.norm(np.asarray(obj) - self._region_pos) > 0.03
        ):
            msg = self._set_region(obj)
            self._region_pos = np.asarray(obj, dtype=float).copy()
            if msg is not None and _purged_count(msg) != 0:
                time.sleep(1.5)  # purge propagation (2 Hz world push)
        hop = min(self.HOP_MAX, max(self.HOP_MIN, 0.4 * (gap - self.standoff)))
        hop = min(hop, gap - self.standoff)  # never step inside the standoff
        step = tool[:2] + to_t / gap * hop
        z = max(float(obj[2]) + 0.03, self.Z_FLOOR)
        bearing = float(np.arctan2(obj[1] - step[1], obj[0] - step[0]))
        quat = list(yaw_about_world_z(HOME_QUAT_XYZW, bearing))  # WRIST FLAT
        plan = None
        for step_len in (hop, hop / 2.0):
            tgt = [float(tool[0] + to_t[0] / gap * step_len),
                   float(tool[1] + to_t[1] / gap * step_len), z]
            plan = self.demo.plan_pose_from(tgt, quat, None)
            if plan is not None and plan.success:
                break
            plan = None
        if plan is None:
            self._status("no plan for the next hop — holding, will retry")
            self.pending_retry = True
            return
        if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
            self._status("hop winds the arm (family flip) — refused, holding")
            return
        if self._run(plan, "approach", obj):
            self._status(
                "SERVOING to %s [%.2f, %.2f, %.2f] — gap %.2f m, hop %.2f m"
                % (self.target, obj[0], obj[1], obj[2], gap, step_len)
            )

    def _search(self):
        if self.handle is not None:
            self._heartbeat()
            return  # let the current motion finish; frames keep coming
        if time.monotonic() < self._dwell_until:
            self._heartbeat()
            return  # parked dwell — acquisition frames at this pose
        if self.last_known is not None and self._backoffs < 2:
            # lost sight while closing in: step BACK along the approach
            # line, camera still on the last known spot, before any
            # glance tour (field 2026-08-19: a glance planned from a
            # fully extended arm swept joint_1 at full reach)
            tool = self._tool_pos()
            if tool is not None:
                away = tool[:2] - np.asarray(self.last_known[:2])
                n = float(np.linalg.norm(away))
                if n > 1e-6:
                    step = tool[:2] + away / n * self.BACKOFF_M
                    bearing = float(np.arctan2(self.last_known[1] - step[1],
                                               self.last_known[0] - step[0]))
                    tgt = [float(step[0]), float(step[1]),
                           max(float(tool[2]), 0.15)]
                    quat = list(yaw_about_world_z(HOME_QUAT_XYZW, bearing))
                    plan = self.demo.plan_pose_from(tgt, quat, None)
                    if (
                        plan is not None
                        and plan.success
                        and max(joint_travel(plan.trajectory).values())
                        <= self.WIND_RAD
                    ):
                        self._backoffs += 1
                        if self._run(plan, "glance"):
                            self._status(
                                "lost sight — stepping back to re-look "
                                "(%d/2)" % self._backoffs
                            )
                            return
        if self.last_known is not None:
            # seed the search at the last place the target was seen
            bearing = float(np.arctan2(self.last_known[1], self.last_known[0]))
            self.last_known = None
            pos, quat = glance_pose(bearing, np.radians(55.0))
            label = "last known position"
        else:
            bearing, pitch = GLANCES[self.search_i % len(GLANCES)]
            self.search_i += 1
            pos, quat = glance_pose(bearing, pitch)
            label = "glance %d/%d" % ((self.search_i - 1) % len(GLANCES) + 1,
                                      len(GLANCES))
        plan = self.demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            return  # unplannable pose: next loop tries the next one
        if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
            return
        if self._run(plan, "glance"):
            self._status("SEARCHING for %s (%s)" % (self.target, label))

    # ---------------------------------------------------------------- tick
    def tick(self):
        now = time.monotonic()
        if not self._startup_clear_done and self.ignore_cli.service_is_ready():
            # a crashed predecessor may have left an ignore region — a
            # fresh controller starts with a whole world (audit 2026-08-19)
            self._startup_clear_done = True
            self.region_on = True
            self.clear_region()
        # target changes (service or param) land between frames
        if self._pending_text is not None:
            text, self._pending_text = self._pending_text, None
            if text == "":
                self.target = None
                self.stop_goal()
                self._drop_target_state()
                self.clear_region()
            else:
                cls = parse_target(text)
                if cls is None:
                    self._status("cannot resolve %r to a known class — "
                                 "still %s" % (text, self.target or "IDLE"))
                elif cls != self.target:
                    self.stop_goal()  # a retarget IS a decision change
                    self._drop_target_state()
                    self.clear_region()
                    self.target = cls
                    if self.model is None:
                        self._status("loading detector for %s..." % cls)
                        self.model = load_detector(self.weights)
        self._harvest_result()
        if self.target is None:
            self._status("IDLE — set a target via ~/set_target")
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.1)  # bounded idle rate, not callback rate
            return
        moving = self.result_fut is not None
        shot = self.grab.shot(timeout_s=0.5, strict=moving)
        if shot is not None:
            self._last_data_t = now
            self._perceive(self._localize(shot), now, parked=not moving)
        elif not moving and not self.grab.missing():
            self._last_data_t = now  # streams alive, frame just late
        if now - self._last_data_t > self.BLIND_S:
            # blind: no autonomous patrol without perception (audit
            # 2026-08-19: a dead camera meant endless glance motion)
            self.stop_goal()
            self._status("no camera data for %.0f s (%s) — motion paused"
                         % (now - self._last_data_t,
                            ", ".join(self.grab.missing()) or "TF"))
            return
        if self.belief is None:
            self.clear_region()
            self._search()
            return
        age = now - self.belief["stamp"]
        if age > self.LOST_S:
            if self.goal_kind == "approach":
                self._heartbeat()
                return  # camera can't see mid-approach; judge on arrival
            self._status("%s lost — stepping back, then searching near "
                         "its last position" % self.target)
            self.last_known = self.belief["pos"].copy()
            self._backoffs = 0
            self.stop_goal()
            self.belief = None
            self.pending_move = None
            self.goal_obj = None
            self.clear_region()
            return
        pos = self.belief["pos"]
        if pos[2] > self.belief["base_z"] + self.LIFT_HOLD:
            # preempt AND clear the region: an in-flight approach would
            # otherwise keep driving toward the spot a hand just reached
            # into, with that spot's voxels purged (audit 2026-08-19)
            self.stop_goal()
            self.clear_region()
            self.goal_obj = None
            self.pending_move = None
            self._status("%s lifted — holding, not chasing a hand"
                         % self.target)
            return
        if age > self.FRESH_S:
            self._heartbeat()
            return  # too old to chase; too young to declare lost
        tool = self._tool_pos()
        gap = (
            float(np.linalg.norm(pos[:2] - tool[:2])) if tool is not None else None
        )
        arrived = gap is not None and gap <= self.standoff + self.ARRIVE_TOL
        moved = (
            self.goal_obj is not None
            and float(np.linalg.norm(pos - self.goal_obj)) > self.DEAD_BAND
        )
        if arrived and not moved:
            if self.handle is None:
                self._status("ARRIVED at %s (gap %.2f m) — holding"
                             % (self.target, gap))
            else:
                self._heartbeat()
            return
        if self.handle is not None and self.goal_kind == "approach" and not moved:
            self._heartbeat()
            return  # hop in flight toward a still-valid target
        if now < self._next_cmd_t:
            self._heartbeat()
            return  # anti-churn cooldown between hop dispatches
        if (
            self._just_acquired
            or (self.pending_retry and self.handle is None)
            or not moved
        ):
            # first hop (double-confirmed acquisition — preempts a
            # glance in flight; owner requirement 2026-08-19), a retry,
            # or a continuation hop toward a target that hasn't moved
            self._just_acquired = False
            self.pending_move = None
            self._servo(pos)
            return
        # target MOVED: commit only when a NEWER frame agrees with the
        # pending one — one frame never re-aims the arm (audit 2026-08-19:
        # comparing against a stale belief copy made one sighting enough)
        if (
            self.pending_move is not None
            and now - self.pending_move[1] < 1.0
            and self.belief["stamp"] > self.pending_move[1]
            and np.linalg.norm(pos - self.pending_move[0]) <= 0.05
        ):
            self.pending_move = None
            self._servo(pos)
        else:
            self.pending_move = (pos.copy(), now)

    def shutdown(self):
        try:
            self.stop_goal()
        except Exception:
            pass
        try:
            self.clear_region()
        except Exception:
            pass


def main():
    from rclpy.signals import SignalHandlerOptions

    # keep SIGINT as a normal KeyboardInterrupt: rclpy's own handler
    # would shut the context down BEFORE our cleanup can cancel the
    # active goal and clear the ignore region (audit 2026-08-19)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("seeker")
    seeker = Seeker(node)
    print(
        "*** SEEKER up — autonomous once a target is set (owner decision "
        "2026-08-19: no gates). Planner execute param + executor gates + "
        "the human on the e-stop are the safety layers. Ctrl+C stops. ***"
    )
    try:
        while rclpy.ok():
            try:
                seeker.tick()
            except SystemExit as e:
                # helper hard-exits (planner briefly gone, driver
                # relaunched wrong) must DEGRADE, not kill the
                # controller (audit 2026-08-19)
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
