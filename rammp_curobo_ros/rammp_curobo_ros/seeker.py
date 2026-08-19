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
            target BELIEF (position + freshness + confidence).
  DECIDE    from the belief alone: fresh belief far from the last
            commanded standoff -> approach it; stale belief -> search
            (visit glance poses — interruptible: a detection mid-motion
            retargets immediately); lifted target or unsafe geometry ->
            hold; no target set -> idle.
  ACT       at most one execution goal in flight; when the decision
            changes, the active goal is preempted through the
            executor's verified stop+hold and a fresh plan starts from
            wherever the arm is. Every plan goes through cuRobo against
            the live perceived world; winding (joint-family-flip) plans
            are refused; ≤0.25 speed.

Safety posture (owner decision 2026-08-19): autonomous and immediate —
no typed gates or countdowns. The planner's execute param, every
executor gate, and the human on the physical e-stop are the layers
that remain. A target lifted off its surface is HELD, never chased
(the ignore region follows the target; chasing a hand-held object
would exclude the hand's nearest voxels from collision checking).

Needs: planner (execute:=true), cameras node, arm bringup, D405 driver
with align_depth.enable:=true. Vocabulary: 80 COCO classes + synonyms;
weights ~/yolo11s-seg.pt (never downloaded).
"""

import time

import numpy as np
import rclpy
from std_msgs.msg import String

from rammp_curobo_ros.cameras import ensure_sensor_params
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
    standoff_pose,
    track_update,
)
from rammp_curobo_ros.tour_demo import TourDemo


def viewpoint_key(cam_trans, grid=0.05):
    """Quantized camera position — 'distinct viewpoint' for clustering.

    Replaces the scripted scan's glance index: two sightings only
    triangulate when the camera has MOVED between them (>= one 5 cm
    grid cell); sightings from a parked camera share every systematic
    error (audit 2026-08-18) and collapse to one viewpoint here."""
    return tuple(int(round(float(v) / grid)) for v in cam_trans)


class Seeker:
    """The perceive-decide-act loop. One instance per arm."""

    FRESH_S = 1.5      # belief younger than this is actionable
    LOST_S = 4.0       # belief older than this is dropped -> search
    DEAD_BAND = 0.05   # m; smaller target moves are noise, not commands
    LEASH = 0.35       # m; a sighting farther from the belief is a decoy
    LIFT_HOLD = 0.15   # m above acquisition height -> held, not chased
    BUF_TTL = 10.0     # s; acquisition sighting buffer
    WIND_RAD = 3.5     # joint-travel above this = family flip, refused

    def __init__(self, node):
        self.node = node
        p = node.declare_parameter
        self.conf = float(p("conf", 0.4).value)
        self.sure_conf = float(p("sure_conf", 0.8).value)
        self.standoff = float(p("standoff", 0.18).value)
        self.speed = float(p("speed", 0.25).value)
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
        self._last_status = None

        self.model = None
        self.target = None          # resolved COCO class or None
        self._pending_text = initial if initial else None

        self.belief = None          # dict(pos, stamp, conf, base_z)
        self.buf = []               # acquisition: (vkey, pos, extent, conf, t)
        self.last_extent = np.array([0.06, 0.20])
        self.pending_move = None    # (pos, t) awaiting 2nd-frame agreement
        self.last_sure = None       # (pos, t) awaiting 2nd sure frame

        self.handle = None          # active execution goal
        self.result_fut = None
        self.goal_kind = None       # "approach" | "glance"
        self.goal_obj = None        # commanded target position (approach)
        self.pending_retry = False
        self.search_i = 0
        self.region_on = False

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
        if text != self._last_status:
            self._last_status = text
            self.node.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def _set_region(self, center):
        if not self.ignore_cli.service_is_ready():
            return None
        req = self._SetIgnoreRegion.Request()
        req.center.x, req.center.y, req.center.z = (float(v) for v in center)
        d = [float(max(self.last_extent[0], 0.05)) + 0.04] * 2 + [
            float(max(self.last_extent[1], 0.05)) + 0.04
        ]
        req.dims.x, req.dims.y, req.dims.z = d
        fut = self.ignore_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        res = fut.result()
        if res is not None and res.success:
            self.region_on = True
        return res.message if res is not None else None

    def clear_region(self):
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
        out = []
        for xyxy, cf, mask in detect_all(self.model, frame, self.target, self.conf):
            loc = box_to_center(xyxy, depth, mask=mask, **intr)
            if loc is None:
                continue
            center, extent = loc
            out.append((vkey, rot @ center + trans, np.asarray(extent), cf))
        return out

    def _perceive(self, sights, now):
        if self.belief is not None:
            near = track_update(self.belief["pos"], sights,
                                max_jump=self.LEASH, min_move=0.0)
            if near is not None:
                s = min(sights, key=lambda s: np.linalg.norm(s[1] - near))
                self.belief.update(pos=near, stamp=now, conf=s[3])
                self.last_extent = np.maximum(self.last_extent, s[2])
            return
        # acquisition — two INDEPENDENT confirmations, two ways to get them:
        # a sure-confidence pair of consecutive frames, or sightings from
        # two distinct (moved-camera) viewpoints in the recent buffer
        self.buf = [b for b in self.buf if now - b[4] < self.BUF_TTL]
        self.buf.extend((v, p, e, c, now) for v, p, e, c in sights)
        sure = [s for s in sights if s[3] >= self.sure_conf]
        if sure:
            best = max(sure, key=lambda s: s[3])
            if self.last_sure is not None and now - self.last_sure[1] < 1.0 and (
                np.linalg.norm(best[1] - self.last_sure[0]) <= 0.05
            ):
                self._acquire(best[1], best[3], best[2])
                return
            self.last_sure = (best[1], now)
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
        self._status("ACQUIRED %s at [%.2f, %.2f, %.2f]"
                     % (self.target, pos[0], pos[1], pos[2]))

    # -------------------------------------------------------------- decide
    def _approach(self, obj):
        if not self.stop_goal():
            return
        msg = self._set_region(obj)
        if msg is not None and _purged_count(msg) != 0:
            time.sleep(1.5)  # purge propagation (2 Hz world push)
        plan = None
        for st, zmin in ((self.standoff, 0.12), (self.standoff, 0.22),
                         (self.standoff + 0.08, 0.30)):
            so = standoff_pose(obj, standoff=st, z_min=zmin)
            if so is None:
                self._status("target too close to the base — holding")
                return
            pos, quat, gap = so
            plan = self.demo.plan_pose_from(pos, quat, None)
            if plan is not None and plan.success:
                break
            plan = None
        if plan is None:
            self._status("no approach plan at any pose — holding")
            self.pending_retry = True
            return
        if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
            self._status("plan winds the arm (family flip) — refused, holding")
            return
        if self._run(plan, "approach", obj):
            self._status("APPROACHING %s -> [%.2f, %.2f, %.2f] (gap %.2f m)"
                         % (self.target, obj[0], obj[1], obj[2], gap))

    def _search(self):
        if self.handle is not None:
            return  # let the current motion finish; frames keep coming
        bearing, pitch = GLANCES[self.search_i % len(GLANCES)]
        self.search_i += 1
        pos, quat = glance_pose(bearing, pitch)
        plan = self.demo.plan_pose_from(pos, quat, None)
        if plan is None or not plan.success:
            return  # unplannable glance: next loop tries the next one
        if max(joint_travel(plan.trajectory).values()) > self.WIND_RAD:
            return
        if self._run(plan, "glance"):
            self._status("SEARCHING for %s (glance %d/%d)"
                         % (self.target, (self.search_i - 1) % len(GLANCES) + 1,
                            len(GLANCES)))

    # ---------------------------------------------------------------- tick
    def tick(self):
        now = time.monotonic()
        # target changes (service or param) land between frames
        if self._pending_text is not None:
            text, self._pending_text = self._pending_text, None
            if text == "":
                self.target = None
                self.belief = None
                self.stop_goal()
                self.clear_region()
            else:
                cls = parse_target(text)
                if cls is not None and cls != self.target:
                    self.target = cls
                    self.belief = None
                    self.buf = []
                    self.pending_move = None
                    self.clear_region()
                    if self.model is None:
                        self.model = load_detector(self.weights)
        self._harvest_result()
        if self.target is None:
            self._status("IDLE — set a target via ~/set_target")
            rclpy.spin_once(self.node, timeout_sec=0.2)
            return
        moving = self.result_fut is not None
        shot = self.grab.shot(timeout_s=0.5, strict=moving)
        if shot is not None:
            self._perceive(self._localize(shot), now)
        if self.belief is None:
            self.clear_region()
            self._search()
            return
        age = now - self.belief["stamp"]
        if age > self.LOST_S:
            self._status("%s lost — searching from last known position"
                         % self.target)
            self.belief = None
            self.pending_move = None
            self.clear_region()
            self.stop_goal()
            return
        pos = self.belief["pos"]
        if pos[2] > self.belief["base_z"] + self.LIFT_HOLD:
            self._status("%s lifted — holding, not chasing a hand"
                         % self.target)
            self.pending_move = None
            return
        if age > self.FRESH_S:
            return  # belief usable for search seeding but too old to chase
        want = (
            self.goal_obj is None
            or self.goal_kind == "glance"
            or float(np.linalg.norm(pos - self.goal_obj)) > self.DEAD_BAND
            or (self.pending_retry and self.handle is None)
        )
        if not want:
            if self.handle is None:
                self._status("HOLDING at %s [%.2f, %.2f, %.2f]"
                             % (self.target, pos[0], pos[1], pos[2]))
            return
        # commit a move only when two consecutive frames agree (a single
        # frame never moves the arm — audit 2026-08-18)
        if self.pending_retry and self.handle is None:
            self.pending_move = None
            self._approach(pos)
            return
        if (
            self.pending_move is not None
            and now - self.pending_move[1] < 1.0
            and np.linalg.norm(pos - self.pending_move[0]) <= 0.05
        ):
            self.pending_move = None
            self._approach(pos)
        else:
            self.pending_move = (pos.copy(), now)

    def shutdown(self):
        try:
            self.stop_goal()
        finally:
            self.clear_region()


def main():
    rclpy.init()
    node = rclpy.create_node("seeker")
    seeker = Seeker(node)
    print(
        "*** SEEKER up — autonomous once a target is set (owner decision "
        "2026-08-19: no gates). Planner execute param + executor gates + "
        "the human on the e-stop are the safety layers. Ctrl+C stops. ***"
    )
    try:
        while rclpy.ok():
            seeker.tick()
    except KeyboardInterrupt:
        pass
    except BaseException:
        seeker.shutdown()
        raise
    seeker.shutdown()
    print("\nseeker stopped — arm holds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nseeker stopped — arm holds")
