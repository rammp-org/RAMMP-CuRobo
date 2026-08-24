"""cameras — continuous perceived-obstacle world for the planner.

Subscribes depth + camera_info for each configured camera (sensor-data
QoS), runs the pure perception pipeline (rammp_curobo.perception) at
`rate_hz`, and replaces the planner's perceived boxes via
/rammp_curobo/update_world_boxes. Publishes ~/world_markers so RViz shows
exactly what the planner believes. Never commands motion.

    ros2 run rammp_curobo_ros cameras
    ros2 run rammp_curobo_ros cameras --ros-args -p "cameras:=['camera_orbbec_bench.yaml']"

Camera YAML schema (same as the 2026-08 scan pipeline): depth_topic,
info_topic, min_range, max_range, and EITHER tf_frame (optical frame in
TF) OR parent_frame + mount_xyz + mount_quat_xyzw (fixed mount; the
calibration script writes this form for the Orbbec).
"""

import os
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rammp_curobo.config import resolve_config
from rammp_curobo.perception import (
    BoxTracker,
    SelfRegistrar,
    VoxelAccumulator,
    boxes_changed,
    cluster_cells,
    depth_to_points,
    in_box_mask,
    load_self_model,
    quat_to_mat,
    robot_mask,
    robot_mask_spheres,
    self_model_spheres,
    transform_points,
    visible_free_cells,
    workspace_crop,
)
from rammp_curobo.scene import load_scene
from rammp_curobo_interfaces.srv import SetIgnoreRegion, UpdateWorldBoxes

ARM_CHAIN = [
    "base_link",
    "shoulder_link",
    "half_arm_1_link",
    "half_arm_2_link",
    "forearm_link",
    "spherical_wrist_1_link",
    "spherical_wrist_2_link",
    "bracelet_link",
    "end_effector_link",
]


def load_camera_config(name_or_path):
    """Resolve a camera YAML by path or packaged name (config/ dir)."""
    candidates = [os.path.expanduser(name_or_path)]
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("rammp_curobo_ros")
        candidates.append(os.path.join(share, "config", name_or_path))
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(here, "config", name_or_path))
    for c in candidates:
        if os.path.isfile(c):
            with open(c) as f:
                cfg = yaml.safe_load(f)
            spec = cfg.get("sensor_params")
            if spec is not None and (
                not isinstance(spec, dict)
                or not isinstance(spec.get("node"), str)
                or not isinstance(spec.get("params"), dict)
                or not spec["params"]
            ):
                # validate HERE, with the filename — a malformed block
                # would otherwise KeyError inside the node's timer and
                # kill the whole perceived world (audit 2026-08-19)
                sys.exit(
                    "sensor_params in %s must be "
                    "{node: <str>, params: {<name>: <value>, ...}}" % c
                )
            return cfg
    sys.exit(
        "camera config %r not found (tried: %s).\nIf you just added it, "
        "colcon globs config/*.yaml at BUILD time — rebuild:\n  colcon "
        "build --symlink-install --packages-select rammp_curobo_ros\n"
        "The Orbbec config is written by scripts/calibrate_orbbec.py."
        % (name_or_path, candidates)
    )


def _sensor_param_request(spec):
    """Build the SetParameters request a `sensor_params` YAML block asks for."""
    from rcl_interfaces.msg import Parameter, ParameterValue
    from rcl_interfaces.srv import SetParameters

    req = SetParameters.Request()
    for name, value in spec["params"].items():
        v = ParameterValue()
        if isinstance(value, bool):
            v.type, v.bool_value = 1, value
        elif isinstance(value, int):
            v.type, v.integer_value = 2, value
        elif isinstance(value, float):
            v.type, v.double_value = 3, value
        else:
            v.type, v.string_value = 4, str(value)
        req.parameters.append(Parameter(name=name, value=v))
    return req


def ensure_sensor_params(node, cfg, timeout_s=3.0):
    """Blocking-push of the YAML's `sensor_params: {node, params}` to the
    driver. The D405 is passive stereo: its permissive defaults
    HALLUCINATE depth on textureless surfaces (field 2026-08-19 — blank
    bench read 0.3 m at a true 0.7 m); High Accuracy makes holes
    instead, which the accumulator handles. Non-fatal on absence."""
    from rcl_interfaces.srv import SetParameters

    spec = cfg.get("sensor_params")
    if not spec:
        return True
    client = node.create_client(SetParameters, spec["node"] + "/set_parameters")
    try:
        if not client.wait_for_service(timeout_sec=timeout_s):
            node.get_logger().warn(
                "sensor_params: %s not reachable — set %s on the driver "
                "manually (depth quality contract unenforced)"
                % (spec["node"], spec["params"])
            )
            return False
        future = client.call_async(_sensor_param_request(spec))
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        res = future.result()
        if res is None or not all(r.successful for r in res.results):
            node.get_logger().warn(
                "sensor_params: %s refused %s" % (spec["node"], spec["params"])
            )
            return False
        node.get_logger().info(
            "sensor_params: %s <- %s" % (spec["node"], spec["params"])
        )
        return True
    finally:
        node.destroy_client(client)


def cropped_points(depth, intr, rot, trans, stride, min_range, max_range,
                   xy_extent, min_z, max_z):
    """Deproject + transform + workspace crop. No masking — the raw view,
    which is what camera registration needs to see the arm."""
    pts = depth_to_points(
        depth,
        intr["fx"],
        intr["fy"],
        intr["cx"],
        intr["cy"],
        stride=stride,
        min_range=min_range,
        max_range=max_range,
    )
    pts = transform_points(pts, rot, trans)
    return workspace_crop(pts, xy_extent=xy_extent, min_z=min_z, max_z=max_z)


def process_camera_points(
    depth,
    intr,
    rot,
    trans,
    stride,
    min_range,
    max_range,
    xy_extent,
    min_z,
    max_z,
    link_pts,
    self_radius,
    ignore_region,
    baseline_boxes,
    self_spheres=None,
    self_margin=0.08,
):
    """One camera frame -> filtered base_link points. Pure (testable).

    Self-filter: the SPHERE model when given (cuRobo's own collision
    spheres in base_link; `self_margin` is added to every radius and only
    has to absorb camera-pose error), else the link-origin capsules of
    radius `self_radius`.
    """
    pts = cropped_points(depth, intr, rot, trans, stride, min_range, max_range,
                         xy_extent, min_z, max_z)
    if len(pts) and self_spheres is not None and len(self_spheres):
        pts = pts[robot_mask_spheres(pts, self_spheres, self_margin)]
    elif len(pts) and link_pts is not None:
        pts = pts[robot_mask(pts, link_pts, self_radius)]
    if len(pts) and ignore_region is not None:
        pts = pts[~in_box_mask(pts, ignore_region["center"], ignore_region["dims"])]
    for box in baseline_boxes:
        if not len(pts):
            break
        pts = pts[~in_box_mask(pts, box["position"], box["dims"], inflate=0.01)]
    return pts


def decayable_cells(cells, voxel, frames):
    """Union of every camera's provably-empty voxel set this tick.

    frames: (rot, trans, depth, intr, min_range, max_range) per processed
    camera frame. An empty frame list decays NOTHING — a wrist camera
    that saw no usable frame this tick must not erode the world.
    """
    out = set()
    for rot, trans, depth, intr, min_range, max_range in frames:
        out |= visible_free_cells(
            cells,
            voxel,
            rot,
            trans,
            depth,
            intr["fx"],
            intr["fy"],
            intr["cx"],
            intr["cy"],
            min_range=min_range,
            max_range=max_range,
        )
    return out


class _ViewServer:
    """Live MJPEG debug view — this bench has no monitor (field
    2026-08-17: the cv2 window attempt core-dumped on the headless
    Jetson), so 'a window' is a browser tab: http://<jetson>:<port>/
    streams what the camera sees with the perceived world drawn on top.
    Read-only and best-effort; never load-bearing for perception."""

    def __init__(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._lock = threading.Lock()
        self._jpeg = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body = (
                        b"<html><head><title>cameras view</title></head>"
                        b"<body style='margin:0;background:#111'>"
                        b"<img src='/stream' style='width:100%'>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=f"
                )
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            buf = outer._jpeg
                        if buf is not None:
                            self.wfile.write(
                                b"--f\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: %d\r\n\r\n" % len(buf)
                            )
                            self.wfile.write(buf)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.15)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def update(self, bgr):
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()


class _CameraInput:
    """Latest depth frame + intrinsics for one configured camera."""

    def __init__(self, node, cfg):
        self.cfg = cfg
        self.depth = None
        self.stamp = None
        self.ros_stamp = None
        self.info = None
        # sensor-contract enforcement state (see _assert_sensor_params)
        self.sp_done = not cfg.get("sensor_params")
        self.sp_future = None
        self.sp_client = None
        self.sp_sent_at = 0.0
        node.create_subscription(
            CameraInfo,
            cfg["info_topic"],
            self._info_cb,
            qos_profile_sensor_data,
            callback_group=node.cb_group,
        )
        node.create_subscription(
            Image,
            cfg["depth_topic"],
            self._depth_cb,
            qos_profile_sensor_data,
            callback_group=node.cb_group,
        )

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.info = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            d = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32)
                / 1000.0
            )
        elif msg.encoding == "32FC1":
            d = (
                np.frombuffer(msg.data, dtype=np.float32)
                .reshape(msg.height, msg.width)
                .copy()
            )
        else:
            return
        self.depth = d
        self.stamp = time.monotonic()
        self.ros_stamp = msg.header.stamp

    def fresh(self, max_age):
        return (
            self.depth is not None
            and self.info is not None
            and time.monotonic() - self.stamp < max_age
        )


class CamerasNode(Node):
    def __init__(self):
        super().__init__("cameras")
        self.cb_group = ReentrantCallbackGroup()
        p = self.declare_parameter
        self.rate_hz = float(p("rate_hz", 2.0).value)
        self.baseline = str(p("baseline", "world_real_bench.yaml").value)
        self.voxel = float(p("voxel", 0.03).value)
        # Self-filter. With a self-model, `self_radius` is the MARGIN added
        # to every collision-sphere radius (camera-pose error + TF/depth
        # skew); without one it is the old capsule radius. "" disables
        # the sphere model.
        self.self_radius = float(p("self_radius", 0.08).value)
        self.self_model_name = str(p("self_model", "self_model_gen3_2f85.yaml").value)
        # Solve each FIXED camera's translation error off the arm at
        # startup, from a few still frames, and apply it in memory. The
        # config on disk is untouched; the corrected line is logged.
        self.auto_register = bool(p("auto_register", True).value)
        self.register_frames = int(p("register_frames", 8).value)
        # 20 was sized for the WRIST camera's handful of clusters. A fixed
        # ENVIRONMENT camera sees the whole bench: 60-70 clusters, of which
        # 20 threw two-thirds of the scene away — obstacles the planner
        # never saw (field 2026-08-20). collision_cache_obb is 60 and the
        # real-bench baseline uses 2, so 50 fits with headroom.
        self.max_boxes = int(p("max_boxes", 50).value)
        self.min_voxels = int(p("min_voxels", 8).value)
        # ticks a voxel must be seen before it counts as an obstacle.
        # 3 at 2 Hz = 1.5 s to react; a reactive demo wants 2 at 5 Hz
        # (0.4 s) and pays for it with less transient rejection.
        self.occupied_at = int(p("occupied_at", 3).value)
        self.stride = int(p("stride", 4).value)
        self.xy_extent = float(p("xy_extent", 1.2).value)
        self.min_z = float(p("min_z", 0.03).value)
        self.max_z = float(p("max_z", 1.3).value)
        self.max_motion_mm = float(p("max_motion_mm", 3.0).value)
        self._moving_skips = 0
        self._tick_count = 0
        cam_names = list(p("cameras", ["camera_d405_wrist.yaml"]).value)

        # Two nodes of this name silently break each other's subscriptions:
        # the second one's arrival makes the first report "no fresh depth
        # frames" while the driver is happily publishing at 30 Hz (field
        # 2026-08-21, an hour lost to it). The planner node has guarded
        # this since 2026-08-13; so does this one now.
        time.sleep(1.0)  # let discovery see an already-running peer
        peers = [
            name
            for name, _ns in self.get_node_names_and_namespaces()
            if name == self.get_name()
        ]
        if len(peers) > 1:
            raise SystemExit(
                "another '%s' node is already running — two of them break "
                "each other's depth subscriptions (the symptom is 'no fresh "
                "depth frames' while the driver is fine). Stop the other one "
                "first: ros2 node list. NOTE sweep_demo.launch.py starts a "
                "cameras node of its own." % self.get_name()
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cams = []
        for n in cam_names:
            cfg = load_camera_config(n)
            cfg["_name"] = n                     # for log lines that name the file
            self.cams.append(_CameraInput(self, cfg))
        self.self_model = None
        if self.self_model_name:
            try:
                self.self_model = load_self_model(resolve_config(self.self_model_name))
                self.get_logger().info(
                    "self-model %s: %d spheres over %d frames, margin %.2f m"
                    % (self.self_model_name,
                       sum(len(v) for v in self.self_model.values()),
                       len(self.self_model), self.self_radius)
                )
            except Exception as exc:
                self.get_logger().warn(
                    "self-model %s unavailable (%s) — falling back to %.2f m "
                    "link capsules" % (self.self_model_name, exc, self.self_radius)
                )
        self._registrars = {
            id(cam): SelfRegistrar(frames=self.register_frames)
            for cam in self.cams
            if self.auto_register and cam.cfg.get("parent_frame") == "base_link"
        }
        self.acc = VoxelAccumulator(
            voxel=self.voxel, occupied_at=self.occupied_at
        )
        self.tracker = BoxTracker()
        self.ignore_region = None
        self._last_sent = None
        self._pending = None
        self._warned = set()

        scene = load_scene(resolve_config(self.baseline))
        self.baseline_boxes = [
            {"position": o.position, "dims": o.dims} for o in scene.obstacles
        ]
        self.get_logger().info(
            "baseline %s: %d obstacle(s) stripped from depth"
            % (self.baseline, len(self.baseline_boxes))
        )

        self.world_client = self.create_client(
            UpdateWorldBoxes,
            "/rammp_curobo/update_world_boxes",
            callback_group=self.cb_group,
        )
        self.create_service(
            SetIgnoreRegion,
            "~/set_ignore_region",
            self._set_ignore_cb,
            callback_group=self.cb_group,
        )
        self.markers_pub = self.create_publisher(MarkerArray, "~/world_markers", 1)

        self._view_boxes = {}
        self._view_color = None
        self.view = bool(p("view", True).value)
        if self.view:
            port = int(p("view_port", 8766).value)
            try:
                self._view_server = _ViewServer(port)
            except OSError as e:
                self.get_logger().warn(
                    "view server failed (%s) — continuing headless" % e
                )
                self.view = False
            else:
                cfg0 = self.cams[0].cfg
                if "/depth/" in cfg0.get("depth_topic", ""):
                    ns = cfg0["depth_topic"].rsplit("/depth/", 1)[0]
                    self.create_subscription(
                        Image,
                        ns + "/color/image_raw",
                        self._view_color_cb,
                        qos_profile_sensor_data,
                        callback_group=self.cb_group,
                    )
                self.create_timer(0.2, self._view_tick, callback_group=self.cb_group)
                self.get_logger().info(
                    "live view: http://<this-host>:%d/ (param view:=false "
                    "to disable)" % port
                )

        self.create_timer(1.0 / self.rate_hz, self._tick, callback_group=self.cb_group)
        self.get_logger().info(
            "cameras up: %s @ %.1f Hz, voxel %.0f mm"
            % (cam_names, self.rate_hz, self.voxel * 1000)
        )

    # ------------------------------------------------------------------ TF
    def _base_from(self, frame, stamp=None, strict=False):
        """base_T_frame at `stamp` (a builtin_interfaces Time msg) or latest.

        The stamp matters for a WRIST camera: pairing a depth frame with
        'latest' TF smears points whenever the arm moves (same time-skew
        class as the calibration Accept-time bug). Non-strict lookups fall
        back to latest when the buffer can't serve the stamp; strict ones
        return None instead — the stillness check NEEDS the true stamped
        pose (a latest-fallback would compare latest-vs-latest and call a
        moving camera still).
        """
        whens = [Time.from_msg(stamp)] if stamp is not None else []
        if not strict:
            whens.append(rclpy.time.Time())
        for when in whens:
            try:
                tr = self.tf_buffer.lookup_transform("base_link", frame, when)
            except Exception:
                continue
            q, t = tr.transform.rotation, tr.transform.translation
            return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        return None

    def _camera_pose(self, cfg, stamp=None, strict=False):
        """(R, t) base_T_optical, or None while TF is incomplete."""
        if cfg.get("tf_frame"):
            return self._base_from(cfg["tf_frame"], stamp=stamp, strict=strict)
        parent = self._base_from(cfg["parent_frame"], stamp=stamp, strict=strict)
        if parent is None and cfg["parent_frame"] == "base_link":
            parent = (np.eye(3), np.zeros(3))  # fixed camera needs no TF
        if parent is None:
            return None
        r_p, t_p = parent
        mx = np.asarray(cfg["mount_xyz"], dtype=float)
        qx, qy, qz, qw = cfg["mount_quat_xyzw"]
        return r_p @ quat_to_mat(qx, qy, qz, qw), r_p @ mx + t_p

    def _assert_sensor_params(self, cam):
        """Non-blocking sensor-contract enforcement, retried each tick.
        Fire-and-harvest across ticks; re-arms when the driver's service
        drops (a restarted driver reverts to defaults); a 5 s deadline
        drops a future a dead driver will never answer."""
        from rcl_interfaces.srv import SetParameters

        spec = cam.cfg.get("sensor_params")
        if not spec:
            return
        unreach_key = "sp-unreach:%s" % spec["node"]
        refused_key = "sp-refused:%s" % spec["node"]
        if cam.sp_done:
            if cam.sp_client is not None and not cam.sp_client.service_is_ready():
                cam.sp_done = False
                self.get_logger().warn(
                    "sensor_params: %s went away — will re-assert %s when "
                    "it returns (a restarted driver reverts to defaults)"
                    % (spec["node"], spec["params"])
                )
            return
        if cam.sp_future is not None:
            if not cam.sp_future.done():
                if time.monotonic() - cam.sp_sent_at > 5.0:
                    # a dead server never answers; a restarted one can't
                    # answer the OLD request — drop future AND client
                    cam.sp_future.cancel()
                    cam.sp_future = None
                    self.destroy_client(cam.sp_client)
                    cam.sp_client = None
                    self.get_logger().warn(
                        "sensor_params: %s call timed out — retrying"
                        % spec["node"]
                    )
                return
            res = cam.sp_future.result()
            cam.sp_future = None
            if res is not None and res.results and all(
                r.successful for r in res.results
            ):
                cam.sp_done = True
                self._warned.discard(unreach_key)
                self._warned.discard(refused_key)
                self.get_logger().info(
                    "sensor_params: %s <- %s" % (spec["node"], spec["params"])
                )
            elif refused_key not in self._warned:
                self._warned.add(refused_key)
                self.get_logger().warn(
                    "sensor_params: %s refused %s — retrying"
                    % (spec["node"], spec["params"])
                )
            return
        if cam.sp_client is None:
            cam.sp_client = self.create_client(
                SetParameters,
                spec["node"] + "/set_parameters",
                callback_group=self.cb_group,
            )
        if not cam.sp_client.service_is_ready():
            if unreach_key not in self._warned:
                self._warned.add(unreach_key)
                self.get_logger().warn(
                    "sensor_params: %s not reachable yet — will keep trying "
                    "(depth quality contract unenforced until then)"
                    % spec["node"]
                )
            return
        cam.sp_sent_at = time.monotonic()
        cam.sp_future = cam.sp_client.call_async(_sensor_param_request(spec))

    def _camera_still(self, cam):
        """False while the camera was moving around the frame's stamp.

        A frame captured mid-motion places obstacles at smeared positions
        even with stamped TF (rolling shutter + median-of-frames). Checks
        the pose at the stamp and at two lookbacks (0.075 s / 0.15 s) —
        the midpoint catches direction reversals whose NET displacement is
        near zero — and folds rotation in as its worst-case point sweep at
        max_range: this bracket's optical center sits only ~7.5 cm off the
        tool axis, so a pure wrist twist moves the camera origin 8x slower
        than it sweeps the scene (audit 2026-08-18).
        """
        if cam.cfg.get("parent_frame") == "base_link" or cam.ros_stamp is None:
            return True  # fixed camera: always still
        stamp_t = Time.from_msg(cam.ros_stamp)
        if stamp_t.nanoseconds < int(0.2e9):
            # zero/near-epoch stamps: the driver isn't stamping (tf2 would
            # read Time(0) as 'latest'), and the lookback would underflow —
            # distrust the frame rather than crash (audit 2026-08-18).
            return False
        poses = []
        for dt in (0.0, 0.075, 0.15):
            when = (stamp_t - Duration(seconds=dt)).to_msg()
            got = self._camera_pose(cam.cfg, stamp=when, strict=True)
            if got is None:
                return False  # can't prove stillness -> don't trust it
            poses.append(got)
        limit = self.max_motion_mm / 1000.0
        reach = float(cam.cfg.get("max_range", 1.2))
        for (r1, t1), (r0, t0) in zip(poses[:-1], poses[1:]):
            drift = float(np.linalg.norm(t1 - t0))
            cosang = (np.trace(r0.T @ r1) - 1.0) / 2.0
            ang = float(np.arccos(np.clip(cosang, -1.0, 1.0)))
            if drift + ang * reach > limit:
                self._moving_skips += 1
                return False
        return True

    def _link_points(self, stamp=None):
        """Arm-link origins in base_link, at `stamp` when TF can serve it.

        The stamp matters the moment the ARM moves: masking depth from
        20-75 ms ago against the arm's pose NOW aims the self-filter where
        the arm has already got to, so the real arm's trailing edge leaks
        in as phantom obstacles the planner then dodges. Same time-skew
        class as the camera pose (which _camera_pose already stamps);
        non-strict lookups fall back to latest, so a TF buffer that cannot
        serve the stamp behaves exactly as before.
        """
        pts = []
        for name in ARM_CHAIN:
            got = self._base_from(name, stamp=stamp)
            if got is None:
                return None  # no bringup running — skip self-filtering
            pts.append(got[1])
        got = self._base_from("end_effector_link", stamp=stamp)
        if got is None:
            return None
        r_ee, t_ee = got
        pts.append(t_ee + r_ee @ np.array([0.0, 0.0, 0.18]))  # gripper capsule
        return pts

    def _self_spheres(self, stamp=None):
        """cuRobo's collision spheres in base_link at `stamp`, or None when
        the self-model is off or a frame cannot be resolved."""
        if self.self_model is None:
            return None
        link_tf = {}
        for link in self.self_model:
            got = self._base_from(link, stamp=stamp)
            if got is None:
                return None
            link_tf[link] = got
        spheres, missing = self_model_spheres(self.self_model, link_tf)
        return None if missing else spheres

    def _try_register(self, cam, pose, spheres):
        reg = self._registrars.get(id(cam))
        if reg is None or reg.done:
            return
        self._register(
            cam,
            cropped_points(
                cam.depth, cam.info, pose[0], pose[1], self.stride,
                float(cam.cfg.get("min_range", 0.12)),
                float(cam.cfg.get("max_range", 1.2)),
                self.xy_extent, self.min_z, self.max_z,
            ),
            spheres,
            pose[1],
        )

    def _register(self, cam, pts, spheres, cam_origin):
        """Feed one still, unmasked frame to this camera's registrar; apply
        the solved shift to the mount the moment it passes the gates."""
        reg = self._registrars.get(id(cam))
        if reg is None or reg.done or spheres is None:
            return
        out = reg.feed(pts, spheres, cam_origin)
        if out is None:
            return
        if out["ok"]:
            cfg = cam.cfg
            new = [float(v) for v in np.asarray(cfg["mount_xyz"], float) + out["shift"]]
            self.get_logger().info(
                "camera registered off the arm: shift %s m (%d points, rms "
                "%.1f mm). Using it for this session; to make it permanent "
                "set in %s:\n    mount_xyz: [%.5f, %.5f, %.5f]"
                % (np.round(out["shift"], 4), out["used"], out["rms"] * 1000,
                   cfg.get("_name", "the camera config"), new[0], new[1], new[2])
            )
            cfg["mount_xyz"] = new
            # everything accumulated so far was placed with the OLD mount
            n = self.acc.reset()
            self.tracker = BoxTracker()
            self.get_logger().info("world restarted clean (%d voxels dropped)" % n)
        elif out["final"]:
            self.get_logger().warn(
                "camera NOT registered after %d tries: %s. Obstacles carry "
                "the calibration's pose error; raise self_radius if the arm "
                "appears in its own map." % (out["tries"], out["reason"])
            )

    # ---------------------------------------------------------------- tick
    def _tick(self):
        all_pts, frames_used, saw_frame, had_fresh = [], [], False, False
        link_pts = self._link_points()
        if link_pts is None and "tf" not in self._warned:
            self._warned.add("tf")
            self.get_logger().warn(
                "no arm TF — self-filter OFF (fine without a bringup; the "
                "arm will cluster as an obstacle if one IS running)"
            )
        if link_pts is not None and "tf" in self._warned:
            # self-filter just armed: voxels accumulated from the
            # UNFILTERED arm can never decay (the arm still occupies them,
            # no camera can see through) — start the world over clean.
            self._warned.discard("tf")
            n = self.acc.reset()
            self.tracker = BoxTracker()
            self.get_logger().info(
                "arm TF appeared — self-filter ON, %d unfiltered voxels "
                "dropped, world restarts clean" % n
            )
        for cam in self.cams:
            self._assert_sensor_params(cam)
            if not cam.fresh(max_age=2.0 / self.rate_hz):
                continue
            had_fresh = True
            if not self._camera_still(cam):
                continue
            pose = self._camera_pose(cam.cfg, stamp=cam.ros_stamp)
            if pose is None:
                continue
            saw_frame = True
            # mask against where the arm was WHEN THIS FRAME WAS TAKEN
            cam_links = self._link_points(stamp=cam.ros_stamp) or link_pts
            spheres = self._self_spheres(stamp=cam.ros_stamp) if link_pts is not None else None
            if spheres is not None and id(cam) in self._registrars:
                # a registration bug must never take the perceived world
                # down with it — the planner would keep the last sticky
                # world and nothing downstream would know the feed died
                try:
                    self._try_register(cam, pose, spheres)
                except Exception as exc:
                    self.get_logger().error(
                        "camera registration failed (%s) — continuing with "
                        "the configured mount" % exc
                    )
                    self._registrars.pop(id(cam), None)
                pose = self._camera_pose(cam.cfg, stamp=cam.ros_stamp) or pose
            frames_used.append(
                (
                    pose[0],
                    pose[1],
                    cam.depth,
                    cam.info,
                    float(cam.cfg.get("min_range", 0.12)),
                    float(cam.cfg.get("max_range", 1.2)),
                )
            )
            all_pts.append(
                process_camera_points(
                    cam.depth,
                    cam.info,
                    pose[0],
                    pose[1],
                    stride=self.stride,
                    min_range=float(cam.cfg.get("min_range", 0.12)),
                    max_range=float(cam.cfg.get("max_range", 1.2)),
                    xy_extent=self.xy_extent,
                    min_z=self.min_z,
                    max_z=self.max_z,
                    link_pts=cam_links,
                    self_radius=self.self_radius,
                    ignore_region=self.ignore_region,
                    baseline_boxes=self.baseline_boxes,
                    self_spheres=spheres,
                    self_margin=self.self_radius,
                )
            )
        if not saw_frame:
            if had_fresh:
                # frames ARE arriving; they were gated (camera moving, or
                # stamped TF unavailable) — say so instead of blaming the
                # driver (audit 2026-08-18: the old message misdiagnosed
                # every arm-motion episode)
                if "gated" not in self._warned:
                    self._warned.add("gated")
                    self.get_logger().info(
                        "depth frames gated (camera moving / stamped TF "
                        "unavailable) — world holds until the camera is "
                        "still (%d gated so far)" % self._moving_skips
                    )
            elif "frames" not in self._warned:
                self._warned.add("frames")
                self.get_logger().warn(
                    "no fresh depth frames — is the camera driver running?"
                )
            return
        self._warned.discard("frames")
        self._warned.discard("gated")
        pts = np.vstack(all_pts) if all_pts else np.empty((0, 3))
        # decay is scoped to what THIS tick's frames can prove empty — a
        # wrist camera looking away must not erode the remembered world
        decay = decayable_cells(self.acc.known_cells(), self.voxel, frames_used)
        self.acc.update(pts, decay_cells=decay)
        self._tick_count += 1
        boxes, total = cluster_cells(
            self.acc.occupied_cells(),
            self.voxel,
            min_voxels=self.min_voxels,
            max_boxes=self.max_boxes,
        )
        if total > len(boxes):
            self.get_logger().warn(
                "%d clusters found, capped to nearest %d" % (total, len(boxes))
            )
        named = self.tracker.assign(boxes)
        self._publish_markers(named)
        if self._last_sent is not None and not boxes_changed(named, self._last_sent):
            return
        self._send_world(named)

    def _send_world(self, named):
        if self._pending is not None and not self._pending.done():
            return  # previous update still in flight — next tick retries
        if not self.world_client.service_is_ready():
            if "planner" not in self._warned:
                self._warned.add("planner")
                self.get_logger().warn("planner update_world_boxes not available yet")
            return
        self._warned.discard("planner")
        req = UpdateWorldBoxes.Request()
        req.baseline = self.baseline
        for name, b in sorted(named.items()):
            req.names.append(name)
            req.centers.append(
                Point(x=b["center"][0], y=b["center"][1], z=b["center"][2])
            )
            req.dims.append(Vector3(x=b["dims"][0], y=b["dims"][1], z=b["dims"][2]))
        snapshot = {k: dict(v) for k, v in named.items()}

        def _done(fut):
            res = fut.result() if fut.exception() is None else None
            if res is not None and res.success:
                self._last_sent = snapshot
                self.get_logger().info(res.message)
            else:
                msg = res.message if res is not None else str(fut.exception())
                self.get_logger().error("world update failed: %s" % msg)

        self._pending = self.world_client.call_async(req)
        self._pending.add_done_callback(_done)

    # -------------------------------------------------------------- view
    def _view_color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            self._view_color = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()

    def _view_tick(self):
        import cv2

        cam = self.cams[0]
        img = None
        if self._view_color is not None:
            img = self._view_color.copy()
        elif cam.depth is not None:
            top = max(float(cam.cfg.get("max_range", 0.9)), 0.1)
            d = cam.depth.copy()
            d[~np.isfinite(d)] = 0.0
            dv = np.clip(d / top * 255, 0, 255).astype(np.uint8)
            img = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO)
        if img is None:
            img = np.zeros((240, 424, 3), np.uint8)
            cv2.putText(img, "no frames yet", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        # perceived boxes projected into the image (latest TF + depth
        # intrinsics on the color frame: a few px off — a debug view for
        # human eyes, never load-bearing)
        pose = self._camera_pose(cam.cfg)
        if pose is not None and cam.info is not None:
            r, t = pose
            fx, fy = cam.info["fx"], cam.info["fy"]
            cx, cy = cam.info["cx"], cam.info["cy"]
            h, w = img.shape[:2]
            for name, b in sorted(self._view_boxes.items()):
                c = np.asarray(b["center"], dtype=float)
                half = np.asarray(b["dims"], dtype=float) / 2.0
                corners = c + np.array(
                    [
                        [sx, sy, sz]
                        for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1])
                        for sz in (-half[2], half[2])
                    ]
                )
                pc = (corners - t) @ r  # base -> optical (row form of R^T)
                if (pc[:, 2] <= 0.05).any():
                    continue
                us = fx * pc[:, 0] / pc[:, 2] + cx
                vs = fy * pc[:, 1] / pc[:, 2] + cy
                x1, x2 = int(us.min()), int(us.max())
                y1, y2 = int(vs.min()), int(vs.max())
                if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
                    continue
                cv2.rectangle(
                    img,
                    (max(x1, 0), max(y1, 0)),
                    (min(x2, w - 1), min(y2, h - 1)),
                    (0, 80, 255),
                    2,
                )
                cv2.putText(img, name, (max(x1, 0), max(y1 - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
        cv2.putText(img, "DEPTH OBSTACLE MAP (no object detection here)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
        status = "obstacles %d | gated %d | ignore %s" % (
            len(self._view_boxes),
            self._moving_skips,
            "ON" if self.ignore_region is not None else "off",
        )
        cv2.putText(img, status, (8, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        self._view_server.update(img)

    def _publish_markers(self, named):
        self._view_boxes = dict(named)
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        for i, (name, b) in enumerate(sorted(named.items())):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "perceived", i, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = b["center"]
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = b["dims"]
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.3, 0.1, 0.55
            m.text = name
            arr.markers.append(m)
        self.markers_pub.publish(arr)

    # ------------------------------------------------------------- services
    def _set_ignore_cb(self, request, response):
        d = [request.dims.x, request.dims.y, request.dims.z]
        if not any(d):
            self.ignore_region = None
            response.message = "ignore region cleared"
        else:
            self.ignore_region = {
                "center": [request.center.x, request.center.y, request.center.z],
                "dims": d,
            }
            # purge what's ALREADY mapped there: frustum-scoped decay can
            # never erase a still-present object (it blocks its own
            # see-through), so masking future hits alone would leave the
            # grasp target's cuboid standing forever (audit 2026-08-18)
            purged = self.acc.clear_box(
                self.ignore_region["center"], d, inflate=self.voxel
            )
            response.message = (
                "ignoring %.2f x %.2f x %.2f m at (%.2f, %.2f, %.2f); "
                "%d mapped voxels purged"
                % (
                    d[0],
                    d[1],
                    d[2],
                    request.center.x,
                    request.center.y,
                    request.center.z,
                    purged,
                )
            )
        response.success = True
        self.get_logger().info(response.message)
        return response


def main(args=None):
    import signal
    import threading

    from rclpy.signals import SignalHandlerOptions

    # Own the SIGINT. rclpy's default handler shuts the context down under
    # the spinning executor, and spin() then dies inside a subscription
    # take with a pybind RuntimeError ("Unable to convert call argument")
    # — seen on every launch Ctrl+C on the bench. A flag we poll is clean.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    node = CamerasNode()
    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
