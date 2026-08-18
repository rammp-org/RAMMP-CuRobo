#!/usr/bin/env python3
"""Calibrate a FIXED camera's pose in base_link — no fiducials.

The target is the arm's own fingertip. Close the gripper (the fingertips
meet at ~tool_frame), THE HUMAN moves the arm to N poses spread through
the camera view IN 3D (vary height too, not just a plane); at each pose
press ENTER, then click the fingertip midpoint in the frozen color frame.
The aligned depth at the click deprojects to a camera-frame point, TF
gives the same point in base_link, and a rigid Kabsch fit over all pairs
solves base_T_camera. The result is only written if the RMS residual is
under --max-residual (default 0.02 m).

    export ROS_LOCALHOST_ONLY=1
    # terminal 1: arm bringup (kortex).  terminal 2:
    ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true
    # terminal 3 (no display needed — clicking happens in a browser):
    python3 scripts/calibrate_camera_extrinsics.py --poses 8

The script serves the frozen frame at http://<jetson>:8765 — open that in
a browser on any machine on the lab network, click the fingertip on the
photo, hit Accept (or Skip). This Jetson is headless, hence no cv2 window.

Depth sees the fingertip SURFACE, not its center — the click point is
pushed --surface-bias (default 1 cm) further along the viewing ray.
Writes rammp_curobo_ros/config/camera_orbbec_bench.yaml (depth aligned to
the color frame by depth_registration, so the color extrinsic IS the
depth extrinsic). Rebuild rammp_curobo_ros afterwards so the share/ copy
updates.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join(
    REPO, "rammp_curobo_ros", "config", "camera_orbbec_bench.yaml"
)


def solve_rigid(base_pts, cam_pts):
    """Kabsch: paired points -> (R, t, rms) with base_pt ~= R @ cam_pt + t.

    The det() guard rejects the reflection solution that plain SVD can
    return for noisy or near-planar point sets.
    """
    base = np.asarray(base_pts, dtype=float)
    cam = np.asarray(cam_pts, dtype=float)
    cb, cc = base.mean(axis=0), cam.mean(axis=0)
    h = (cam - cc).T @ (base - cb)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = cb - rot @ cc
    rms = float(np.sqrt(np.mean(np.sum((base - (cam @ rot.T + t)) ** 2, axis=1))))
    return rot, t, rms


def solve_rigid_robust(base_pts, cam_pts, min_keep=5):
    """solve_rigid with iterative outlier trimming.

    A click that lands on the silhouette edge can sample the BACKGROUND
    depth and put that 'fingertip' half a metre off — one such pair ruins
    a plain least-squares fit. Solve, drop the worst pair while its
    residual exceeds max(3 cm, 3x the median), re-solve; never go below
    min_keep pairs. Returns (R, t, rms, residuals, dropped) where
    residuals covers ALL input pairs (in input order) against the final
    fit and dropped lists the rejected indices.
    """
    base = np.asarray(base_pts, dtype=float)
    cam = np.asarray(cam_pts, dtype=float)
    kept = list(range(len(base)))
    dropped = []
    while True:
        rot, t, _ = solve_rigid(base[kept], cam[kept])
        res_kept = np.linalg.norm(base[kept] - (cam[kept] @ rot.T + t), axis=1)
        worst = int(np.argmax(res_kept))
        limit = max(0.03, 3.0 * float(np.median(res_kept)))
        if res_kept[worst] > limit and len(kept) > min_keep:
            dropped.append(kept.pop(worst))
            continue
        rms = float(np.sqrt(np.mean(res_kept**2)))
        residuals = np.linalg.norm(base - (cam @ rot.T + t), axis=1)
        return rot, t, rms, residuals, sorted(dropped)


def spread_check(base_pts):
    """None if the poses span 3D well; else a human-readable complaint.

    Kabsch needs non-collinear points for any rotation at all, and
    non-coplanar spread for a well-conditioned one.
    """
    base = np.asarray(base_pts, dtype=float)
    svals = np.linalg.svd(base - base.mean(axis=0), compute_uv=False)
    if svals[1] < 0.02:
        return "poses are nearly COLLINEAR — the rotation is unsolvable"
    if svals[2] < 0.03:
        return (
            "poses are nearly COPLANAR (3rd axis spread %.0f mm) — vary the "
            "HEIGHT of the fingertip between poses" % (svals[2] * 1000)
        )
    return None


_PAGE = """<!doctype html><html><head><title>calibrate</title><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:12px}
#wrap{position:relative;display:inline-block}
#im{max-width:100%%;display:block}
#mark{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;
  border:2px solid red;border-radius:50%%;pointer-events:none;display:none}
button{font-size:1.1em;margin:8px 8px 0 0;padding:6px 18px}
#msg{margin-top:8px}
</style></head><body>
<h3 id="head">calibration — waiting for a pose (press ENTER in the terminal)</h3>
<div id="wrap"><img id="im"><div id="mark"></div></div><br>
<button id="ok" disabled>Accept click</button>
<button id="skip" disabled>Skip pose</button>
<div id="msg">Click the point where the CLOSED fingertips meet.</div>
<script>
let pose=-1, uv=null;
const im=document.getElementById('im'), mark=document.getElementById('mark');
im.onclick=e=>{
  const r=im.getBoundingClientRect();
  const sx=im.naturalWidth/r.width, sy=im.naturalHeight/r.height;
  uv=[Math.round((e.clientX-r.left)*sx), Math.round((e.clientY-r.top)*sy)];
  mark.style.left=(e.clientX-r.left)+'px'; mark.style.top=(e.clientY-r.top)+'px';
  mark.style.display='block'; document.getElementById('ok').disabled=false;
};
function decide(body){
  fetch('/decide',{method:'POST',body:JSON.stringify(body)});
  document.getElementById('ok').disabled=true;
  document.getElementById('skip').disabled=true;
  mark.style.display='none'; uv=null;
  document.getElementById('head').textContent=
    'recorded — move the arm, then ENTER in the terminal';
}
document.getElementById('ok').onclick=()=>{ if(uv) decide({u:uv[0],v:uv[1]}); };
document.getElementById('skip').onclick=()=>decide({skip:true});
setInterval(async()=>{
  try{
    const s=await (await fetch('/status')).json();
    if(s.armed && s.pose!==pose){
      pose=s.pose; im.src='/frame.jpg?p='+pose;
      document.getElementById('head').textContent='pose '+pose+' — click the fingertip';
      document.getElementById('skip').disabled=false;
      document.getElementById('ok').disabled=true;
      mark.style.display='none'; uv=null;
    }
  }catch(e){}
},700);
</script></body></html>"""


class BrowserClickUI:
    """Serves the frozen frame over HTTP; the human clicks in a browser.

    No display needed on this machine — built after the bench Jetson
    turned out to be headless (X window attempt core-dumped on GDM auth).
    """

    def __init__(self, port):
        import http.server
        import json
        import threading

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._state = {"jpeg": None, "pose": 0, "armed": False, "decision": None}
        state, lock, event = self._state, self._lock, self._event

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _send(self, code, body, ctype="text/html"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/frame.jpg"):
                    with lock:
                        jpeg = state["jpeg"]
                    if jpeg is None:
                        self._send(404, b"no frame yet", "text/plain")
                    else:
                        self._send(200, jpeg, "image/jpeg")
                elif self.path.startswith("/status"):
                    import json as _json

                    with lock:
                        body = _json.dumps(
                            {"pose": state["pose"], "armed": state["armed"]}
                        ).encode()
                    self._send(200, body, "application/json")
                else:
                    self._send(200, _PAGE.encode())

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                try:
                    data = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    data = {}
                with lock:
                    if not state["armed"]:
                        self._send(409, b"not armed", "text/plain")
                        return
                    if data.get("skip"):
                        state["decision"] = None
                    elif "u" in data and "v" in data:
                        state["decision"] = (int(data["u"]), int(data["v"]))
                    else:
                        self._send(400, b"bad body", "text/plain")
                        return
                    state["armed"] = False
                event.set()
                self._send(200, b"ok", "text/plain")

        self._srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def get_click(self, jpeg_bytes):
        """Arm the page with a frame; block until Accept/Skip. (u, v) or None."""
        with self._lock:
            self._state["jpeg"] = jpeg_bytes
            self._state["pose"] += 1
            self._state["armed"] = True
            self._state["decision"] = None
        self._event.clear()
        while not self._event.wait(timeout=0.5):
            pass  # loop keeps Ctrl-C responsive
        with self._lock:
            return self._state["decision"]


def _lan_ip():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        s.close()


def mat_to_quat_xyzw(R):
    """3x3 rotation matrix -> xyzw quaternion (Shepperd's method)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        return [float(x), float(y), float(z), float(w)]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4.0
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    w = (R[k, j] - R[j, k]) / s
    return [float(q[0]), float(q[1]), float(q[2]), float(w)]


def main():
    ap = argparse.ArgumentParser(
        description="fiducial-free extrinsic calibration (click the fingertip)"
    )
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--depth-info-topic", default="/camera/depth/camera_info")
    # tool_frame (the fingertip midpoint) exists only in the PLANNER's
    # kinematics — the ROS URDF/TF ends at end_effector_link (wrist
    # flange), so look that up and add the fixed flange->fingertip offset
    # (cuRobo kinova_gen3_7dof.urdf: tool_frame = end_effector_link +
    # [0, 0, 0.120], no rotation).
    ap.add_argument("--tool-frame", default="end_effector_link")
    ap.add_argument(
        "--tip-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.120],
        metavar=("X", "Y", "Z"),
        help="fingertip midpoint in the tool frame's local axes (m)",
    )
    ap.add_argument("--max-residual", type=float, default=0.02)
    ap.add_argument(
        "--surface-bias",
        type=float,
        default=0.010,
        help="metres added along the viewing ray (depth sees the fingertip "
        "surface, tool_frame is its center)",
    )
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument(
        "--port", type=int, default=8765, help="HTTP port for the click page"
    )
    ap.add_argument(
        "--pairs-file",
        default=os.path.expanduser("~/.ros/rammp_curobo/calib_pairs.json"),
        help="recorded point pairs are saved here after every accept",
    )
    ap.add_argument(
        "--resolve-from",
        default=None,
        metavar="JSON",
        help="skip collection; re-solve from a saved pairs file",
    )
    args = ap.parse_args()

    import cv2
    import rclpy
    import yaml
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    from rammp_curobo.perception import quat_to_mat

    class Grab(Node):
        def __init__(self):
            super().__init__("calibrate_camera_extrinsics")
            self.img = None
            self.depths = []
            self.info = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(
                Image, args.color_topic, self._img_cb, qos_profile_sensor_data
            )
            self.create_subscription(
                Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data
            )
            self.create_subscription(
                CameraInfo, args.info_topic, self._info_cb, qos_profile_sensor_data
            )

        def _img_cb(self, msg):
            if msg.encoding in ("rgb8", "bgr8"):
                a = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                self.img = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()

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
            self.depths.append(d)
            del self.depths[:-5]

        def _info_cb(self, msg):
            self.info = msg

        def grab(self, n_depth=3, timeout_s=10.0):
            """Fresh color frame + median of the NEXT n_depth depth frames."""
            self.img = None
            self.depths.clear()
            t0 = self.get_clock().now()
            while (
                self.img is None or len(self.depths) < n_depth or self.info is None
            ):
                rclpy.spin_once(self, timeout_sec=0.2)
                if (self.get_clock().now() - t0).nanoseconds > timeout_s * 1e9:
                    sys.exit(
                        "no frames on %s / %s — is the camera driver running "
                        "with depth_registration:=true?"
                        % (args.color_topic, args.depth_topic)
                    )
            return self.img, np.median(np.stack(self.depths[-n_depth:]), axis=0)

        def tool_position(self):
            tr = self.tf_buffer.lookup_transform(
                "base_link", args.tool_frame, rclpy.time.Time()
            )
            q, t = tr.transform.rotation, tr.transform.translation
            rot = quat_to_mat(q.x, q.y, q.z, q.w)
            return np.array([t.x, t.y, t.z]) + rot @ np.asarray(
                args.tip_offset, dtype=float
            )

    ui = None  # created only when collecting (binds the HTTP port)

    def click_point(img, depth, k):
        """Serve the frozen frame to the browser; deprojected click or None."""
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            sys.exit("JPEG encode failed")
        while True:
            uv = ui.get_click(jpeg.tobytes())
            if uv is None:
                return None
            u, v = uv
            patch = depth[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4]
            patch = patch[(patch > 0.05) & np.isfinite(patch)]
            if not len(patch):
                print("  no valid depth at that click — click again in the browser")
                continue
            # A click near the finger's silhouette mixes finger and
            # BACKGROUND depths; the plain median can pick the wall behind
            # and place the "fingertip" half a metre off (field, 2026-08-18:
            # one such pair blew an 8-pose solve to 15 cm RMS). Keep only
            # the nearest cluster — the finger is always the foreground.
            near = patch[patch < patch.min() + 0.03]
            z = float(np.median(near))
            p = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
            # push from the fingertip SURFACE to its center, along the ray
            p *= (np.linalg.norm(p) + args.surface_bias) / np.linalg.norm(p)
            return p

    node = None
    if args.resolve_from:
        with open(args.resolve_from) as f:
            d = json.load(f)
        base_pts = [np.asarray(p, dtype=float) for p in d["base"]]
        cam_pts = [np.asarray(p, dtype=float) for p in d["cam"]]
        print(
            "re-solving from %d saved pairs (%s) — no camera/arm needed"
            % (len(base_pts), args.resolve_from)
        )
    else:
        rclpy.init()
        node = Grab()
        ui = BrowserClickUI(args.port)
        print(
            "CLOSE the gripper first (fingertips together = tool_frame). "
            "%d poses, spread across the view AND in height (coplanar pose "
            "sets weaken the solve).\n\n"
            "  >>> open  http://%s:%d  in a browser on your laptop <<<\n\n"
            "Per pose: move the arm, ENTER here, then click the fingertip "
            "midpoint on the photo in the browser and hit Accept (or Skip). "
            "'done' after >=5 solves early, Ctrl-C aborts."
            % (args.poses, _lan_ip(), args.port)
        )
        base_pts, cam_pts = [], []
        while len(base_pts) < args.poses:
            prompt = "[%d/%d] > " % (len(base_pts) + 1, args.poses)
            if input(prompt).strip() == "done":
                break
            # The robot position MUST come from the same instant as the
            # photo. Field failure (2026-08-18, run 2): TF was read at
            # Accept-click time, seconds after the frame — pressing ENTER
            # while the arm was still settling paired a mid-motion photo
            # with the settled position (26-188 mm skew, 7.6 cm RMS solve).
            # Bracket the frame grab with two TF reads and refuse the pose
            # if the arm moved between them.
            try:
                p_before = node.tool_position()
                img, depth = node.grab()
                p_base = node.tool_position()
            except Exception as exc:
                print(
                    "  no TF base_link->%s (%s) — is the bringup up?"
                    % (args.tool_frame, exc)
                )
                continue
            if np.linalg.norm(p_base - p_before) > 0.003:
                print(
                    "  ARM STILL MOVING (fingertip drifted %.0f mm during the "
                    "capture) — let it settle, then ENTER again"
                    % (np.linalg.norm(p_base - p_before) * 1000)
                )
                continue
            if depth.shape != img.shape[:2]:
                sys.exit(
                    "depth %s and color %s sizes differ — run the driver with "
                    "depth_registration:=true" % (depth.shape, img.shape[:2])
                )
            k = np.array(node.info.k).reshape(3, 3)
            p_cam = click_point(img, depth, k)
            if p_cam is None:
                print("  skipped")
                continue
            cam_pts.append(p_cam)
            base_pts.append(p_base)
            print(
                "  recorded (fingertip %.2f m from camera, %.2f m from base)"
                % (float(np.linalg.norm(p_cam)), float(np.linalg.norm(p_base)))
            )
            os.makedirs(os.path.dirname(args.pairs_file), exist_ok=True)
            with open(args.pairs_file, "w") as f:
                json.dump(
                    {
                        "base": [list(map(float, p)) for p in base_pts],
                        "cam": [list(map(float, p)) for p in cam_pts],
                    },
                    f,
                )
    if len(base_pts) < 5:
        sys.exit("only %d poses — need at least 5" % len(base_pts))
    complaint = spread_check(base_pts)
    if complaint:
        sys.exit("BAD POSE SPREAD: %s. Re-run with better spread." % complaint)
    rot, t, rms, residuals, dropped = solve_rigid_robust(base_pts, cam_pts)
    for i, r in enumerate(residuals):
        print(
            "  pose %d: residual %4.0f mm%s"
            % (i + 1, r * 1000, "  << DROPPED (outlier)" if i in dropped else "")
        )
    kept = len(base_pts) - len(dropped)
    print(
        "base_T_camera t = [%.4f, %.4f, %.4f], RMS residual %.4f m "
        "over %d poses (%d dropped)" % (t[0], t[1], t[2], rms, kept, len(dropped))
    )
    print("sanity: tape-measure the camera against those numbers (base frame).")
    if not args.resolve_from:
        print(
            "(pairs saved to %s — re-solve without re-clicking via "
            "--resolve-from)" % args.pairs_file
        )
    if rms > args.max_residual:
        sys.exit(
            "RESIDUAL %.4f m > %.3f m — NOT writing. Click more carefully, "
            "keep the gripper fully closed, spread the poses wider."
            % (rms, args.max_residual)
        )
    if kept < 5:
        sys.exit("only %d poses survived outlier trimming — collect more" % kept)
    cfg = {
        "depth_topic": args.depth_topic,
        "info_topic": args.depth_info_topic,
        "parent_frame": "base_link",
        "mount_xyz": [float(v) for v in t],
        "mount_quat_xyzw": mat_to_quat_xyzw(rot),
        "min_range": 0.25,
        "max_range": 1.5,
        "calibrated": "fingertip-click Kabsch, RMS %.4f m, %d poses (%d dropped)"
        % (rms, kept, len(dropped)),
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote %s — REBUILD rammp_curobo_ros so the share/ copy updates" % args.out)
    # quat_to_mat imported for parity with the node's YAML consumption —
    # keep the round-trip honest if conventions ever drift
    assert np.allclose(quat_to_mat(*cfg["mount_quat_xyzw"]), rot, atol=1e-6)
    if node is not None:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
