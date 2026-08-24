"""GraspGenX client: masked depth in, 6-DoF grasp poses in base_link out.

Talks the GraspGenX ZMQ wire protocol directly (msgpack + msgpack_numpy),
so this repo depends on a reachable SOCKET, not on GraspGenX being
importable. Start the server once, out-of-tree:

    cd ~/GraspGenX && ~/graspgen_venv/bin/python client-server/graspgenx_server.py \
        --config ext/graspgenx_checkpoints/release --assets_dir assets \
        --port 5556 --default_gripper robotiq_2f_85

FRAMES (the part that silently ruins grasps if you get it wrong):
GraspGenX returns 4x4 poses in the CAMERA optical frame, in the gripper's
canonical frame — +Z the approach axis, +X the closing direction. For the
Robotiq 2F-85 that frame is EXACTLY our `end_effector_link`: GraspGenX's
gripper.urdf and kinova_gen3_7dof.urdf both join the 2F-85 base with the
same +90 deg Z offset, so the usual grasp_to_tool rotation cancels. Our
cuRobo ee_link is `tool_frame` = end_effector_link + [0, 0, 0.120], so
the whole conversion is a 12 cm push along the grasp's own +Z:

    T_base_tool = T_base_cam @ T_grasp @ translate(0, 0, tool_offset)

Caveat carried deliberately: GraspGenX declares the 2F-85 fingertip at
z = 0.136 while our tool_frame sits at 0.120 — 16 mm shallower. That is
`tool_offset` and it is a knob, not a constant of nature.
"""

import numpy as np

from rammp_curobo.perception import mat_to_quat_xyzw  # noqa: F401  (re-export: seeker imports it here)

# Robotiq 2F-85 sweep-volume conditioning, read out of GraspGenX's own
# gripper config so the client needs no GraspGenX assets.
ROBOTIQ_2F85_SWEEP = {
    "extents_open": [0.085, 0.032, 0.036],
    "offset_open": [0.0, 0.0, 0.130],
    "extents_mid": [0.046, 0.032, 0.036],
    "offset_mid": [0.0, 0.0, 0.143],
    "gripper_type": 1,
    "fingertip_depth": 0.136,
}


def to_base(grasp_cam, rot_cam, trans_cam, tool_offset=0.120):
    """Camera-frame 4x4 grasp -> (pos, quat_xyzw) goal for tool_frame."""
    t_base_cam = np.eye(4)
    t_base_cam[:3, :3] = rot_cam
    t_base_cam[:3, 3] = trans_cam
    push = np.eye(4)
    push[2, 3] = tool_offset
    t = t_base_cam @ np.asarray(grasp_cam, dtype=float) @ push
    return t[:3, 3], mat_to_quat_xyzw(t[:3, :3])


def pregrasp(pos, rot3, standoff=0.10):
    """Back off along the grasp's own approach axis (-Z)."""
    return np.asarray(pos, dtype=float) - np.asarray(rot3, dtype=float)[:, 2] * standoff


def quat_to_mat3(q):
    from rammp_curobo.perception import quat_to_mat

    return quat_to_mat(*q)


class GraspClient:
    """Minimal GraspGenX ZMQ client (REQ/REP, msgpack)."""

    def __init__(self, endpoint="tcp://127.0.0.1:5556", timeout_ms=20000):
        import msgpack
        import msgpack_numpy
        import zmq

        self._msgpack = msgpack
        msgpack_numpy.patch()
        self._ctx = zmq.Context.instance()
        self._zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._sock = None

    def _connect(self):
        if self._sock is not None:
            return
        s = self._ctx.socket(self._zmq.REQ)
        s.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        s.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(self._zmq.LINGER, 0)
        s.connect(self.endpoint)
        self._sock = s

    def _request(self, payload):
        """One REQ/REP round trip. A timeout closes the socket — a REQ
        socket that missed its reply is wedged for every later call."""
        self._connect()
        try:
            self._sock.send(self._msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except Exception:
            self._sock.close()
            self._sock = None
            raise
        res = self._msgpack.unpackb(raw, raw=False)
        # success returns the payload directly; only failures carry "error"
        if "error" in res:
            raise RuntimeError(res["error"])
        return res

    def grasps_from_mask(self, depth_m, intrinsics, mask, planner="graspmoe",
                         num_grasps=200, threshold=0.5, topk=10):
        """(K,4,4) camera-frame poses + (K,) scores, best first.

        depth_m: (H,W) float32 METERS (0/NaN invalid) — uint16 is refused
        upstream. mask: (H,W) int, 0 = ignore; our YOLO instance mask as 1.
        """
        depth = np.asarray(depth_m, dtype=np.float32)
        mask = np.asarray(mask).astype(np.int32)
        if mask.shape != depth.shape:
            raise ValueError("mask %s != depth %s" % (mask.shape, depth.shape))
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        res = self._request({
            "action": "infer_scene_depth",
            "depth": depth,
            "intrinsics": k,
            "instance_mask": mask,
            "sweep_volume_params": dict(ROBOTIQ_2F85_SWEEP),
            "min_object_points": 100,
            "num_grasps": int(num_grasps),
            "planner": planner,
        })
        # scene replies are PARALLEL LISTS (msgpack rejects int map keys):
        # instance_ids / grasps / confidences. We mask one object, so take
        # the instance with the most grasps.
        best_g, best_c = None, None
        for g_i, c_i in zip(res.get("grasps") or [], res.get("confidences") or []):
            g = np.asarray(g_i, dtype=float).reshape(-1, 4, 4)
            c = np.asarray(c_i, dtype=float).reshape(-1)
            if best_g is None or len(g) > len(best_g):
                best_g, best_c = g, c
        if best_g is None or len(best_g) == 0:
            return np.zeros((0, 4, 4)), np.zeros((0,))
        keep = best_c >= threshold
        g, c = best_g[keep], best_c[keep]
        order = np.argsort(-c)[:topk]
        return g[order], c[order]
