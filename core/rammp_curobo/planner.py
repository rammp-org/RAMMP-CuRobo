"""CuRoboPlanner: the pure-Python planning core (no ROS imports).

    from rammp_curobo import CuRoboPlanner

    planner = CuRoboPlanner.from_config('gen3.yaml')
    res = planner.plan_to_pose([0.45, 0.0, 0.35], [1, 0, 0, 0], quat_order='wxyz')
    res = planner.plan_to_joints(planner.home_pose)
    planner.update_world([{'name': 'box', 'position': [0.5, 0, 0], 'dims': [0.1, 0.1, 0.1]}])

Everything cuRobo stays behind this class; results come back as plain
numpy-backed dataclasses (types.PlanResult / types.Trajectory).

Jetson-specific decisions inherited from RAMMP-Kinova's field notes (do not
"clean these up" without re-testing on the Orin):
  * torch linalg is routed to MAGMA and the graph (PRM) planner stays OFF —
    the Jetson torch wheel's cuSOLVER path lacks cusolverDnXsyevBatched and
    dies inside torch.svd (DT_EXCEPTION).
  * plan_to_joints tries native plan_single_js first (verified working on
    the torch 2.10 jp6/cu126 wheel) and falls back to FK-pose planning on
    failure; the graph fallback that killed js planning on older wheels
    stays hard-disabled either way.
  * cuRobo velocity_scale is never passed (plan at 1.0); slow execution is
    time dilation at the execution layer (retime.py).
  * Only the TRIMMED interpolated plan leaves this class — cuRobo result
    buffers are padded and a stale tail is the violent-motion failure mode.
"""

import logging
import math
import time

import numpy as np

from rammp_curobo import geometry
from rammp_curobo.config import load_planner_config, resolve_config
from rammp_curobo.robot_config import load_robot_config
from rammp_curobo.scene import Scene, load_scene, merged_scene, scene_from_obstacles
from rammp_curobo.types import PlanResult, Trajectory
from rammp_curobo.validate import validate_trajectory
from rammp_curobo.world import make_world_config

log = logging.getLogger("rammp_curobo")


class CuRoboPlanner:
    """One cuRobo MotionGen instance wrapped in a config-driven API.

    Construction is HEAVY (CUDA init + kernel warmup, tens of seconds on
    the Orin the first time) — build once, keep it alive.
    """

    def __init__(self, config, config_dir=None):
        self._cfg = config
        self._config_dir = config_dir
        p = config["planner"]
        self.joint_names = list(config["joint_names"])
        self.home_pose = [float(v) for v in config["home_pose_rad"]]
        self.interpolation_dt = float(p["interpolation_dt"])
        self.max_attempts = int(p["max_attempts"])
        self.finetune_attempts = int(p["finetune_attempts"])
        self.enable_finetune = bool(p["enable_finetune"])
        self.enable_graph = bool(p["enable_graph"])
        self.collision_cache_obb = int(p["collision_cache_obb"])
        self.collision_cache_mesh = int(p["collision_cache_mesh"])
        self.collision_activation_distance = float(p["collision_activation_distance"])
        self.world_padding = float(p["world_padding"])
        self.no_pad_names = frozenset(p["no_pad_names"] or [])
        self.joint_space_method = str(p["joint_space_method"])
        self.limit_clamp_rad = float(p["limit_clamp_rad"])
        self.tool_spin_deg = float(config["tool"]["spin_deg"])
        self.tool_tip_offset = float(config["tool"]["tip_offset_m"])
        self.execution = dict(config["execution"])

        robot_path = resolve_config(config["robot"], relative_to=config_dir)
        world_path = resolve_config(config["world"], relative_to=config_dir)
        self._robot_cfg = load_robot_config(
            robot_path, home_pose_rad=self.home_pose, joint_names=self.joint_names
        )
        self._scene = load_scene(world_path)
        self._init_curobo(warmup=bool(p["warmup"]))

    @classmethod
    def from_config(cls, name_or_path):
        """Build from a planner YAML (packaged name like 'gen3.yaml' or a
        filesystem path). See configs/gen3.yaml for the schema."""
        cfg, cfg_dir = load_planner_config(name_or_path)
        return cls(cfg, config_dir=cfg_dir)

    # ------------------------------------------------------------ cuRobo setup
    def _init_curobo(self, warmup=True):
        import torch

        try:
            # Newer warp-lang (>=~1.6) needs the torch interop imported
            # explicitly; cuRobo v0.7.8 assumes implicit wp.torch and
            # crashes in its mesh collision checker otherwise.
            import warp.torch  # noqa: F401
        except ImportError:
            pass
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except Exception:
            pass
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        self._torch = torch
        self._tensor_args = TensorDeviceType()
        world = make_world_config(
            self._scene,
            padding=self.world_padding,
            no_pad_names=self.no_pad_names,
            cache_obb=self.collision_cache_obb,
        )
        log.info("Loading cuRobo MotionGen (%s)...", self._cfg["robot"])
        cfg = MotionGenConfig.load_from_robot_config(
            self._robot_cfg,
            world,
            tensor_args=self._tensor_args,
            interpolation_dt=self.interpolation_dt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": self.collision_cache_obb,
                "mesh": self.collision_cache_mesh,
            },
            collision_activation_distance=self.collision_activation_distance,
        )
        self._motion_gen = MotionGen(cfg)
        # cuRobo consumes start states in ITS cspace order, not by name —
        # plan_single does not reorder. Remap by name at every boundary.
        self._curobo_joint_names = list(self._motion_gen.kinematics.joint_names)
        if set(self._curobo_joint_names) != set(self.joint_names):
            raise RuntimeError(
                "cuRobo joints %s != configured joints %s"
                % (self._curobo_joint_names, self.joint_names)
            )
        if warmup:
            log.info("Warming up cuRobo (first-run kernel compile)...")
            self._motion_gen.warmup(enable_graph=self.enable_graph)
            log.info("cuRobo warmup complete.")

    # ---------------------------------------------------------------- planning
    def plan_to_pose(
        self,
        position,
        quaternion,
        start=None,
        quat_order="xyzw",
        apply_tool_correction=None,
    ):
        """Plan a collision-free trajectory to an end-effector pose.

        position: [x, y, z] metres in the robot base frame.
        quaternion: orientation of cuRobo's ee_link (tool_frame — for the
            Gen3 config that is 0.120 m beyond the wrist flange, roughly the
            fingertip midpoint). quat_order says how it is packed: 'xyzw'
            (ROS, the default) or 'wxyz' (cuRobo).
        start: joint positions in controller order; None = configured home.
        apply_tool_correction: apply the configured tool spin/tip-offset
            calibration (authored-fingertip goals). None = apply whenever
            the config carries non-zero values.
        """
        t0 = time.monotonic()
        if quat_order == "xyzw":
            wxyz = geometry.xyzw_to_wxyz(quaternion)
        elif quat_order == "wxyz":
            wxyz = [float(v) for v in quaternion]
        else:
            raise ValueError("quat_order must be 'xyzw' or 'wxyz'")
        n = float(np.linalg.norm(wxyz))
        if n < 1e-6:
            return PlanResult.failure("BAD_GOAL", "zero-length quaternion")
        wxyz = [v / n for v in wxyz]
        xyz = [float(v) for v in position]

        correct = apply_tool_correction
        if correct is None:
            correct = bool(self.tool_spin_deg or self.tool_tip_offset)
        if correct and self.tool_spin_deg:
            wxyz = geometry.spin_about_tool(wxyz, self.tool_spin_deg)
        if correct and self.tool_tip_offset:
            xyz = geometry.tip_to_tool(xyz, wxyz, self.tool_tip_offset)

        from curobo.types.math import Pose

        start, err = self._clamp_to_limits(start, "start")
        if err is not None:
            return PlanResult.failure("START_OUTSIDE_LIMITS", err)
        start_state = self._start_state(start)
        goal = Pose(position=self._tensor([xyz]), quaternion=self._tensor([wxyz]))
        try:
            result = self._motion_gen.plan_single(
                start_state, goal, self._plan_config()
            )
        except Exception as exc:
            return PlanResult.failure(
                "EXCEPTION",
                "cuRobo plan_single raised: %s" % exc,
                timing=time.monotonic() - t0,
            )
        return self._finish(result, t0)

    def plan_to_joints(self, q_goal, start=None, method=None):
        """Plan a collision-free trajectory to a joint configuration.

        method 'auto' (default): try native plan_single_js — exact joint
        goal, no elbow-family surprises (verified working on the Jetson's
        torch 2.10 jp6/cu126 wheel; the graph fallback that killed it on
        older wheels stays hard-disabled either way) — and on any failure
        fall back to 'fk_pose'. method 'fk_pose': plan in POSE space to the
        FK of q_goal; reaches the same tool pose but redundancy may land a
        DIFFERENT joint vector — goal_mismatch_rad records the gap, check
        it before executing anything that assumes specific joints. method
        'js': force native plan_single_js with no fallback (what 'auto'
        tries first).
        """
        t0 = time.monotonic()
        q_goal = [float(v) for v in q_goal]
        if len(q_goal) != len(self.joint_names):
            raise ValueError(
                "q_goal has %d values for %d joints"
                % (len(q_goal), len(self.joint_names))
            )
        ref = self.home_pose if start is None else [float(v) for v in start]
        q_goal = self._nearest_branch(q_goal, ref)
        q_goal, err = self._clamp_to_limits(q_goal, "goal")
        if err is not None:
            return PlanResult.failure("GOAL_OUTSIDE_LIMITS", err)
        start, err = self._clamp_to_limits(start, "start")
        if err is not None:
            return PlanResult.failure("START_OUTSIDE_LIMITS", err)
        method = method or self.joint_space_method

        if method == "auto":
            res = self.plan_to_joints(q_goal, start=start, method="js")
            if res.success:
                return res
            log.warning(
                "plan_single_js failed (%s) — falling back to " "FK-pose planning",
                res.status,
            )
            res = self.plan_to_joints(q_goal, start=start, method="fk_pose")
            res.timing = time.monotonic() - t0
            return res
        if method == "js":
            goal = self._curobo_state(q_goal)
            try:
                result = self._motion_gen.plan_single_js(
                    self._start_state(start), goal, self._plan_config()
                )
            except Exception as exc:
                return PlanResult.failure(
                    "EXCEPTION",
                    "cuRobo plan_single_js raised: %s" % exc,
                    timing=time.monotonic() - t0,
                )
            return self._finish(result, t0, q_goal=q_goal)
        if method == "fk_pose":
            pos, wxyz = self.fk(q_goal, quat_order="wxyz")
            res = self.plan_to_pose(
                pos, wxyz, start=start, quat_order="wxyz", apply_tool_correction=False
            )
            res.timing = time.monotonic() - t0
            if res.joint_traj is not None:
                res.goal_mismatch_rad = float(
                    np.abs(res.final_joints - np.asarray(q_goal)).max()
                )
            return res
        raise ValueError(
            "joint-space method must be 'auto', 'js', or " "'fk_pose', got %r" % method
        )

    def update_world(self, world, ignore=()):
        """Replace the collision world.

        world: a scene YAML path/name, a Scene, or a list of obstacle dicts
        ({'name', 'position', and 'dims' | 'type'+'radius'(+'height')}).
        ignore: prop names to leave out (the object being reached for).
        Guards refuse empty worlds and cache overflow — both are silent or
        cryptic failures inside cuRobo v0.7.8.
        """
        if isinstance(world, Scene):
            scene = world
        elif isinstance(world, (list, tuple)):
            scene = scene_from_obstacles(world)
        else:
            scene = load_scene(resolve_config(world, relative_to=self._config_dir))
        wc = make_world_config(
            scene,
            padding=self.world_padding,
            ignore=frozenset(ignore),
            no_pad_names=self.no_pad_names,
            cache_obb=self.collision_cache_obb,
        )
        self._motion_gen.update_world(wc)
        self._scene = scene
        log.info(
            "Collision world: %d boxes%s",
            len(wc.cuboid),
            " (ignoring: %s)" % ", ".join(sorted(ignore)) if ignore else "",
        )

    def update_world_boxes(self, boxes, baseline=None):
        """Continuous-perception entry: merge perceived AABBs onto the
        static baseline and swap the collision world.

        boxes: dicts with name + position + dims (base frame, metres).
        baseline: world YAML name/path to (re)load as the static scene;
        None keeps the current one (the initial world until a baseline is
        ever named). Callers serialize with planning themselves (the ROS
        node holds its plan lock around this).
        """
        if baseline:
            self._baseline_scene = load_scene(
                resolve_config(baseline, relative_to=self._config_dir)
            )
        if getattr(self, "_baseline_scene", None) is None:
            self._baseline_scene = self._scene
        self.update_world(merged_scene(self._baseline_scene, boxes))

    # ----------------------------------------------------------------- queries
    @property
    def scene(self):
        return self._scene

    @property
    def world_name(self):
        """The world reference the planner was configured with (a name or
        path string; informational — update_world does not change it)."""
        return str(self._cfg.get("world", ""))

    def fk(self, q, quat_order="xyzw"):
        """Forward kinematics of a controller-order joint vector ->
        (position [x,y,z], quaternion in `quat_order`) of the ee_link."""
        state = self._motion_gen.kinematics.get_state(
            self._tensor([self._to_curobo_order(q)])
        )
        pos = [float(v) for v in state.ee_position[0].tolist()]
        wxyz = [float(v) for v in state.ee_quaternion[0].tolist()]
        return pos, (wxyz if quat_order == "wxyz" else geometry.wxyz_to_xyzw(wxyz))

    def joint_limits(self):
        """{'position': (2, dof) [lower; upper], 'velocity': (dof,)} in
        CONTROLLER joint order (numpy, radians)."""
        lim = self._motion_gen.kinematics.get_joint_limits()
        order = [self._curobo_joint_names.index(n) for n in self.joint_names]
        pos = lim.position.detach().cpu().numpy()[:, order].astype(float)
        vel = lim.velocity.detach().cpu().numpy().astype(float)
        vmax = (np.abs(vel).max(axis=0) if vel.ndim == 2 else np.abs(vel))[order]
        return {"position": pos, "velocity": vmax}

    def check_state_valid(self, q):
        """(feasible, detail) for one controller-order joint vector against
        joint limits, self-collision, and the CURRENT collision world."""
        state = self._curobo_state(q)
        try:
            metrics = self._motion_gen.check_constraints(state)
            ok = bool(metrics.feasible.all().item())
            return ok, "" if ok else "constraint violation at state"
        except Exception as exc:
            return False, "check_constraints failed: %s" % exc

    # ---------------------------------------------------------------- plumbing
    def _tensor(self, data):
        return self._torch.tensor(
            data, device=self._tensor_args.device, dtype=self._tensor_args.dtype
        )

    def _to_curobo_order(self, q):
        """Controller-order joint vector -> cuRobo cspace order (by name —
        cuRobo consumes states in ITS order and never reorders)."""
        by_name = dict(zip(self.joint_names, [float(v) for v in q]))
        return [by_name[n] for n in self._curobo_joint_names]

    def _curobo_state(self, q):
        """Controller-order joint vector -> single-state CuJointState."""
        from curobo.types.robot import JointState as CuJointState

        return CuJointState.from_position(
            self._tensor([self._to_curobo_order(q)]),
            joint_names=list(self._curobo_joint_names),
        )

    def _nearest_branch(self, q_goal, q_ref):
        """Shift continuous joints' goals by 2*pi onto the branch nearest
        the start.

        The kortex driver reports continuous joints wrapped to (-pi, pi];
        at home, joint_3 sits exactly on that boundary, so the same
        physical angle can read +pi or -pi between boots (observed live).
        A goal authored as +pi with the arm reading -pi would otherwise
        plan a full winding revolution. Only joints whose model range
        spans more than 2*pi are shifted, and never outside the limits.
        """
        lim = self.joint_limits()["position"]
        out = []
        for j, (g, r) in enumerate(zip(q_goal, q_ref)):
            lo, hi = float(lim[0][j]), float(lim[1][j])
            if hi - lo > 2 * math.pi + 1e-6:
                cand = g + 2 * math.pi * round((r - g) / (2 * math.pi))
                if lo <= cand <= hi and abs(cand - g) > 1e-9:
                    log.info(
                        "goal %s shifted %+.3f rad to the 2*pi branch "
                        "nearest the start",
                        self.joint_names[j],
                        cand - g,
                    )
                    g = cand
            out.append(float(g))
        return out

    def _clamp_to_limits(self, q, what):
        """Clamp a controller-order joint vector into the model's position
        limits, or explain why that's not okay.

        Real arms park slightly OUTSIDE conservative model limits — the
        Gen3's factory park pose puts joint_4 ~0.8 deg past cuRobo's URDF
        bound, and cuRobo hard-refuses such a start. Violations up to
        `limit_clamp_rad` are clamped inward (the executor's start-state
        tolerance absorbs the difference); anything larger is a genuine
        configuration problem and is refused with the joint named.

        Returns (clamped_q or None, error or None). q may be None (= home).
        """
        if q is None:
            return None, None
        lim = self.joint_limits()["position"]
        arr = np.asarray([float(v) for v in q], dtype=float)
        clamped = np.clip(arr, lim[0], lim[1])
        deltas = np.abs(clamped - arr)
        j = int(np.argmax(deltas))
        worst = float(deltas[j])
        if worst > self.limit_clamp_rad:
            return None, (
                "%s: %s is %.3f rad outside the model position limits "
                "[%.3f, %.3f] (max clamp %.3f) — jog the arm inside its "
                "planning limits or fix the request"
                % (
                    what,
                    self.joint_names[j],
                    worst,
                    lim[0][j],
                    lim[1][j],
                    self.limit_clamp_rad,
                )
            )
        if worst > 1e-9:
            log.info(
                "%s: %s clamped %.4f rad inward to the model position limit",
                what,
                self.joint_names[j],
                worst,
            )
            return clamped.tolist(), None
        return arr.tolist(), None

    def _start_state(self, q=None):
        q = self.home_pose if q is None else [float(v) for v in q]
        if len(q) != len(self.joint_names):
            raise ValueError(
                "start has %d values for %d joints" % (len(q), len(self.joint_names))
            )
        return self._curobo_state(q)

    def _plan_config(self):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        kw = {}
        if not self.enable_graph:
            # cuRobo silently ENABLES the graph planner after 3 failed
            # attempts unless this is None — and the graph planner is the
            # exact thing the Jetson wheel cannot run. Never let it engage.
            kw["enable_graph_attempt"] = None
        return MotionGenPlanConfig(
            max_attempts=self.max_attempts,
            enable_graph=self.enable_graph,
            enable_finetune_trajopt=self.enable_finetune,
            finetune_attempts=self.finetune_attempts,
            **kw,
        )

    def _finish(self, result, t0, q_goal=None):
        """cuRobo MotionGenResult -> validated PlanResult."""
        if result is None or not bool(result.success.item()):
            status = (
                "no result"
                if result is None
                else str(getattr(result, "status", None) or "unknown")
            )
            return PlanResult.failure(
                status, self._explain(status), timing=time.monotonic() - t0
            )

        # ONLY the trimmed interpolated plan: result buffers are padded and
        # the untrimmed tail can be stale garbage from a previous plan.
        plan = result.get_interpolated_plan()
        ordered = plan.get_ordered_joint_state(self.joint_names)
        pos = ordered.position.detach().cpu().numpy().astype(float).copy()
        vel = (
            ordered.velocity.detach().cpu().numpy().astype(float).copy()
            if ordered.velocity is not None
            else None
        )
        acc = (
            ordered.acceleration.detach().cpu().numpy().astype(float).copy()
            if ordered.acceleration is not None
            else None
        )
        traj = Trajectory(
            joint_names=list(self.joint_names),
            positions=pos,
            velocities=vel,
            accelerations=acc,
            dt=float(result.interpolation_dt),
        )

        lim = self.joint_limits()
        problems = validate_trajectory(traj, lim["position"], lim["velocity"])
        problems += self._recheck_constraints(plan)
        if problems:
            return PlanResult.failure(
                "LIBRARY_VALIDATION_FAILED",
                "plan rejected by post-validation: " + "; ".join(problems),
                timing=time.monotonic() - t0,
            )

        res = PlanResult(
            success=True,
            joint_traj=traj,
            timing=time.monotonic() - t0,
            error=None,
            status="OK",
            validated=True,
            final_joints=pos[-1].copy(),
        )
        if q_goal is not None:
            res.goal_mismatch_rad = float(
                np.abs(pos[-1] - np.asarray(q_goal, dtype=float)).max()
            )
        return res

    def _recheck_constraints(self, plan_curobo_order):
        """Independent belt-and-braces: re-check every interpolated state
        against limits/self-collision/world via cuRobo's constraint checker.
        A checker failure is reported as a problem (fail closed) — this
        gate exists for the real arm."""
        try:
            metrics = self._motion_gen.check_constraints(plan_curobo_order)
            if bool(metrics.feasible.all().item()):
                return []
            feas = metrics.feasible.detach().cpu().numpy().reshape(-1)
            bad = int((~feas.astype(bool)).sum())
            return [
                "constraint re-check flagged %d/%d states infeasible" % (bad, feas.size)
            ]
        except Exception as exc:
            return ["constraint re-check unavailable: %s" % exc]

    @staticmethod
    def _explain(status):
        s = str(status).upper().replace(" ", "_")
        if "IK" in s:
            return (
                "no collision-free joint solution AT the goal — move it "
                "away from obstacles or relax the orientation. The goal "
                "is the tool_frame (fingertip midpoint); the flange sits "
                "12 cm behind it along the tool axis."
            )
        if "INVALID_START" in s:
            return (
                "the start state collides with the world (or exceeds "
                "limits) — check the world config against where the arm "
                "actually is."
            )
        if "FINETUNE" in s:
            return (
                "a collision-free path WAS found but retiming failed "
                "(goal near a kinematic limit) — raise finetune_attempts "
                "or move the goal slightly."
            )
        if "TRAJOPT" in s:
            return (
                "the goal is reachable but no collision-free PATH was "
                "found — clear the approach corridor or plan via an "
                "intermediate goal."
            )
        return "planning failed (%s)" % status
