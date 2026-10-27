"""CuRoboPlanner: the pure-Python planning core (no ROS imports).

    from rammp_curobo import CuRoboPlanner

    planner = CuRoboPlanner.from_config('gen3.yaml', urdf_path='robot_description.urdf')
    res = planner.plan_to_pose([0.45, 0.0, 0.35], [1, 0, 0, 0], q_now, quat_order='wxyz')
    res = planner.plan_to_joints(q_goal, q_now)
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
from rammp_curobo.config import PLANNER_DEFAULTS, load_planner_config, resolve_config
from rammp_curobo import constraints
from rammp_curobo.constraints import PoseConstraint, ViaPoint
from rammp_curobo.resample import resample_quintic
from rammp_curobo.robot_config import load_robot_config
from rammp_curobo.scene import Scene, load_layers, load_scene, scene_from_obstacles
from rammp_curobo.types import PlanResult, Trajectory
from rammp_curobo.urdf_model import ROOT_LINK
from rammp_curobo.validate import validate_trajectory
from rammp_curobo.world import (
    compose_cuboids,
    point_box_distance,
    sphere_box_penetration,
    summarize,
    to_world_config,
)

log = logging.getLogger("rammp_curobo")


def _boxes(hits):
    """_colliding_boxes' hits as text: "person/lap (154 mm), person/legs_0
    (12 mm)"."""
    return ", ".join("%s (%.0f mm)" % (key, depth * 1e3) for key, _, depth in hits)


class CuRoboPlanner:
    """One cuRobo MotionGen instance wrapped in a config-driven API.

    Construction is HEAVY (CUDA init + kernel warmup, tens of seconds on
    the Orin the first time) — build once, keep it alive.
    """

    def __init__(self, config, config_dir=None, *, urdf_path=None):
        self._cfg = config
        self._config_dir = config_dir
        p = config["planner"]
        self.joint_names = list(config["joint_names"])
        self.interpolation_dt = float(p["interpolation_dt"])
        self.max_attempts = int(p["max_attempts"])
        self.finetune_attempts = int(p["finetune_attempts"])
        self.enable_finetune = bool(p["enable_finetune"])
        self.enable_graph = bool(p["enable_graph"])
        self.collision_cache_obb = int(p["collision_cache_obb"])
        self.collision_cache_mesh = int(p["collision_cache_mesh"])
        self.collision_activation_distance = float(p["collision_activation_distance"])
        self.world_padding = float(p["world_padding"])
        # One tolerance, two jobs: it gates the start-vs-goal check AND the
        # verification of the returned trajectory, because the start is part
        # of "held throughout".
        self.constraint_tolerance_rad = math.radians(
            float(p["constraint_tolerance_deg"])
        )
        self.no_pad_names = frozenset(p["no_pad_names"] or [])
        self.joint_space_method = str(p["joint_space_method"])
        self.limit_clamp_rad = float(p["limit_clamp_rad"])
        self.joint_limit_margin_rad = float(p["joint_limit_margin_rad"])
        if not self.joint_limit_margin_rad >= 0.0:
            raise ValueError(
                "joint_limit_margin_rad must be >= 0, got %r"
                % p["joint_limit_margin_rad"]
            )
        self.tool_spin_deg = float(config["tool"]["spin_deg"])
        self.tool_tip_offset = float(config["tool"]["tip_offset_m"])

        robot_path = resolve_config(config["robot"], relative_to=config_dir)
        world_path = resolve_config(config["world"], relative_to=config_dir)
        # The arm's model: the URDF from /arm/robot_description, as a file.
        self._robot_cfg = load_robot_config(robot_path, urdf_path=urdf_path)
        # The frame every Cartesian input and trajectory is in. cuRobo calls the
        # root of the chain it plans in "base_link"; for us that is arm_mount,
        # the gravity-aligned root, NOT the arm's own arm_base_link. A robot
        # YAML naming another root would plan (and stamp) in that frame, which
        # kinova-gen3-ros2 refuses only at execution: refuse it here instead.
        self.base_frame = str(self._robot_cfg["kinematics"]["base_link"])
        if self.base_frame != ROOT_LINK:
            raise ValueError(
                "robot config %s: kinematics.base_link is %r, expected %r (the "
                "URDF's root)" % (robot_path, self.base_frame, ROOT_LINK)
            )
        # The link plan_to_pose goals and fk() poses are for (arm_tool_frame).
        self.ee_link = str(self._robot_cfg["kinematics"]["ee_link"])
        # The IK seed / null-space reference, in controller order. A planner
        # fact (it fixes the elbow family), NOT a statement about where the
        # arm belongs — this planner has no such opinion (issue #6).
        cspace = self._robot_cfg["kinematics"]["cspace"]
        by_name = dict(zip(cspace["joint_names"], cspace["retract_config"]))
        self.retract_pose = [float(by_name[n]) for n in self.joint_names]
        # cuRobo plans (IK, trajopt, the constraint re-check) inside limits
        # narrowed by the margin, so no plan reaches a URDF limit.
        cspace["position_limit_clip"] = self.joint_limit_margin_rad
        # Composed into every world, set_world pushes included, as loaded
        # here: a file edited later changes nothing until a restart. A config
        # dict built for v1.0.0, which had no such key, lists none.
        self._layers = load_layers(
            config.get("layers", []),
            relative_to=config_dir,
            listed_in="planner config in %s" % config_dir,
        )
        for layer in self._layers:
            log.info(
                "Config layer %s: %d boxes, %s, %s",
                layer.name,
                len(layer.obstacles),
                "excludable" if layer.excludable else "not excludable",
                layer.path,
            )
        self._scene = load_scene(world_path, frame=self.base_frame)
        # Pure Python, so a bad layer, an overflow or a duplicate box fails
        # here, before any GPU work.
        self._compose(self._scene, allow_empty=True)
        self._init_curobo(warmup=bool(p["warmup"]))

    @classmethod
    def from_config(cls, name_or_path, *, urdf_path=None):
        """Build from a planner YAML (packaged name like 'gen3.yaml' or a
        filesystem path). See configs/gen3.yaml for the schema. urdf_path is
        the arm's model as a file (urdf_model.write_urdf)."""
        cfg, cfg_dir = load_planner_config(name_or_path)
        return cls(cfg, config_dir=cfg_dir, urdf_path=urdf_path)

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
        cuboids, sources = self._compose(self._scene, allow_empty=True)
        # cuRobo warms up by planning from the retract, and a start in
        # collision returns INVALID_START before any solver runs: where the
        # config's layers collide with the retract (on the chair it is 154 mm
        # inside the person), warm-up would compile nothing. So MotionGen
        # starts in the scene's own world, its layers included, and the
        # config's layers load after warm-up. The CUDA graphs warm-up records
        # stay valid: the box cache is allocated at full size here,
        # update_world writes into it in place, and the kernels read the box
        # count from the device.
        warm_first = bool(self._layers) and warmup
        warm = (
            self._compose(self._scene, layers=(), allow_empty=True)[0]
            if warm_first
            else cuboids
        )
        log.info("Loading cuRobo MotionGen (%s)...", self._cfg["robot"])
        cfg = MotionGenConfig.load_from_robot_config(
            self._robot_cfg,
            to_world_config(warm, allow_empty=True),
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
        self._fixed_spheres = self._joint_independent_spheres()
        self._check_fixed_spheres(cuboids, sources)
        if warmup:
            if warm_first:
                log.info(
                    "warming up in the scene-only world (%d %s); config layers "
                    "load after warm-up",
                    len(warm),
                    "box" if len(warm) == 1 else "boxes",
                )
            else:
                log.info("Warming up cuRobo (first-run kernel compile)...")
            self._motion_gen.warmup(enable_graph=self.enable_graph)
            log.info("cuRobo warmup complete.")
        if warm_first:
            self._motion_gen.update_world(to_world_config(cuboids))
        self._world_cuboids, self._world_sources = cuboids, sources
        self._warm_boxes = len(warm)
        log.info("Collision world: %s", self.world_summary())
        ok, detail = self.check_state_valid(self.retract_pose)
        if not ok:
            log.warning(
                "the IK seed (retract_config) is not a valid state in this "
                "world: %s — plans still work from valid starts; a chair seed "
                "is #20",
                detail,
            )

    def _compose(self, scene, ignore=frozenset(), layers=None, allow_empty=False):
        """compose_cuboids with this planner's padding, exemptions and cache,
        and its config layers unless `layers` says otherwise. `allow_empty`
        is for construction only (world_empty.yaml): never on a push."""
        return compose_cuboids(
            scene,
            padding=self.world_padding,
            ignore=ignore,
            no_pad_names=self.no_pad_names,
            cache_obb=self.collision_cache_obb,
            layers=self._layers if layers is None else layers,
            allow_empty=allow_empty,
        )

    def _joint_independent_spheres(self):
        """The arm's collision spheres that no joint moves, (n, 4): x, y, z
        and the radius cuRobo checks. They are the ones FK puts in the same
        place at the retract and at two seeded configurations within the
        limits: arm_base_link's, and arm_shoulder_link's on the arm_joint_1 axis."""
        lim = self.joint_limits()["position"]
        rng = np.random.default_rng(0)
        states = [self.retract_pose] + [rng.uniform(lim[0], lim[1]) for _ in range(2)]
        spheres = np.stack([self._spheres(q) for q in states])
        moved = np.linalg.norm(spheres[..., :3] - spheres[0, :, :3], axis=2).max(axis=0)
        return spheres[0, (moved <= 1e-6) & (spheres[0, :, 3] > 0)]

    def _check_fixed_spheres(self, cuboids, sources):
        """Refuse a world in which a padded layer box reaches into a sphere
        that no joint moves: every state, the start of every plan, would be
        in collision. Scene boxes are cuRobo's to judge, as in v1.0.0."""
        layered = {key: box for key, box in cuboids.items() if sources[key] != "scene"}
        hits = sphere_box_penetration(self._fixed_spheres, layered)
        if hits:
            key = max(hits, key=hits.get)
            # The padding the intruding layers were generated for, as they
            # state it (meta.padding_assumed_m), falling back to the default
            # for a layer that states none. Read from the layers rather than
            # assumed, so this and compose_cuboids' check cannot disagree.
            # The padding is to blame when it is above that and every box that
            # intrudes clears the spheres at it; otherwise a box reaches them
            # by itself, and restoring the padding would not help.
            claims = {
                layer.padding_assumed
                for layer in self._layers + list(self._scene.layers)
                if layer.name in {sources[k] for k in hits}
                and layer.padding_assumed is not None
            }
            assumed = (
                min(claims) if claims else PLANNER_DEFAULTS["planner"]["world_padding"]
            )
            grown = 2.0 * (self.world_padding - assumed)
            shrunk = {
                k: dict(layered[k], dims=[d - grown for d in layered[k]["dims"]])
                for k in hits
            }
            if grown > 0 and not sphere_box_penetration(self._fixed_spheres, shrunk):
                fix = (
                    "the layers were generated for %g m — regenerate them or "
                    "restore world_padding" % assumed
                )
            else:
                fix = "move or shrink that box, or regenerate its layer"
            raise ValueError(
                "layer box %s intrudes %.1f mm into the arm's fixed base/shoulder "
                "spheres (world_padding %.3f m): every start state would be "
                "infeasible; %s" % (key, hits[key] * 1e3, self.world_padding, fix)
            )

    # ---------------------------------------------------------------- planning
    def plan_to_pose(
        self,
        position,
        quaternion,
        start,
        quat_order="xyzw",
        apply_tool_correction=None,
        constraint=None,
        via=None,
    ):
        """Plan a collision-free trajectory to an end-effector pose.

        position: [x, y, z] metres in arm_mount (base_frame).
        quaternion: orientation of cuRobo's ee_link (arm_tool_frame — for the
            Gen3 config that is 0.120 m beyond the wrist flange, roughly the
            fingertip midpoint). quat_order says how it is packed: 'xyzw'
            (ROS, the default) or 'wxyz' (cuRobo).
        start: joint positions in controller order. REQUIRED — the planner
            does not know where the arm is and will not guess.
        apply_tool_correction: apply the configured tool spin/tip-offset
            calibration (authored-fingertip goals). None = apply whenever
            the config carries non-zero values.
        constraint: PoseConstraint — how much of the tool's orientation to
            hold for the whole trajectory (keeping a carried object level).
            None = free.
        via: ViaPoint — one blended intermediate target offset from the
            goal along a tool axis. The path passes NEAR it without
            stopping; it is a cost, not a waypoint to hit. None = none.
            An active hold and an active via together are refused as
            BAD_CONSTRAINT: cuRobo's approach metric zeroes the per-step
            pose cost before tstep_fraction, so the hold would not be
            applied for most of the path.
        """
        t0 = time.monotonic()
        try:
            self._pose_cost_kwargs(constraint, via)
        except ValueError as exc:
            return PlanResult.failure("BAD_CONSTRAINT", str(exc))
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
            return PlanResult.failure(*err)

        # Refuse a self-contradictory request BEFORE planning. cuRobo does
        # not check a base-frame hold's start itself, so without this a
        # contradictory goal would only surface as a hold violation after a
        # wasted plan.
        if constraint is not None and constraint.is_active():
            ok, why = self.constraint_satisfied_at_start(
                start, xyz, wxyz, constraint, quat_order="wxyz"
            )
            if not ok:
                return PlanResult.failure(
                    "CONSTRAINT_UNSATISFIABLE", why, timing=time.monotonic() - t0
                )
            if why:
                log.warning("%s", why)

        start_state = self._start_state(start)
        goal = Pose(position=self._tensor([xyz]), quaternion=self._tensor([wxyz]))
        try:
            result = self._motion_gen.plan_single(
                start_state, goal, self._plan_config(constraint, via)
            )
        except Exception as exc:
            # cuRobo resets the pose cost metric only on a normal return, so
            # a raise would leave this plan's hold or via in the rollouts and
            # silently constrain the next, unconstrained plan.
            self._reset_pose_cost_metric()
            return PlanResult.failure(
                "EXCEPTION",
                "cuRobo plan_single raised: %s" % exc,
                timing=time.monotonic() - t0,
            )
        out = self._with_failure_note(
            self._finish(result, t0), result, start, goal_point=xyz
        )
        # IK can land a continuous joint on a 2*pi branch away from the
        # start, and the plan then goes the long way round -- a full
        # revolution for a goal at the start's own angle (#14). cuRobo
        # already seeds IK with the start, but +/-pi cost the same and
        # plan_single takes no seed of its own. So: if the plan ENDS off the
        # start's branch, replan in joint space to the end moved onto it.
        # A plan that ends on the branch can only have detoured, not wound.
        # 'js' only: 'auto' could fall back into this path. A success without
        # final_joints (a subclass's _finish may return one) has no branch.
        if out.success and out.final_joints is not None:
            final = [float(v) for v in out.final_joints]
            if self._nearest_branch(final, start) != final:
                held = (constraint is not None and constraint.is_active()) or (
                    via is not None and via.is_active()
                )
                if held:
                    # a joint-space replan would drop the hold or via
                    log.warning(
                        "plan ends on a 2*pi branch away from the start "
                        "(it goes the long way round); not replanned "
                        "under a hold or via"
                    )
                else:
                    res = self.plan_to_joints(final, start, method="js")
                    if res.success:
                        res.goal_mismatch_rad = None
                        res.timing = time.monotonic() - t0
                        return res
                    out.error = (
                        "plan ends on a 2*pi branch away from the start (it "
                        "goes the long way round) and the replan onto the "
                        "start's branch failed (%s); returning it as planned"
                        % res.status
                    )
                    log.warning("%s", out.error)
        # A hold is a COST, so a returned plan is a proposal, not a promise.
        # Measure it and refuse rather than hand over a trajectory that quietly
        # tipped the thing we were asked to keep upright.
        if out.success and constraint is not None and constraint.is_active():
            bad = self._verify_hold(out.joint_traj, wxyz, constraint)
            if bad is not None:
                return PlanResult.failure(
                    "CONSTRAINT_VIOLATED", bad, timing=time.monotonic() - t0
                )
        return out

    def _reset_pose_cost_metric(self):
        from curobo.rollout.cost.pose_cost import PoseCostMetric

        self._motion_gen.update_pose_cost_metric(PoseCostMetric.reset_metric())

    def _verify_hold(self, traj, goal_wxyz, constraint):
        """None if the trajectory honoured the hold, else why it did not.

        Measures the held deviation from the goal orientation with the SAME
        metric the start check uses, so one tolerance means one thing.

        Then checks that the plan ENDS at the goal's full orientation, yaw
        included. LEVEL frees yaw along the way, never at the goal: cuRobo's
        terminal cost keeps full weights, but its own success test allows
        sin(err/2) < 0.05 (about 5.7 deg), so a plan it calls successful can
        still miss the requested yaw by more than this tolerance.

        ONE kinematics call for the whole trajectory, then numpy: the joint
        reorder is a precomputed index permutation and the quaternion maths is
        vectorised. A per-waypoint Python loop would put ~100 GPU round trips
        on the planning path, which is not an acceptable cost for a check that
        runs on every constrained plan.
        """
        if traj is None or traj.n_points == 0:
            return None
        # controller order -> cuRobo cspace order, as an index permutation
        # rather than _to_curobo_order's per-row dict rebuild.
        idx = [self.joint_names.index(n) for n in self._curobo_joint_names]
        # ascontiguousarray is REQUIRED, not defensive. cuRobo's fused
        # kinematics kernel asserts joint_vec.is_contiguous(), and the tensor
        # built from this column selection arrives non-contiguous, so the
        # batched call dies with an INTERNAL ASSERT FAILED from
        # kinematics_fused_kernel.cu. The single-row fk() path never hit it.
        # Measured in the container: as-is contiguous=False -> assert;
        # ascontiguousarray -> (N, 4) quaternions back.
        rows = np.ascontiguousarray(np.asarray(traj.positions, dtype=float)[:, idx])
        state = self._motion_gen.kinematics.get_state(self._tensor(rows))
        wxyz = state.ee_quaternion.detach().cpu().numpy().astype(float)
        dev = self._hold_deviation_batch(
            wxyz, geometry.wxyz_to_xyzw(goal_wxyz), constraint
        )
        worst_k = int(np.argmax(dev))
        worst = float(dev[worst_k])
        tol = self.constraint_tolerance_rad
        if worst <= tol:
            end = float(
                self._hold_deviation_batch(
                    wxyz[-1:],
                    geometry.wxyz_to_xyzw(goal_wxyz),
                    PoseConstraint(hold=constraints.HOLD_FIXED),
                )[0]
            )
            if end <= tol:
                return None
            return (
                "the plan ends %.2f deg from the goal orientation (limit "
                "%.2f). The '%s' hold frees nothing at the goal: every "
                "component of the goal orientation, yaw included, must be "
                "reached." % (math.degrees(end), math.degrees(tol), constraint.name())
            )
        what = (
            "tilt (roll/pitch)"
            if constraint.hold == constraints.HOLD_LEVEL
            else "orientation"
        )
        return (
            "the plan does not honour the '%s' hold: worst %s deviation "
            "%.2f deg at waypoint %d of %d (limit %.2f). cuRobo's hold is a "
            "cost, not a clamp, so a plan can come back having bent it; this "
            "refuses rather than executing it."
            % (
                constraint.name(),
                what,
                math.degrees(worst),
                worst_k,
                traj.n_points,
                math.degrees(tol),
            )
        )

    def plan_to_joints(self, q_goal, start, method=None):
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
        q_goal = self._nearest_branch(q_goal, [float(v) for v in start])
        q_goal, err = self._clamp_to_limits(q_goal, "goal")
        if err is not None:
            return PlanResult.failure(*err)
        start, err = self._clamp_to_limits(start, "start")
        if err is not None:
            return PlanResult.failure(*err)
        method = method or self.joint_space_method

        if method == "auto":
            res = self.plan_to_joints(q_goal, start=start, method="js")
            if res.success:
                return res
            # the error too: it names the boxes the joint goal reaches into,
            # which the fallback's result, about a pose, cannot
            log.warning(
                "plan_single_js failed (%s: %s) — falling back to FK-pose planning",
                res.status,
                res.error,
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
            res = self._finish(result, t0, q_goal=q_goal)
            return self._with_failure_note(res, result, start, q_goal=q_goal)
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
        """Replace the scene of the collision world. The layers the planner
        config lists stay: they are composed into every world.

        world: a scene YAML path/name, a Scene, or a list of obstacle dicts
        ({'name', 'position', and 'dims' | 'type'+'radius'(+'height')}).
        ignore: names to leave out of this world only: props (the object
        being reached for) and excludable layers ({'person'}). A bare
        string is one name.
        Refused before cuRobo sees it, so the world installed before stays:
        whatever compose_cuboids refuses (empty worlds and cache overflow
        among it — both silent or cryptic failures inside cuRobo v0.7.8),
        and a layer box that reaches into the arm's fixed spheres.
        """
        if isinstance(world, Scene):
            scene = world
            if scene.base_frame != self.base_frame:
                raise ValueError(
                    "scene base_frame is %r, expected %r (the planning frame)"
                    % (scene.base_frame, self.base_frame)
                )
        elif isinstance(world, (list, tuple)):
            scene = scene_from_obstacles(world, base_frame=self.base_frame)
        else:
            scene = load_scene(
                resolve_config(world, relative_to=self._config_dir),
                frame=self.base_frame,
            )
        cuboids, sources = self._compose(scene, ignore=ignore)
        self._check_fixed_spheres(cuboids, sources)
        self._motion_gen.update_world(to_world_config(cuboids))
        # only now that cuRobo holds it: a refused world leaves the planner
        # describing the world cuRobo kept
        self._scene = scene
        self._world_cuboids, self._world_sources = cuboids, sources
        names = [ignore] if isinstance(ignore, str) else sorted(ignore)
        log.info(
            "Collision world: %s%s",
            self.world_summary(),
            " (ignoring: %s)" % ", ".join(names) if names else "",
        )

    # ----------------------------------------------------------------- queries
    @property
    def scene(self):
        return self._scene

    @property
    def world_name(self):
        """The world reference the planner was configured with (a name or
        path string; informational — update_world does not change it)."""
        return str(self._cfg.get("world", ""))

    @property
    def world_boxes(self):
        """The installed collision world, box by box: [{'name', 'layer',
        'dims', 'pose'}]. 'name' is the box's key in the world ('table',
        'obj_mug', 'person/lap'), 'layer' is 'scene' or the layer the box
        comes from, 'dims' are padded, and 'pose' is [x, y, z, qw, qx, qy,
        qz] in the base frame: the boxes cuRobo was given."""
        return [
            {
                "name": key,
                "layer": self._world_sources[key],
                "dims": list(box["dims"]),
                "pose": list(box["pose"]),
            }
            for key, box in self._world_cuboids.items()
        ]

    def world_summary(self):
        """Where the installed world's boxes come from, on one line, e.g.
        "25 boxes: scene 0 | config layers: platform 16, person 8 | scene
        layers: floor 1"."""
        return summarize(self._world_sources, [layer.name for layer in self._layers])

    def fk(self, q, quat_order="xyzw"):
        """Forward kinematics of a controller-order joint vector ->
        (position [x,y,z], quaternion in `quat_order`) of the ee_link."""
        state = self._motion_gen.kinematics.get_state(
            self._tensor([self._to_curobo_order(q)])
        )
        pos = [float(v) for v in state.ee_position[0].tolist()]
        wxyz = [float(v) for v in state.ee_quaternion[0].tolist()]
        return pos, (wxyz if quat_order == "wxyz" else geometry.wxyz_to_xyzw(wxyz))

    # The old HOLD_TOL_RAD / HOLD_TOL_M constants are gone: the rotational
    # one became the configurable constraint_tolerance_deg, and the linear one
    # has nothing left to gate now that position is never held.

    def constraint_satisfied_at_start(
        self, start, position, quaternion, constraint, quat_order="xyzw"
    ):
        """Can this constrained plan even be attempted from `start`?

        A hold keeps the held components AT THE GOAL'S VALUE, so a start
        that disagrees with the goal on them asks for two orientations at
        once; 'keep the tool level' is two-phase: level it with an
        unconstrained move, then transport under the hold. cuRobo does not
        check this itself for a base-frame hold (its start check in
        update_pose_cost_metric only runs the orientation test when
        projecting to the goal frame), so this is the only gate.

        `quat_order` describes only the caller's own `quaternion` argument
        (the goal) — the start pose is always read from `fk` in xyzw, so a
        `quat_order="wxyz"` caller never mixes conventions between the two
        operands fed to `rotvec_between`.


        Returns (ok, reason). reason is None when clean, a 'marginal: ...'
        string when inside tolerance but close to it, and an explanation
        when not. There is no longer an out-of-scope case: holds are always
        base-frame, and a via is not judged here at all.
        """
        constraint = constraint or PoseConstraint()
        if not constraint.is_active():
            return True, None
        # No frame branch any more: holds are always base-frame, so there is no
        # "cannot pre-check this mode" case left to decline. And no via branch:
        # a via's hold engages at tstep_fraction, NOT at the start, so folding
        # it in here refused plans that cuRobo accepts happily — a false
        # refusal this check used to emit through the hold-check service.
        try:
            constraint.validate()
        except ValueError as exc:
            return False, str(exc)

        pos = [float(v) for v in position]
        if not all(math.isfinite(v) for v in pos):
            return False, "goal position is non-finite"

        if quat_order == "wxyz":
            goal_quat = geometry.wxyz_to_xyzw(quaternion)
        else:
            goal_quat = [float(v) for v in quaternion]
        if not all(math.isfinite(v) for v in goal_quat):
            return False, "goal quaternion is non-finite"
        quat_norm = math.sqrt(sum(v * v for v in goal_quat))
        if quat_norm < 1e-6:
            # mirrors plan_to_pose's own BAD_GOAL guard on a zero-length
            # quaternion — this pre-check must refuse the same malformed
            # goal, not report a confident "satisfied".
            return False, "goal quaternion is zero-length"

        _cur_pos, cur_quat = self.fk(start, quat_order="xyzw")
        # Via the CLASS, not self: it is a staticmethod, and this method is
        # also called unbound with a stub as `self` in the offline tests.
        err, what = CuRoboPlanner._hold_deviation(cur_quat, goal_quat, constraint)
        tol = self.constraint_tolerance_rad
        ratio = err / tol if tol > 0.0 else 0.0

        if ratio > 1.0:
            return False, (
                "the goal's %s differs from the start by %.2f deg (limit "
                "%.2f): a hold keeps the orientation AT THE GOAL'S VALUE, so "
                "a goal that disagrees with the start is asking for two "
                "different orientations at once. Either pass the start's "
                "orientation as the goal's, or move there with an "
                "unconstrained plan first."
                % (what, math.degrees(err), math.degrees(tol))
            )
        if ratio > 0.8:
            return True, (
                "marginal: the goal's %s is %.2f deg from the start (limit "
                "%.2f, %.0f%% of it)"
                % (what, math.degrees(err), math.degrees(tol), 100.0 * ratio)
            )
        return True, None

    @staticmethod
    def _hold_deviation_batch(a_wxyz, b_xyzw, constraint):
        """Vectorised _hold_deviation: (N,) radians for (N,4) wxyz against one
        xyzw goal.

        Same closed forms as the scalar version, so the two cannot drift apart
        in convention; see _hold_deviation for the derivation. Both measures
        are invariant under conjugating e, which is why `a` may be the
        waypoints here and the current pose in the scalar version.
        """
        ax, ay, az, aw = a_wxyz[:, 1], a_wxyz[:, 2], a_wxyz[:, 3], a_wxyz[:, 0]
        bx, by, bz, bw = [float(v) for v in b_xyzw]
        ew = bw * aw + bx * ax + by * ay + bz * az
        ez = bz * aw - bw * az - bx * ay + by * ax
        if constraint.hold == constraints.HOLD_LEVEL:
            return 2.0 * np.arccos(np.minimum(1.0, np.hypot(ew, ez)))
        return 2.0 * np.arccos(np.minimum(1.0, np.abs(ew)))

    @staticmethod
    def _hold_deviation(quat_a, quat_b, constraint):
        """(radians, what) — how far apart two orientations are, counting only
        what the hold actually holds.

        LEVEL zeroes rz in hold_vec_weight, so cuRobo is indifferent to rotation
        about base Z: it treats the whole ORBIT {Rz(psi) * goal} as satisfying
        the hold. The honest deviation is therefore the geodesic distance from
        the orientation to that orbit,

            d = min over psi of angle(q, Rz(psi) * goal)
              = 2 * acos( sqrt(e_w^2 + e_z^2) )     for e = q (x) conj(goal)

        which has a closed form because maximising |w| of e (x) Rz(-psi) over
        the unit circle (c, s) gives sqrt(e_w^2 + e_z^2) directly.

        It replaces hypot(rx, ry), which was WRONG in exactly the case LEVEL is
        for. That over-reported the true tilt by (psi/2)/sin(psi/2) where psi is
        the yaw difference -- up to 1.571x at psi = 180 deg. Since LEVEL frees
        yaw without bound, large yaw is the normal case, not an edge: a plan
        honouring the hold to a genuine 1.3 deg while yawing 180 deg measured
        2.04 deg and was refused as CONSTRAINT_VIOLATED. Verified against a
        full-circle brute-force minimisation over 200 random orientation pairs,
        agreeing to the brute force's own grid resolution.

        Note the earlier evidence for hypot could not have caught this: a pure
        yaw reads exactly 0 under BOTH formulas (it is the one case where hypot
        is exact), and batch-vs-scalar agreement proves implementation
        consistency, not that the quantity is the right one.

        FIXED holds all three, so it is the full angle, 2 * acos(|e_w|).

        The same measure serves the start check and the trajectory verification,
        so "within tolerance" means one thing.
        """
        ax, ay, az, aw = [float(v) for v in quat_a]
        bx, by, bz, bw = [float(v) for v in quat_b]
        # e = b (x) conj(a), components in the BASE frame -- same convention as
        # geometry.rotvec_between. Only e_w and e_z are needed; both measures
        # below are invariant under conjugating e, so the a/b order is free.
        ew = bw * aw + bx * ax + by * ay + bz * az
        ez = bz * aw - bw * az - bx * ay + by * ax
        if constraint.hold == constraints.HOLD_LEVEL:
            return 2.0 * math.acos(min(1.0, math.hypot(ew, ez))), "tilt"
        return 2.0 * math.acos(min(1.0, abs(ew))), "orientation"

    def joint_limits(self):
        """{'position': (2, dof) [lower; upper], 'velocity': (dof,),
        'acceleration': (dof,)} in CONTROLLER joint order (numpy, radians).
        Positions are the URDF's: cuRobo's own are narrowed by
        joint_limit_margin_rad, which is widened back here."""
        lim = self._motion_gen.kinematics.get_joint_limits()
        order = [self._curobo_joint_names.index(n) for n in self.joint_names]
        pos = lim.position.detach().cpu().numpy()[:, order].astype(float)
        pos += np.array([[-1.0], [1.0]]) * self.joint_limit_margin_rad
        vel = lim.velocity.detach().cpu().numpy().astype(float)
        vmax = (np.abs(vel).max(axis=0) if vel.ndim == 2 else np.abs(vel))[order]
        acc = np.abs(lim.acceleration.detach().cpu().numpy().astype(float))
        amax = (acc.max(axis=0) if acc.ndim == 2 else acc)[order]
        return {"position": pos, "velocity": vmax, "acceleration": amax}

    def check_state_valid(self, q):
        """(feasible, detail) for one controller-order joint vector against
        joint limits, self-collision, and the CURRENT collision world. The
        detail of an infeasible state names the world boxes it reaches
        into, deepest first."""
        state = self._curobo_state(q)
        try:
            metrics = self._motion_gen.check_constraints(state)
            ok = bool(metrics.feasible.all().item())
        except Exception as exc:
            return False, "check_constraints failed: %s" % exc
        if ok:
            return True, ""
        try:
            hits = self._colliding_boxes(q)
        except Exception as exc:
            # it only explains: the state is infeasible all the same
            return False, (
                "constraint violation at state (attribution unavailable: %s)" % exc
            )
        return False, "constraint violation at state" + (
            "; collides with " + _boxes(hits) if hits else ""
        )

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

    def _spheres(self, q):
        """cuRobo's collision spheres at one controller-order joint vector,
        (n, 4) in the base frame: x, y, z and the radius with the robot
        config's buffer. A sphere cuRobo leaves out has a radius <= 0."""
        state = self._motion_gen.kinematics.get_state(
            self._tensor([self._to_curobo_order(q)])
        )
        return state.link_spheres_tensor[0].detach().cpu().numpy().astype(float)

    def _colliding_boxes(self, q):
        """The boxes of the installed world that state `q` reaches into,
        deepest first: [(key, layer, depth_m)].

        The test cuRobo's feasibility check makes, on the CPU against the
        composed boxes. Switching boxes off in cuRobo to find the culprit
        would change the live world, and a state that hits two layers stays
        infeasible with either one off, so neither would be named.
        """
        spheres = self._spheres(q)
        hits = sphere_box_penetration(spheres[spheres[:, 3] > 0], self._world_cuboids)
        return sorted(
            ((key, self._world_sources[key], depth) for key, depth in hits.items()),
            key=lambda hit: -hit[2],
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

        cuRobo hard-refuses a start outside its limits, so small
        model-vs-arm disagreements, up to `limit_clamp_rad`, are clamped
        inward; anything larger is refused with the joint named. The Gen3's
        park pose (arm_joint_4 ~ -2.674) is 0.10 rad past the URDF's +-2.57
        and is refused (tracked in kinova-gen3-ros2#49).

        cuRobo plans inside joint_limit_margin_rad of every limit, so a
        joint within the margin is refused too, never moved: shifting a
        start would put the plan's first point off the arm's measured q.

        Returns (clamped_q or None, (status, error) or None). q may be None
        (nothing to clamp).
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
                "%s_OUTSIDE_LIMITS" % what.upper(),
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
                ),
            )
        room = np.minimum(clamped - lim[0], lim[1] - clamped)
        k = int(np.argmin(room))
        if room[k] < self.joint_limit_margin_rad:
            return None, (
                "%s_IN_LIMIT_MARGIN" % what.upper(),
                "%s: %s is %.3f rad from its position limit [%.3f, %.3f], "
                "inside joint_limit_margin_rad %.3f — jog it away from the "
                "limit first"
                % (
                    what,
                    self.joint_names[k],
                    max(float(room[k]), 0.0),
                    lim[0][k],
                    lim[1][k],
                    self.joint_limit_margin_rad,
                ),
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

    def _start_state(self, q):
        q = [float(v) for v in q]
        if len(q) != len(self.joint_names):
            raise ValueError(
                "start has %d values for %d joints" % (len(q), len(self.joint_names))
            )
        return self._curobo_state(q)

    @staticmethod
    def _effective_hold_vec_weight(constraint, via):
        """The 6-vector [rx, ry, rz, x, y, z] cuRobo will actually apply
        once `via` is folded in.

        An INACTIVE via point changes nothing: the effective hold is just
        `constraint.hold_vec_weight()`. An ACTIVE via point replaces it with
        cuRobo's grasp-approach metric, which holds every component except
        its own `linear_axis` -- but only from tstep_fraction on, because
        update_offset_waypoint zeroes the per-step weights before that.
        A hold combined with a via would therefore not be applied for most
        of the path, so the combination raises ValueError rather than
        quietly dropping the hold.
        """
        constraint = constraint or PoseConstraint()
        via = via or ViaPoint()
        if not via.is_active():
            return constraint.hold_vec_weight()
        if constraint.is_active():
            raise ValueError(
                "an orientation hold cannot be combined with a via point: "
                "cuRobo's approach metric zeroes the path cost before "
                "tstep_fraction=%.2f, so the '%s' hold would not be applied "
                "for that part of the path" % (via.tstep_fraction, constraint.name())
            )
        hold = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        hold[3 + via.linear_axis] = 0.0
        return hold

    @staticmethod
    def _pose_cost_kwargs(constraint, via):
        """Plain-value kwargs describing the PoseCostMetric to build, or
        None if unconstrained.

        Deliberately tensor-free so the mapping is testable without a GPU;
        _plan_config does the tensor / PoseCostMetric(-subclass) work.

        Verified against real cuRobo v0.7.8 by introspection on the Jetson:
        `PoseCostMetric.__init__` takes NO `linear_axis` kwarg, and
        `offset_position` is a 3-vector, not a scalar — passing them as
        plain values straight to `PoseCostMetric(...)` (an earlier version
        of this function did) raises `PoseCostMetric.__init__() got an
        unexpected keyword argument 'linear_axis'`. So this function hands
        back only PLAIN values; `_plan_config` builds the via-point case
        through cuRobo's own supported entry point for it,
        `PoseCostMetric.create_grasp_approach_metric(...)`, using
        `offset_position` / `linear_axis` / `offset_tstep_fraction` from
        here as that classmethod's own (scalar) arguments.
        """
        constraint = constraint or PoseConstraint()
        via = via or ViaPoint()
        constraint.validate()
        via.validate()
        if not constraint.is_active() and not via.is_active():
            return None

        kw = {
            "hold_partial_pose": True,
            "hold_vec_weight": CuRoboPlanner._effective_hold_vec_weight(
                constraint, via
            ),
            # The BASE frame for a hold: "level" means level with the world,
            # and only the base frame can express that. Not used on the via
            # path -- create_grasp_approach_metric ignores the argument and
            # projects to the goal frame (see ViaPoint).
            "project_to_goal_frame": False,
        }
        if via.is_active():
            kw["offset_position"] = float(via.offset_m)
            kw["linear_axis"] = int(via.linear_axis)
            kw["offset_tstep_fraction"] = float(via.tstep_fraction)
        return kw

    def _plan_config(self, constraint=None, via=None):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        kw = {}
        if not self.enable_graph:
            # cuRobo silently ENABLES the graph planner after 3 failed
            # attempts unless this is None — and the graph planner is the
            # exact thing the Jetson wheel cannot run. Never let it engage.
            kw["enable_graph_attempt"] = None

        metric_kw = self._pose_cost_kwargs(constraint, via)
        if metric_kw is not None:
            from curobo.rollout.cost.pose_cost import PoseCostMetric

            if (via or ViaPoint()).is_active():
                # The supported path for an approach — it is what cuRobo's
                # own tests use, and it builds the 3-vector offset_position
                # PoseCostMetric.__init__ actually requires internally.
                kw["pose_cost_metric"] = PoseCostMetric.create_grasp_approach_metric(
                    offset_position=metric_kw["offset_position"],
                    linear_axis=metric_kw["linear_axis"],
                    tstep_fraction=metric_kw["offset_tstep_fraction"],
                    tensor_args=self._tensor_args,
                )
            else:
                metric_kw = dict(metric_kw)
                metric_kw["hold_vec_weight"] = self._tensor(
                    metric_kw["hold_vec_weight"]
                )
                kw["pose_cost_metric"] = PoseCostMetric(**metric_kw)

        return MotionGenPlanConfig(
            max_attempts=self.max_attempts,
            enable_graph=self.enable_graph,
            enable_finetune_trajopt=self.enable_finetune,
            finetune_attempts=self.finetune_attempts,
            **kw,
        )

    def _finish(self, result, t0, q_goal=None):
        """cuRobo MotionGenResult -> validated PlanResult.

        Subclasses override it with this signature, v1.0.0's, and call
        super()._finish(result, t0, q_goal=q_goal) (RAMMP-generalized's
        StationaryPlanner), so it takes no new argument: what a failure's
        error says about the world is added after it, by
        _with_failure_note."""
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
        linear = result.get_interpolated_plan()
        plan, consistent = self._resample(linear, result)
        recheck = self._recheck_constraints(plan)
        if recheck and consistent:
            # The spline can leave the hull of cuRobo's knots: a goal its IK
            # put ON a (margin-narrowed) limit is passed by up to ~0.7 mrad,
            # which the re-check rejects. cuRobo's chords never leave the
            # hull, so ship its own plan (no accelerations) through the same
            # gate instead; it still keeps the margin from the URDF limit.
            log.warning(
                "quintic resample failed the constraint re-check (%s); "
                "using cuRobo's linear plan, without accelerations",
                recheck[0],
            )
            plan, consistent = linear, False
            recheck = self._recheck_constraints(plan)
        ordered = plan.get_ordered_joint_state(self.joint_names)
        pos = ordered.position.detach().cpu().numpy().astype(float).copy()
        vel = (
            ordered.velocity.detach().cpu().numpy().astype(float).copy()
            if ordered.velocity is not None
            else None
        )
        # Unresampled, cuRobo's accelerations disagree with its positions
        # (#15): ship none, so a consumer stays on cubic rather than trust them.
        acc = (
            ordered.acceleration.detach().cpu().numpy().astype(float).copy()
            if consistent and ordered.acceleration is not None
            else None
        )
        traj = Trajectory(
            joint_names=list(self.joint_names),
            positions=pos,
            velocities=vel,
            accelerations=acc,
            dt=float(result.interpolation_dt),
            frame_id=self.base_frame,
        )

        lim = self.joint_limits()
        problems = validate_trajectory(
            traj,
            lim["position"],
            lim["velocity"],
            acceleration_limits=lim.get("acceleration"),
        )
        problems += recheck
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

    def _resample(self, plan, result):
        """Replace cuRobo's LINEAR_CUDA resample with a quintic-spline one
        through the same knots (#15) — BEFORE validation, so the re-check
        sees exactly the states that ship. Returns (plan, consistent):
        cuRobo's own plan and False when there are no knots or they do not
        reproduce it."""

        def arr(x):
            return x.detach().cpu().numpy().astype(float).reshape(-1, x.shape[-1])

        knots = getattr(result, "optimized_plan", None)
        out = None
        if knots is not None:
            out = resample_quintic(
                arr(knots.position),
                arr(plan.position),
                float(result.interpolation_dt),
            )
        if out is None:
            log.warning(
                "dense plan is not a linear resample of its knots; kept as-is, "
                "without accelerations"
            )
            return plan, False
        plan = plan.clone()
        for dst, src in zip((plan.position, plan.velocity, plan.acceleration), out):
            dst.copy_(self._torch.as_tensor(src, dtype=dst.dtype, device=dst.device))
        return plan, True

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

    def _with_failure_note(self, res, result, start, q_goal=None, goal_point=None):
        """`res`, which _finish made of cuRobo's `result`, with a failure's
        error, _explain(status), followed by what the world has to do with
        it: the boxes the `start` state or the joint goal `q_goal` reach
        into, the boxes around the tool point `goal_point` of a pose goal,
        and where the world's boxes come from. Added after _finish, so an
        override of _finish cannot leave it out."""
        if res.success or (result is not None and bool(result.success.item())):
            # nothing failed, or post-validation or an override refused a
            # plan cuRobo found and its error says why
            return res
        # cuRobo's status, read as _finish reads it: an override may report
        # the failure under a status of its own, which names neither the
        # start cuRobo refused nor the goal point no arm could hold
        status = (
            "no result"
            if result is None
            else str(getattr(result, "status", None) or "unknown")
        )
        try:
            note = self._failure_note(status, start, q_goal, goal_point)
        except Exception as exc:
            # it only explains: the failure is reported all the same
            note = " (attribution unavailable: %s)" % exc
        # where the world's boxes come from needs no FK, so it is said
        # whatever the attribution came to. An override's failure may have
        # no error (PlanResult allows None): then the note is the error
        try:
            world = " [world: %s]" % self.world_summary()
        except Exception as exc:
            world = " [world summary unavailable: %s]" % exc
        res.error = (res.error or "") + note + world
        return res

    def _failure_note(self, status, start, q_goal, goal_point):
        """Which boxes of the installed world a failed plan's start or joint
        goal reach into, or its tool point lies in or near, as text to follow
        _explain(status)."""
        s = str(status).upper().replace(" ", "_")
        note = ""
        if "INVALID_START" in s and start is not None:
            hits = self._colliding_boxes(start)
            note += (
                " [start collides with %s]" % _boxes(hits)
                if hits
                else " [start: no world box penetrated — self-collision or "
                "joint limits]"
            )
        if q_goal is not None:
            hits = self._colliding_boxes(q_goal)
            if hits:
                note += " [goal collides with %s]" % _boxes(hits)
        if "IK" in s and goal_point is not None:
            # a pose goal fixes the tool point, not the arm that would hold
            # it: boxes around that point are only a lead
            distance = sorted(
                point_box_distance(goal_point, self._world_cuboids).items(),
                key=lambda item: item[1],
            )
            inside = [key for key, d in distance if d < 0]
            near = [key for key, d in distance if 0 <= d < 0.05]
            where = (["inside " + ", ".join(inside)] if inside else []) + (
                ["within 5 cm of " + ", ".join(near)] if near else []
            )
            if where:
                note += " [goal point %s] (heuristic: tool point only)" % "; ".join(
                    where
                )
        return note

    @staticmethod
    def _explain(status):
        s = str(status).upper().replace(" ", "_")
        if "IK" in s:
            return (
                "no collision-free joint solution AT the goal — move it "
                "away from obstacles or relax the orientation. The goal "
                "is arm_tool_frame (fingertip midpoint) in arm_mount; the "
                "flange sits 12 cm behind it along the tool axis."
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
