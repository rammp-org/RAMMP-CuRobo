"""A held object, on the GPU (issue #23): spheres attached in the tool frame change what the planner avoids, and
detaching restores the unattached behaviour exactly. Skips cleanly without CUDA/cuRobo."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from rammp_curobo import CuRoboPlanner  # noqa: E402
from rammp_curobo.config import resolve_config  # noqa: E402
from rammp_curobo.scene import Obstacle, Scene, load_scene  # noqa: E402
from rammp_curobo.geometry import (  # noqa: E402
    euler_deg_to_quat_xyzw,
    tool_axis,
    xyzw_to_wxyz,
)

UTENSIL = [
    [0.0, 0.0, z, 0.012] for z in (0.04, 0.08, 0.12, 0.15)
]  # a 15 cm utensil along the tool's z


@pytest.fixture(scope="module")
def planner():
    p = CuRoboPlanner.from_config("gen3_real_held.yaml")
    yield p
    p.detach_spheres()


def _goal(planner):
    """A tool pose turned well away from the start, and where a 15 cm utensil's tip would be there. Pose plans: cuRobo
    repeats them exactly (its joint-space plans vary run to run), so the restore can be checked exactly."""
    q = list(planner.retract_pose)
    q[0] += 1.0
    q[1] += 0.25
    pos, quat = planner.fk(q)
    tip = np.asarray(pos) + np.asarray(tool_axis(xyzw_to_wxyz(quat))) * 0.15
    return [float(v) for v in pos], [float(v) for v in quat], tip


def test_the_link_is_baked_with_its_budget(planner):
    assert planner.attached_capacity == 8
    assert planner.attached == []


def test_attached_utensil_is_avoided_and_detach_restores(planner):
    start = list(planner.retract_pose)
    pos, quat, tip = _goal(planner)
    scene = planner.scene
    box = Obstacle(
        {
            "name": "test_box_at_the_utensil_tip",
            "position": [float(v) for v in tip],
            "dims": [0.04, 0.04, 0.04],
        }
    )
    try:
        planner.update_world(
            Scene(
                base_frame=scene.base_frame,
                obstacles=list(scene.obstacles) + [box],
                targets=scene.targets,
                objects=scene.objects,
            )
        )
        bare = planner.plan_to_pose(pos, quat, start=start)
        assert bare.success, "the bare gripper clears the box: %s" % bare.status
        planner.attach_spheres(UTENSIL)
        assert planner.attached == UTENSIL
        held = planner.plan_to_pose(pos, quat, start=start)
        assert not held.success, (
            "the goal puts the attached utensil's tip inside the box: must be refused (%s)"
            % held.status
        )
        planner.detach_spheres()
        again = planner.plan_to_pose(pos, quat, start=start)
        assert again.success
        assert (
            again.joint_traj.positions.shape == bare.joint_traj.positions.shape
            and np.allclose(
                again.joint_traj.positions, bare.joint_traj.positions, atol=1e-6
            )
        ), "detaching restores the plan"
    finally:
        planner.detach_spheres()
        planner.update_world(scene)


def test_self_collision_free_at_home_and_every_scene_target(planner):
    """The audit the baked spheres were tuned against (self-collision-free at home and at every scene target), for
    the held model: in a world of the pedestal alone, every sim-kitchen target the default model reaches from the
    retract pose, the held model reaches too -- with nothing attached and with the 15 cm utensil attached. cuRobo
    only returns self-collision-free plans, so each success is a self-collision-free state at that target."""
    scene = planner.scene
    mount = Scene(
        base_frame=scene.base_frame,
        obstacles=[o for o in scene.obstacles if o.name == "pedestal"],
        targets=[],
        objects=[],
    )
    default = CuRoboPlanner.from_config("gen3_real.yaml")
    targets = load_scene(resolve_config("world_sim_kitchen.yaml")).targets
    start = list(planner.retract_pose)
    try:
        default.update_world(mount)
        planner.update_world(mount)
        assert planner.check_state_valid(start)[0]
        planner.attach_spheres(UTENSIL)
        assert planner.check_state_valid(start)[
            0
        ], "the utensil collides with the arm at home"
        planner.detach_spheres()
        reached = []
        for t in targets:
            quat = list(euler_deg_to_quat_xyzw(t.rpy_deg))
            if not default.plan_to_pose(t.position, quat, start=start).success:
                continue
            reached.append(t.name)
            assert planner.plan_to_pose(t.position, quat, start=start).success, t.name
            planner.attach_spheres(UTENSIL)
            try:
                r = planner.plan_to_pose(t.position, quat, start=start)
                assert r.success, "%s with the utensil: %s" % (t.name, r.status)
            finally:
                planner.detach_spheres()
        assert reached, "the default model reached none of the scene's targets"
    finally:
        planner.detach_spheres()
        planner.update_world(scene)


def test_the_default_config_refuses_to_attach():
    p = CuRoboPlanner.from_config("gen3_real.yaml")
    assert p.attached_capacity == 0
    with pytest.raises(ValueError, match="held-object config"):
        p.attach_spheres(UTENSIL)


def test_attach_refuses_bad_input(planner):
    with pytest.raises(ValueError):
        planner.attach_spheres([[0, 0, 0.1, 0.01]] * 9)  # over the budget
    with pytest.raises(ValueError):
        planner.attach_spheres([[0, 0, 0.1, 0.0]])  # r <= 0
    assert planner.attached == []
