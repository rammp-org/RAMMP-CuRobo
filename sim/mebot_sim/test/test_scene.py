"""The sim's MuJoCo scene (mebot_sim/scene.py), on the CPU.

It needs MuJoCo, the menagerie Gen3 and 2F-85, the humanoid of ROS's
mujoco_vendor and the pinned MEBot URDF (from the cache, a Demo-Software
clone or GitHub); without them it skips (conftest.py's assets).
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mebot_sim import scene, stack  # noqa: E402

DATA = stack.generator().load_data()
MOUNT = np.array(DATA["mount"]["arm_base_in_mebot"]["position"])
SEAT = DATA["person"]["seat"]
SITTING = DATA["person"]["ansur_in"]["sitting_height"]["inches"] * 0.0254
POSTURES = ("upright_lap", "lean_lap", "forearms_forward")


@pytest.fixture(scope="module")
def model(assets):
    return scene.build(assets, "upright_lap")


def _geoms(m, prefix):
    return [g for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)]


def test_the_scene_follows_the_recipe(model):
    m = model
    overlays = _geoms(m, "overlay/")
    # the design-phase prototype's counts (spec 8)
    assert (m.nbody, m.ngeom - len(overlays), m.njnt, m.nu) == (69, 108, 15, 8)
    assert m.opt.impratio == 10 and m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    # the gripper is in the bracelet, as g_
    assert m.body(m.body_parentid[m.body("g_base").id]).name == "bracelet_link"
    # menagerie's base and shoulder meshes overlap by design
    pair = (m.body("base_link").id << 16) + m.body("shoulder_link").id
    assert pair in m.exclude_signature
    # every arm actuator is menagerie's position actuator on its joint
    for name in stack.JOINTS:
        a = m.actuator(name)
        assert m.joint(a.trnid[0]).name == name and a.gainprm[0] > 0


def test_the_mebot_is_its_urdf_with_every_joint_deleted(model):
    m = model
    mebot = [
        g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("mebot/")
    ]
    kinds = [int(m.geom_type[g]) for g in mebot]
    assert len(mebot) == 46
    assert kinds.count(mujoco.mjtGeom.mjGEOM_BOX) == 40
    assert kinds.count(mujoco.mjtGeom.mjGEOM_CYLINDER) == 6
    assert not [
        j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith("mebot/")
    ]
    # attached at -mount: the root link's frame is the mebot frame
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    assert d.xpos[m.body("mebot/mebot").id] == pytest.approx(-MOUNT)


def test_the_floor_is_the_floor_layers_top(assets, model):
    floor = assets.layers["floor"].obstacles[0]
    top = floor.position[2] + floor.dims[2] / 2
    assert model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom("floor").pos[2] == pytest.approx(top)
    assert top == pytest.approx(-0.4593)


def test_the_overlays_are_the_padded_layers_and_touch_nothing(assets, model):
    m = model
    overlays = _geoms(m, "overlay/")
    assert len(overlays) == 25  # floor 1, platform 16, person 8
    for layer in assets.layers.values():
        for box in layer.obstacles:
            g = m.geom("overlay/%s/%s" % (layer.name, box.name))
            # padded by the chair config's 0.02 per side, as the planner pads
            assert 2 * g.size == pytest.approx(np.add(box.dims, 0.04))
            assert g.pos == pytest.approx(box.position)
            assert (g.contype, g.conaffinity, g.group) == (0, 0, 2)
            assert g.rgba[3] == pytest.approx(0.2)
            r, gr, b = g.rgba[:3]
            if layer.name == "platform":  # blue
                assert b > max(r, gr)
            elif layer.name == "person":  # orange
                assert r > gr > b
            else:  # grey
                assert r == gr == b


@pytest.mark.parametrize("posture", POSTURES)
def test_the_standin_sits_in_the_seat(assets, posture):
    geoms, scale = scene.standin(assets, posture)

    def extent(g, axis, side):
        """The geom's bound along `axis` in base_link: side -1 low, +1 high."""
        if g["type"] == mujoco.mjtGeom.mjGEOM_SPHERE:
            return g["pos"][axis] + side * g["size"][0]
        half = np.asarray(g["mat"]).reshape(3, 3)[:, 2] * g["size"][1]
        ends = (g["pos"][axis] - half[axis], g["pos"][axis] + half[axis])
        return (max if side > 0 else min)(ends) + side * g["size"][0]

    named = {g["name"]: g for g in geoms}
    butt, head = named["butt"], named["head"]
    pan = SEAT["pan_top_z"] - MOUNT[2]
    # on the pan, at the occupant's sitting height, on the centreline
    assert extent(butt, 2, -1) == pytest.approx(pan)
    assert extent(head, 2, +1) - pan == pytest.approx(SITTING)
    assert butt["pos"][1] == pytest.approx(SEAT["centreline_y"] - MOUNT[1])
    # its back against the rear of the backrest's CAD range (spec 6.4)
    back = min(
        extent(named[n], 0, -1) for n in ("butt", "torso", "waist_upper", "waist_lower")
    )
    assert back == pytest.approx(min(SEAT["backrest_face_x"]) - MOUNT[0])
    # spec 6.4: x1.229 upright, x1.254 leaning
    assert scale == pytest.approx(1.254 if posture == "lean_lap" else 1.229, abs=5e-4)


def test_the_standin_can_sit_at_the_front_of_the_backrest_range(assets):
    front = max(SEAT["backrest_face_x"])
    rear, _ = scene.standin(assets, "upright_lap")
    forward, _ = scene.standin(assets, "upright_lap", backrest_face_x=front)
    shift = [f["pos"] - r["pos"] for f, r in zip(forward, rear)]
    assert np.allclose(shift, [front - min(SEAT["backrest_face_x"]), 0, 0])


def test_contacts_are_counted_against_three_groups(assets, model):
    arm, groups = scene.contact_groups(model)
    m = model

    def under(body, root):
        while body:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    root = m.body("base_link").id
    assert all(under(m.geom_bodyid[g], root) for g in arm)
    # the gripper's included
    assert any(m.body(m.geom_bodyid[g]).name.startswith("g_") for g in arm)
    assert len(groups["mebot"]) == 46 and len(groups["floor"]) == 1
    assert len(groups["standin"]) == len(scene.standin(assets, "upright_lap")[0])
    # visual geoms and the overlays are in no group
    every = set(arm).union(*map(set, groups.values()))
    assert every.isdisjoint(_geoms(m, "overlay/"))
    assert all(m.geom_contype[g] or m.geom_conaffinity[g] for g in every)

    # an empty chair
    assert len(scene.contact_groups(scene.build(assets, None))[1]["standin"]) == 0
