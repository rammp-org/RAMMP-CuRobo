"""The MEBot sim's MuJoCo scene, built with MjSpec in the arm's base_link
frame, which is MuJoCo's world frame here.

The arm follows the recipe of RAMMP-Kinova's mujoco_sim build_scene.py,
re-implemented by reference: the menagerie Gen3, with the 2F-85 in its
bracelet. Around it is what the arm works beside on the chair:

  * the MEBot drive base, through MuJoCo's own URDF importer, from the very
    bytes the layers were generated from (the generator's fetch_pinned):
    its 46 collision primitives exactly, cylinders as cylinders, with every
    joint deleted, since the layers model it at all joints zero;
  * a floor plane at the floor layer's top;
  * a seated stand-in whose shape owes nothing to the person layer: the
    stock DeepMind humanoid, posed, scaled to the occupant's sitting height
    and seated with the anchors of scripts/mebot_layers.yaml;
  * the chair planner's boxes, padded as it pads them, as overlays.

Contacts with the true shapes are what the sim reports. The overlays only
show what the planner was told: nothing collides with them.
"""

import os

import mujoco
import numpy as np

from mebot_sim import stack

MENAGERIE = "~/mujoco_menagerie"
HUMANOID = "/opt/ros/humble/opt/mujoco_vendor/model/humanoid/humanoid.xml"
GROUPS = ("mebot", "standin", "floor")
# The menagerie Gen3 README's attachment turned 90 degrees about the
# gripper's axis, as the RAMMP arm's gripper is mounted.
GRIPPER_POS = [0.0, 0.0, -0.06149039]
GRIPPER_QUAT = [0.0, 0.0, 1.0, 0.0]
OVERLAY_RGBA = {
    "platform": [0.15, 0.45, 1.0, 0.2],
    "person": [1.0, 0.55, 0.05, 0.2],
    "floor": [0.55, 0.55, 0.55, 0.2],
}
MEBOT_RGBA = [0.25, 0.27, 0.32, 1.0]
SKIN_RGBA = [0.93, 0.74, 0.62, 1.0]


class Assets:
    """What every scene is built from, read once.

    The menagerie models and the humanoid are read by path. From this
    checkout: the pinned URDF, scripts/mebot_layers.yaml (the mount, the
    seat, the occupant), the stand-in's postures (config/scenarios.yaml),
    and the chair planner's start-up world as it composes it
    (gen3_mebot.yaml): `boxes`, [(key, layer, {dims, pose})] padded, and
    `layers`, {name: Layer} unpadded.
    """

    def __init__(self, menagerie=MENAGERIE, humanoid=HUMANOID):
        menagerie = os.path.expanduser(menagerie)
        self.gen3 = os.path.join(menagerie, "kinova_gen3", "gen3.xml")
        self.gripper = os.path.join(menagerie, "robotiq_2f85", "2f85.xml")
        self.humanoid = os.path.expanduser(humanoid)
        for path in (self.gen3, self.gripper, self.humanoid):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "%s not found (the launch's menagerie:= and humanoid:= say "
                    "where to look)" % path
                )
        generator = stack.generator()
        self.data = generator.load_data()
        self.urdf, self.urdf_origin = generator.fetch_pinned(self.data["source"])
        self.mount = np.array(self.data["mount"]["arm_base_in_mebot"]["position"])
        occupant = self.data["person"]["ansur_in"]
        self.sitting_height = occupant["sitting_height"]["inches"] * generator.INCH
        self.postures = stack.scenarios()["standin"]
        self.boxes, self.layers = _chair_world()


def _chair_world(config="gen3_mebot.yaml"):
    """The chair planner's start-up world as it composes it: its boxes,
    padded, with each one's layer, and its layers by name."""
    stack.core()
    from rammp_curobo.config import load_planner_config, resolve_config
    from rammp_curobo.scene import load_layers, load_scene
    from rammp_curobo.world import compose_cuboids

    cfg, where = load_planner_config(config)
    layers = load_layers(cfg["layers"], relative_to=where, listed_in=config)
    world = load_scene(resolve_config(cfg["world"], relative_to=where))
    p = cfg["planner"]
    cuboids, sources = compose_cuboids(
        world,
        p["world_padding"],
        no_pad_names=frozenset(p["no_pad_names"] or []),
        cache_obb=p["collision_cache_obb"],
        layers=layers,
    )
    boxes = [(key, sources[key], box) for key, box in cuboids.items()]
    return boxes, {layer.name: layer for layer in layers + world.layers}


def mebot_spec(urdf):
    """The MEBot as MuJoCo's URDF importer reads the pinned bytes: collision
    primitives only, every link its own body (fusestatic off, so the links
    keep their names) and every joint deleted."""
    text = urdf.decode("ascii")
    head = text.index(">", text.index("<robot")) + 1
    text = (
        text[:head]
        + '<mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>'
        + text[head:]
    )
    spec = mujoco.MjSpec.from_string(text)
    for joint in list(spec.joints):
        spec.delete(joint)
    return spec


def _bounds(geom):
    """A capsule's or a sphere's axis-aligned bounds: (low, high)."""
    pos, radius = geom["pos"], geom["size"][0]
    if geom["type"] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos - radius, pos + radius
    half = np.asarray(geom["mat"]).reshape(3, 3)[:, 2] * geom["size"][1]
    ends = pos - half, pos + half
    return np.minimum(*ends) - radius, np.maximum(*ends) + radius


def standin(assets, posture, backrest_face_x=None, sitting_height=None):
    """The seated stand-in's geoms in base_link, and its scale.

    The humanoid is posed (seated, then the posture's joints and pitch),
    scaled uniformly so that butt bottom to head top is `sitting_height`
    (metres; by default the occupant's, the sitting height
    scripts/mebot_layers.yaml sizes the person layer's top for), and placed
    with the butt on the seat pan and on the occupant's centreline and the
    back against the backrest face. That face lies somewhere in a CAD range
    (mebot frame); the default is the rear of it, since at the front a
    95th-percentile body and the arm's bench retract interpenetrate (spec
    6.4).

    Returns ([{name, type, size, pos, mat}], scale), `mat` being the geom's
    orientation as a flattened 3x3 matrix.
    """
    seat = assets.data["person"]["seat"]
    if backrest_face_x is None:
        backrest_face_x = min(seat["backrest_face_x"])
    if sitting_height is None:
        sitting_height = assets.sitting_height
    pose = assets.postures["postures"][posture]
    m = mujoco.MjModel.from_xml_path(assets.humanoid)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for name, degrees in {**assets.postures["seated"], **pose["joints"]}.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = np.radians(degrees)
    # the free root joint's orientation: a pitch about y leans it forward
    half = np.radians(pose.get("pitch_deg", 0)) / 2
    root = m.jnt_qposadr[m.joint("root").id]
    d.qpos[root + 3 : root + 7] = [np.cos(half), 0.0, np.sin(half), 0.0]
    mujoco.mj_kinematics(m, d)
    geoms = [
        {
            "name": m.geom(g).name,
            "type": int(m.geom_type[g]),
            "size": m.geom_size[g].copy(),
            "pos": d.geom_xpos[g].copy(),
            "mat": d.geom_xmat[g].copy(),
        }
        for g in range(m.ngeom)
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE  # its own floor
    ]
    named = {g["name"]: g for g in geoms}
    butt = _bounds(named["butt"])[0][2]
    scale = sitting_height / (_bounds(named["head"])[1][2] - butt)
    back = min(
        _bounds(named[n])[0][0] for n in ("butt", "torso", "waist_upper", "waist_lower")
    )
    anchor = [backrest_face_x, seat["centreline_y"], seat["pan_top_z"]] - assets.mount
    offset = anchor - scale * np.array([back, named["butt"]["pos"][1], butt])
    for g in geoms:
        g["pos"] = scale * g["pos"] + offset
        g["size"] = scale * g["size"]
    return geoms, scale


def _menagerie_spec(path):
    """A menagerie model as an MjSpec ready to be merged: its mesh files made
    absolute (an attached model's would resolve against the parent's
    directory) and its keyframes deleted (they no longer fit once merged)."""
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.join(os.path.dirname(os.path.abspath(path)), spec.meshdir or "")
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.normpath(os.path.join(base, mesh.file))
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def build(assets, posture, offscreen=(960, 720)):
    """The scene as an MjModel, the stand-in in `posture` (None: an empty
    chair). `offscreen` is the largest picture the live view renders."""
    spec = _menagerie_spec(assets.gen3)
    gripper = _menagerie_spec(assets.gripper)
    # the 2F-85's contact options, set before the attach so they win its merge
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # menagerie's base_link and shoulder_link meshes overlap by design; as a
    # live contact pair they brake joint_1
    exclude = spec.add_exclude()
    exclude.bodyname1, exclude.bodyname2 = "base_link", "shoulder_link"
    bracelet = spec.body("bracelet_link").add_frame(pos=GRIPPER_POS, quat=GRIPPER_QUAT)
    bracelet.attach_body(gripper.body("base"), "g_", "")

    world = spec.worldbody
    mebot = world.add_frame(pos=list(-assets.mount), quat=[1.0, 0.0, 0.0, 0.0])
    mebot.attach_body(mebot_spec(assets.urdf).body("mebot"), "mebot/", "")
    for body in spec.bodies:
        if body.name.startswith("mebot/"):
            for geom in body.geoms:
                geom.rgba = MEBOT_RGBA
    slab = assets.layers["floor"].obstacles[0]
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[slab.dims[0] / 2, slab.dims[1] / 2, 0.1],
        pos=[slab.position[0], slab.position[1], slab.position[2] + slab.dims[2] / 2],
        material="grid",
    )
    if posture is not None:
        body = world.add_body(name="standin")
        for g in standin(assets, posture)[0]:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, g["mat"])
            body.add_geom(
                name="standin/" + g["name"],
                type=g["type"],
                size=g["size"],
                pos=g["pos"],
                quat=quat,
                rgba=SKIN_RGBA,
            )
    for key, layer, box in assets.boxes:
        x, y, z, qw, qx, qy, qz = box["pose"]
        world.add_geom(
            name="overlay/" + key,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[d / 2 for d in box["dims"]],
            pos=[x, y, z],
            quat=[qw, qx, qy, qz],
            rgba=OVERLAY_RGBA[layer],
            contype=0,
            conaffinity=0,
            group=2,
        )
    _style(spec, offscreen)
    return spec.compile()


def _style(spec, offscreen):
    """What the live view needs of the model: a picture buffer as large as
    it renders, light, a sky, a checker floor and visible contacts."""
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.quality.shadowsize = 4096
    spec.visual.scale.contactwidth = 0.06
    spec.visual.scale.contactheight = 0.015
    spec.visual.rgba.contactpoint = [1.0, 0.0, 0.0, 1.0]
    spec.worldbody.add_light(
        pos=[1.5, -1.5, 2.5],
        dir=[-0.5, 0.5, -1.0],
        diffuse=[0.55, 0.55, 0.55],
        specular=[0.1, 0.1, 0.1],
        castshadow=True,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=512,
        height=3072,
        rgb1=[0.78, 0.84, 0.92],
        rgb2=[0.30, 0.36, 0.45],
    )
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300,
        height=300,
        rgb1=[0.82, 0.82, 0.80],
        rgb2=[0.72, 0.72, 0.70],
    )
    grid = spec.add_material(name="grid")
    grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    grid.texrepeat = [12, 12]


def contact_groups(m):
    """(arm, groups): the arm's and the gripper's collision geoms, and
    {group: geom ids} for the shapes a contact is reported against."""
    root = m.body("base_link").id

    def on_arm(body):
        while body:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    collides = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    arm = [g for g in collides if on_arm(m.geom_bodyid[g])]
    groups = {group: [] for group in GROUPS}
    for g in collides:
        body = m.body(m.geom_bodyid[g]).name
        if m.geom(g).name == "floor":
            groups["floor"].append(g)
        elif body == "standin":
            groups["standin"].append(g)
        elif body.startswith("mebot/"):
            groups["mebot"].append(g)
    return np.array(arm), {k: np.array(v, dtype=int) for k, v in groups.items()}
