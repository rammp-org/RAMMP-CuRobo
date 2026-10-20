"""Scene + layers -> cuRobo collision world, with the v0.7.8 guard rails.

A world is composed in pure Python (compose_cuboids): the scene's own boxes,
then the layers the planner config lists, then the layers the scene lists.
Its guards all run there, before cuRobo is imported; to_world_config then
refuses an empty world again, before its own import, and checks that
cuRobo holds exactly the boxes composed. A refused world is never loaded,
so the world loaded before it stays.

cuRobo v0.7.8 sharp edges this module exists to blunt (the first three
field-verified in RAMMP-Kinova):
  * Cylinder/Sphere entries in a WorldConfig are SILENTLY DROPPED by the
    collision checkers — every body goes in as a cuboid bounding box.
  * update_world() with ZERO cuboids silently keeps the previous world
    (early return before the disable line) — an empty world is refused here,
    except at construction (`allow_empty`), where there is no old world.
  * More cuboids than the collision cache raises from inside cuRobo —
    refused here with a message that names the fix.
  * Boxes are keyed by name, so two boxes of one name COLLAPSE into one —
    refused here, naming both.
  * A NaN position, a zero-norm quaternion or a negative dim loads as an
    ENABLED box that can never collide, with no error or log (a CPU probe
    of v0.7.8) — every number is checked here, and quaternions are only
    ever made from rpy.
  * WorldConfig.from_dict skips a top-level key it does not know, so a
    misspelt one gives an EMPTY world — to_world_config checks that cuRobo
    holds exactly the boxes it was given.
"""

import math
from collections import Counter

import numpy as np

from rammp_curobo.geometry import euler_deg_to_quat_xyzw

# v1.0.0's words: callers match 'empty'
_EMPTY = (
    "collision world is empty; refusing to build it — cuRobo would "
    "silently keep the previous world on update"
)


def world_cuboids(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset()):
    """The scene's obstacles + props as padded cuboid dicts.

    `padding` is per SIDE (each dim grows by 2*padding). Names in
    `no_pad_names` (the arm's own pedestal) are exempt — that box is
    margin-sized to stay under the robot's base spheres, and padding it
    would put every start state in collision. Props in `ignore` are left
    out entirely (the object being reached for cannot also be dodged).
    """
    pad = 2.0 * float(padding)
    cuboids = {}
    for o in scene.obstacles:
        x, y, z, w = euler_deg_to_quat_xyzw(o.rpy_deg)
        p = 0.0 if o.name in no_pad_names else pad
        cuboids[o.name] = {
            "dims": [d + p for d in o.dims],
            "pose": [o.position[0], o.position[1], o.position[2], w, x, y, z],
        }
    for o in scene.objects:
        if o.name in ignore:
            continue
        x, y, z, w = euler_deg_to_quat_xyzw(o.rpy_deg)
        # 'obj_' prefix so a prop can share a name with an obstacle.
        cuboids["obj_" + o.name] = {
            "dims": [d + pad for d in o.bounding_dims()],
            "pose": [o.position[0], o.position[1], o.position[2], w, x, y, z],
        }
    return cuboids


def compose_cuboids(
    scene,
    padding=0.02,
    ignore=frozenset(),
    no_pad_names=frozenset(),
    cache_obb=60,
    *,
    layers=(),
    allow_empty=False,
):
    """The collision world of `scene` and the config `layers`, guarded.

    Returns (cuboids, sources). `cuboids` is in world_cuboids' format, in
    the order scene obstacles, scene props, config layers, scene layers;
    `sources` maps each key to 'scene' or to its layer's name, so that a
    box's layer is never read off its key. Scene boxes keep their v1.0.0
    keys, and a layer box is '<layer>/<box>'. Every layer box is padded:
    `no_pad_names` exempts scene obstacles only (the pedestal), as in
    v1.0.0. A file that the config and the scene both list is one layer,
    in the config's place and as the config loaded it. world_cuboids is
    v1.0.0's composition, kept as it was: a world without layers composes
    exactly as it did.

    `ignore` leaves out props, as in v1.0.0, and whole layers, if they
    are excludable. A bare string is one name.

    Refused, all before cuRobo is imported: a padding below 0 or not
    finite; one layer from two files; an ignored layer that is not
    excludable, or whose name a prop shares; a position or rpy_deg that is
    not 3 finite numbers, or dims that are not 3 finite numbers above 0;
    two boxes of one key; a scene with no content of its own; more boxes
    than `cache_obb`.

    `allow_empty` is for building a planner only, where there is no old
    world for cuRobo to keep: the scene then need not bring content of its
    own, and may compose to no boxes at all. Never pass it on update.
    """
    padding = float(padding)
    # each box is checked before it is padded, so the padding is checked
    # here: below 0 it would shrink a thin box past 0, and a NaN or an inf
    # would spread to every box
    if not (math.isfinite(padding) and padding >= 0):
        raise ValueError("world_padding must be a finite number >= 0, got %r" % padding)
    if isinstance(ignore, str):
        ignore = [ignore]  # one name, not the set of its letters
    ignore = frozenset(ignore)
    layers = list(layers)
    chosen = []
    for layer in layers + list(scene.layers):
        # one file is one layer, where it is first listed: a file the config
        # lists composes as the copy loaded with the config, whatever the
        # file says now. A Layer built in code may have no path: no file
        if layer.path is not None and any(layer.path == other.path for other in chosen):
            continue
        for other in chosen:
            if other.name == layer.name:
                raise ValueError(
                    "layer %r is defined by two files: %s and %s"
                    % (layer.name, other.path, layer.path)
                )
        chosen.append(layer)
    props = {o.name for o in scene.objects}
    for layer in chosen:
        if layer.name not in ignore:
            continue
        if layer.name in props:
            raise ValueError(
                "ignore=%r names both a prop and a layer; rename the prop" % layer.name
            )
        if not layer.excludable:
            raise ValueError(
                "ignore=%r: layer %r is not excludable" % (layer.name, layer.name)
            )

    pad = 2.0 * padding
    cuboids, sources, owners = {}, {}, {}

    def add(key, source, owner, box, dims, p):
        # before padding: padded, a dim of -0.01 would make a box 3 cm wide
        for field, v, positive in (
            ("position", box.position, False),
            ("rpy_deg", box.rpy_deg, False),
            ("dims", dims, True),
        ):
            if len(v) != 3 or not all(
                math.isfinite(x) and (x > 0 or not positive) for x in v
            ):
                raise ValueError(
                    "collision box %r (%s): %r must be 3 finite numbers%s, got %r"
                    % (key, owner, field, " greater than 0" if positive else "", v)
                )
        if key in owners:
            raise ValueError(
                "two collision boxes are named %r: %s and %s; rename one — a "
                "world holds one box per name" % (key, owners[key], owner)
            )
        x, y, z, w = euler_deg_to_quat_xyzw(box.rpy_deg)
        cuboids[key] = {
            "dims": [d + p for d in dims],
            "pose": [box.position[0], box.position[1], box.position[2], w, x, y, z],
        }
        sources[key] = source
        owners[key] = owner

    for i, o in enumerate(scene.obstacles):
        p = 0.0 if o.name in no_pad_names else pad
        add(o.name, "scene", "scene obstacles[%d]" % i, o, o.dims, p)
    for i, o in enumerate(scene.objects):
        if o.name not in ignore:
            # as the box cuRobo gets: a radius of 0 is a dim of 0
            owner = "scene objects[%d]" % i
            add("obj_" + o.name, "scene", owner, o, o.bounding_dims(), pad)
    for layer in chosen:
        if layer.name in ignore:
            continue
        for i, o in enumerate(layer.obstacles):
            owner = "layer %r obstacles[%d] in %s" % (layer.name, i, layer.path)
            add("%s/%s" % (layer.name, o.name), layer.name, owner, o, o.dims, pad)

    # The scene must bring content of its own, whatever the config adds: a
    # push cut short would otherwise plan as the configured layers alone. A
    # file the scene lists is its own, though the config lists it too, and
    # it counts as it composed: as the config's copy, boxes and name.
    listed = {layer.path for layer in scene.layers} - {None}
    own = (
        len(scene.obstacles)
        + sum(o.name not in ignore for o in scene.objects)
        + sum(
            len(layer.obstacles)
            for layer in chosen
            if layer.name not in ignore
            and (layer not in layers or layer.path in listed)
        )
    )
    if not own and not allow_empty:
        configured = [layer.name for layer in chosen if layer in layers]
        if configured:
            raise ValueError(
                "collision world is empty apart from the configured layers "
                "(%s): the pushed scene has no obstacles, objects or layers — "
                "refusing it" % ", ".join(configured)
            )
        raise ValueError(_EMPTY)
    n = len(cuboids)
    if n > int(cache_obb):
        counts = Counter(sources.values())
        raise ValueError(
            "%d collision boxes > collision_cache_obb=%d (%s); raise the cache "
            "setting in the planner config or shrink a layer"
            % (n, cache_obb, ", ".join("%s %d" % c for c in counts.items()))
        )
    return cuboids, sources


def to_world_config(cuboids, allow_empty=False):
    """cuRobo's WorldConfig of composed `cuboids`, checked to hold those
    boxes and nothing else (see the module docstring). The only function
    in this module that imports cuRobo, and it refuses an empty world
    before it does: one built anywhere but compose_cuboids would otherwise
    reach update_world, which keeps the world loaded before. `allow_empty`
    is compose_cuboids', for building a planner only."""
    if not cuboids and not allow_empty:
        raise ValueError(_EMPTY)
    from curobo.geom.types import WorldConfig

    wc = WorldConfig.from_dict({"cuboid": cuboids})
    want = {"cuboid": len(cuboids), "sphere": 0, "cylinder": 0, "capsule": 0, "mesh": 0}
    held = {kind: len(getattr(wc, kind)) for kind in want}
    if held != want:
        raise ValueError(
            "cuRobo's WorldConfig holds %s for %d composed boxes; only those "
            "boxes may reach its collision checkers" % (held, len(cuboids))
        )
    return wc


def make_world_config(
    scene,
    padding=0.02,
    ignore=frozenset(),
    no_pad_names=frozenset(),
    cache_obb=60,
    *,
    layers=(),
    allow_empty=False,
):
    """A guarded cuRobo WorldConfig for this scene and the config `layers`:
    compose_cuboids, then to_world_config."""
    cuboids, _ = compose_cuboids(
        scene,
        padding,
        ignore,
        no_pad_names,
        cache_obb,
        layers=layers,
        allow_empty=allow_empty,
    )
    return to_world_config(cuboids, allow_empty=allow_empty)


def sphere_box_penetration(spheres, cuboids):
    """{key: depth_m} for each box of `cuboids` (composed, so padded) that
    one of `spheres` ((N, 4), or (..., 4) for a batch: x, y, z and a radius
    that includes any buffer) reaches into, by its deepest sphere. Other
    boxes are left out.

    The test cuRobo's feasibility check makes at activation distance 0: a
    sphere collides when its radius exceeds its centre's signed distance
    to the box. Made here on the CPU, it names the boxes a state hits
    without touching the world loaded in cuRobo.
    """
    spheres = np.asarray(spheres, dtype=float)
    # read 4 numbers at a time, any other last dimension would be other
    # spheres; and one sphere that is not finite would hide every hit, since
    # no maximum taken over it is above 0
    if spheres.shape[-1:] != (4,):
        raise ValueError(
            "spheres must have shape (..., 4): x, y, z and radius; got %s"
            % (spheres.shape,)
        )
    spheres = spheres.reshape(-1, 4)
    finite = np.isfinite(spheres).all(axis=1)
    if not finite.all():
        raise ValueError(
            "spheres must be finite, got %s" % spheres[~finite][0].tolist()
        )
    depth = spheres[:, 3:] - _signed_distances(spheres[:, :3], cuboids)
    deepest = depth.max(axis=0, initial=-np.inf)
    return {key: float(d) for key, d in zip(cuboids, deepest) if d > 0}


def point_box_distance(point, cuboids):
    """{key: signed distance in metres} from `point` to each box of
    `cuboids`: positive outside, minus the depth to the nearest face
    inside."""
    d = _signed_distances(np.asarray(point, dtype=float).reshape(1, 3), cuboids)
    return dict(zip(cuboids, d[0].tolist()))


def _signed_distances(points, cuboids):
    """(N, M) signed distances from N points to the M oriented boxes of
    `cuboids`, each posed [x, y, z, qw, qx, qy, qz]."""
    boxes = list(cuboids.values())
    pose = np.array([b["pose"] for b in boxes], dtype=float).reshape(-1, 7)
    half = np.array([b["dims"] for b in boxes], dtype=float).reshape(-1, 3) / 2
    w, x, y, z = pose[:, 3:].T
    # 2 / |q|^2 rather than 2: cuRobo normalizes the quaternions it is given
    s = 2.0 / (w * w + x * x + y * y + z * z)
    rot = np.array(  # (3, 3, M): box frame -> base frame
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )
    # every point in every box's frame: R^T (p - centre)
    local = np.einsum("jim,nmj->nmi", rot, points[:, None, :] - pose[:, :3])
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
    return outside + np.minimum(q.max(axis=2), 0.0)


def summarize(sources, config_layer_names):
    """Where the boxes of a composed world come from, on one line, e.g.
    "25 boxes: scene 0 | config layers: platform 16, person 8 | scene
    layers: floor 1". `sources` is compose_cuboids' map, and
    `config_layer_names` the layers the planner config lists: one that
    this world leaves out (ignore=) is still named, with 0."""
    counts = Counter(sources.values())
    config = ["%s %d" % (name, counts[name]) for name in config_layer_names]
    scene = [
        "%s %d" % (name, n)
        for name, n in counts.items()
        if name != "scene" and name not in config_layer_names
    ]
    n = len(sources)
    return "%d %s: scene %d | config layers: %s | scene layers: %s" % (
        n,
        "box" if n == 1 else "boxes",
        counts["scene"],
        ", ".join(config) or "none",
        ", ".join(scene) or "none",
    )
