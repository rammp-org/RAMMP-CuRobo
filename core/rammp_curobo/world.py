"""Scene -> cuRobo collision world, with the v0.7.8 guard rails.

cuRobo v0.7.8 sharp edges this module exists to blunt (all field-verified
in RAMMP-Kinova):
  * Cylinder/Sphere entries in a WorldConfig are SILENTLY DROPPED by the
    collision checkers — every body goes in as a cuboid bounding box.
  * update_world() with ZERO cuboids silently keeps the previous world
    (early return before the disable line) — an empty world is refused here.
  * More cuboids than the collision cache raises from inside cuRobo —
    refused here with a message that names the fix.
"""

from rammp_curobo.config import PLANNER_DEFAULTS
from rammp_curobo.geometry import euler_deg_to_quat_xyzw


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


def make_world_config(
    scene,
    padding=0.02,
    ignore=frozenset(),
    no_pad_names=frozenset(),
    # default tracks the config so a caller that forgets to pass the cache
    # size gets the same cap the planner was built with
    cache_obb=PLANNER_DEFAULTS["planner"]["collision_cache_obb"],
):
    """A guarded cuRobo WorldConfig for this scene (see module docstring)."""
    from curobo.geom.types import WorldConfig

    cuboids = world_cuboids(
        scene, padding=padding, ignore=ignore, no_pad_names=no_pad_names
    )
    n = len(cuboids)
    if n < 1:
        raise ValueError(
            "collision world is empty; refusing to build it — cuRobo would "
            "silently keep the previous world on update"
        )
    if n > int(cache_obb):
        raise ValueError(
            "%d collision boxes > collision_cache_obb=%d; raise the cache "
            "setting in the planner config" % (n, cache_obb)
        )
    return WorldConfig.from_dict({"cuboid": cuboids})
