"""S0: the committed layers against the true shapes, on the CPU.

    ros2 run mebot_sim check_layers

No ROS graph and no planner: the MuJoCo scene the sim flies in (scene.py),
and the layer files the generator writes, as the planner loads them.

  * G0, registration. MuJoCo's own parse of the pinned MEBot URDF is sampled
    on every primitive: a box on the outer layer of a 9 x 9 x 9 lattice, a
    cylinder on rims of 72 points at its radius and at half of it, at 9
    heights. Every sample must lie in a box of the committed platform layer,
    unpadded; with every box shrunk by 1 mm, some must not, or the check
    could not fail. The samples cover only what MuJoCo parsed, so its parse
    must hold every primitive of the URDF as the generator reads it, link
    by link, box or cylinder.
  * G1, stand-in coverage. The seated stand-in, in each posture and with its
    back at either end of the backrest face's CAD range, is sampled through
    its volume on a 6 mm grid. Every point must lie in the person layer as
    the chair planner pads it, or in the arm's carved column, where the
    arm's own base is: any other point is a gap, a part of the body the
    planner does not keep the arm out of. The gap of the unpadded layer is
    reported beside it. The column is the generator's, and a column carved
    too wide, or a body too small, leaves no gap either. So, with bounds
    from config/scenarios.yaml's thresholds, covered_pct of the body or more
    must lie in the layer itself, and the body must be the whole humanoid at
    the scale that seats the occupant: its volume within body_volume_pct of
    the humanoid's own times the scale cubed. Both follow the occupant that
    scripts/mebot_layers.yaml sizes. Spec 8's G1 is the gap alone.

Exit status 1 if a gate fails.
"""

import argparse
import collections
import math
import sys

import mujoco
import numpy as np

from mebot_sim import scene, stack

LATTICE = 9  # samples along a box's edge, or a cylinder's height
ANGLES = 72  # samples around a cylinder's rim
SHRINK_M = 0.001
GRID_M = 0.006


def surface_samples(m, d, geoms):
    """Points on each of `geoms` of model `m`, posed in `d`, in the world
    frame. The MEBot's primitives are boxes and cylinders, as its URDF has
    them: anything else is refused."""
    out = []
    for g in geoms:
        size = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            index = np.stack(
                np.meshgrid(*[np.arange(LATTICE)] * 3, indexing="ij"), axis=-1
            ).reshape(-1, 3)
            index = index[((index == 0) | (index == LATTICE - 1)).any(axis=1)]
            local = (2.0 * index / (LATTICE - 1) - 1.0) * size
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            angle = np.linspace(0.0, 2 * math.pi, ANGLES, endpoint=False)
            local = np.concatenate(
                [
                    np.stack(
                        [r * np.cos(angle), r * np.sin(angle), np.full(ANGLES, z)],
                        axis=1,
                    )
                    for z in np.linspace(-size[1], size[1], LATTICE)
                    for r in (size[0], size[0] / 2)
                ]
            )
        else:
            raise ValueError(
                "geom %r is neither a box nor a cylinder" % (m.geom(g).name or g)
            )
        out.append(local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(out) if out else np.empty((0, 3))


def bounds(obstacles, grow=0.0):
    """Each box's (low, high) corners, grown by `grow` metres a side; a
    negative `grow` shrinks it. The checks read the boxes as axis-aligned in
    base_link, as the generator writes them, so a rotated one is refused."""
    out = []
    for o in obstacles:
        if any(o.rpy_deg):
            raise ValueError(
                "%s is rotated (rpy_deg %s): these checks read a layer's boxes "
                "as axis-aligned, as the generator writes them" % (o.name, o.rpy_deg)
            )
        half = np.asarray(o.dims) / 2 + grow
        out.append((np.asarray(o.position) - half, np.asarray(o.position) + half))
    return out


def _inside(points, boxes):
    """Which of `points` lie in one of `boxes`, (low, high) corners each."""
    inside = np.zeros(len(points), dtype=bool)
    for low, high in boxes:
        inside |= ((points >= low) & (points <= high)).all(axis=1)
    return inside


def registration(m, platform, shrink=0.0):
    """How the MEBot of model `m` (scene.build) sits in the layer `platform`
    with its boxes shrunk by `shrink` a side: {geoms, primitives, samples,
    outside}, primitives counting its geoms by (link, "box" or "cylinder")
    as urdf_primitives does."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    geoms = scene.contact_groups(m)[1]["mebot"]
    points = surface_samples(m, d, geoms)
    outside = ~_inside(points, bounds(platform.obstacles, -shrink))
    primitives = collections.Counter(
        (
            m.body(m.geom_bodyid[g]).name.split("/", 1)[1],
            "box" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else "cylinder",
        )
        for g in geoms
    )
    return {
        "geoms": len(geoms),
        "primitives": primitives,
        "samples": len(points),
        "outside": int(outside.sum()),
    }


def urdf_primitives(assets):
    """The pinned URDF's collision primitives as the generator reads them,
    counted by (link, "box" or "cylinder")."""
    return collections.Counter(
        (p["link"], p["kind"])
        for p in stack.generator().urdf_primitives(assets.urdf, assets.mount)
    )


def _volume(g, grid):
    """Points through the volume of a stand-in geom on a lattice `grid` apart
    in its own frame, in base_link."""
    if g["type"] not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError(
            "stand-in geom %r is neither a capsule nor a sphere" % g["name"]
        )
    radius = g["size"][0]
    half = g["size"][1] if g["type"] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
    across = np.arange(-radius + grid / 2, radius, grid)
    along = np.arange(-(half + radius) + grid / 2, half + radius, grid)
    local = np.stack(np.meshgrid(across, across, along, indexing="ij"), axis=-1)
    local = local.reshape(-1, 3)
    # within the radius of the capsule's axis segment (a sphere's is a point)
    axis = np.clip(local[:, 2], -half, half)
    off = np.hypot(np.hypot(local[:, 0], local[:, 1]), local[:, 2] - axis)
    return local[off <= radius] @ np.asarray(g["mat"]).reshape(3, 3).T + g["pos"]


def volume_samples(geoms, grid=GRID_M):
    """Points through the volume of each of `geoms`, scene.standin's capsules
    and spheres, in base_link. Where geoms overlap, the place is sampled once
    for each."""
    return np.concatenate([_volume(g, grid) for g in geoms])


def humanoid_m3(assets):
    """The stock humanoid's own volume, as scene.standin takes it before it
    poses and scales it: its capsules and spheres, exactly, where they
    overlap counted once for each, as volume_samples samples them."""
    m = mujoco.MjModel.from_xml_path(assets.humanoid)
    total = 0.0
    for g in range(m.ngeom):
        kind = int(m.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:  # its own floor
            continue
        if kind not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError(
                "humanoid geom %r is neither a capsule nor a sphere" % m.geom(g).name
            )
        radius = float(m.geom_size[g][0])
        half = float(m.geom_size[g][1]) if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0
        total += math.pi * radius**2 * (2 * half + 4 / 3 * radius)
    return total


def coverage(geoms, boxes, column, grid=GRID_M):
    """How the body made of `geoms` sits in `boxes`, (low, high) corners
    each. A point of volume_samples is covered when it lies in a box, else
    it lies in the arm's `column`, (low, high), or in a gap.

    Returns {body_m3, covered_pct, column_m3, gap_m3, gap_points, gap_where},
    gap_where counting the gap's points by geom."""
    total = covered = in_column = 0
    where = {}
    for g in geoms:
        points = volume_samples([g], grid)
        inside = _inside(points, boxes)
        column_only = ~inside & _inside(points, [column])
        total += len(points)
        covered += int(inside.sum())
        in_column += int(column_only.sum())
        gap = len(points) - int(inside.sum()) - int(column_only.sum())
        if gap:
            where[g["name"]] = gap
    gap = sum(where.values())
    cell = grid**3
    return {
        "body_m3": total * cell,
        "covered_pct": 100.0 * covered / total,
        "column_m3": in_column * cell,
        "gap_m3": gap * cell,
        "gap_points": gap,
        "gap_where": where,
    }


def chair_padding():
    """How far the chair planner pads every layer box, a side."""
    stack.core()
    from rammp_curobo.config import load_planner_config

    return load_planner_config(stack.CHAIR_CONFIG)[0]["planner"]["world_padding"]


def column(assets):
    """The arm's carved column, (low, high) corners in base_link: the box
    around the arm's joint-independent spheres that the generator kept every
    person box out of."""
    generator = stack.generator()
    low, high = generator.generate(assets.data, assets.urdf)["person"]["column"]
    return np.multiply(low, generator.TICK), np.multiply(high, generator.TICK)


def _named(primitives):
    """A Counter of (link, kind), in words."""
    return ["%s (%s)" % p for p in sorted(primitives.elements())]


def cases(assets):
    """S0's cases, as stack.case: the registration (G0), then the stand-in in
    each posture with its back at each end of the backrest face's range
    (G1). They check the layer files the generator writes, whatever the
    chair's config lists."""
    m = scene.build(assets, None)
    layers = stack.generated_layers()
    platform = layers["platform"]
    whole = registration(m, platform)
    shrunk = registration(m, platform, SHRINK_M)
    urdf = urdf_primitives(assets)
    missing, extra = urdf - whole["primitives"], whole["primitives"] - urdf
    shortfall = "; ".join(
        "%s %s" % (what, ", ".join(_named(these)))
        for what, these in (("missing", missing), ("extra", extra))
        if these
    )
    out = [
        stack.case(
            "S0 registration",
            "MuJoCo's MEBot | platform, unpadded",
            "%d of %d samples of %s primitives outside; shrunk %g mm: %d%s"
            % (
                whole["outside"],
                whole["samples"],
                whole["geoms"] if shortfall else "the URDF's %d" % whole["geoms"],
                SHRINK_M * 1e3,
                shrunk["outside"],
                "; the URDF has %d: %s" % (sum(urdf.values()), shortfall)
                if shortfall
                else "",
            ),
            {"G0": whole["outside"] == 0 and shrunk["outside"] > 0 and not shortfall},
            geoms=whole["geoms"],
            missing=_named(missing),
            extra=_named(extra),
            samples=whole["samples"],
            outside=whole["outside"],
            outside_shrunk=shrunk["outside"],
        )
    ]
    person = layers["person"].obstacles
    padded, unpadded = bounds(person, chair_padding()), bounds(person)
    carved = column(assets)
    faces = sorted(assets.data["person"]["seat"]["backrest_face_x"])
    thresholds = stack.scenarios()["thresholds"]
    unscaled = humanoid_m3(assets)
    for posture in assets.postures["postures"]:
        for face in faces:
            geoms, scale = scene.standin(assets, posture, face)
            # a numpy float: a verdict compared from it would be a numpy
            # bool, which the result file's JSON cannot hold
            scale = float(scale)
            c = coverage(geoms, padded, carved)
            bare = coverage(geoms, unpadded, carved)
            # the whole humanoid, at the scale that seats the occupant
            expected = unscaled * scale**3
            off_pct = abs(c["body_m3"] / expected - 1) * 100
            sized = off_pct <= thresholds["body_volume_pct"]
            covered = c["covered_pct"] >= thresholds["covered_pct"]
            out.append(
                stack.case(
                    "S0 %s %.4f" % (posture, face),
                    "stand-in | person, padded",
                    "gap %.6f m3 (unpadded %.6f); of %.4f m3%s, %.2f %% covered%s, "
                    "%.5f in the column"
                    % (
                        c["gap_m3"],
                        bare["gap_m3"],
                        c["body_m3"],
                        ""
                        if sized
                        else " (not the humanoid's %.4f at x%.3f)" % (expected, scale),
                        c["covered_pct"],
                        "" if covered else " (under %g)" % thresholds["covered_pct"],
                        c["column_m3"],
                    ),
                    {"G1": c["gap_points"] == 0 and covered and sized},
                    posture=posture,
                    backrest_face_x=face,
                    scale=scale,
                    body_m3=c["body_m3"],
                    humanoid_m3=expected,
                    covered_pct=c["covered_pct"],
                    column_m3=c["column_m3"],
                    gap_m3=c["gap_m3"],
                    gap_where=c["gap_where"],
                    unpadded_gap_m3=bare["gap_m3"],
                )
            )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--menagerie",
        default=scene.MENAGERIE,
        help="a mujoco_menagerie checkout (default: %(default)s)",
    )
    parser.add_argument(
        "--humanoid",
        default=scene.HUMANOID,
        help="the DeepMind humanoid the stand-in is posed from",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        results = cases(scene.Assets(args.menagerie, args.humanoid))
    except (OSError, ValueError) as exc:
        print("check_layers: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    for case in results:
        print(stack.case_line(case))
    failed = sum(not case["passed"] for case in results)
    print("S0: %d cases, %d failed" % (len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
