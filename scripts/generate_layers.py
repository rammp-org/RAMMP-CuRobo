#!/usr/bin/env python3
"""Generate the MEBot obstacle layers from the pinned RAMMP Gen1.2 URDF
(available in Demo-Software; scripts/mebot_layers.yaml holds the pin and a
link, plus the to-do to move to the Gen1.5 specs).

The planner's collision world is boxes only (cuRobo v0.7.8 silently drops
everything else), so the wheelchair and its occupant reach it as box layers.
This script builds them on the CPU from scripts/mebot_layers.yaml and writes
core/rammp_curobo/configs/layer_mebot_{platform,person,floor}.yaml:

  * platform: the MEBot drive base. The URDF's collision primitives (boxes
    and cylinders, every joint at zero) are bounded exactly, placed in the
    arm's mount frame, arm_mount, and merged pairwise down to a box budget. A merge only
    ever grows a box, so the layer covers every primitive (asserted); what
    merging adds is measured exactly and held to a budget.
  * person: any seated occupant from the 5th-percentile woman to the
    95th-percentile man, plus a margin, as three segments carved around the
    arm's joint-independent spheres.
  * floor: a slab whose top is the URDF's lowest point.

No padded box of any layer may come near those fixed spheres, or the planner
sees every state as a collision; a layer that does is not written.

The URDF is not vendored: it is fetched by commit and verified by sha256.
After one outward snap onto a 0.1 mm grid all geometry is integers, and the
text is printed from them, so amd64 and arm64 write the same bytes and the
committed files can be diff-asserted (`make layers`, and a regeneration test).

Run, with this checkout's rammp_curobo the one that is imported (the arm's
spheres are read from its robot config; another checkout's is refused):
    python3 scripts/generate_layers.py            # report, then write the files
    python3 scripts/generate_layers.py --check    # write nothing; diff instead
"""

import argparse
import contextlib
import difflib
import hashlib
import http.client
import math
import os
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from rammp_curobo.config import PACKAGED_CONFIG_DIR

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "mebot_layers.yaml"

# The grid, in metres. Positions are printed in half ticks (5 decimals) and
# dims in ticks (4 decimals), so the data file's grid_m must be this.
TICK = 1e-4
# The snap's outward bias. FK goes through libm's sin and cos, which may
# differ by an ulp between amd64 and arm64; a bound sitting on a grid line
# would then floor or ceil to different ticks. Biased, it lands a whole tick
# out on both.
SNAP_BIAS = 1e-9
INCH = 0.0254  # metres

JOINT_TYPES = ("fixed", "revolute", "continuous", "prismatic")


def load_data(path=None):
    """The generator's inputs (default: scripts/mebot_layers.yaml)."""
    with open(path or DATA, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- pinned fetch
def _write_atomic(path, payload):
    """Replace `path` with `payload` in one step: no reader sees half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(partial, "wb") as f:
            f.write(payload)
        os.replace(partial, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise


def fetch_pinned(source, cache_dir=None, clone=None, offline=False):
    """The pinned file's bytes and where they came from: (bytes, origin).

    Sources in order: the cache, a local clone (`clone`, $RAMMP_DEMO_SOFTWARE,
    then ~/Demo-Software) and, unless `offline`, GitHub raw. Whatever serves
    the bytes, their sha256 is checked against `source["sha256"]`; verified
    bytes are cached. A clone or URL serving other bytes is a ValueError.
    When no source has the file, FileNotFoundError lists every one tried.
    """
    want = source["sha256"]
    cache_dir = cache_dir or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "rammp_curobo",
        "cad",
    )
    cached = os.path.join(cache_dir, want + ".urdf")
    tried, remarks = [], []
    try:
        with open(cached, "rb") as f:
            raw = f.read()
    except OSError as exc:
        tried.append("cache %s: %s" % (cached, exc.strerror))
    else:
        if hashlib.sha256(raw).hexdigest() == want:
            return raw, "cache %s" % cached
        tried.append("cache %s: sha256 mismatch, ignored" % cached)
        remarks.append("the cached file was ignored: sha256 mismatch")

    raw = None
    clone = (
        clone
        or os.environ.get("RAMMP_DEMO_SOFTWARE")
        or os.path.expanduser("~/Demo-Software")
    )
    blob = "%s:%s" % (source["commit"], source["path"])
    try:
        raw = subprocess.run(
            ["git", "-C", clone, "cat-file", "blob", blob],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        origin = "clone %s" % clone
    except subprocess.CalledProcessError as exc:
        reason = exc.stderr.decode(errors="replace").strip()
        tried.append("clone %s: %s" % (clone, reason))
    except (OSError, subprocess.TimeoutExpired) as exc:
        tried.append("clone %s: %s" % (clone, exc))

    if raw is None:
        url = "https://raw.githubusercontent.com/%s/%s/%s" % (
            source["repo"],
            source["commit"],
            source["path"],
        )
        if offline:
            tried.append("%s: not asked (offline)" % url)
        else:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    raw = response.read()
                origin = url
            except (OSError, http.client.HTTPException) as exc:
                tried.append("%s: %s" % (url, exc))
    if raw is None:
        raise FileNotFoundError(
            "pinned %s is not available; tried: %s" % (source["path"], "; ".join(tried))
        )

    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise ValueError(
            "sha256 mismatch from %s: got %s, pinned %s" % (origin, got, want)
        )
    try:
        _write_atomic(cached, raw)
    except OSError as exc:
        remarks.append("cache not written: %s" % exc)
    if remarks:
        origin += " (%s)" % "; ".join(remarks)
    return raw, origin


# ----------------------------------------------------------- the integer grid
def _snap_down(v):
    return math.floor((v - SNAP_BIAS) / TICK)


def _snap_up(v):
    return math.ceil((v + SNAP_BIAS) / TICK)


def _snap(lo, hi):
    """Bounds in metres as a box in ticks, (lo, hi), grown outward to the grid."""
    return tuple(_snap_down(v) for v in lo), tuple(_snap_up(v) for v in hi)


def _volume(box):
    lo, hi = box
    return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])


def _bbox(a, b):
    return (
        tuple(min(a[0][k], b[0][k]) for k in range(3)),
        tuple(max(a[1][k], b[1][k]) for k in range(3)),
    )


def _overlap(a, b):
    edges = [min(a[1][k], b[1][k]) - max(a[0][k], b[0][k]) for k in range(3)]
    return 0 if min(edges) <= 0 else edges[0] * edges[1] * edges[2]


def _contains(outer, inner):
    return all(
        outer[0][k] <= inner[0][k] and inner[1][k] <= outer[1][k] for k in range(3)
    )


def union_stats(boxes):
    """Exact (union volume, sum of volumes, most boxes over one point), ticks^3.

    Coordinate compression: the boxes' faces cut space into cells, each either
    wholly inside or wholly outside every box.
    """
    cuts = [sorted({b[side][k] for b in boxes for side in (0, 1)}) for k in range(3)]
    index = [{v: i for i, v in enumerate(c)} for c in cuts]
    count = np.zeros([len(c) - 1 for c in cuts], dtype=np.int64)
    for lo, hi in boxes:
        count[tuple(slice(index[k][lo[k]], index[k][hi[k]]) for k in range(3))] += 1
    edge = [np.diff(np.asarray(c, dtype=np.int64)) for c in cuts]
    cell = edge[0][:, None, None] * edge[1][None, :, None] * edge[2][None, None, :]
    return int((cell * (count > 0)).sum()), int((cell * count).sum()), int(count.max())


# ---------------------------------------------------------------- URDF and FK
def _transform(xyz, rpy):
    """A URDF origin as a 4x4 pose; fixed-axis rpy is Rz(yaw) Ry(pitch) Rx(roll).

    Plain floats, here and in _mul: no BLAS between the URDF and the snap.
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
        [-sp, cp * sr, cp * cr, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]


def _numbers(text, count, what):
    try:
        values = [float(v) for v in text.split()]
    except (AttributeError, ValueError):
        values = []
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError(
            "%s: expected %d finite number(s), got %r" % (what, count, text)
        )
    return values


def _origin(element, what):
    if element is None:
        return _transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    return _transform(
        _numbers(element.get("xyz", "0 0 0"), 3, what + " origin xyz"),
        _numbers(element.get("rpy", "0 0 0"), 3, what + " origin rpy"),
    )


def _link_poses(robot, mount_xyz):
    """{link name: (element, arm_mount pose)} with every joint held at zero."""
    links = {}
    for link in robot.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError("link %r is unnamed or defined twice" % name)
        links[name] = link

    parent_joint, children = {}, {}
    for joint in robot.findall("joint"):
        what = "joint %r" % joint.get("name")
        kind = joint.get("type")
        if kind not in JOINT_TYPES:
            raise ValueError(
                "%s: type %r is not supported, only %s"
                % (what, kind, ", ".join(JOINT_TYPES))
            )
        if joint.find("mimic") is not None:
            raise ValueError("%s: <mimic> is not supported" % what)
        if kind in ("revolute", "prismatic"):
            limit = joint.find("limit")
            lower, upper = (
                0.0
                if limit is None
                else _numbers(limit.get(k, "0"), 1, "%s limit" % what)[0]
                for k in ("lower", "upper")
            )
            if not lower <= 0.0 <= upper:
                raise ValueError(
                    "%s: 0 is outside its limits [%s, %s], and every joint is"
                    " held at zero" % (what, lower, upper)
                )
        ends = []
        for tag in ("parent", "child"):
            end = joint.find(tag)
            ends.append(None if end is None else end.get("link"))
            if ends[-1] not in links:
                raise ValueError(
                    "%s: its %s link %r is not defined" % (what, tag, ends[-1])
                )
        parent, child = ends
        if child in parent_joint:
            raise ValueError(
                "link %r is the child of two joints, %r and %r"
                % (child, parent_joint[child], joint.get("name"))
            )
        parent_joint[child] = joint.get("name")
        # At zero a joint is its origin, whatever its type and axis.
        children.setdefault(parent, []).append(
            (child, _origin(joint.find("origin"), what))
        )

    roots = [name for name in links if name not in parent_joint]
    if len(roots) != 1:
        raise ValueError(
            "the URDF must have exactly one root link, found %s" % (roots or "none")
        )
    # The root sits at -mount, so every pose below is already in arm_mount.
    poses = {roots[0]: _transform([-v for v in mount_xyz], [0.0, 0.0, 0.0])}
    reached = list(roots)
    while reached:
        parent = reached.pop()
        for child, origin in children.get(parent, ()):
            poses[child] = _mul(poses[parent], origin)
            reached.append(child)
    loose = sorted(set(links) - set(poses))
    if loose:
        raise ValueError(
            "links %s are in a joint loop, not under the root %r" % (loose, roots[0])
        )
    return {name: (link, poses[name]) for name, link in links.items()}


def _writes_plain(name):
    """Whether `name` can be a layer box name, written bare in a flow mapping."""
    try:
        return "/" not in name and yaml.safe_load("{name: %s}" % name) == {"name": name}
    except yaml.YAMLError:
        return False


def urdf_primitives(urdf_bytes, mount_xyz, perturb=0.0):
    """Every collision primitive of the URDF at all joints zero, in arm_mount.

    One dict per <collision>, in URDF order: `name` (the link's, or
    <link>_<i> when a link has several), `link`, `kind` ("box" with size
    [x, y, z], or "cylinder" with size [radius, length]), `pose` (4x4,
    arm_mount from the primitive), `lo` and `hi` (its exact axis-aligned
    bounds, metres) and `box` (those bounds snapped outward, ticks).

    arm_mount is the URDF's root frame moved by -`mount_xyz`. Anything that
    cannot be bounded exactly or held at zero is refused, naming the link or
    joint. `perturb` scales every bound by (1 + perturb): it stands in for
    another platform's libm when testing that the output does not change.
    """
    primitives = []
    links = _link_poses(ET.fromstring(urdf_bytes), mount_xyz)
    for link_name, (link, link_pose) in links.items():
        collisions = link.findall("collision")
        for index, collision in enumerate(collisions):
            what = "link %r collision %d" % (link_name, index)
            geometry = collision.find("geometry")
            shapes = [] if geometry is None else list(geometry)
            if len(shapes) != 1 or shapes[0].tag not in ("box", "cylinder"):
                found = ", ".join("<%s>" % s.tag for s in shapes) or "empty"
                raise ValueError(
                    "%s: %s geometry is not supported, only one <box> or one"
                    " <cylinder> (mesh is the stretch goal)" % (what, found)
                )
            pose = _mul(link_pose, _origin(collision.find("origin"), what))
            if shapes[0].tag == "box":
                size = _numbers(shapes[0].get("size"), 3, what + " box size")
                # |R| maps the half-size to the half-extent of its bounding
                # box, whatever the rotation (the CAD's origins all carry a
                # -1.6e-7 rad yaw).
                half = [
                    sum(abs(pose[i][j]) * size[j] / 2 for j in range(3))
                    for i in range(3)
                ]
            else:
                size = [
                    _numbers(shapes[0].get(k), 1, "%s cylinder %s" % (what, k))[0]
                    for k in ("radius", "length")
                ]
                radius, length = size
                # Exact circumscription: along axis i the cylinder's own axis
                # `a` reaches |a_i| L/2 and its end discs r sqrt(1 - a_i^2).
                # Boxing a meshed cylinder instead (cuRobo's get_obb_world
                # goes through a 32-gon) measures across the flats: 0.9 mm
                # short per side on a drive wheel.
                half = [
                    abs(pose[i][2]) * length / 2
                    + radius * math.sqrt(max(0.0, 1.0 - pose[i][2] * pose[i][2]))
                    for i in range(3)
                ]
            if min(size) <= 0:
                raise ValueError("%s: size must be positive, got %s" % (what, size))
            name = link_name if len(collisions) == 1 else "%s_%d" % (link_name, index)
            if not _writes_plain(name):
                raise ValueError(
                    "%s: %r cannot be written as a layer box name" % (what, name)
                )
            lo = [(pose[i][3] - half[i]) * (1 + perturb) for i in range(3)]
            hi = [(pose[i][3] + half[i]) * (1 + perturb) for i in range(3)]
            primitives.append(
                {
                    "name": name,
                    "link": link_name,
                    "kind": shapes[0].tag,
                    "size": size,
                    "pose": pose,
                    "lo": lo,
                    "hi": hi,
                    "box": _snap(lo, hi),
                }
            )
    if not primitives:
        raise ValueError("the URDF has no collision primitives")
    names = [p["name"] for p in primitives]
    for p in primitives:
        if names.count(p["name"]) > 1:
            raise ValueError(
                "link %r: the box name %r is used twice (a link with several"
                " collisions names them <link>_<i>)" % (p["link"], p["name"])
            )
    return primitives


# ------------------------------------------------------------------- keep-out
def keep_out_spheres(data):
    """The arm's joint-independent spheres in arm_mount: [(centre, radius, link)].

    arm_base_link's spheres never move. arm_shoulder_link's turn about the
    arm_joint_1 axis, so the ones ON that axis stay put too; one off it is refused, since
    it sweeps a ring this keep-out would not cover. Radii include the robot
    config's collision_sphere_buffer, as cuRobo's do. A sphere that is not
    finite, in the config or through arm_shoulder_link's origin, is refused too.

    The column is placed for a level mount: arm_mount -> arm_base_link
    (arm_base_joint) taken as the identity, as it is on the MEBot. A tilted
    mount moves the column in arm_mount; the planner's FK-based guard, which
    refuses a layer box touching the live column, covers that case.
    """
    keep_out = data["keep_out"]
    path = PACKAGED_CONFIG_DIR / keep_out["robot_config"]
    with open(path, encoding="utf-8") as f:
        kinematics = yaml.safe_load(f)["robot_cfg"]["kinematics"]
    buffer = float(kinematics["collision_sphere_buffer"])

    def listed(link):
        """The link's spheres in its own frame: (centre, radius with the buffer)."""
        for index, s in enumerate(kinematics["collision_spheres"][link]):
            c = [float(v) for v in s["center"]]
            radius = float(s["radius"]) + buffer
            # A NaN compares false with everything, so min() and max() pass
            # over it: the sphere would drop out of the person's column and of
            # every clearance.
            if not all(math.isfinite(v) for v in (*c, radius)):
                raise ValueError(
                    "%s: %s sphere %d is not finite: center %s, radius %s, buffer %s"
                    % (path, link, index, c, s["radius"], buffer)
                )
            yield c, radius

    spheres = [
        (tuple(c), radius, "arm_base_link") for c, radius in listed("arm_base_link")
    ]
    origin = keep_out["shoulder_link_origin"]
    # ... and so would every arm_shoulder_link sphere, placed by a pose that is
    # not finite.
    if not all(math.isfinite(v) for v in (*origin["xyz"], *origin["rpy"])):
        raise ValueError("keep_out shoulder_link_origin %s is not finite" % origin)
    pose = _transform(origin["xyz"], origin["rpy"])
    for c, radius in listed("arm_shoulder_link"):
        if c[0] != 0.0 or c[1] != 0.0:
            raise ValueError(
                "%s: the arm_shoulder_link sphere at %s is off the arm_joint_1 axis, so it"
                " is not joint-independent" % (path, c)
            )
        centre = tuple(
            sum(pose[i][k] * c[k] for k in range(3)) + pose[i][3] for i in range(3)
        )
        spheres.append((centre, radius, "arm_shoulder_link"))
    return spheres


def clearance(box, spheres, pad):
    """How far `box` (ticks), padded by `pad`, stays from the spheres, metres.

    The least of (distance from a sphere's centre to the padded box) minus its
    radius; a centre inside the box counts as minus its depth to the nearest
    face. NaN if a sphere is not a number, which no gate passes.
    """
    lo = [v * TICK - pad for v in box[0]]
    hi = [v * TICK + pad for v in box[1]]
    least = math.inf
    for centre, radius, _ in spheres:
        # min() and max() pass over a NaN: the sphere would count for nothing.
        if any(math.isnan(v) for v in (*centre, radius)):
            return math.nan
        gaps = [min(max(c, a), b) - c for c, a, b in zip(centre, lo, hi)]
        depth = min(min(c - a, b - c) for c, a, b in zip(centre, lo, hi))
        distance = -depth if depth > 0 else math.sqrt(sum(g * g for g in gaps))
        least = min(least, distance - radius)
    return least


# -------------------------------------------------------------------- platform
def merge(primitives, max_boxes, spheres, pad, margin):
    """Merge the primitives' boxes down to `max_boxes`: [{name, box, members}].

    Boxes inside another box are absorbed first, which costs nothing. Then,
    greedily, the pair whose bounding box adds the least volume that neither
    covers is merged. Pairwise cost, not the empty volume added to the whole
    layer: that objective lets boxes pile up on each other, and cuRobo sums
    the collision cost of every box a sphere is in. A merge whose padded box
    would come within `margin` of an arm keep-out sphere is never made.

    Deterministic: ties go to the smaller bounding box, then to the first pair
    in primitive-name order. Each box is named after its largest member.
    """
    volume = {p["name"]: _volume(p["box"]) for p in primitives}
    ordered = sorted(primitives, key=lambda p: p["name"])
    clusters = [{"box": p["box"], "members": [p["name"]]} for p in ordered]

    def absorb():
        i = 0
        while i < len(clusters):
            outer = clusters[i]
            for j, inner in enumerate(clusters):
                if inner is not outer and _contains(outer["box"], inner["box"]):
                    outer["members"] += inner["members"]
                    del clusters[j]
                    i = -1  # the list changed under the scan: start over
                    break
            i += 1

    absorb()
    # "not <=" where ">" would read better: a budget that is NaN is never met.
    while not len(clusters) <= max_boxes:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i]["box"], clusters[j]["box"]
                both = _bbox(a, b)
                added = _volume(both) - (_volume(a) + _volume(b) - _overlap(a, b))
                key = (added, _volume(both), i, j)
                if best is not None and key >= best[0]:
                    continue
                if clearance(both, spheres, pad) >= margin:
                    best = (key, both)
        if best is None:
            raise ValueError(
                "cannot merge %d boxes down to %s without touching the arm's"
                " keep-out" % (len(clusters), max_boxes)
            )
        (_, _, i, j), both = best
        clusters[i] = {
            "box": both,
            "members": clusters[i]["members"] + clusters[j]["members"],
        }
        del clusters[j]
        absorb()

    for cluster in clusters:
        cluster["members"].sort()
        cluster["name"] = max(cluster["members"], key=lambda name: (volume[name], name))
    return sorted(clusters, key=lambda c: c["name"])


def _certify(primitives, boxes):
    """Raise unless every primitive's snapped bounds lie inside its box.

    The integer half of the zero-under-coverage certificate: a primitive's
    solid is inside its exact bounds, those are inside the snapped bounds, and
    this checks the snapped bounds against the box that took the primitive in.
    """
    owner = {name: box for box in boxes for name in box["members"]}
    for p in primitives:
        box = owner.get(p["name"])
        if box is None:
            raise ValueError("containment: primitive %r is in no box" % p["name"])
        if not _contains(box["box"], p["box"]):
            raise ValueError(
                "containment: primitive %r is not inside its box %r"
                % (p["name"], box["name"])
            )


def _platform(data, primitives, spheres):
    budget = data["platform"]
    boxes = merge(
        primitives,
        budget["max_boxes"],
        spheres,
        data["padding_assumed_m"],
        data["keep_out_margin_m"],
    )
    _certify(primitives, boxes)
    primitive_union, _, _ = union_stats([p["box"] for p in primitives])
    union, total, max_stack = union_stats([b["box"] for b in boxes])
    # Integer round-half-up to 0.1 %, so the number written is the one gated.
    over = union - primitive_union
    over_pct = (over * 1000 + primitive_union // 2) // primitive_union / 10
    # "not <=" here too: a NaN budget passes nothing.
    if not over_pct <= budget["max_over_pct"]:
        raise ValueError(
            "platform: %d boxes add %.1f %% to the primitives' bounding boxes,"
            " above max_over_pct %s" % (len(boxes), over_pct, budget["max_over_pct"])
        )
    return {
        "primitives": primitives,
        "boxes": boxes,
        "primitive_union": primitive_union,
        "union": union,
        "total": total,
        "max_stack": max_stack,
        "over_pct": over_pct,
        # What boxing the wheels adds over the true shapes: (4 - pi) r^2 L each.
        "cylinder_boxing_m3": sum(
            (4.0 - math.pi) * p["size"][0] * p["size"][0] * p["size"][1]
            for p in primitives
            if p["kind"] == "cylinder"
        ),
    }


# ---------------------------------------------------------------------- person
def _person_segments(data):
    """The occupant's three segments in the mebot frame, metres: (name, lo, hi).

    Each size is the one, over every occupant the layer holds, that puts the
    faces it sets furthest out (scripts/mebot_layers.yaml says which): the
    back of the knees takes the shortest thigh, every other face the largest
    size. The margin then goes on every face.
    """
    person = data["person"]
    size = {key: row["inches"] * INCH for key, row in person["ansur_in"].items()}
    seat, m = person["seat"], person["margin_m"]
    # A size that is not positive, or a negative margin, shrinks a segment and
    # may leave it with volume, which is all the gate on the boxes looks at.
    # An infinite one cannot be snapped to the grid, and float's OverflowError
    # would name no key. "not 0 < x < inf": a NaN is refused too, which a
    # max() below would pass over.
    for key, row in person["ansur_in"].items():
        if not 0 < row["inches"] < math.inf:
            raise ValueError(
                "person ansur_in %s: size must be a finite number above 0,"
                " got %s inches" % (key, row["inches"])
            )
    if not 0 <= m < math.inf:
        raise ValueError("person margin_m must be a finite number >= 0, got %s" % m)
    pan, centre = seat["pan_top_z"], seat["centreline_y"]
    # The backrest face is somewhere in this range. The chest, the knees and
    # the feet take its front end and the back of the knees its rear end, so
    # the segments hold the occupant wherever in the range the face is.
    # Sorted: written front first, the range would shrink every segment.
    back_min, back_max = sorted(seat["backrest_face_x"])
    shoulder = size["shoulder_breadth_bideltoid"] / 2
    hip = size["hip_breadth"] / 2
    chest_front = back_max + max(size["abdominal_depth"], size["chest_depth"]) + m
    # thighs, or forearms resting on them or on the armrests
    lap_height = max(
        size["thigh_clearance"], size["sitting_elbow_height"] + size["hand_breadth"]
    )
    return [
        (
            "torso_head",
            [seat["backrest_rear_x"] - m, centre - shoulder - m, pan - m],
            [chest_front, centre + shoulder + m, pan + size["sitting_height"] + m],
        ),
        (
            "lap",
            [chest_front, centre - shoulder - m, pan - m],
            [
                back_max + size["buttock_knee_length"] + m,
                centre + shoulder + m,
                pan + lap_height + m,
            ],
        ),
        # Lower legs and feet: below the pan, from the back of the knee to the
        # front of the knee plus a foot. The shorter the thigh, the nearer the
        # seat the shins hang, so the back face is the shortest thigh's. Its
        # top is the others' bottom face.
        (
            "legs",
            [
                back_min + size["buttock_popliteal_length"] - m,
                centre - hip - m,
                pan - size["popliteal_height"] - m,
            ],
            [
                back_max + size["buttock_knee_length"] + size["foot_length"] + m,
                centre + hip + m,
                pan - m,
            ],
        ),
    ]


def subtract(box, cut):
    """`box` minus `cut` as disjoint boxes: slabs in x, then y, then z."""
    lo, hi = list(box[0]), list(box[1])
    if any(cut[1][k] <= lo[k] or cut[0][k] >= hi[k] for k in range(3)):
        return [box]
    pieces = []
    for k in range(3):
        if cut[0][k] > lo[k]:
            below = list(hi)
            below[k] = cut[0][k]
            pieces.append((tuple(lo), tuple(below)))
            lo[k] = cut[0][k]
        if cut[1][k] < hi[k]:
            above = list(lo)
            above[k] = cut[1][k]
            pieces.append((tuple(above), tuple(hi)))
            hi[k] = cut[1][k]
    return pieces


def _person(data, spheres):
    mount = data["mount"]["arm_base_in_mebot"]["position"]
    reach = data["padding_assumed_m"] + data["keep_out_margin_m"]
    # The column no person box may enter: the fixed spheres' bounding box, out
    # to where a padded box would still clear them by the margin. A person
    # cannot occupy the arm's base, and a padded box there would make every
    # state a collision. It stops at the column's top: a full-height corridor
    # would under-size the person exactly where the arm works.
    column = _snap(
        [min(c[k] - r for c, r, _ in spheres) - reach for k in range(3)],
        [max(c[k] + r for c, r, _ in spheres) + reach for k in range(3)],
    )
    envelope, boxes = [], []
    for name, lo, hi in _person_segments(data):
        whole = _snap(
            [v - o for v, o in zip(lo, mount)], [v - o for v, o in zip(hi, mount)]
        )
        envelope.append((name, whole))
        pieces = subtract(whole, column)
        for k, piece in enumerate(pieces):
            piece_name = name if len(pieces) == 1 else "%s_%d" % (name, k)
            boxes.append({"name": piece_name, "box": piece})
    envelope_union, _, _ = union_stats([box for _, box in envelope])
    union, total, _ = union_stats([b["box"] for b in boxes])
    return {
        "column": column,
        "envelope": envelope,
        "boxes": sorted(boxes, key=lambda b: b["name"]),
        "envelope_union": envelope_union,
        "union": union,
        "total": total,
    }


# ----------------------------------------------------------------------- floor
def _floor(data, primitives):
    floor = data["floor"]
    half = round(floor["half_extent_m"] / TICK)
    # Rounded UP: outward for the floor solid, whose top this is.
    top = _snap_up(min(p["lo"][2] for p in primitives))
    bottom = top - round(floor["thickness_m"] / TICK)
    return {
        "boxes": [{"name": "floor", "box": ((-half, -half, bottom), (half, half, top))}]
    }


# ------------------------------------------------------------------------ text
def _decimal(n, digits):
    """The integer `n`, in units of 10^-digits, as an exact decimal string."""
    whole, fraction = divmod(abs(n), 10**digits)
    return "%s%d.%0*d" % ("-" if n < 0 else "", whole, digits, fraction)


def _m3(ticks3):
    """ticks^3 (1e-12 m^3) as m^3 with 6 decimals, rounded half up in integers."""
    return _decimal((ticks3 + 500000) // 1000000, 6)


def _flow(mapping, *keys):
    """`keys` of a data mapping as a YAML flow mapping, in the order given.

    The order is fixed here, not taken from the data file, so the same inputs
    always print the same bytes. (A list of numbers prints as YAML already.)
    """
    return "{%s}" % ", ".join("%s: %s" % (key, mapping[key]) for key in keys)


def _box_line(box):
    lo, hi = box["box"]
    line = "  - {name: %s, position: [%s], dims: [%s]}" % (
        box["name"],
        # (lo + hi) / 2 ticks is (lo + hi) * 5 units of 0.01 mm: exact
        ", ".join(_decimal((a + b) * 5, 5) for a, b in zip(lo, hi)),
        ", ".join(_decimal(b - a, 4) for a, b in zip(lo, hi)),
    )
    # A `members:` key would be refused by the layer loader, so the links a
    # box stands for ride in a comment.
    others = [name for name in box.get("members", ()) if name != box["name"]]
    return line + ("  # + " + " ".join(others) if others else "")


def _layer_text(layer, excludable, about, meta, boxes):
    lines = [
        "# GENERATED by scripts/generate_layers.py from scripts/mebot_layers.yaml"
        " -- do not hand-edit; `make layers`.",
        "# " + about,
        "# Metres in the arm's mount frame, arm_mount, UNPADDED: the planner adds"
        " world_padding per side.",
        "layer: %s" % layer,
        "excludable: %s" % ("true" if excludable else "false"),
        "meta:",
    ]
    lines += ["  " + line for line in meta] + ["obstacles:"]
    lines += [_box_line(box) for box in boxes]
    return "\n".join(lines) + "\n"


def _pin_meta(data):
    keys = ("repo", "commit", "path", "sha256")
    return ["source:"] + ["  %s: %s" % (k, data["source"][k]) for k in keys]


def _frame_meta(data):
    mount = data["mount"]["arm_base_in_mebot"]
    return [
        "arm_base_in_mebot: %s" % _flow(mount, "position", "rpy_deg"),
        "grid_m: %s" % data["grid_m"],
        "padding_assumed_m: %s" % data["padding_assumed_m"],
    ]


def _platform_text(data, layer):
    meta = _pin_meta(data) + _frame_meta(data)
    meta += [
        "joints: all zero",
        "keep_out_margin_m: %s" % data["keep_out_margin_m"],
        "budget: %s" % _flow(data["platform"], "max_boxes", "max_over_pct"),
        "coverage:",
        "  primitives: %d" % len(layer["primitives"]),
        "  under_m3: 0.000000  # certificate: every primitive's bounding box is"
        " inside its box",
        "  primitives_bbox_union_m3: %s" % _m3(layer["primitive_union"]),
        "  layer_m3: %s" % _m3(layer["union"]),
        "  over_m3: %s" % _m3(layer["union"] - layer["primitive_union"]),
        "  over_pct: %.1f" % layer["over_pct"],
        "  cylinder_boxing_m3: %.6f  # analytic: (4 - pi) r^2 L summed over the"
        " cylinders" % layer["cylinder_boxing_m3"],
        "  overlap_m3: %s" % _m3(layer["total"] - layer["union"]),
        "  max_stack: %d" % layer["max_stack"],
        "keep_out_clearance_m: %.4f" % layer["clearance_m"],
    ]
    about = (
        "The MEBot drive base: the pinned URDF's collision primitives, merged"
        " into boxes."
    )
    return _layer_text("platform", False, about, meta, layer["boxes"])


def _person_text(data, layer):
    person = data["person"]
    rows = sorted(person["ansur_in"].items(), key=lambda item: item[1]["row"])
    seat = ("pan_top_z", "backrest_face_x", "backrest_rear_x", "centreline_y")
    meta = _frame_meta(data)
    meta += ["keep_out_margin_m: %s" % data["keep_out_margin_m"], "ansur_in:"]
    meta += [
        "  %s: %s" % (key, _flow(row, "row", "sex", "percentile", "inches"))
        for key, row in rows
    ]
    meta += [
        "margin_m: %s" % person["margin_m"],
        "seat: %s" % _flow(person["seat"], *seat),
        "volumes:",
        "  envelope_m3: %s" % _m3(layer["envelope_union"]),
        "  layer_m3: %s" % _m3(layer["union"]),
        "  carved_m3: %s" % _m3(layer["envelope_union"] - layer["union"]),
        "  overlap_m3: %s" % _m3(layer["total"] - layer["union"]),
        "keep_out_clearance_m: %.4f" % layer["clearance_m"],
    ]
    about = (
        "The MEBot's occupant: any seated adult from the ANSUR II 5th-percentile"
        " woman to the 95th-percentile man, plus a margin, carved around the"
        " arm's fixed spheres."
    )
    return _layer_text("person", True, about, meta, layer["boxes"])


def _floor_text(data, layer):
    meta = _pin_meta(data) + _frame_meta(data)
    meta += [
        "joints: all zero",
        "floor: %s" % _flow(data["floor"], "half_extent_m", "thickness_m"),
        "keep_out_clearance_m: %.4f" % layer["clearance_m"],
    ]
    about = (
        "The ground under the MEBot: a slab whose top is the pinned URDF's"
        " lowest point."
    )
    return _layer_text("floor", False, about, meta, layer["boxes"])


# -------------------------------------------------------------------- generate
def generate(data, urdf_bytes, perturb=0.0):
    """The three layer files as text, with the numbers behind the report.

    Returns {"files": {output path: text}, "platform": ..., "person": ...,
    "floor": ...}. Each layer has `boxes` ([{name, box}], ticks, sorted by
    name) and `clearance_m` (to the arm's fixed spheres, padded); volumes are
    exact, in ticks^3. platform adds `primitives`, `primitive_union`, `union`,
    `total`, `max_stack`, `over_pct` and `cylinder_boxing_m3`; person adds
    `column`, `envelope` ([(segment, box)] before the carve), `envelope_union`,
    `union` and `total`.

    Every gate runs here, and any failure is a ValueError: a result may be
    written as it is. `perturb` is passed to urdf_primitives.
    """
    digest = hashlib.sha256(urdf_bytes).hexdigest()
    if digest != data["source"]["sha256"]:
        raise ValueError(
            "the URDF's sha256 is %s, not the pinned %s"
            % (digest, data["source"]["sha256"])
        )
    if data["grid_m"] != TICK:
        raise ValueError(
            "grid_m %s is not supported: the layer files are printed on a %s m"
            " grid" % (data["grid_m"], TICK)
        )
    # Below zero it would shrink every box before its clearance is measured,
    # and the column carved out of the person; infinite, it overflows the
    # grid. "not 0 <= x < inf": a NaN is refused too.
    if not 0 <= data["padding_assumed_m"] < math.inf:
        raise ValueError(
            "padding_assumed_m must be a finite number >= 0, got %s"
            % data["padding_assumed_m"]
        )
    # Below zero the keep-out gate passes a padded box inside the arm's fixed
    # spheres: data would switch it off. "not >=": a NaN is refused too.
    if not data["keep_out"]["min_clearance_m"] >= 0:
        raise ValueError(
            "keep_out min_clearance_m must not be negative, got %s"
            % data["keep_out"]["min_clearance_m"]
        )
    mount = data["mount"]["arm_base_in_mebot"]
    if any(mount["rpy_deg"]):
        raise ValueError(
            "mount rpy_deg %s is not supported: the boxes are axis-aligned in"
            " arm_mount, so the arm must be mounted square to the chair"
            % mount["rpy_deg"]
        )
    # The arm's spheres come from the rammp_curobo that is imported, and a
    # host may have an editable install of another checkout: the layers would
    # be gated against that tree's robot config and written to this one. An
    # installed package (the image) is under no core/ and is no checkout.
    package = PACKAGED_CONFIG_DIR.parents[1]
    if package.name == "core" and package != REPO / "core":
        raise ValueError(
            "rammp_curobo is imported from %s, not this checkout (%s); run with"
            " PYTHONPATH=%s" % (package, REPO / "core", REPO / "core")
        )

    spheres = keep_out_spheres(data)
    primitives = urdf_primitives(urdf_bytes, mount["position"], perturb)
    result = {
        "platform": _platform(data, primitives, spheres),
        "person": _person(data, spheres),
        "floor": _floor(data, primitives),
    }
    pad, gate = data["padding_assumed_m"], data["keep_out"]["min_clearance_m"]
    for name, layer in result.items():
        # The seat's anchors and the floor's sizes are data: a backrest whose
        # rear is in front of the chest, or a negative thickness, turns a box
        # inside out, and a zero leaves it flat.
        for box in layer["boxes"]:
            dims = [hi - lo for lo, hi in zip(*box["box"])]
            if min(dims) <= 0:
                raise ValueError(
                    "layer %s: box %r would have dims [%s]; a box without volume"
                    " is one cuRobo never sees"
                    % (name, box["name"], ", ".join(_decimal(d, 4) for d in dims))
                )
        layer["clearance_m"], nearest = min(
            (clearance(box["box"], spheres, pad), box["name"]) for box in layer["boxes"]
        )
        # "not >=": a NaN gate passes nothing.
        if not layer["clearance_m"] >= gate:
            raise ValueError(
                "layer %s: box %r, padded by %s m, clears the arm's fixed spheres"
                " by %.1f mm; at least %.1f mm is required"
                % (name, nearest, pad, layer["clearance_m"] * 1e3, gate * 1e3)
            )

    texts = {"platform": _platform_text, "person": _person_text, "floor": _floor_text}
    result["files"] = {}
    for name, text_of in texts.items():
        text = text_of(data, result[name])
        if not text.isascii():
            raise ValueError("layer %s: the file would hold non-ASCII text" % name)
        result["files"][data["outputs"][name]] = text
    return result


# ------------------------------------------------------------------------- CLI
def _report(data, result, origin):
    source = data["source"]
    platform, person, floor = result["platform"], result["person"], result["floor"]
    kinds = [p["kind"] for p in platform["primitives"]]
    lines = [
        "source: %s" % origin,
        "  %s@%s %s" % (source["repo"], source["commit"], source["path"]),
        "keep-out: the arm's fixed spheres from %s; no padded box within %.1f mm"
        % (
            PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"],
            data["keep_out"]["min_clearance_m"] * 1e3,
        ),
        "platform: %d primitives (%d box, %d cylinder) -> %d boxes (budget %d)"
        % (
            len(kinds),
            kinds.count("box"),
            kinds.count("cylinder"),
            len(platform["boxes"]),
            data["platform"]["max_boxes"],
        ),
    ]
    for box in platform["boxes"]:
        others = [name for name in box["members"] if name != box["name"]]
        lines.append(
            "  %s%s" % (box["name"], " + " + " ".join(others) if others else "")
        )
    lines += [
        "  coverage: under 0 (certificate); over %s m3 = %.1f %% of the primitives'"
        " bounding boxes (budget %s %%)"
        % (
            _m3(platform["union"] - platform["primitive_union"]),
            platform["over_pct"],
            data["platform"]["max_over_pct"],
        ),
        "  overlap %s m3, max stack %d; cylinder boxing %.6f m3"
        % (
            _m3(platform["total"] - platform["union"]),
            platform["max_stack"],
            platform["cylinder_boxing_m3"],
        ),
        "  keep-out: %.1f mm" % (platform["clearance_m"] * 1e3),
        "person: %d boxes; envelope %s m3, layer %s m3, carved %s m3, overlap %s m3"
        % (
            len(person["boxes"]),
            _m3(person["envelope_union"]),
            _m3(person["union"]),
            _m3(person["envelope_union"] - person["union"]),
            _m3(person["total"] - person["union"]),
        ),
        "  keep-out: %.1f mm" % (person["clearance_m"] * 1e3),
        "floor: top at %s m" % _decimal(floor["boxes"][0]["box"][1][2], 4),
        "  keep-out: %.1f mm" % (floor["clearance_m"] * 1e3),
    ]
    return "\n".join(lines)


def _diff(path, text):
    """A unified diff from the file at `path` to `text`; empty if they match."""
    try:
        with open(path, "rb") as f:
            current = f.read()
    except FileNotFoundError:
        current = b""
    if current == text.encode("ascii"):
        return ""
    return "".join(
        difflib.unified_diff(
            current.decode("utf-8", "replace").splitlines(keepends=True),
            text.splitlines(keepends=True),
            str(path),
            "%s (regenerated)" % path,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the MEBot obstacle layers from the pinned URDF."
    )
    parser.add_argument(
        "--data", default=str(DATA), help="inputs (default: %(default)s)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 with a diff if the files are not what would"
        " be written",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never ask GitHub: the cache or a local clone must have the URDF",
    )
    parser.add_argument(
        "--clone",
        help="a Demo-Software clone (default: $RAMMP_DEMO_SOFTWARE, then"
        " ~/Demo-Software)",
    )
    parser.add_argument(
        "--cache",
        help="where verified URDFs are kept (default:"
        " ${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/cad)",
    )
    args = parser.parse_args(argv)
    try:
        data = load_data(args.data)
        urdf, origin = fetch_pinned(
            data["source"], cache_dir=args.cache, clone=args.clone, offline=args.offline
        )
        result = generate(data, urdf)
    except (OSError, ValueError) as exc:
        print("generate_layers: %s" % exc, file=sys.stderr)
        return 1
    print(_report(data, result, origin))

    files = {REPO / path: text for path, text in result["files"].items()}
    if args.check:
        diffs = [_diff(path, text) for path, text in files.items()]
        sys.stdout.writelines(diffs)
        return 1 if any(diffs) else 0
    for path, text in files.items():
        _write_atomic(path, text.encode("ascii"))
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
