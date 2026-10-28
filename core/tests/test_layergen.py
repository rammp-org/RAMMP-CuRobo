"""The layer generator (scripts/generate_layers.py), offline and CPU only.

The generator's logic is exercised on synthetic URDFs and a throwaway git
clone, so these tests need no network, no cache and no Demo-Software checkout.
The committed layer files are checked too: the person byte for byte and
against every occupant of its range (it needs no URDF), and every layer
against the packaged robot config here, and against the pinned URDF wherever
it can be had (in CI, from GitHub).
"""

import bisect
import hashlib
import http.client
import importlib.util
import io
import itertools
import math
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import yaml

from rammp_curobo.config import PACKAGED_CONFIG_DIR
from rammp_curobo.scene import Scene, load_layer
from rammp_curobo.world import compose_cuboids, point_box_distance

REPO = Path(__file__).resolve().parents[2]


def _load_generator():
    # scripts/ is not a package: the generator is loaded from its path, the way
    # the sim loads it.
    spec = importlib.util.spec_from_file_location(
        "generate_layers", REPO / "scripts" / "generate_layers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

QUARTER = "1.5707963267948966"
X, Y, Z = 0, 1, 2
LO, HI = 0, 1
SEGMENTS = ("hips", "torso", "head", "thighs", "forearms", "legs")
# The occupants the person layer holds: each ANSUR II size, in
# inches, from the smaller sex's 5th percentile to the larger sex's 95th, with
# its row in the NCSU table.
OCCUPANTS_IN = {
    "sitting_height": (31.61, 38.46),  # row 8
    "sitting_elbow_height": (7.44, 11.46),  # row 11
    "thigh_clearance": (5.71, 8.15),  # row 12
    "buttock_knee_length": (21.30, 26.34),  # row 13
    "buttock_popliteal_length": (17.36, 21.57),  # row 14
    "popliteal_height": (13.78, 18.54),  # row 16
    "shoulder_breadth_bideltoid": (15.98, 22.32),  # row 17
    "hip_breadth": (12.13, 15.75),  # row 19
    "chest_depth": (8.07, 11.73),  # row 20
    "abdominal_depth": (7.24, 12.60),  # row 21
    "elbow_fingertip_length": (15.91, 20.47),  # row 23
    "head_length": (7.01, 8.31),  # row 26
    "head_breadth": (5.51, 6.42),  # row 27
    "hand_breadth": (2.83, 3.78),  # row 29
    "foot_length": (8.90, 11.54),  # row 30
}
# The envelope the layer is built for: each size at the end of its range that
# puts its face furthest out. That is the 95th percentile, but for the back of
# the knees, which the shortest thigh brings nearest the seat.
OCCUPANT_IN = {key: high for key, (_, high) in OCCUPANTS_IN.items()}
OCCUPANT_IN["buttock_popliteal_length"] = OCCUPANTS_IN["buttock_popliteal_length"][0]
# ... and three faces the small end puts furthest out (the chin, which the
# torso and the head meet at, and the thighs' top under the forearms)
for _key in (
    "sitting_height",
    "head_length",
    "thigh_clearance",
    "sitting_elbow_height",
):
    OCCUPANT_IN[_key + "_low"] = OCCUPANTS_IN[_key][0]


def _inputs():
    """The generator's inputs, with the geometry the layers were designed for:
    the RAMMS mount, 0.02 m of padding and the ANSUR II occupant range.

    scripts/mebot_layers.yaml gives only their shape. It is where a measured
    mount or a tuned size goes, and such an edit must not turn these tests red.
    """
    data = gen.load_data()
    data["mount"]["arm_base_in_mebot"] = {
        "position": [0.505, -0.265, 0.376],
        "rpy_deg": [0, 0, 0],
    }
    data["mount"]["arm_base_joint"] = {"rpy_deg": [0, 0, 0]}
    data["padding_assumed_m"] = 0.02
    data["keep_out_margin_m"] = 0.005
    data["keep_out"]["min_clearance_m"] = 0.004
    data["platform"] = {"max_boxes": 16, "max_over_pct": 15.0}
    person = data["person"]
    for key, inches in OCCUPANT_IN.items():
        person["ansur_in"][key]["inches"] = inches
    person["margin_m"] = 0.05
    # the RAMMS mount is on the chair's right
    person["arm_side"] = -1
    person["seat"] = {
        "pan_top_z": 0.4705,
        "backrest_face_x": [0.1587, 0.2733],
        "backrest_rear_x": -0.1502,
        "centreline_y": 0,
    }
    # The RAMMS seat as the RAMMS model has it, its armrest on the left, across
    # the chair from that mount's arm
    data["seat"] = {
        "keep_out_margin_m": 0.035,
        "boxes": {
            "base": [[0.1587, -0.2950, 0.2177], [0.7011, 0.2962, 0.4705]],
            "backrest": [[-0.1502, -0.2950, 0.4705], [0.2733, 0.2962, 0.8497]],
            "armrest": [[0.2192, 0.2237, 0.5925], [0.7334, 0.3679, 0.7953]],
        },
    }
    data["floor"] = {"half_extent_m": 1.5, "thickness_m": 0.1}
    return data


@pytest.fixture
def data():
    return _inputs()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No network, and neither the user's cache nor their Demo-Software clone."""

    def no_network(*args, **kwargs):
        raise AssertionError("the network must not be touched")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("RAMMP_DEMO_SOFTWARE", str(tmp_path / "no-clone"))
    return tmp_path


# --------------------------------------------------------------- URDF builders
def _origin(xyz, rpy):
    given = [(k, v) for k, v in (("xyz", xyz), ("rpy", rpy)) if v is not None]
    return "<origin%s/>" % "".join(' %s="%s"' % kv for kv in given) if given else ""


def _shape(xml, xyz=None, rpy=None):
    origin = _origin(xyz, rpy)
    return "<collision>%s<geometry>%s</geometry></collision>" % (origin, xml)


def _box(size, xyz=None, rpy=None):
    return _shape('<box size="%s"/>' % size, xyz, rpy)


def _cylinder(radius, length, xyz=None, rpy=None):
    return _shape('<cylinder radius="%s" length="%s"/>' % (radius, length), xyz, rpy)


def _link(name, *collisions):
    return '<link name="%s">%s</link>' % (name, "".join(collisions))


def _joint(name, kind, parent, child, xyz=None, rpy=None, extra=""):
    ends = '<parent link="%s"/><child link="%s"/>' % (parent, child)
    body = _origin(xyz, rpy) + ends + extra
    return '<joint name="%s" type="%s">%s</joint>' % (name, kind, body)


def _limit(lower=None, upper=None):
    given = [(k, v) for k, v in (("lower", lower), ("upper", upper)) if v is not None]
    ends = "".join(' %s="%s"' % kv for kv in given)
    return '<limit%s effort="1" velocity="1"/>' % ends


def _urdf(*elements):
    body = "\n".join(elements)
    return ('<?xml version="1.0"?>\n<robot name="t">\n%s\n</robot>\n' % body).encode()


CUBE = _box("0.1 0.1 0.1")
# One 0.1 m cube at the URDF origin: at the real mount it is far from the arm.
FAR_BOX = _urdf(_link("chassis", CUBE))


def _generate(data, urdf, **kwargs):
    data["source"]["sha256"] = hashlib.sha256(urdf).hexdigest()
    return gen.generate(data, urdf, **kwargs)


def _obstacle_lines(text):
    lines = text.splitlines()
    return lines[lines.index("obstacles:") + 1 :]


def _box_lines(*rows):
    return ["  - {name: %s, position: [%s], dims: [%s]}" % row for row in rows]


def _clone(repo, path, content):
    """A git repository holding `content` at `path` in `commit`: (repo, commit).

    The clone has moved on since, as a developer's will have: at HEAD the file
    holds other bytes.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # the user's git config (hooks, signing, autocrlf) must not shape the blob
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    identity = ["-c", "user.name=t", "-c", "user.email=t@t"]

    def git(*args):
        command = ["git", "-C", str(repo), *identity, *args]
        done = subprocess.run(command, check=True, capture_output=True, env=env)
        return done.stdout.decode().strip()

    (repo / path).parent.mkdir(parents=True)
    (repo / path).write_bytes(content)
    git("init", "-q")
    git("add", path)
    git("commit", "-q", "-m", "model")
    commit = git("rev-parse", "HEAD")
    (repo / path).write_bytes(content + b"<!-- edited since -->\n")
    git("commit", "-q", "-a", "-m", "model, edited")
    return repo, commit


# ------------------------------------------------------------------------ FK
def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    about_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    about_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    about_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


def test_fk_and_mount_on_a_synthetic_urdf(data):
    urdf = _urdf(
        _link(
            "base",
            _box("0.2 0.4 0.6", xyz="0.1 0.2 0.3"),
            _box("0.02 0.02 0.02", xyz="0.00003 0 -1"),
        ),
        # no <origin> on the collision: the link frame itself
        _link("arm", _box("0.2 0.4 0.6")),
        _joint("yaw", "fixed", "base", "arm", xyz="1 0 0", rpy="0 0 " + QUARTER),
        _link("wheel", _cylinder("0.1", "0.04", xyz="0 0 0.25")),
        # revolute without an <axis>: at zero only the joint's origin acts
        _joint(
            "swing",
            "revolute",
            "arm",
            "wheel",
            xyz="0.5 0 0",
            rpy=QUARTER + " 0 0",
            extra=_limit(-1, 1),
        ),
        # roll AND yaw: Rz*Ry*Rx sends x->y, y->z, z->x; Rx*Ry*Rz would not
        _link("rod", _box("0.02 0.04 0.06", rpy="%s 0 %s" % (QUARTER, QUARTER))),
        # no <limit>: lower and upper default to 0, which zero satisfies
        _joint(
            "slide",
            "prismatic",
            "wheel",
            "rod",
            xyz="0 0 0.1",
            extra='<axis xyz="0 0 1"/>',
        ),
        _link("mast", _box("0.2 0.4 0.6", xyz="0 0 0.5")),
        _joint(
            "spin",
            "continuous",
            "base",
            "mast",
            xyz="-1 0 0",
            rpy="0 %s 0" % QUARTER,
            extra='<axis xyz="0 1 0"/>',
        ),
    )
    # Hand-computed in the URDF's root frame. arm: yawed a quarter turn at
    # (1, 0, 0). wheel: its frame is at (1, 0.5, 0) with z along root x, so the
    # cylinder sits at (1.25, 0.5, 0) lying along x. rod: 0.1 further along x,
    # its box axes cycled twice (x->z, y->x, z->y). mast: pitched a quarter
    # turn at (-1, 0, 0), so its z offset points along root x.
    want = {
        "base_0": ("base", "box", (0.0, 0.0, 0.0), (0.2, 0.4, 0.6)),
        "base_1": ("base", "box", (-0.00997, -0.01, -1.01), (0.01003, 0.01, -0.99)),
        "arm": ("arm", "box", (0.8, -0.1, -0.3), (1.2, 0.1, 0.3)),
        "wheel": ("wheel", "cylinder", (1.23, 0.4, -0.1), (1.27, 0.6, 0.1)),
        "rod": ("rod", "box", (1.08, 0.47, -0.01), (1.12, 0.53, 0.01)),
        "mast": ("mast", "box", (-0.8, -0.2, -0.1), (-0.2, 0.2, 0.1)),
    }
    mount = [1.0, -2.0, 0.5]

    primitives = {p["name"]: p for p in gen.urdf_primitives(urdf, mount)}
    assert sorted(primitives) == sorted(want)
    for name, (link, kind, lo, hi) in want.items():
        p = primitives[name]
        assert (p["link"], p["kind"]) == (link, kind), name
        # arm_mount is the root frame moved by -mount
        in_base = [[v - m for v, m in zip(bound, mount)] for bound in (lo, hi)]
        assert p["lo"] == pytest.approx(in_base[LO], abs=1e-12), name
        assert p["hi"] == pytest.approx(in_base[HI], abs=1e-12), name
    assert primitives["wheel"]["size"] == [0.1, 0.04]  # radius, length
    assert primitives["rod"]["size"] == [0.02, 0.04, 0.06]
    # The pose is arm_mount from the primitive: the rod's axes land on z, x, y.
    rod = np.array(primitives["rod"]["pose"])
    assert rod[:3, :3] == pytest.approx(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert rod[:3, 3] == pytest.approx([1.1 - 1.0, 0.5 + 2.0, 0.0 - 0.5])

    # All three angles at once, which no quarter turn above combines: a
    # joint's frame is Rz(yaw) Ry(pitch) Rx(roll), a point of its child turns
    # with it, and a box's bounds are those of its eight corners.
    rpy, at = (0.7, -0.4, 1.1), (0.3, -0.2, 0.5)
    offset, size = (0.1, 0.2, 0.3), (0.2, 0.4, 0.6)
    tilted = _urdf(
        _link("hub"),
        _link("slab", _box("%s %s %s" % size, "%s %s %s" % offset)),
        _joint("tilt", "fixed", "hub", "slab", "%s %s %s" % at, "%s %s %s" % rpy),
    )
    (slab,) = gen.urdf_primitives(tilted, [0.0, 0.0, 0.0])
    turn = _rpy_matrix(*rpy)
    pose = np.array(slab["pose"])
    assert pose[:3, :3] == pytest.approx(turn, abs=1e-12)
    assert pose[:3, 3] == pytest.approx(turn @ offset + at, abs=1e-12)
    signs = [(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    corners = (np.array(signs) * size / 2 + offset) @ turn.T + at
    assert slab["lo"] == pytest.approx(corners.min(axis=0), abs=1e-12)
    assert slab["hi"] == pytest.approx(corners.max(axis=0), abs=1e-12)

    # The same boxes as written: snapped outward to 0.1 mm (a bound on a grid
    # line moves a whole tick; base_1's x bounds are off the grid), centres in
    # 0.05 mm steps, dims in 0.1 mm steps.
    data["mount"]["arm_base_in_mebot"]["position"] = mount
    data["floor"] = {"half_extent_m": 2.0, "thickness_m": 0.2}
    files = _generate(data, urdf)["files"]
    assert _obstacle_lines(files[data["outputs"]["platform"]]) == _box_lines(
        ("arm", "0.00000, 2.00000, -0.50000", "0.4002, 0.2002, 0.6002"),
        ("base_0", "-0.90000, 2.20000, -0.20000", "0.2002, 0.4002, 0.6002"),
        ("base_1", "-0.99995, 2.00000, -1.50000", "0.0201, 0.0202, 0.0202"),
        ("mast", "-1.50000, 2.00000, -0.50000", "0.6002, 0.4002, 0.2002"),
        ("rod", "0.10000, 2.50000, -0.50000", "0.0402, 0.0602, 0.0202"),
        ("wheel", "0.25000, 2.50000, -0.50000", "0.0402, 0.2002, 0.2002"),
    )
    # The floor's top is the lowest point (base_1's bottom, -1.51, on a grid
    # line) rounded UP a tick to -1.5099; the slab is centred under the arm,
    # here 4 x 4 x 0.2.
    assert _obstacle_lines(files[data["outputs"]["floor"]]) == _box_lines(
        ("floor", "0.00000, 0.00000, -1.60990", "4.0000, 4.0000, 0.2000")
    )


def test_cylinder_bounding_box_is_exact():
    # The MEBot's wheels: rolled a quarter turn, so the axis lies along y.
    wheel = _cylinder("0.184", "0.075", rpy=QUARTER + " 0 0")
    (p,) = gen.urdf_primitives(_urdf(_link("wheel", wheel)), [0.0, 0.0, 0.0])
    dims = [hi - lo for lo, hi in zip(p["lo"], p["hi"])]
    assert dims == pytest.approx([0.368, 0.075, 0.368], abs=1e-12)

    radius, length = 0.1, 0.3
    centre, rpy = (0.3, -0.2, 0.5), (0.7, -0.4, 1.1)
    tilted = _cylinder(radius, length, "%s %s %s" % centre, "%s %s %s" % rpy)
    (p,) = gen.urdf_primitives(_urdf(_link("tilted", tilted)), [0.0, 0.0, 0.0])
    # The exact solid, sampled: both rims densely (a cylinder's extremes lie
    # on its rims) and the inside coarsely.
    fine = np.linspace(0.0, 2.0 * np.pi, 200000, endpoint=False)
    coarse = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rings = [(radius, fine, z) for z in (-length / 2, length / 2)]
    rings += [
        (r, coarse, z)
        for r in (0.0, radius / 2)
        for z in np.linspace(-length / 2, length / 2, 9)
    ]
    local = [
        np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], axis=1)
        for r, a, z in rings
    ]
    points = np.concatenate(local) @ _rpy_matrix(*rpy).T + centre
    lo, hi = np.array(p["lo"]), np.array(p["hi"])
    assert (points >= lo - 1e-12).all() and (points <= hi + 1e-12).all()
    # ... and no looser than it has to be: every face touches the solid.
    assert (points.min(axis=0) - lo < 1e-9).all()
    assert (hi - points.max(axis=0) < 1e-9).all()


def test_unsupported_input_is_refused(data, tmp_path, monkeypatch):
    def pair(joint):
        return _urdf(_link("a", CUBE), _link("b", CUBE), joint)

    refused = [
        # geometry a box layer cannot bound exactly
        (_urdf(_link("hull", _shape('<mesh filename="h.stl"/>'))), ["hull", "mesh"]),
        (_urdf(_link("ball", _shape('<sphere radius="0.1"/>'))), ["ball", "sphere"]),
        (
            _urdf(_link("pill", _shape('<capsule radius="0.1" length="0.2"/>'))),
            ["pill", "capsule"],
        ),
        (_urdf(_link("hollow", "<collision><geometry/></collision>")), ["hollow"]),
        (_urdf(_link("bare", "<collision/>")), ["bare"]),
        # two shapes in one <geometry>: reading the first would drop the other
        (
            _urdf(_link("two", _shape('<box size="0.1 0.1 0.1"/><box size="1 1 1"/>'))),
            ["two", "<box>, <box>"],
        ),
        # sizes cuRobo would turn into a box that can never collide
        (_urdf(_link("nanbox", _box("0.1 nan 0.1"))), ["nanbox"]),
        (_urdf(_link("infbox", _box("0.1 inf 0.1"))), ["infbox"]),
        (_urdf(_link("flat", _box("0.1 0 0.1"))), ["flat"]),
        (_urdf(_link("inverted", _box("0.1 -0.1 0.1"))), ["inverted"]),
        (_urdf(_link("short", _box("0.1 0.1"))), ["short"]),
        (_urdf(_link("worded", _box("0.1 wide 0.1"))), ["worded"]),
        (_urdf(_link("unsized", _shape("<box/>"))), ["unsized"]),
        (_urdf(_link("thin", _cylinder("0", "0.1"))), ["thin"]),
        (_urdf(_link("endless", _cylinder("0.1", "inf"))), ["endless"]),
        # joints that cannot simply be held at zero
        (
            pair(_joint("follower", "fixed", "a", "b", extra='<mimic joint="x"/>')),
            ["follower", "mimic"],
        ),
        (pair(_joint("skate", "planar", "a", "b")), ["skate", "planar"]),
        (pair(_joint("drift", "floating", "a", "b")), ["drift", "floating"]),
        (
            pair(_joint("lifted", "prismatic", "a", "b", extra=_limit(0.1, 0.4))),
            ["lifted"],
        ),
        (
            pair(_joint("sunk", "revolute", "a", "b", extra=_limit(-0.4, -0.1))),
            ["sunk"],
        ),
        # not one tree of links
        (
            _urdf(_link("left", CUBE), _link("right", CUBE)),
            ["left", "right", "one root"],
        ),
        # a ring that is the whole URDF: no link is left to be the root
        (
            pair(_joint("ab", "fixed", "a", "b") + _joint("ba", "fixed", "b", "a")),
            ["one root", "none"],
        ),
        (_urdf(_link("twin", CUBE), _link("twin", CUBE)), ["twin"]),
        (_urdf("<link>%s</link>" % CUBE), ["unnamed"]),
        (pair(_joint("dangling", "fixed", "a", "ghost")), ["dangling", "ghost"]),
        (
            _urdf(
                _link("a", CUBE),
                _link("b", CUBE),
                _link("shared", CUBE),
                _joint("ab", "fixed", "a", "b"),
                _joint("one", "fixed", "a", "shared"),
                _joint("two", "fixed", "b", "shared"),
            ),
            ["shared"],
        ),
        (
            _urdf(
                _link("root", CUBE),
                _link("ring_a", CUBE),
                _link("ring_b", CUBE),
                _joint("ab", "fixed", "ring_a", "ring_b"),
                _joint("ba", "fixed", "ring_b", "ring_a"),
            ),
            ["ring_a", "ring_b"],
        ),
        # names that cannot be written as a layer's box names
        (_urdf(_link("a, b", CUBE)), ["a, b"]),
        (_urdf(_link("null", CUBE)), ["null"]),
        (_urdf(_link("wheel[1]", CUBE)), ["wheel[1]"]),
        (_urdf(_link("x/y", CUBE)), ["x/y"]),
        (
            _urdf(
                _link("pair", CUBE, CUBE),
                _link("pair_0", CUBE),
                _joint("j", "fixed", "pair", "pair_0"),
            ),
            ["pair_0"],
        ),
        # nothing to cover
        (_urdf(_link("frame_only")), ["collision"]),
    ]
    for urdf, named in refused:
        with pytest.raises(ValueError) as caught:
            gen.urdf_primitives(urdf, [0.0, 0.0, 0.0])
        for part in named:
            assert part in str(caught.value), (named, str(caught.value))

    # A limit that leaves one end out has that end at 0, URDF's default: zero
    # is inside it, and the joint is held there like any other.
    for ends in ({"lower": -0.4}, {"upper": 0.4}):
        hinge = _joint("hinge", "revolute", "a", "b", extra=_limit(**ends))
        assert len(gen.urdf_primitives(pair(hinge), [0.0, 0.0, 0.0])) == 2

    # A rotated mount, about any axis and either way: the boxes are
    # axis-aligned in arm_mount.
    for rpy_deg in ([90, 0, 0], [0, 90, 0], [0, 0, 90], [0, -0.5, 0]):
        rotated = _inputs()
        rotated["mount"]["arm_base_in_mebot"]["rpy_deg"] = rpy_deg
        with pytest.raises(ValueError, match="rpy_deg"):
            _generate(rotated, FAR_BOX)

    # An arm side that is neither side, or not the side the mount is on (the
    # RAMMS mount is on the right, -1), or a mount on the centreline, which
    # is on neither: the forearms would be narrowed on the armrest's side.
    for side in (0, 2, 1, -2):
        wrong = _inputs()
        wrong["person"]["arm_side"] = side
        with pytest.raises(ValueError, match="arm_side %r must be" % side):
            _generate(wrong, FAR_BOX)
    centred = _inputs()
    centred["mount"]["arm_base_in_mebot"]["position"][Y] = 0.0
    for side in (1, -1):
        centred["person"]["arm_side"] = side
        with pytest.raises(ValueError, match="arm_side"):
            _generate(centred, FAR_BOX)

    # A grid the printed decimals do not match.
    coarse = _inputs()
    coarse["grid_m"] = 0.001
    with pytest.raises(ValueError, match="grid_m"):
        _generate(coarse, FAR_BOX)

    # Data that would write a box without volume, which cuRobo never sees: a
    # floor with no thickness, thighs turned inside out by a knee nearer the
    # backrest than the chest's front, a seat part written high corner first.
    flat = _inputs()
    flat["floor"]["thickness_m"] = 0.0
    with pytest.raises(ValueError, match="layer floor: box 'floor'"):
        _generate(flat, FAR_BOX)
    inverted = _inputs()
    inverted["person"]["ansur_in"]["buttock_knee_length"]["inches"] = 5.0
    with pytest.raises(ValueError, match="layer person: box 'thighs'"):
        _generate(inverted, FAR_BOX)
    backwards = _inputs()
    lo, hi = backwards["seat"]["boxes"]["backrest"]
    backwards["seat"]["boxes"]["backrest"] = [hi, lo]
    with pytest.raises(ValueError, match="layer seat: box 'backrest'"):
        _generate(backwards, FAR_BOX)

    # Data no gate on the boxes can see, because the boxes keep their volume:
    # a body size that is not positive (a foot of -11.54 in puts the legs'
    # front face 0.59 m back) or not a number (a hand breadth drops out of a
    # max(), and the lap's top by 0.18 m) or a margin below zero under-sizes
    # the person, and a padding below zero measures every clearance from a
    # shrunken box. An infinite one cannot be put on the grid at all, and
    # float's OverflowError would name no key.
    sizes = [(key, -inches) for key, inches in OCCUPANT_IN.items()]
    sizes += [(key, math.inf) for key in OCCUPANT_IN]
    for key, inches in sizes + [("hand_breadth", 0.0), ("hand_breadth", math.nan)]:
        shrunk = _inputs()
        shrunk["person"]["ansur_in"][key]["inches"] = inches
        with pytest.raises(ValueError, match="ansur_in %s: " % key):
            _generate(shrunk, FAR_BOX)
    for bad in (-0.01, math.nan, math.inf):
        thin = _inputs()
        thin["person"]["margin_m"] = bad
        with pytest.raises(ValueError, match="person margin_m"):
            _generate(thin, FAR_BOX)
        shrinking = _inputs()
        shrinking["padding_assumed_m"] = bad
        with pytest.raises(ValueError, match="padding_assumed_m"):
            _generate(shrinking, FAR_BOX)
    # Zero is a choice, though: the occupant as tabulated, under a planner
    # that does not pad.
    exact = _inputs()
    exact["person"]["margin_m"] = exact["padding_assumed_m"] = 0.0
    _generate(exact, FAR_BOX)

    # A budget or a gate that is not a number holds nothing back.
    for block, key in (
        ("platform", "max_boxes"),
        ("platform", "max_over_pct"),
        ("keep_out", "min_clearance_m"),
    ):
        unset = _inputs()
        unset[block][key] = math.nan
        with pytest.raises(ValueError, match="nan"):
            _generate(unset, FAR_BOX)
    # Nor does a gate below zero: it passes a padded box inside the arm's fixed
    # spheres, so data would switch the keep-out off. A cube at the arm's base
    # is 85 mm inside them.
    at_the_arm = _urdf(_link("chassis", _box("0.1 0.1 0.1", "0.505 -0.265 0.376")))
    for gate in (-0.001, -1.0):
        lax = _inputs()
        lax["keep_out"]["min_clearance_m"] = gate
        with pytest.raises(ValueError, match="keep_out min_clearance_m"):
            _generate(lax, at_the_arm)

    # Bytes that are not the pinned ones (the data file pins the real URDF).
    with pytest.raises(ValueError, match="sha256"):
        gen.generate(_inputs(), FAR_BOX)

    # A name that would put non-ASCII bytes in a layer file.
    with pytest.raises(ValueError, match="ASCII"):
        _generate(_inputs(), _urdf(_link("caf\u00e9", CUBE)))

    # rammp_curobo imported from another checkout (this host has an editable
    # install of one): its robot config is not the one beside the layers being
    # written. An installed package, as in the image, is no checkout.
    def packaged_under(parent):
        configs = tmp_path / parent / "rammp_curobo" / "configs"
        configs.mkdir(parents=True)
        shutil.copy(PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"], configs)
        return configs

    with monkeypatch.context() as patched:
        patched.setattr(gen, "PACKAGED_CONFIG_DIR", packaged_under("site-packages"))
        _generate(_inputs(), FAR_BOX)
        patched.setattr(gen, "PACKAGED_CONFIG_DIR", packaged_under("elsewhere/core"))
        with pytest.raises(ValueError) as caught:
            _generate(_inputs(), FAR_BOX)
    for part in (
        str(tmp_path / "elsewhere" / "core"),
        str(REPO / "core"),
        "PYTHONPATH",
    ):
        assert part in str(caught.value)

    # A fixed sphere that is not finite, in the robot config or through the
    # origin that places arm_shoulder_link: min() and max() pass over a NaN, so
    # the sphere would drop out of the keep-out without a word, and the layers
    # be carved and gated without it.
    robot = tmp_path / "robot.yaml"
    packaged = (PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"]).read_text()
    for link, index, key, value in (
        ("arm_base_link", 0, "radius", math.nan),
        ("arm_base_link", 1, "center", [math.nan, 0.0, 0.125]),
        ("arm_shoulder_link", 1, "radius", math.inf),
        ("arm_shoulder_link", 2, "radius", math.nan),
        ("arm_shoulder_link", 2, "center", [0.0, 0.0, math.nan]),
    ):
        config = yaml.safe_load(packaged)
        spheres = config["robot_cfg"]["kinematics"]["collision_spheres"]
        spheres[link][index][key] = value
        robot.write_text(yaml.safe_dump(config))
        lost = _inputs()
        lost["keep_out"]["robot_config"] = str(robot)
        with pytest.raises(ValueError) as caught:
            _generate(lost, FAR_BOX)
        for part in (str(robot), "%s sphere %d" % (link, index)):
            assert part in str(caught.value), (part, str(caught.value))
    for key in ("xyz", "rpy"):
        lost = _inputs()
        lost["keep_out"]["shoulder_link_origin"][key][X] = math.nan
        with pytest.raises(ValueError, match="shoulder_link_origin"):
            _generate(lost, FAR_BOX)

    # A shoulder sphere off the arm_joint_1 axis, in x or in y, moves with
    # arm_joint_1: it is not part of the fixed column, and treating it as fixed
    # would be wrong.
    for off_axis in ([0.01, 0.0, -0.04], [0.0, 0.01, -0.04]):
        spheres = {
            "arm_base_link": [{"center": [0.0, 0.0, 0.065], "radius": 0.075}],
            "arm_shoulder_link": [{"center": off_axis, "radius": 0.07}],
        }
        kinematics = {"collision_sphere_buffer": 0.005, "collision_spheres": spheres}
        robot.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kinematics}}))
        data["keep_out"]["robot_config"] = str(robot)
        with pytest.raises(ValueError, match="arm_shoulder_link"):
            gen.keep_out_spheres(data)


# --------------------------------------------------------------------- fetch
def test_pinned_fetch_verifies_sha256_and_uses_the_cache(isolated, monkeypatch):
    tmp_path = isolated
    # CRLF, like the real file: the hash is of the raw bytes.
    content = b"<robot name='pinned'/>\r\n"
    clone, commit = _clone(tmp_path / "clone", "urdf/model.urdf", content)
    source = {
        "repo": "example/chair",
        "commit": commit,
        "path": "urdf/model.urdf",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    name = source["sha256"] + ".urdf"
    url = "https://raw.githubusercontent.com/example/chair/%s/urdf/model.urdf" % commit
    cache = tmp_path / "cache"
    missing = tmp_path / "missing"

    # From the clone (the argument, not $RAMMP_DEMO_SOFTWARE): verified, then
    # cached (in the directory given, not under $XDG_CACHE_HOME).
    raw, origin = gen.fetch_pinned(source, cache_dir=cache, clone=clone)
    assert raw == content
    assert origin.startswith("clone") and str(clone) in origin
    assert (cache / name).read_bytes() == content
    assert os.listdir(cache) == [name]  # no temp file left behind
    assert not (tmp_path / "xdg").exists()

    # A cache hit asks neither git nor the network.
    def no_git(*args, **kwargs):
        raise AssertionError("git must not be called")

    with monkeypatch.context() as patched:
        patched.setattr(subprocess, "run", no_git)
        raw, origin = gen.fetch_pinned(source, cache_dir=cache, clone=clone)
    assert raw == content
    assert origin.startswith("cache") and str(cache / name) in origin

    # A corrupt cache file is ignored, said so, and replaced.
    (cache / name).write_bytes(b"garbage")
    raw, origin = gen.fetch_pinned(source, cache_dir=cache, clone=clone)
    assert raw == content
    assert str(clone) in origin and "ignored: sha256 mismatch" in origin
    assert (cache / name).read_bytes() == content

    # A wrong hash is an error naming the origin and both hashes, and the
    # bytes are not cached.
    wrong = dict(source, sha256=hashlib.sha256(b"something else").hexdigest())
    with pytest.raises(ValueError) as caught:
        gen.fetch_pinned(wrong, cache_dir=cache, clone=clone)
    for part in (str(clone), wrong["sha256"], source["sha256"]):
        assert part in str(caught.value)
    assert os.listdir(cache) == [name]

    # No source left (offline forbids the URL): the error lists every one.
    with pytest.raises(FileNotFoundError) as caught:
        gen.fetch_pinned(
            source, cache_dir=tmp_path / "empty", clone=missing, offline=True
        )
    for part in (str(tmp_path / "empty" / name), str(missing), url):
        assert part in str(caught.value)
    # A cache file that fails its hash is listed as that, not as missing.
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / name).write_bytes(b"garbage")
    with pytest.raises(FileNotFoundError, match="sha256 mismatch, ignored"):
        gen.fetch_pinned(source, cache_dir=stale, clone=missing, offline=True)

    # GitHub raw is the last resort, verified like the clone.
    asked = []

    def urlopen(url, data=None, timeout=None):
        asked.append((url, timeout))
        return io.BytesIO(content)

    with monkeypatch.context() as patched:
        patched.setattr(urllib.request, "urlopen", urlopen)
        web = tmp_path / "web"
        raw, origin = gen.fetch_pinned(source, cache_dir=web, clone=missing)
        assert (raw, origin) == (content, url)
        assert asked == [(url, 30)]
        assert (web / name).read_bytes() == content
        with pytest.raises(ValueError, match="raw.githubusercontent.com"):
            gen.fetch_pinned(wrong, cache_dir=web, clone=missing)

    # GitHub out of reach, or the body cut short: no source is left either,
    # and the error says what each one answered.
    class CutShort(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(self.getvalue())

    def unreachable(url, data=None, timeout=None):
        raise urllib.error.URLError("no route to host")

    def cut_short(url, data=None, timeout=None):
        return CutShort(content[:9])

    for answer, said in ((unreachable, "no route to host"), (cut_short, "9 bytes")):
        with monkeypatch.context() as patched:
            patched.setattr(urllib.request, "urlopen", answer)
            with pytest.raises(FileNotFoundError) as caught:
                gen.fetch_pinned(source, cache_dir=tmp_path / "empty", clone=missing)
        for part in (str(tmp_path / "empty" / name), str(missing), url, said):
            assert part in str(caught.value)

    # git missing, or hung (it gets 30 s): the next source is asked.
    for failure in (
        FileNotFoundError(2, "No such file or directory", "git"),
        subprocess.TimeoutExpired("git", 30),
    ):
        limits = []

        def git_fails(command, failure=failure, **kwargs):
            limits.append(kwargs.get("timeout"))
            raise failure

        with monkeypatch.context() as patched:
            patched.setattr(subprocess, "run", git_fails)
            patched.setattr(urllib.request, "urlopen", urlopen)
            fresh = tmp_path / type(failure).__name__
            fetched = gen.fetch_pinned(source, cache_dir=fresh, clone=clone)
        assert fetched == (content, url) and limits == [30]

    # Defaults: the clone is $RAMMP_DEMO_SOFTWARE, then ~/Demo-Software; the
    # cache is ${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/cad.
    monkeypatch.setenv("RAMMP_DEMO_SOFTWARE", str(clone))
    raw, origin = gen.fetch_pinned(source, offline=True)
    assert raw == content and str(clone) in origin
    in_xdg = tmp_path / "xdg" / "rammp_curobo" / "cad" / name
    assert in_xdg.read_bytes() == content

    monkeypatch.delenv("RAMMP_DEMO_SOFTWARE")
    monkeypatch.delenv("XDG_CACHE_HOME")
    home = tmp_path / "home"
    shutil.copytree(clone, home / "Demo-Software")
    raw, origin = gen.fetch_pinned(source, offline=True)
    assert raw == content and str(home / "Demo-Software") in origin
    in_home = home / ".cache" / "rammp_curobo" / "cad" / name
    assert in_home.read_bytes() == content

    # A cache that cannot be written (a read-only HOME in a container): the
    # verified bytes are still returned, and the origin says so.
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache directory should go")
    raw, origin = gen.fetch_pinned(source, cache_dir=blocker / "cad", clone=clone)
    assert raw == content
    assert str(clone) in origin and "cache not written" in origin

    # A write that fails at its last step, the rename, leaves no temp file
    # behind and the file it was to replace as it was.
    def refuse(src, dst):
        raise OSError("rename refused")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", refuse)
        raw, origin = gen.fetch_pinned(source, cache_dir=stale, clone=clone)
    assert raw == content and "cache not written: rename refused" in origin
    assert os.listdir(stale) == [name]
    assert (stale / name).read_bytes() == b"garbage"


# --------------------------------------------------------------------- merge
def _prim(name, lo, hi):
    return {"name": name, "box": (tuple(lo), tuple(hi))}


def test_merge_meets_the_budget_and_never_touches_the_keep_out(data, monkeypatch):
    sphere = [((0.0, 0.0, 0.0), 0.05, "arm_base_link")]
    pad, margin = 0.02, 0.005

    # The measure: distance from a sphere to the PADDED box, minus the radius.
    face_on = ((1000, -500, -500), (2000, 500, 500))
    assert gen.clearance(face_on, sphere, 0.02) == pytest.approx(0.1 - 0.02 - 0.05)
    corner = ((1000, 1000, 1000), (2000, 2000, 2000))
    assert gen.clearance(corner, sphere, 0.0) == pytest.approx(0.1 * 3**0.5 - 0.05)
    # a centre inside the box counts as minus the depth to the nearest face
    around = ((-1000, -1000, -1000), (1000, 1000, 3000))
    assert gen.clearance(around, sphere, 0.02) == pytest.approx(-0.12 - 0.05)
    # A sphere that is not a number makes the answer one too, wherever it is
    # in the list: a gate passes no NaN, and min() would pass over this one.
    for centre, radius in (
        ((math.nan, 0.0, 0.0), 0.05),
        ((0.0, 0.0, math.nan), 0.05),
        ((0.0, 0.0, 0.0), math.nan),
    ):
        lost = [(centre, radius, "arm_base_link")]
        for box in (face_on, around):
            for spheres in (lost, lost + sphere, sphere + lost):
                assert math.isnan(gen.clearance(box, spheres, 0.02))

    # Boxes in ticks along x (0.1 mm): a, c, d in front of the sphere, b behind
    # it, e inside a. Any box holding b and another crosses the sphere.
    a = _prim("a", (1000, -500, -500), (2000, 500, 500))
    b = _prim("b", (-2000, -500, -500), (-1000, 500, 500))
    c = _prim("c", (3000, -600, -500), (4000, 600, 500))
    d = _prim("d", (5000, -500, -500), (6000, 500, 500))
    e = _prim("e", (1200, -100, -100), (1800, 100, 100))
    prims = [a, b, c, d, e]

    def merged(max_boxes, order=prims):
        clusters = gen.merge(order, max_boxes, sphere, pad, margin)
        assert all(gen.clearance(c["box"], sphere, pad) >= margin for c in clusters)
        return {c["name"]: (c["box"], c["members"]) for c in clusters}

    # e is absorbed whatever the budget: four boxes where five were allowed.
    assert merged(5) == {
        "a": (a["box"], ["a", "e"]),
        "b": (b["box"], ["b"]),
        "c": (c["box"], ["c"]),
        "d": (d["box"], ["d"]),
    }
    # a+c and c+d add the same empty volume and have the same bounding volume;
    # the tie goes to the first pair in name order. The box is named after its
    # largest member.
    assert merged(3) == {
        "b": (b["box"], ["b"]),
        "c": (((1000, -600, -500), (4000, 600, 500)), ["a", "c", "e"]),
        "d": (d["box"], ["d"]),
    }
    assert merged(3, order=[d, e, c, b, a]) == merged(3)
    assert merged(2) == {
        "b": (b["box"], ["b"]),
        "c": (((1000, -600, -500), (6000, 600, 500)), ["a", "c", "d", "e"]),
    }
    # One box would have to cross the sphere: refused, not approximated.
    with pytest.raises(ValueError, match="keep-out"):
        gen.merge(prims, 1, sphere, pad, margin)

    # At equal cost the pair with the smaller bounding box goes first, though
    # the other is first in name order: both gaps are 100 ticks of the same
    # cross-section, between long boxes (f, g) and between short ones (j, k).
    # Of members equal in volume, the last in name order names the box.
    f = _prim("f", (0, 0, 0), (1000, 100, 100))
    g = _prim("g", (1100, 0, 0), (2100, 100, 100))
    j = _prim("j", (10000, 0, 0), (10100, 100, 100))
    k = _prim("k", (10200, 0, 0), (10300, 100, 100))
    short_first = gen.merge([f, g, j, k], 3, [], pad, margin)
    assert {c["name"]: c["members"] for c in short_first} == {
        "f": ["f"],
        "g": ["g"],
        "k": ["j", "k"],
    }

    # The boxes come back in name order, the order they are written in: s and
    # y merge at s's place in the list, ahead of t, and take y's name.
    s = _prim("s", (0, 0, 0), (10, 10, 10))
    t = _prim("t", (5000, 0, 0), (5100, 100, 100))
    y = _prim("y", (20, 0, 0), (500, 100, 100))
    assert [c["name"] for c in gen.merge([s, t, y], 2, [], pad, margin)] == ["t", "y"]

    # Coming within the margin is enough to skip a merge, without containing
    # the sphere: the padded box of p+q would clear it by 4.9 mm at x 0.0749,
    # and does by 5.1 mm at x 0.0751.
    def beside(x):
        return [
            _prim("p", (x, 2000, -500), (1000, 3000, 500)),
            _prim("q", (x, -3000, -500), (1000, -2000, 500)),
        ]

    with pytest.raises(ValueError, match="keep-out"):
        gen.merge(beside(749), 1, sphere, pad, margin)
    (both,) = gen.merge(beside(751), 1, sphere, pad, margin)
    assert both["box"] == ((751, -3000, -500), (1000, 3000, 500))

    # Off a face, padding and margin do not add up: padding grows the box on
    # every axis, and the margin is measured from the grown box. The box of
    # m+n turns an edge to the sphere, as far out in x as in y: padded, it
    # would clear the sphere by 4.9 mm at 0.0588 and does by 5.2 mm at 0.0590.
    # (5 mm of padding under a 20 mm margin would draw that line at 0.0545,
    # and 25 mm of padding under none at 0.0604.)
    def edge_on(xy):
        return [
            _prim("m", (xy, 2000, -500), (1000, 3000, 500)),
            _prim("n", (2000, xy, -500), (3000, 1000, 500)),
        ]

    with pytest.raises(ValueError, match="keep-out"):
        gen.merge(edge_on(588), 1, sphere, pad, margin)
    (both,) = gen.merge(edge_on(590), 1, sphere, pad, margin)
    assert both["box"] == ((590, 590, -500), (3000, 3000, 500))

    # A pair costs the empty volume its bounding box adds; what the two cover
    # together counts once. u and v make an L (they share a corner) and w lies
    # beside u: counted twice, the corner would make closing the L (u+v) look
    # cheaper than the strip between u and w.
    u = _prim("u", (0, 0, 0), (1000, 100, 100))
    v = _prim("v", (0, 0, 0), (100, 1000, 100))
    w = _prim("w", (0, -905, 0), (1000, -805, 100))
    strip = gen.merge([u, v, w], 2, [], pad, margin)
    assert sorted(c["members"] for c in strip) == [["u", "w"], ["v"]]

    # After a merge, what the new box swallowed is absorbed again: r, in the
    # L's far corner, is dearer to merge with u or v than they are with each
    # other, and ends inside their box without taking a slot of the budget.
    r = _prim("r", (900, 900, 0), (1000, 1000, 100))
    (closed,) = gen.merge([u, v, r], 2, [], pad, margin)
    assert closed["box"] == ((0, 0, 0), (1000, 1000, 100))
    assert closed["members"] == ["r", "u", "v"]

    # A box takes in every box inside it, not only the first one found.
    crate = _prim("crate", (0, 0, 0), (1000, 1000, 1000))
    nut = _prim("nut", (100, 100, 100), (200, 200, 200))
    washer = _prim("washer", (700, 700, 700), (800, 800, 800))
    (packed,) = gen.merge([crate, nut, washer], 3, [], pad, margin)
    assert packed["members"] == ["crate", "nut", "washer"]

    # Inside on two axes is not inside: a post through a plate sticks out
    # along the third, from one face of the plate or from the other.
    for axis in (X, Y, Z):
        for lo, hi in ((-900, 100), (0, 1000)):
            plate_hi, post_lo, post_hi = [1000] * 3, [400] * 3, [500] * 3
            plate_hi[axis], post_lo[axis], post_hi[axis] = 100, lo, hi
            plate = _prim("plate", (0, 0, 0), plate_hi)
            post = _prim("post", post_lo, post_hi)
            apart = gen.merge([plate, post], 2, [], pad, margin)
            assert [c["members"] for c in apart] == [["plate"], ["post"]], axis

    # As written: a merged box carries its largest member's name, and the
    # other members ride in a comment.
    hull_and_peg = _urdf(
        _link("hull", _box("0.2 0.2 0.2", "1 0 0")),
        _link("peg", _box("0.05 0.05 0.05")),
        _joint("stem", "fixed", "hull", "peg", "1.3 0 0"),
    )
    roomy = _inputs()
    roomy["platform"] = {"max_boxes": 1, "max_over_pct": 1000.0}
    files = _generate(roomy, hull_and_peg)["files"]
    (line,) = _obstacle_lines(files[roomy["outputs"]["platform"]])
    hull = ("hull", "0.60750, 0.26500, -0.37600", "0.4252, 0.2002, 0.2002")
    assert line == _box_lines(hull)[0] + "  # + peg"

    # generate() hands the merge the arm's own spheres, the padding and the
    # margin. Three cubes in a row along y, just in front of the arm (it is
    # at 0.505, -0.265, 0.376 here), and a foot that keeps the floor far
    # below. The cheapest pair straddles the arm: their padded box would
    # clear the base sphere by 4.4 mm. That passes the 4 mm gate, so only
    # the merge's 5 mm margin turns it down, and the dearer pair is merged.
    row = ("0.6595 -0.565 0.441", "0.6595 0.035 0.441", "0.6595 2.035 0.441")
    cubes = [_box("0.1 0.1 0.1", at) for at in row + ("3.505 -0.265 -1.5",)]
    crowded = _inputs()
    crowded["platform"] = {"max_boxes": 3, "max_over_pct": 5000.0}
    handed = []
    sound = gen.merge

    def recorded(primitives, *budget_and_keep_out):
        handed.append(budget_and_keep_out)
        return sound(primitives, *budget_and_keep_out)

    with monkeypatch.context() as patched:
        patched.setattr(gen, "merge", recorded)
        platform = _generate(crowded, _urdf(_link("chair", *cubes)))["platform"]
    (lo_a, hi_a), (lo_b, hi_b) = (p["box"] for p in platform["primitives"][:2])
    around = tuple(map(min, lo_a, lo_b)), tuple(map(max, hi_a, hi_b))
    turned_down = gen.clearance(around, gen.keep_out_spheres(crowded), 0.02)
    assert 0.004 <= turned_down < 0.005
    assert [box["members"] for box in platform["boxes"]] == [
        ["chair_0"],
        ["chair_1", "chair_2"],
        ["chair_3"],
    ]
    # That gap is face-on, where padding and margin count only as their sum:
    # which of the two is which is read off the call.
    assert handed == [(3, gen.keep_out_spheres(crowded), 0.02, 0.005)]

    # A primitive that is itself inside the arm's column cannot be merged
    # away from it: the layer is refused.
    at_the_arm = _urdf(_link("chassis", _box("0.1 0.1 0.1", "0.505 -0.265 0.376")))
    with pytest.raises(ValueError) as caught:
        _generate(data, at_the_arm)
    assert "platform" in str(caught.value) and "chassis" in str(caught.value)

    # Every layer is held to the gate. A shelf 2.5 m in front of the arm and
    # level with it is far enough itself, but the floor goes under the URDF's
    # lowest point, and here that slab runs through the arm.
    shelf = _urdf(_link("shelf", _box("0.1 0.1 0.1", "3.005 -0.265 0.626")))
    with pytest.raises(ValueError, match="layer floor: box 'floor'"):
        _generate(data, shelf)


# ------------------------------------------------------------------ coverage
def _meta(urdf, **budget):
    data = _inputs()
    data["platform"] = budget
    files = _generate(data, urdf)["files"]
    return data, yaml.safe_load(files[data["outputs"]["platform"]])["meta"]


def test_over_coverage_is_exact_on_known_cases():
    def cube(x):
        return ((x, 0, 0), (x + 10, 10, 10))

    # (union, sum of volumes, most boxes over one point), in ticks^3
    assert gen.union_stats([cube(0), cube(20)]) == (2000, 2000, 1)
    assert gen.union_stats([cube(0), cube(5)]) == (1500, 2000, 2)
    assert gen.union_stats([cube(0), cube(5), cube(8)]) == (1800, 3000, 3)

    def two_cubes(x):
        cubes = [_box("0.1 0.1 0.1", "1 0 0"), _box("0.1 0.1 0.1", "%s 0 0" % x)]
        return _urdf(_link("chassis", *cubes))

    # Two 0.1 m cubes 0.25 m apart, merged into one box. Their bounds sit on
    # grid lines, so each snaps to 1002 ticks and the gap to 2498:
    #   primitives 2 * 1002^3    = 2 012 024 016 ticks^3
    #   layer      4502 * 1002^2 = 4 520 026 008
    #   over       2498 * 1002^2 = 2 508 001 992  (124.65 %, written 124.7)
    apart = two_cubes(1.35)
    data, meta = _meta(apart, max_boxes=1, max_over_pct=124.7)
    assert meta["coverage"] == {
        "primitives": 2,
        "under_m3": 0.0,
        "primitives_bbox_union_m3": 0.002012,
        "layer_m3": 0.00452,
        "over_m3": 0.002508,
        "over_pct": 124.7,
        "cylinder_boxing_m3": 0.0,
        "overlap_m3": 0.0,
        "max_stack": 1,
    }
    # The rest of meta: where the numbers came from and what they were held to.
    pin = ("repo", "commit", "path", "sha256")
    assert meta["source"] == {k: data["source"][k] for k in pin}
    assert meta["arm_base_in_mebot"] == data["mount"]["arm_base_in_mebot"]
    assert meta["grid_m"] == 0.0001
    assert meta["padding_assumed_m"] == data["padding_assumed_m"] == 0.02
    assert meta["keep_out_margin_m"] == data["keep_out_margin_m"]
    assert meta["budget"] == {"max_boxes": 1, "max_over_pct": 124.7}
    # The box's padded corner (0.4249, 0.1949, -0.3059) is nearest the base
    # sphere at (0, 0, 0.065), radius 0.075 + 0.005.
    nearest = math.sqrt(0.4249**2 + 0.1949**2 + (0.065 + 0.3059) ** 2) - 0.080
    assert meta["keep_out_clearance_m"] == pytest.approx(nearest, abs=5e-5)

    # The budget gates the number that is written.
    for budget in (124.6, 15.0):
        with pytest.raises(ValueError, match="124.7"):
            _meta(apart, max_boxes=1, max_over_pct=budget)
    _, unmerged = _meta(apart, max_boxes=2, max_over_pct=0.0)
    assert unmerged["coverage"]["over_pct"] == 0.0
    # 0.2 m apart the gap is 1998 ticks: 99.7006 %, written 99.7 and within a
    # budget of 99.7.
    _, meta = _meta(two_cubes(1.3), max_boxes=1, max_over_pct=99.7)
    assert meta["coverage"]["over_pct"] == 99.7

    # Two cubes overlapping by half, kept as two boxes: the overlap (502 ticks
    # of x) is counted once in the volumes and reported on its own.
    #   union   1502 * 1002^2 = 1 508 014 008
    #   overlap  502 * 1002^2 =   504 010 008
    _, meta = _meta(two_cubes(1.05), max_boxes=2, max_over_pct=0.0)
    assert meta["coverage"] == {
        "primitives": 2,
        "under_m3": 0.0,
        "primitives_bbox_union_m3": 0.001508,
        "layer_m3": 0.001508,
        "over_m3": 0.0,
        "over_pct": 0.0,
        "cylinder_boxing_m3": 0.0,
        "overlap_m3": 0.000504,
        "max_stack": 2,
    }

    # A wheel: what its box adds over the true cylinder is (4 - pi) r^2 L.
    wheel = _urdf(_link("wheel", _cylinder("0.1", "0.2", rpy=QUARTER + " 0 0")))
    _, meta = _meta(wheel, max_boxes=1, max_over_pct=0.0)
    assert meta["coverage"]["cylinder_boxing_m3"] == 0.001717
    assert meta["coverage"]["primitives_bbox_union_m3"] == 0.008024  # 2002^3

    # Volumes are rounded half up: a 0.05 m cube is 502^3 = 126 506 008 ticks^3.
    _, meta = _meta(
        _urdf(_link("peg", _box("0.05 0.05 0.05"))), max_boxes=1, max_over_pct=0.0
    )
    assert meta["coverage"]["layer_m3"] == 0.000127


@pytest.mark.parametrize("defect", ["drop", "shrink_1_tick", "shift_1mm"])
def test_containment_certificate_catches_a_defective_layer(data, monkeypatch, defect):
    urdf = _urdf(
        _link(
            "chassis",
            _box("0.1 0.1 0.1", "1 0 0"),
            _box("0.1 0.1 0.1", "1.3 0 0"),
            _box("0.1 0.1 0.1", "1 0.5 0"),
        )
    )
    data["platform"] = {"max_boxes": 2, "max_over_pct": 100.0}
    _generate(data, urdf)  # sound as generated

    # One box wrong along one axis at a time: its low or its high face one
    # tick inside, or the whole box 1 mm (10 ticks) to either side.
    moves = {
        "drop": [],
        "shrink_1_tick": [(1, 0), (0, -1)],
        "shift_1mm": [(10, 10), (-10, -10)],
    }[defect]
    defects = [(axis, move) for axis in (X, Y, Z) for move in moves] or [None]
    sound = gen.merge
    for moved in defects:

        def defective(*args, moved=moved):
            clusters = sound(*args)
            if moved is None:
                return clusters[1:]
            axis, (lo_by, hi_by) = moved
            lo, hi = (list(corner) for corner in clusters[0]["box"])
            lo[axis] += lo_by
            hi[axis] += hi_by
            clusters[0]["box"] = (tuple(lo), tuple(hi))
            return clusters

        monkeypatch.setattr(gen, "merge", defective)
        with pytest.raises(ValueError, match="containment: primitive 'chassis_"):
            _generate(data, urdf)


# -------------------------------------------------------------------- person
def _occupied(boxes, cuts):
    """Which cells of the grid cut at `cuts` are covered by any of `boxes`."""
    filled = np.zeros([len(c) - 1 for c in cuts], dtype=bool)
    for lo, hi in boxes:
        cells = tuple(slice(c.index(lo[k]), c.index(hi[k])) for k, c in enumerate(cuts))
        filled[cells] = True
    return filled


def _clearance(boxes, spheres, pad):
    lo = np.array([b[LO] for b in boxes]) * 1e-4 - pad
    hi = np.array([b[HI] for b in boxes]) * 1e-4 + pad
    best = math.inf
    for centre, radius, _ in spheres:
        outside = np.linalg.norm(np.clip(centre, lo, hi) - centre, axis=1)
        depth = np.minimum(centre - lo, hi - centre).min(axis=1)
        best = min(best, (np.where(depth > 0, -depth, outside) - radius).min())
    return best


def _faces(data):
    envelope = dict(_generate(data, FAR_BOX)["person"]["envelope"])
    return {
        (segment, side, axis): box[side][axis]
        for segment, box in envelope.items()
        for side in (LO, HI)
        for axis in (X, Y, Z)
    }


def test_person_boxes_are_the_envelope_minus_the_arm_column(data):
    # The carve, on a case small enough to read: slabs in x, then y, then z.
    cube, core = ((0, 0, 0), (30, 30, 30)), ((10, 10, 10), (20, 20, 20))
    assert gen.subtract(cube, core) == [
        ((0, 0, 0), (10, 30, 30)),
        ((20, 0, 0), (30, 30, 30)),
        ((10, 0, 0), (20, 10, 30)),
        ((10, 20, 0), (20, 30, 30)),
        ((10, 10, 0), (20, 20, 10)),
        ((10, 10, 20), (20, 20, 30)),
    ]
    # a cut elsewhere, or only touching a face, leaves the box whole: no slabs
    assert gen.subtract(cube, ((40, 10, 10), (50, 20, 20))) == [cube]
    assert gen.subtract(cube, ((30, 10, 10), (50, 20, 20))) == [cube]
    assert gen.subtract(core, cube) == []

    # The arm's fixed spheres, from the packaged robot config: arm_base_link's as
    # they are, arm_shoulder_link's carried through arm_joint_1's origin (a half turn
    # about x, so a sphere at local z sits at `up - z`), all with the buffer.
    spheres = gen.keep_out_spheres(data)
    robot = PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"]
    kinematics = yaml.safe_load(robot.read_text())["robot_cfg"]["kinematics"]
    base = kinematics["collision_spheres"]["arm_base_link"]
    shoulder = kinematics["collision_spheres"]["arm_shoulder_link"]
    up = data["keep_out"]["shoulder_link_origin"]["xyz"][Z]
    links = ["arm_base_link"] * len(base) + ["arm_shoulder_link"] * len(shoulder)
    assert [link for _, _, link in spheres] == links
    centres = np.array([c for c, _, _ in spheres])
    radii = np.array([r for _, r, _ in spheres])
    assert np.abs(centres[:, :Z]).max() < 1e-5  # all on the arm_joint_1 axis
    heights = [s["center"][Z] for s in base] + [up - s["center"][Z] for s in shoulder]
    assert centres[:, Z] == pytest.approx(heights, abs=1e-6)
    buffer = kinematics["collision_sphere_buffer"]
    assert radii == pytest.approx([s["radius"] + buffer for s in base + shoulder])

    margin = data["keep_out_margin_m"]
    # The RAMMS mount and 0.02 m padding, where the column stands in the torso and
    # the legs, and a mount that stands it in other segments, with more
    # padding.
    for mount, pad in (([0.505, -0.265, 0.376], 0.02), ([0.65, 0.1, 0.30], 0.03)):
        placed = _inputs()
        placed["mount"]["arm_base_in_mebot"]["position"] = mount
        placed["person"]["arm_side"] = 1 if mount[1] > 0 else -1
        placed["padding_assumed_m"] = pad
        result = _generate(placed, FAR_BOX)
        person = result["person"]
        column = person["column"]
        envelope = [box for _, box in person["envelope"]]
        boxes = [b["box"] for b in person["boxes"]]
        assert [segment for segment, _ in person["envelope"]] == list(SEGMENTS)

        # The column: the spheres' bounding box grown by padding and margin,
        # snapped outward by at most a tick.
        grown_lo = (centres - radii[:, None]).min(axis=0) - pad - margin
        grown_hi = (centres + radii[:, None]).max(axis=0) + pad + margin
        slack_lo = grown_lo - np.array(column[LO]) * 1e-4
        slack_hi = np.array(column[HI]) * 1e-4 - grown_hi
        assert (slack_lo > 0).all() and (slack_lo < 1e-4 + 2e-9).all()
        assert (slack_hi > 0).all() and (slack_hi < 1e-4 + 2e-9).all()

        # Exactly the envelope minus the column, cell for cell on the integer
        # grid cut at every face involved.
        every = boxes + envelope + [column]
        cuts = [sorted({b[s][k] for b in every for s in (LO, HI)}) for k in (X, Y, Z)]
        whole = _occupied(envelope, cuts)
        carved = whole & ~_occupied([column], cuts)
        assert np.array_equal(_occupied(boxes, cuts), carved)
        assert carved.any() and carved.sum() < whole.sum()  # it did carve

        # Padded, the boxes clear the spheres by the gate and more.
        clear = _clearance(boxes, spheres, pad)
        assert clear >= 0.004
        assert person["clearance_m"] == pytest.approx(clear, abs=1e-9)

        # A whole segment keeps its name; pieces are numbered.
        names = [b["name"] for b in person["boxes"]]
        assert len(set(names)) == len(names)
        for segment in SEGMENTS:
            pieces = sorted(n for n in names if n.startswith(segment))
            numbered = ["%s_%d" % (segment, k) for k in range(len(pieces))]
            assert pieces == [segment] or (len(pieces) > 1 and pieces == numbered)

        # The file says what it was built from, each size with its row, sex
        # and percentile, and its volumes are the cells' (summed here, not by
        # the generator).
        meta = yaml.safe_load(result["files"][placed["outputs"]["person"]])["meta"]
        assert meta["ansur_in"] == placed["person"]["ansur_in"]
        assert meta["margin_m"] == placed["person"]["margin_m"]
        assert meta["seat"] == placed["person"]["seat"]
        assert meta["keep_out_clearance_m"] == pytest.approx(clear, abs=5e-5)
        edge = [np.diff(c) for c in cuts]
        cell = edge[X][:, None, None] * edge[Y][None, :, None] * edge[Z][None, None, :]
        envelope_m3, layer_m3 = (
            float((cell * f).sum()) * 1e-12 for f in (whole, carved)
        )
        summed = sum(float(np.prod(np.subtract(hi, lo))) for lo, hi in boxes) * 1e-12
        assert meta["volumes"] == pytest.approx(
            {
                "envelope_m3": envelope_m3,
                "layer_m3": layer_m3,
                "carved_m3": envelope_m3 - layer_m3,
                "overlap_m3": summed - layer_m3,
            },
            abs=1e-6,
        )

    # Where the boxes are, which the deltas below cannot see: the design
    # occupant at the RAMMS mount is these sixteen boxes. The hips, the torso
    # and the forearms (which reach back to the back) stand beside the arm's
    # column and the legs reach back under the seat to it, and it carves all
    # four. The forearms are hip-wide on the arm's side (-y here) and
    # shoulder-wide on the other.
    files = _generate(_inputs(), FAR_BOX)["files"]
    assert _obstacle_lines(files[data["outputs"]["person"]]) == _box_lines(
        ("forearms_0", "-0.25075, 0.30670, 0.36055", "0.2913, 0.5836, 0.3421"),
        ("forearms_1", "0.22170, 0.30670, 0.36055", "0.2332, 0.5836, 0.3421"),
        ("forearms_2", "0.00000, 0.35180, 0.36055", "0.2102, 0.4934, 0.3421"),
        ("forearms_3", "0.00000, 0.06000, 0.47005", "0.2102, 0.0902, 0.1231"),
        ("head", "-0.07625, 0.26500, 0.87885", "0.6403, 0.2632, 0.4851"),
        ("hips_0", "-0.25075, 0.26500, 0.13895", "0.2913, 0.5002, 0.1891"),
        ("hips_1", "0.12175, 0.26500, 0.13895", "0.0333, 0.5002, 0.1891"),
        ("hips_2", "0.00000, 0.31010, 0.13895", "0.2102, 0.4100, 0.1891"),
        ("legs_0", "0.44280, 0.26500, -0.19095", "0.6754, 0.5002, 0.4711"),
        ("legs_1", "0.00155, 0.31010, -0.19095", "0.2071, 0.4100, 0.4711"),
        ("legs_2", "0.00155, 0.06000, -0.23330", "0.2071, 0.0902, 0.3864"),
        ("thighs", "0.31285, 0.26500, 0.19800", "0.3491, 0.5002, 0.3072"),
        ("torso_0", "-0.25075, 0.26500, 0.58840", "0.2913, 0.6670, 0.7100"),
        ("torso_1", "0.12175, 0.26500, 0.58840", "0.0333, 0.6670, 0.7100"),
        ("torso_2", "0.00000, 0.35180, 0.58840", "0.2102, 0.4934, 0.7100"),
        ("torso_3", "0.00000, 0.01830, 0.67595", "0.2102, 0.1736, 0.5349"),
    )

    # Each parameter moves exactly its faces. Lengths grow by 0.1 m (1000
    # ticks; a breadth moves each side by half of it).
    base_faces = _faces(_inputs())
    grow = 0.1 / 0.0254
    all_z = {(s, side, Z): 1000 for s in SEGMENTS for side in (LO, HI)}
    all_y = {(s, side, Y): 1000 for s in SEGMENTS for side in (LO, HI)}

    def breadth(*segments):
        return {(s, side, Y): (-500, 500)[side] for s in segments for side in (LO, HI)}

    # the chest's front: the torso's and the thighs' faces on it, and the
    # head's, which reaches half its length past it (the forearms start at
    # the back)
    chest = {(s, HI, X): 1000 for s in ("hips", "torso", "head")}
    chest.update({(s, LO, X): 1000 for s in ("thighs",)})
    ansur = {
        # its top; the torso's (the chin) is this less the shortest head
        "sitting_height": {("torso", HI, Z): 1000, ("head", HI, Z): 1000},
        # the lowest chin, the head's bottom
        "sitting_height_low": {("head", LO, Z): 1000},
        "head_length": {("head", LO, Z): -1000, ("head", HI, X): 500},
        "head_length_low": {("torso", HI, Z): -1000},
        "head_breadth": breadth("head"),
        "sitting_elbow_height": {("forearms", HI, Z): 1000},
        # the lowest elbow, where the hips give way to the shoulder-wide torso
        "sitting_elbow_height_low": {("hips", HI, Z): 1000, ("torso", LO, Z): 1000},
        "hand_breadth": {("forearms", HI, Z): 1000},
        "thigh_clearance": {("thighs", HI, Z): 1000},
        # the thinnest thighs, which the forearms rest lowest on
        "thigh_clearance_low": {("forearms", LO, Z): 1000},
        "elbow_fingertip_length": {("forearms", HI, X): 1000},
        "buttock_knee_length": {("thighs", HI, X): 1000, ("legs", HI, X): 1000},
        "buttock_popliteal_length": {("legs", LO, X): 1000},
        "popliteal_height": {("legs", LO, Z): -1000},
        # the forearms: shoulder-wide away from the arm (+y here), hip-wide on
        # its side
        "shoulder_breadth_bideltoid": {**breadth("torso"), ("forearms", HI, Y): 500},
        "hip_breadth": {
            **breadth("hips", "thighs", "legs"),
            ("forearms", LO, Y): -500,
        },
        "abdominal_depth": chest,
        # the feet ahead of the knees, and tucked half a foot behind them
        "foot_length": {("legs", HI, X): 1000, ("legs", LO, X): -500},
    }
    seat = {
        "pan_top_z": all_z,
        # the backrest's rear no longer bounds the person: its back is the face
        "backrest_rear_x": {},
        "centreline_y": all_y,
    }

    def moved(change):
        changed = _inputs()
        change(changed["person"], changed["mount"]["arm_base_in_mebot"]["position"])
        faces = _faces(changed)
        return {f: faces[f] - base_faces[f] for f in faces if faces[f] != base_faces[f]}

    def inches(key, amount):
        def change(person, mount):
            person["ansur_in"][key]["inches"] += amount

        return change

    def anchor(key, amount):
        def change(person, mount):
            person["seat"][key] += amount

        return change

    for key, want in ansur.items():
        assert moved(inches(key, grow)) == want, key
    for key, want in seat.items():
        assert moved(anchor(key, 0.1)) == want, key

    # The arm's mount moves the whole envelope the other way in arm_mount.
    for axis in (X, Y, Z):

        def shift(person, mount, axis=axis):
            mount[axis] += 0.1

        every_face = {(s, side, axis): -1000 for s in SEGMENTS for side in (LO, HI)}
        assert moved(shift) == every_face, axis

    # The smaller term of a max() moves nothing until it is the larger one:
    # the chest behind the abdomen.
    def raise_chest(person, mount):
        rows = person["ansur_in"]
        rows["chest_depth"]["inches"] = rows["abdominal_depth"]["inches"] + grow

    assert moved(inches("chest_depth", grow / 10)) == {}
    assert moved(raise_chest) == chest

    # The backrest face is a range: its front sets the chest, the knees (and
    # the feet) and the fingertips, its rear the back and the back of the
    # knees.
    def backrest(index):
        def change(person, mount):
            person["seat"]["backrest_face_x"][index] += 0.1

        return change

    knees = {("thighs", HI, X): 1000, ("legs", HI, X): 1000}
    fingertips = {("forearms", HI, X): 1000}
    assert moved(backrest(1)) == {**chest, **knees, **fingertips}
    assert moved(backrest(0)) == {
        (s, LO, X): 1000 for s in ("hips", "torso", "head", "forearms", "legs")
    }

    # ... whichever end of the range is written first.
    def front_first(person, mount):
        person["seat"]["backrest_face_x"].reverse()

    assert moved(front_first) == {}

    # The margin is on every face. Where two segments meet (the chest front,
    # the pan underside) the face moves with the segment that owns it.
    def margin_up(person, mount):
        person["margin_m"] += 0.01

    outward = {
        (s, side, k): (-100, 100)[side]
        for s in SEGMENTS
        for side in (LO, HI)
        for k in (X, Y, Z)
    }
    # the chest's front is the thighs' back face, the pan's underside the
    # legs' top, and the lowest elbow less the margin the hips' top (the
    # forearms' back face is the back's, which the margin moves out)
    outward[("thighs", LO, X)] = 100
    outward[("legs", HI, Z)] = -100
    outward[("hips", HI, Z)] = -100
    assert moved(margin_up) == outward

    # Carved back by only 1 mm, the padded person comes within 4 mm of the
    # arm: such a layer is not written. Nor is the default one (5.1 mm clear)
    # under a 6 mm gate.
    tight = _inputs()
    tight["keep_out_margin_m"] = 0.001
    with pytest.raises(ValueError, match="person"):
        _generate(tight, FAR_BOX)
    strict = _inputs()
    strict["keep_out"]["min_clearance_m"] = 0.006
    with pytest.raises(ValueError, match="person"):
        _generate(strict, FAR_BOX)


def test_the_forearms_are_hip_wide_on_the_arms_side_and_shoulder_wide_on_the_other():
    # The armrest is across the chair from the arm: a forearm rests on it,
    # shoulder-wide, on that side, and on the thigh, hip-wide, on the arm's.
    # Mirrored with the mount, and with arm_side.
    for y, side in ((-0.265, -1), (0.265, 1)):
        placed = _inputs()
        placed["mount"]["arm_base_in_mebot"]["position"][Y] = y
        placed["person"]["arm_side"] = side
        envelope = dict(_generate(placed, FAR_BOX)["person"]["envelope"])
        forearms, hips, torso = (envelope[k] for k in ("forearms", "hips", "torso"))
        arm, other = (LO, HI) if side < 0 else (HI, LO)
        assert forearms[arm][Y] == hips[arm][Y], side
        assert forearms[other][Y] == torso[other][Y], side
        # the shoulders are the wider: the forearms are narrower on the arm's
        # side than on the other, by the difference
        centre = (placed["person"]["seat"]["centreline_y"] - y) / 1e-4
        reach = [abs(forearms[s][Y] - centre) for s in (arm, other)]
        rows = placed["person"]["ansur_in"]
        gap = (
            rows["shoulder_breadth_bideltoid"]["inches"] - rows["hip_breadth"]["inches"]
        )
        assert reach[1] - reach[0] == pytest.approx(gap / 2 * 254, abs=2)


def test_seat_boxes_are_its_parts_minus_a_wider_arm_column(data):
    # The seat's parts, as the data gives them in the mebot frame, placed in
    # arm_mount and carved around the arm's fixed spheres: by a column grown
    # by the padding and the seat's OWN margin, wider than the person's, since
    # closer than the planner's collision activation distance those spheres
    # carry a cost no joint can lower (DT_EXCEPTION on nearly every plan).
    spheres = gen.keep_out_spheres(data)
    centres = np.array([c for c, _, _ in spheres])
    radii = np.array([r for _, r, _ in spheres])
    pad, margin = data["padding_assumed_m"], data["seat"]["keep_out_margin_m"]
    assert margin > data["keep_out_margin_m"]
    result = _generate(data, FAR_BOX)
    seat = result["seat"]
    mount = np.array(data["mount"]["arm_base_in_mebot"]["position"])

    # the parts, in arm_mount, snapped outward to the grid
    parts = {
        name: gen._snap(*[[v - o for v, o in zip(c, mount)] for c in corners])
        for name, corners in data["seat"]["boxes"].items()
    }
    assert sorted(parts) == ["armrest", "backrest", "base"]

    # the column: the spheres' bounding box grown by padding and the seat's
    # margin, snapped outward by at most a tick
    column = gen._column(data, spheres, margin)
    grown_lo = (centres - radii[:, None]).min(axis=0) - pad - margin
    grown_hi = (centres + radii[:, None]).max(axis=0) + pad + margin
    slack_lo = grown_lo - np.array(column[LO]) * 1e-4
    slack_hi = np.array(column[HI]) * 1e-4 - grown_hi
    assert (slack_lo > 0).all() and (slack_lo < 1e-4 + 2e-9).all()
    assert (slack_hi > 0).all() and (slack_hi < 1e-4 + 2e-9).all()

    # exactly the parts minus that column, cell for cell
    boxes = [b["box"] for b in seat["boxes"]]
    every = boxes + list(parts.values()) + [column]
    cuts = [sorted({b[s][k] for b in every for s in (LO, HI)}) for k in (X, Y, Z)]
    whole = _occupied(list(parts.values()), cuts)
    carved = whole & ~_occupied([column], cuts)
    assert np.array_equal(_occupied(boxes, cuts), carved)
    assert carved.any() and carved.sum() < whole.sum()  # the base: it did carve

    # padded, the boxes clear the spheres by the seat's margin, and the file
    # says by how much
    clear = _clearance(boxes, spheres, pad)
    assert clear >= margin
    assert seat["clearance_m"] == pytest.approx(clear, abs=1e-9)
    meta = yaml.safe_load(result["files"][data["outputs"]["seat"]])["meta"]
    assert meta["keep_out_margin_m"] == margin
    assert meta["keep_out_clearance_m"] == pytest.approx(clear, abs=5e-5)

    # a part whole keeps its name; pieces are numbered
    names = [b["name"] for b in seat["boxes"]]
    for part in parts:
        pieces = sorted(n for n in names if n == part or n.startswith(part + "_"))
        numbered = ["%s_%d" % (part, k) for k in range(len(pieces))]
        assert pieces == [part] or (len(pieces) > 1 and pieces == numbered)

    # the margin is the seat's own: at the person's, the seat comes as close
    # as the person does
    narrow = _inputs()
    narrow["seat"]["keep_out_margin_m"] = narrow["keep_out_margin_m"]
    near = _generate(narrow, FAR_BOX)
    assert near["seat"]["clearance_m"] < margin
    assert near["seat"]["clearance_m"] == pytest.approx(
        near["person"]["clearance_m"], abs=2e-4
    )


def _union(boxes):
    """The union of `boxes`, (lo, hi) each: the grid cut at their faces, and
    which of its cells they fill."""
    cuts = [sorted({b[side][k] for b in boxes for side in (LO, HI)}) for k in (X, Y, Z)]
    return cuts, _occupied(boxes, cuts)


def _within(box, union):
    """Whether `box`, (lo, hi), lies inside `union` (_union's): inside its
    grid, and every cell the box reaches into filled."""
    cuts, filled = union
    lo, hi = box
    if any(lo[k] < c[0] or hi[k] > c[-1] for k, c in enumerate(cuts)):
        return False
    cells = tuple(
        slice(bisect.bisect_right(c, lo[k]) - 1, bisect.bisect_left(c, hi[k]))
        for k, c in enumerate(cuts)
    )
    return bool(filled[cells].all())


def _segments(data, sizes, backrest):
    """The segments of an occupant of `sizes` (inches, by key), seated as
    `data` seats the person with its back against `backrest`, the backrest
    face's range: [(lo, hi)] in the mebot frame, metres, margin and all."""
    person = data["person"]
    occupant = {
        "ansur_in": {key: {"inches": inches} for key, inches in sizes.items()},
        "seat": dict(person["seat"], backrest_face_x=backrest),
        "margin_m": person["margin_m"],
        "arm_side": person["arm_side"],
    }
    return [(lo, hi) for _, lo, hi in gen._person_segments({"person": occupant})]


def _occupants(data):
    """Every occupant the person layer holds (OCCUPANTS_IN), seated as `data`
    seats the person: each size at either end of its range (4096 occupants),
    with its back at either end of the backrest face's range. Yields (sizes,
    face, segments)."""
    faces = sorted(data["person"]["seat"]["backrest_face_x"])
    keys = sorted(OCCUPANTS_IN)
    for ends in itertools.product((0, 1), repeat=len(keys)):
        sizes = {key: OCCUPANTS_IN[key][end] for key, end in zip(keys, ends)}
        for face in faces:
            yield sizes, face, _segments(data, sizes, [face, face])


def test_the_person_holds_every_occupant_of_its_range():
    # Over-sized for every occupant from the smaller sex's 5th percentile to
    # the larger sex's 95th, not for the largest alone: each of them (every
    # size at either end of its range, 4096 occupants), with its back at
    # either end of the backrest face's CAD range, lies inside the envelope,
    # margin and all.
    placed = _inputs()
    faces = sorted(placed["person"]["seat"]["backrest_face_x"])
    envelope = _union(_segments(placed, OCCUPANT_IN, faces))
    for sizes, face, segments in _occupants(placed):
        for box in segments:
            assert _within(box, envelope), (sizes, face)

    # Sized by the largest occupant alone, four segments leave the smallest
    # one out: the torso and the forearms start above its lower elbows and
    # thinner thighs, and the legs leave its shins and feet behind their back
    # face (the shorter the thigh, the nearer the seat the shins hang). Its
    # head is held: below the largest's chin it is inside the largest's
    # torso.
    largest = {key: high for key, (_, high) in OCCUPANTS_IN.items()}
    smallest = {key: low for key, (low, _) in OCCUPANTS_IN.items()}
    alone = _union(_segments(placed, largest, faces))
    held = [_within(box, alone) for box in _segments(placed, smallest, faces[:1] * 2)]
    assert held == [True, False, True, True, False, False]  # SEGMENTS' order


# --------------------------------------------------------------- determinism
BOX_LINE = re.compile(
    r"^  - \{name: (\w+), position: \[(.+)\], dims: \[(.+)\]\}(  # \+ .+)?$"
)


def test_output_is_deterministic_under_float_noise(data):
    # Bounds ON grid lines (the 0.2 m cube at x 1.0) are where an ulp of FK
    # noise would flip a floor or a ceil; the yawed box and the wheel give
    # bounds off the grid, and the budget forces merges.
    urdf = _urdf(
        _link(
            "chassis",
            _box("0.2 0.2 0.2", "1 0 0"),
            _box("0.3 0.1 0.2", "1.2 0.1 0", "0.3 -0.2 0.5"),
            _box("0.05 0.05 0.05", "1 0 0"),
        ),
        _link("wheel", _cylinder("0.15", "0.05", rpy="1.5707963 0 -1.6292068e-07")),
        _joint("axle", "revolute", "chassis", "wheel", "1 0.2 0", extra=_limit(-1, 1)),
        _link("caster", CUBE),
        _joint("stem", "fixed", "chassis", "caster", "1.5 0 -0.1"),
    )
    data["platform"] = {"max_boxes": 2, "max_over_pct": 100.0}

    def bounds(result):
        return [v for p in result["platform"]["primitives"] for v in p["lo"] + p["hi"]]

    exact = _generate(data, urdf)
    files = exact["files"]
    assert _generate(data, urdf)["files"] == files
    for noise in (1e-12, -1e-12):
        noisy = _generate(data, urdf, perturb=noise)
        # the noise did reach every bound that was then snapped
        assert all(a != b for a, b in zip(bounds(exact), bounds(noisy)))
        assert noisy["files"] == files
    # The key order is the generator's, not the data file's (safe_dump sorts).
    assert gen.generate(yaml.safe_load(yaml.safe_dump(data)), urdf)["files"] == files

    assert sorted(files) == sorted(data["outputs"].values())
    for text in files.values():
        # what pre-commit would otherwise rewrite, breaking the diff-assert
        assert text.isascii() and "\r" not in text and "\t" not in text
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert all(line == line.rstrip() for line in text.splitlines())
        assert text.startswith("# GENERATED by scripts/generate_layers.py")

        layer = yaml.safe_load(text)
        # every file says what it was built for, and how close it comes
        meta = layer["meta"]
        assert meta["arm_base_in_mebot"] == data["mount"]["arm_base_in_mebot"]
        assert meta["grid_m"] == data["grid_m"]
        assert meta["padding_assumed_m"] == data["padding_assumed_m"]
        assert meta["keep_out_clearance_m"] >= data["keep_out"]["min_clearance_m"]

        names = [o["name"] for o in layer["obstacles"]]
        assert names == sorted(names)
        lines = _obstacle_lines(text)
        assert len(lines) == len(names)
        for line in lines:
            match = BOX_LINE.match(line)
            assert match, line
            position = match.group(2).split(", ")
            dims = match.group(3).split(", ")
            assert all(re.fullmatch(r"-?\d+\.\d{5}", v) for v in position), line
            assert all(re.fullmatch(r"\d+\.\d{4}", v) for v in dims), line
            for centre, size in zip(position, dims):
                half = Decimal(size) / 2
                assert half > 0, line
                # both faces on the 0.1 mm grid
                for face in (Decimal(centre) - half, Decimal(centre) + half):
                    assert face % Decimal("0.0001") == 0, line

    platform = yaml.safe_load(files[data["outputs"]["platform"]])
    assert len(platform["obstacles"]) == 2  # the merges happened


# ----------------------------------------------------------------------- CLI
def test_cli_writes_only_after_every_gate_and_check_prints_a_diff(
    isolated, monkeypatch, capsys
):
    tmp_path = isolated
    urdf = _urdf(
        _link("chassis", _box("0.1 0.1 0.1", "1 0 0"), _box("0.1 0.1 0.1", "1.3 0 0"))
    )
    clone, commit = _clone(tmp_path / "clone", "chair.urdf", urdf)
    pinned = hashlib.sha256(urdf).hexdigest()
    # A throwaway checkout with its package beside it, and the command run
    # from another directory: the outputs are relative to the repository root.
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    packaged = repo / "core" / "rammp_curobo" / "configs"
    packaged.mkdir(parents=True)
    elsewhere.mkdir()
    shutil.copy(PACKAGED_CONFIG_DIR / "robot_gen3_2f85.yaml", packaged)
    monkeypatch.setattr(gen, "REPO", repo)
    monkeypatch.setattr(gen, "PACKAGED_CONFIG_DIR", packaged)
    monkeypatch.chdir(elsewhere)
    out = repo / "out"
    layers = ("platform", "person", "seat", "floor")
    names = {layer: "%s.yaml" % layer for layer in layers}

    def run(*flags, into="out", max_boxes=2, sha256=pinned, clone=clone, cache="cache"):
        data = _inputs()
        data["source"] = {
            "repo": "example/chair",
            "commit": commit,
            "path": "chair.urdf",
            "sha256": sha256,
        }
        data["platform"]["max_boxes"] = max_boxes
        data["outputs"] = {layer: "%s/%s" % (into, names[layer]) for layer in names}
        (tmp_path / "data.yaml").write_text(yaml.safe_dump(data))
        argv = ["--data", str(tmp_path / "data.yaml"), "--clone", str(clone)]
        argv += ["--cache", str(tmp_path / cache), *flags]
        return gen.main(argv), capsys.readouterr()

    status, said = run()
    assert status == 0
    assert sorted(os.listdir(out)) == sorted(names.values())  # and no temp files
    written = {layer: (out / name).read_text() for layer, name in names.items()}
    loaded = {layer: yaml.safe_load(text) for layer, text in written.items()}
    assert {layer: doc["layer"] for layer, doc in loaded.items()} == {
        "platform": "platform",
        "person": "person",
        "seat": "seat",
        "floor": "floor",
    }
    assert {layer: doc["excludable"] for layer, doc in loaded.items()} == {
        "platform": False,
        "person": True,
        "seat": False,
        "floor": False,
    }
    boxes = [o["name"] for o in loaded["platform"]["obstacles"]]
    assert boxes == ["chassis_0", "chassis_1"]
    # the report says where the URDF and the arm's spheres came from (on a
    # host with several checkouts, which one) and what was written
    assert str(clone) in said.out
    assert str(packaged / "robot_gen3_2f85.yaml") in said.out
    assert all(str(out / name) in said.out for name in names.values())
    assert os.listdir(tmp_path / "cache") == [pinned + ".urdf"]

    # --check: silent and 0 while the files are what the generator writes.
    status, said = run("--check")
    assert status == 0 and "@@" not in said.out

    # A hand edit: --check exits 1 with a unified diff and leaves the file be.
    edited = written["platform"].replace("excludable: false", "excludable: true")
    (out / names["platform"]).write_text(edited)
    status, said = run("--check")
    assert status == 1
    assert "-excludable: true" in said.out and "+excludable: false" in said.out
    assert "@@" in said.out and str(out / names["platform"]) in said.out
    assert (out / names["platform"]).read_text() == edited

    # A missing file is a difference too.
    (out / names["floor"]).unlink()
    status, said = run("--check")
    assert status == 1 and "+layer: floor" in said.out
    assert not (out / names["floor"]).exists()

    # A wrong pin: non-zero, and the files stay as they were.
    status, said = run(sha256=hashlib.sha256(b"another file").hexdigest())
    assert status != 0 and "sha256" in said.err
    assert (out / names["platform"]).read_text() == edited
    assert not (out / names["floor"]).exists()

    # A write that fails at its last step, the rename: the file is replaced
    # whole or not at all, and no temp file is left beside it.
    def refuse(src, dst):
        raise OSError("rename refused")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", refuse)
        with pytest.raises(OSError, match="rename refused"):
            run()
    capsys.readouterr()
    assert (out / names["platform"]).read_text() == edited
    assert sorted(os.listdir(out)) == sorted(
        names[layer] for layer in ("person", "platform", "seat")
    )

    # Regenerating puts them back.
    status, said = run()
    assert status == 0
    assert {layer: (out / name).read_text() for layer, name in names.items()} == written

    # A failed gate (one box is 99.7 % over, the budget is 15 %): non-zero,
    # and nothing is written, not even the directory.
    status, said = run(into="refused", max_boxes=1)
    assert status != 0 and "99.7" in said.err
    assert not (repo / "refused").exists()

    # --offline with no cache and no clone: non-zero, and GitHub is not asked
    # (the fixture's urlopen would raise).
    status, said = run("--offline", clone=tmp_path / "missing", cache="empty")
    assert status != 0 and "raw.githubusercontent.com" in said.err

    # A cache that cannot be written does not stop the run; the report says so.
    (tmp_path / "blocker").write_text("a file where the cache directory should go")
    status, said = run(cache="blocker/cad")
    assert status == 0 and "cache not written" in said.out

    # Nothing was written relative to the directory the command ran in.
    assert os.listdir(elsewhere) == []

    # Without --data the inputs are scripts/mebot_layers.yaml: it is the real
    # pin that is asked for (and, isolated and offline, not found).
    status = gen.main(["--offline", "--check"])
    assert status != 0
    assert gen.load_data()["source"]["path"] in capsys.readouterr().err


# ----------------------------------------------------------- committed files
def _pinned_urdf(source):
    """The pinned URDF, from the cache, a Demo-Software clone or GitHub raw.

    Where none of them has it the test is skipped, but not in CI (GitHub sets
    CI=true): there a skip would leave the committed layers unchecked.
    """
    try:
        urdf, _ = gen.fetch_pinned(source)
    except FileNotFoundError as exc:
        if os.environ.get("CI") == "true":
            raise
        pytest.skip("the pinned URDF is not available: %s" % exc)
    return urdf


def _committed(layer):
    """Where the committed file of `layer` (platform, person, floor) is."""
    return REPO / gen.load_data()["outputs"][layer]


def test_committed_layers_regenerate_byte_identical():
    # `make layers` as a test, so that CI regenerates the files on both of its
    # architectures: a hand edit fails it, and so does an input changed without
    # regenerating.
    data = gen.load_data()
    files = gen.generate(data, _pinned_urdf(data["source"]))["files"]
    for path, text in files.items():
        committed = (REPO / path).read_bytes().decode("utf-8", "replace")
        assert committed == text, "%s is not what `make layers` writes" % path


def test_committed_person_regenerates_without_the_urdf():
    # The person is made from the data file and the packaged robot config
    # alone, so unlike the test above this runs wherever the suite does: a
    # hand edit fails it, and so does a change to its inputs in
    # scripts/mebot_layers.yaml without `make layers`. The URDF only has to
    # pass the platform's gates, and a cube far from the arm does.
    data = gen.load_data()
    path = data["outputs"]["person"]
    text = _generate(data, FAR_BOX)["files"][path]
    committed = (REPO / path).read_bytes().decode("utf-8", "replace")
    assert committed == text, "%s is not what `make layers` writes" % path


def _true_shapes(urdf, mount):
    """Each collision solid of `urdf` with every joint at zero, in arm_mount
    (the URDF's root at -`mount`): [(link, (lo, hi), points)].

    Worked out apart from the generator: FK in numpy, the exact bounding box
    (lo, hi) of the solid, and points on and in it: a 5 x 5 x 5 lattice
    through a box, 72 angles x 2 radii x 9 heights through a cylinder.
    """
    robot = ET.fromstring(urdf)

    def frame(origin):
        pose = np.eye(4)
        if origin is not None:
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            pose[:3, :3] = _rpy_matrix(*rpy)
            pose[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        return pose

    joint_to = {j.find("child").get("link"): j for j in robot.findall("joint")}

    def place(link):
        if link not in joint_to:  # the root
            pose = np.eye(4)
            pose[:3, 3] = np.negative(mount)
            return pose
        joint = joint_to[link]
        return place(joint.find("parent").get("link")) @ frame(joint.find("origin"))

    ticks = np.linspace(-0.5, 0.5, 5)
    lattice = np.stack(np.meshgrid(ticks, ticks, ticks), -1).reshape(-1, 3)
    angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    shapes = []
    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            pose = place(link.get("name")) @ frame(collision.find("origin"))
            (solid,) = collision.find("geometry")
            if solid.tag == "box":
                local = lattice * [float(v) for v in solid.get("size").split()]
            else:
                radius, length = (float(solid.get(k)) for k in ("radius", "length"))
                heights = np.linspace(-length / 2, length / 2, 9)
                rho, phi, z = np.meshgrid([radius, radius / 2], angles, heights)
                local = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], -1)
            points = local.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
            if solid.tag == "box":  # the lattice holds its corners
                bounds = points.min(axis=0), points.max(axis=0)
            else:  # the rims' reach along each axis, from the cylinder's axis
                axis = pose[:3, 2]
                reach = np.abs(axis) * length / 2
                reach += radius * np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
                bounds = pose[:3, 3] - reach, pose[:3, 3] + reach
            shapes.append((link.get("name"), bounds, points))
    return shapes


def test_committed_platform_covers_every_urdf_primitive():
    # The committed boxes as the planner loads them, against the URDF and the
    # mount that the file itself names.
    path = _committed("platform")
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))["meta"]
    mount = meta["arm_base_in_mebot"]
    assert mount["rpy_deg"] == [0, 0, 0]
    shapes = _true_shapes(_pinned_urdf(meta["source"]), mount["position"])
    assert len(shapes) == meta["coverage"]["primitives"]
    boxes = load_layer(path).obstacles
    # the planner turns a box by its rpy_deg, and the spans lo..hi below
    # hold only for an unturned one, as the generator writes them all
    assert [o.name for o in boxes if any(o.rpy_deg)] == []
    centre = np.array([o.position for o in boxes])
    half = np.array([o.dims for o in boxes]) / 2
    lo, hi = centre - half - 1e-9, centre + half + 1e-9

    def outside(points, inset=0.0):
        """The points no box holds, each box first moved `inset` in from
        every face."""
        held = (points[:, None] >= lo + inset) & (points[:, None] <= hi - inset)
        return points[~held.all(axis=2).any(axis=1)]

    points = np.concatenate([p for _, _, p in shapes])
    assert len(outside(points)) == 0
    # each solid's exact bounding box lies inside one box: the certificate,
    # checked again
    for link, (a, b), _ in shapes:
        assert ((lo <= a) & (b <= hi)).all(axis=1).any(), link
    # and the samples reach the faces: 1 mm in from every face, some are out
    assert len(outside(points, inset=0.001)) > 0


def test_committed_person_holds_every_occupant_of_its_range():
    # The committed boxes as the planner loads them, against every occupant
    # of OCCUPANTS_IN seated as scripts/mebot_layers.yaml seats the person (its
    # mount, seat and margin): with the arm's column, which no box may fill,
    # they hold each one, margin and all. A size tuned in the data file
    # may grow the person, never shrink it below that range. The sim's G1
    # cannot see that on the back of the knees: its three stand-ins pass the
    # layer below, and first fail near a thigh of 18.5 in.
    data = gen.load_data()
    mount = data["mount"]["arm_base_in_mebot"]["position"]
    committed = load_layer(_committed("person")).obstacles
    # the spans lo..hi below hold only for an unturned box, as the generator
    # writes them all
    assert [o.name for o in committed if any(o.rpy_deg)] == []

    def corner(o, side):
        """The low (side -1) or high (+1) corner of box `o`, in ticks."""
        return tuple(
            round((c + side * d / 2) / gen.TICK) for c, d in zip(o.position, o.dims)
        )

    def lost(boxes):
        """The occupants, (sizes, backrest face), that `boxes` (lo, hi in
        ticks) leave out."""
        layer = _union(boxes)
        return [
            (sizes, back)
            for sizes, back, segments in _occupants(data)
            if not all(
                _within([np.subtract(c, mount) / gen.TICK for c in segment], layer)
                for segment in segments
            )
        ]

    column = _generate(data, FAR_BOX)["person"]["column"]
    assert lost([(corner(o, -1), corner(o, 1)) for o in committed] + [column]) == []

    # The layer generated with the back of the knees at the larger sex's 5th
    # percentile (18.07 in, row 14) leaves out every occupant with the
    # shortest thigh and the longest foot (tucked furthest back under the
    # knee) and its back at the rear of the backrest's range, and no other.
    data["person"]["ansur_in"]["buttock_popliteal_length"]["inches"] = 18.07
    person = _generate(data, FAR_BOX)["person"]
    shorter = lost([b["box"] for b in person["boxes"]] + [person["column"]])
    rear = min(data["person"]["seat"]["backrest_face_x"])
    assert len(shorter) == 2 ** (len(OCCUPANTS_IN) - 2) * 2  # both faces: one range
    assert {
        (s["buttock_popliteal_length"], s["foot_length"], b) for s, b in shorter
    } == {(17.36, 11.54, rear)}


def test_committed_layers_meet_budgets_and_clear_the_arm():
    # Neither the URDF nor the network: the committed files and the packaged
    # robot config are all this reads, so it runs wherever the suite runs.
    data = gen.load_data()
    layers = {name: load_layer(_committed(name)) for name in data["outputs"]}
    meta = {
        name: yaml.safe_load(_committed(name).read_text(encoding="utf-8"))["meta"]
        for name in layers
    }
    assert {name: (layer.name, layer.excludable) for name, layer in layers.items()} == {
        "platform": ("platform", False),
        "person": ("person", True),
        "seat": ("seat", False),
        "floor": ("floor", False),
    }
    for layer in layers.values():
        names = [o.name for o in layer.obstacles]
        assert len(set(names)) == len(names), layer.name
    # the platform's budget, and more than one box: a single one is 158 % over,
    # and padded it reaches into the arm's base sphere
    assert 1 < len(layers["platform"].obstacles) <= 16
    assert meta["platform"]["coverage"]["over_pct"] <= 15.0

    # The arm's fixed spheres, worked out again from the packaged robot config:
    # arm_base_link's, and arm_shoulder_link's placed by arm_joint_1's origin, each radius
    # with the buffer.
    robot = PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"]
    robot_cfg = yaml.safe_load(robot.read_text(encoding="utf-8"))["robot_cfg"]
    buffer = robot_cfg["kinematics"]["collision_sphere_buffer"]
    spheres = robot_cfg["kinematics"]["collision_spheres"]
    joint_1 = data["keep_out"]["shoulder_link_origin"]
    turn = _rpy_matrix(*joint_1["rpy"])
    fixed = [(s["center"], s["radius"] + buffer) for s in spheres["arm_base_link"]]
    fixed += [
        (turn @ s["center"] + joint_1["xyz"], s["radius"] + buffer)
        for s in spheres["arm_shoulder_link"]
    ]
    # tilted on arm_mount as the mount the layers name
    tilt = _rpy_matrix(*np.radians(data["mount"]["arm_base_joint"]["rpy_deg"]))
    fixed = [(tilt @ np.asarray(c), r) for c, r in fixed]
    for name, layer in layers.items():
        # padded as the planner pads them
        alone = Scene("arm_mount", [], [], layers=[layer])
        boxes, _ = compose_cuboids(alone, padding=meta[name]["padding_assumed_m"])
        clear = min(min(point_box_distance(c, boxes).values()) - r for c, r in fixed)
        assert clear >= 0.004, name
        # and the file says how close they come
        claimed = meta[name]["keep_out_clearance_m"]
        assert claimed == pytest.approx(clear, abs=5e-5), name
        # the seat, around the arm's base, is kept clear by its own, wider
        # margin: beyond the planner's collision activation distance
        if name == "seat":
            margin = data["seat"]["keep_out_margin_m"]
            assert meta[name]["keep_out_margin_m"] == margin
            assert clear >= margin


def test_outdir_writes_the_layer_set_beside_the_robot_that_uses_it(
    isolated, monkeypatch, capsys
):
    """The layers stop being package data and become a robot's own files
    (issue #19, Swapnil's review of #28), so the generator has to be able to
    write them somewhere that is not this checkout. `--outdir` writes the set
    there by file name, byte for byte what the in-repo copy gets, and writes
    nothing at the declared outputs: one generator, two destinations, no
    second rendering of the text to drift.
    """
    tmp_path = isolated
    urdf = _urdf(
        _link("chassis", _box("0.1 0.1 0.1", "1 0 0"), _box("0.1 0.1 0.1", "1.3 0 0"))
    )
    clone, commit = _clone(tmp_path / "clone", "chair.urdf", urdf)
    repo = tmp_path / "repo"
    packaged = repo / "core" / "rammp_curobo" / "configs"
    packaged.mkdir(parents=True)
    shutil.copy(PACKAGED_CONFIG_DIR / "robot_gen3_2f85.yaml", packaged)
    monkeypatch.setattr(gen, "REPO", repo)
    monkeypatch.setattr(gen, "PACKAGED_CONFIG_DIR", packaged)
    names = {
        layer: "layer_mebot_%s.yaml" % layer
        for layer in ("platform", "person", "seat", "floor")
    }

    def run(*flags):
        data = _inputs()
        data["source"] = {
            "repo": "example/chair",
            "commit": commit,
            "path": "chair.urdf",
            "sha256": hashlib.sha256(urdf).hexdigest(),
        }
        data["platform"]["max_boxes"] = 2
        data["outputs"] = {
            layer: "core/tests/data/layers/%s" % name for layer, name in names.items()
        }
        (tmp_path / "data.yaml").write_text(yaml.safe_dump(data))
        argv = ["--data", str(tmp_path / "data.yaml"), "--clone", str(clone)]
        argv += ["--cache", str(tmp_path / "cache"), *flags]
        status = gen.main(argv)
        capsys.readouterr()
        return status

    # the in-repo copy, as `make layers` writes it
    assert run() == 0
    fixtures = repo / "core" / "tests" / "data" / "layers"
    in_repo = {name: (fixtures / name).read_bytes() for name in names.values()}

    # and the robot's own copy, somewhere else entirely
    robot = tmp_path / "deployments" / "config" / "robots" / "mebot" / "curobo"
    assert run("--outdir", str(robot)) == 0
    assert sorted(p.name for p in robot.iterdir()) == sorted(names.values())
    assert {p.name: p.read_bytes() for p in robot.iterdir()} == in_repo

    # --outdir writes THERE and nowhere else: a stale in-repo copy is not
    # quietly refreshed by a run aimed at a robot
    (fixtures / names["floor"]).write_bytes(b"layer: tampered\n")
    assert run("--outdir", str(robot)) == 0
    assert (fixtures / names["floor"]).read_bytes() == b"layer: tampered\n"

    # and --check reads the copy it is pointed at, so the robot's files can be
    # gated the same way the in-repo ones are
    assert run("--check", "--outdir", str(robot)) == 0
    (robot / names["floor"]).write_bytes(b"layer: tampered\n")
    assert run("--check", "--outdir", str(robot)) == 1
