"""The layer generator (scripts/generate_layers.py), offline and CPU only.

The generator's logic is exercised on synthetic URDFs and a throwaway git
clone, so these tests need no network, no cache and no Demo-Software checkout.
"""

import hashlib
import importlib.util
import io
import math
import os
import re
import shutil
import subprocess
import urllib.request
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import yaml

from rammp_curobo.config import PACKAGED_CONFIG_DIR

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
SEGMENTS = ("torso_head", "lap", "legs")


@pytest.fixture
def data():
    return gen.load_data()


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


def _limit(lower, upper):
    return '<limit lower="%s" upper="%s" effort="1" velocity="1"/>' % (lower, upper)


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
    """A git repository holding `content` at `path`: (repo, commit)."""
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
    return repo, git("rev-parse", "HEAD")


# ------------------------------------------------------------------------ FK
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
        # base_link is the root frame moved by -mount
        in_base = [[v - m for v, m in zip(bound, mount)] for bound in (lo, hi)]
        assert p["lo"] == pytest.approx(in_base[LO], abs=1e-12), name
        assert p["hi"] == pytest.approx(in_base[HI], abs=1e-12), name
    assert primitives["wheel"]["size"] == [0.1, 0.04]  # radius, length
    assert primitives["rod"]["size"] == [0.02, 0.04, 0.06]
    # The pose is base_link from the primitive: the rod's axes land on z, x, y.
    rod = np.array(primitives["rod"]["pose"])
    assert rod[:3, :3] == pytest.approx(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert rod[:3, 3] == pytest.approx([1.1 - 1.0, 0.5 + 2.0, 0.0 - 0.5])

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


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    about_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    about_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    about_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


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


def test_unsupported_input_is_refused(data, tmp_path):
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
        # sizes cuRobo would turn into a box that can never collide
        (_urdf(_link("nanbox", _box("0.1 nan 0.1"))), ["nanbox"]),
        (_urdf(_link("infbox", _box("0.1 inf 0.1"))), ["infbox"]),
        (_urdf(_link("flat", _box("0.1 0 0.1"))), ["flat"]),
        (_urdf(_link("inverted", _box("0.1 -0.1 0.1"))), ["inverted"]),
        (_urdf(_link("short", _box("0.1 0.1"))), ["short"]),
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
        (_urdf(_link("twin", CUBE), _link("twin", CUBE)), ["twin"]),
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

    # A rotated mount: the boxes are axis-aligned in base_link.
    rotated = gen.load_data()
    rotated["mount"]["arm_base_in_mebot"]["rpy_deg"] = [0, 0, 90]
    with pytest.raises(ValueError, match="rpy_deg"):
        _generate(rotated, FAR_BOX)

    # A grid the printed decimals do not match.
    coarse = gen.load_data()
    coarse["grid_m"] = 0.001
    with pytest.raises(ValueError, match="grid_m"):
        _generate(coarse, FAR_BOX)

    # Bytes that are not the pinned ones (the data file pins the real URDF).
    with pytest.raises(ValueError, match="sha256"):
        gen.generate(gen.load_data(), FAR_BOX)

    # A name that would put non-ASCII bytes in a layer file.
    with pytest.raises(ValueError, match="ASCII"):
        _generate(gen.load_data(), _urdf(_link("caf\u00e9", CUBE)))

    # A shoulder sphere off the joint_1 axis moves with joint_1: it is not
    # part of the fixed column, and treating it as fixed would be wrong.
    spheres = {
        "base_link": [{"center": [0.0, 0.0, 0.065], "radius": 0.075}],
        "shoulder_link": [{"center": [0.01, 0.0, -0.04], "radius": 0.07}],
    }
    kinematics = {"collision_sphere_buffer": 0.005, "collision_spheres": spheres}
    robot = tmp_path / "robot.yaml"
    robot.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kinematics}}))
    data["keep_out"]["robot_config"] = str(robot)
    with pytest.raises(ValueError, match="shoulder_link"):
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
    assert str(clone) in origin and "mismatch" in origin
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


# --------------------------------------------------------------------- merge
def _prim(name, lo, hi):
    return {"name": name, "box": (tuple(lo), tuple(hi))}


def test_merge_meets_the_budget_and_never_touches_the_keep_out(data):
    sphere = [((0.0, 0.0, 0.0), 0.05, "base_link")]
    pad, margin = 0.02, 0.005

    # The measure: distance from a sphere to the PADDED box, minus the radius.
    face_on = ((1000, -500, -500), (2000, 500, 500))
    assert gen.clearance(face_on, sphere, 0.02) == pytest.approx(0.1 - 0.02 - 0.05)
    corner = ((1000, 1000, 1000), (2000, 2000, 2000))
    assert gen.clearance(corner, sphere, 0.0) == pytest.approx(0.1 * 3**0.5 - 0.05)
    # a centre inside the box counts as minus the depth to the nearest face
    around = ((-1000, -1000, -1000), (1000, 1000, 3000))
    assert gen.clearance(around, sphere, 0.02) == pytest.approx(-0.12 - 0.05)

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
    f = _prim("f", (0, 0, 0), (1000, 100, 100))
    g = _prim("g", (1100, 0, 0), (2100, 100, 100))
    j = _prim("j", (10000, 0, 0), (10100, 100, 100))
    k = _prim("k", (10200, 0, 0), (10300, 100, 100))
    short_first = gen.merge([f, g, j, k], 3, [], pad, margin)
    assert sorted(c["members"] for c in short_first) == [["f"], ["g"], ["j", "k"]]

    # Coming within the margin is enough to skip a merge, without containing
    # the sphere: the padded box of p+q would touch it (clearance 0) at
    # x 0.07, and clears it by 10 mm at x 0.08.
    def beside(x):
        return [
            _prim("p", (x, 2000, -500), (1000, 3000, 500)),
            _prim("q", (x, -3000, -500), (1000, -2000, 500)),
        ]

    with pytest.raises(ValueError, match="keep-out"):
        gen.merge(beside(700), 1, sphere, pad, margin)
    (both,) = gen.merge(beside(800), 1, sphere, pad, margin)
    assert both["box"] == ((800, -3000, -500), (1000, 3000, 500))

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

    # As written: a merged box carries its largest member's name, and the
    # other members ride in a comment.
    hull_and_peg = _urdf(
        _link("hull", _box("0.2 0.2 0.2", "1 0 0")),
        _link("peg", _box("0.05 0.05 0.05")),
        _joint("stem", "fixed", "hull", "peg", "1.3 0 0"),
    )
    roomy = gen.load_data()
    roomy["platform"] = {"max_boxes": 1, "max_over_pct": 1000.0}
    files = _generate(roomy, hull_and_peg)["files"]
    (line,) = _obstacle_lines(files[roomy["outputs"]["platform"]])
    hull = ("hull", "0.60750, 0.26500, -0.37600", "0.4252, 0.2002, 0.2002")
    assert line == _box_lines(hull)[0] + "  # + peg"

    # A primitive that is itself inside the arm's column cannot be merged
    # away from it: the layer is refused.
    at_the_arm = _urdf(_link("chassis", _box("0.1 0.1 0.1", "0.505 -0.265 0.376")))
    with pytest.raises(ValueError) as caught:
        _generate(data, at_the_arm)
    assert "platform" in str(caught.value) and "chassis" in str(caught.value)


# ------------------------------------------------------------------ coverage
def _meta(urdf, **budget):
    data = gen.load_data()
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

    sound = gen.merge

    def defective(*args):
        clusters = sound(*args)
        (x0, y0, z0), (x1, y1, z1) = clusters[0]["box"]
        if defect == "drop":
            return clusters[1:]
        if defect == "shrink_1_tick":
            clusters[0]["box"] = ((x0, y0, z0), (x1 - 1, y1, z1))
        else:
            clusters[0]["box"] = ((x0 + 10, y0, z0), (x1 + 10, y1, z1))
        return clusters

    monkeypatch.setattr(gen, "merge", defective)
    with pytest.raises(ValueError, match="chassis_"):
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

    # The arm's fixed spheres, from the packaged robot config: base_link's as
    # they are, shoulder_link's carried through joint_1's origin (a half turn
    # about x, so a sphere at local z sits at `up - z`), all with the buffer.
    spheres = gen.keep_out_spheres(data)
    robot = PACKAGED_CONFIG_DIR / data["keep_out"]["robot_config"]
    kinematics = yaml.safe_load(robot.read_text())["robot_cfg"]["kinematics"]
    base = kinematics["collision_spheres"]["base_link"]
    shoulder = kinematics["collision_spheres"]["shoulder_link"]
    up = data["keep_out"]["shoulder_link_origin"]["xyz"][Z]
    links = ["base_link"] * len(base) + ["shoulder_link"] * len(shoulder)
    assert [link for _, _, link in spheres] == links
    centres = np.array([c for c, _, _ in spheres])
    radii = np.array([r for _, r, _ in spheres])
    assert np.abs(centres[:, :Z]).max() < 1e-5  # all on the joint_1 axis
    heights = [s["center"][Z] for s in base] + [up - s["center"][Z] for s in shoulder]
    assert centres[:, Z] == pytest.approx(heights, abs=1e-6)
    buffer = kinematics["collision_sphere_buffer"]
    assert radii == pytest.approx([s["radius"] + buffer for s in base + shoulder])

    margin = data["keep_out_margin_m"]
    # The real mount and padding, where only the torso meets the column, and a
    # mount that stands the column in all three segments, with more padding.
    for mount, pad in (([0.505, -0.265, 0.376], 0.02), ([0.65, 0.1, 0.30], 0.03)):
        placed = gen.load_data()
        placed["mount"]["arm_base_in_mebot"]["position"] = mount
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

        # The file says what it was built from, and its volumes are the
        # cells' (summed here, not by the generator).
        meta = yaml.safe_load(result["files"][placed["outputs"]["person"]])["meta"]
        rows = placed["person"]["ansur_p95_in"]
        assert meta["ansur_p95_in"] == {key: row["inches"] for key, row in rows.items()}
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

    # Each parameter moves exactly its faces. Lengths grow by 0.1 m (1000
    # ticks; a breadth moves each side by half of it).
    base_faces = _faces(gen.load_data())
    grow = 0.1 / 0.0254
    all_z = {(s, side, Z): 1000 for s in SEGMENTS for side in (LO, HI)}
    all_y = {(s, side, Y): 1000 for s in SEGMENTS for side in (LO, HI)}
    wider = {("torso_head", LO, Y): -500, ("torso_head", HI, Y): 500}
    wider.update({("lap", LO, Y): -500, ("lap", HI, Y): 500})
    chest = {("torso_head", HI, X): 1000, ("lap", LO, X): 1000}
    ansur = {
        "sitting_height": {("torso_head", HI, Z): 1000},
        "sitting_elbow_height": {("lap", HI, Z): 1000},
        "hand_breadth": {("lap", HI, Z): 1000},
        "buttock_knee_length": {("lap", HI, X): 1000, ("legs", HI, X): 1000},
        "buttock_popliteal_length": {("legs", LO, X): 1000},
        "popliteal_height": {("legs", LO, Z): -1000},
        "shoulder_breadth_bideltoid": wider,
        "hip_breadth": {("legs", LO, Y): -500, ("legs", HI, Y): 500},
        "abdominal_depth": chest,
        "foot_length": {("legs", HI, X): 1000},
    }
    seat = {
        "pan_top_z": all_z,
        "backrest_rear_x": {("torso_head", LO, X): 1000},
        "centreline_y": all_y,
    }

    def moved(change):
        changed = gen.load_data()
        change(changed["person"], changed["mount"]["arm_base_in_mebot"]["position"])
        faces = _faces(changed)
        return {f: faces[f] - base_faces[f] for f in faces if faces[f] != base_faces[f]}

    def inches(key, amount):
        def change(person, mount):
            person["ansur_p95_in"][key]["inches"] += amount

        return change

    def anchor(key, amount):
        def change(person, mount):
            person["seat"][key] += amount

        return change

    for key, want in ansur.items():
        assert moved(inches(key, grow)) == want, key
    for key, want in seat.items():
        assert moved(anchor(key, 0.1)) == want, key

    # The arm's mount moves the whole envelope the other way in base_link.
    for axis in (X, Y, Z):

        def shift(person, mount, axis=axis):
            mount[axis] += 0.1

        every_face = {(s, side, axis): -1000 for s in SEGMENTS for side in (LO, HI)}
        assert moved(shift) == every_face, axis

    # The smaller term of a max() moves nothing until it is the larger one:
    # the thighs under the forearms, the chest behind the abdomen.
    def raise_thigh(person, mount):
        rows = person["ansur_p95_in"]
        forearms = (
            rows["sitting_elbow_height"]["inches"] + rows["hand_breadth"]["inches"]
        )
        rows["thigh_clearance"]["inches"] = forearms + grow

    def raise_chest(person, mount):
        rows = person["ansur_p95_in"]
        rows["chest_depth"]["inches"] = rows["abdominal_depth"]["inches"] + grow

    assert moved(inches("thigh_clearance", grow / 10)) == {}
    assert moved(inches("chest_depth", grow / 10)) == {}
    assert moved(raise_thigh) == {("lap", HI, Z): 1000}
    assert moved(raise_chest) == chest

    # The backrest face is a range: its front sets the chest and the knees
    # (and the feet), its rear sets the back of the knees.
    def backrest(index):
        def change(person, mount):
            person["seat"]["backrest_face_x"][index] += 0.1

        return change

    knees = {("lap", HI, X): 1000, ("legs", HI, X): 1000}
    assert moved(backrest(1)) == {**chest, **knees}
    assert moved(backrest(0)) == {("legs", LO, X): 1000}

    # The margin is on every face. Where two segments meet (the chest front,
    # the pan underside) the face moves with the segment that owns it.
    def margin_up(person, mount):
        person["margin_m"] += 0.01

    assert moved(margin_up) == {
        ("torso_head", LO, X): -100,
        ("torso_head", HI, X): 100,
        ("torso_head", LO, Y): -100,
        ("torso_head", HI, Y): 100,
        ("torso_head", LO, Z): -100,
        ("torso_head", HI, Z): 100,
        ("lap", LO, X): 100,
        ("lap", HI, X): 100,
        ("lap", LO, Y): -100,
        ("lap", HI, Y): 100,
        ("lap", LO, Z): -100,
        ("lap", HI, Z): 100,
        ("legs", LO, X): -100,
        ("legs", HI, X): 100,
        ("legs", LO, Y): -100,
        ("legs", HI, Y): 100,
        ("legs", LO, Z): -100,
        ("legs", HI, Z): -100,
    }

    # Carved back by only 1 mm, the padded person comes within 4 mm of the
    # arm: such a layer is not written. Nor is the real one (5.1 mm clear)
    # under a 6 mm gate.
    tight = gen.load_data()
    tight["keep_out_margin_m"] = 0.001
    with pytest.raises(ValueError, match="person"):
        _generate(tight, FAR_BOX)
    strict = gen.load_data()
    strict["keep_out"]["min_clearance_m"] = 0.006
    with pytest.raises(ValueError, match="person"):
        _generate(strict, FAR_BOX)


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

    files = _generate(data, urdf)["files"]
    assert _generate(data, urdf)["files"] == files
    mount = data["mount"]["arm_base_in_mebot"]["position"]
    exact = [p["lo"] + p["hi"] for p in gen.urdf_primitives(urdf, mount)]
    for noise in (1e-12, -1e-12):
        noisy = [p["lo"] + p["hi"] for p in gen.urdf_primitives(urdf, mount, noise)]
        assert all(a != b for a, b in zip(sum(exact, []), sum(noisy, [])))
        assert _generate(data, urdf, perturb=noise)["files"] == files
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
def test_cli_writes_only_after_every_gate_and_check_prints_a_diff(isolated, capsys):
    tmp_path = isolated
    urdf = _urdf(
        _link("chassis", _box("0.1 0.1 0.1", "1 0 0"), _box("0.1 0.1 0.1", "1.3 0 0"))
    )
    clone, commit = _clone(tmp_path / "clone", "chair.urdf", urdf)
    pinned = hashlib.sha256(urdf).hexdigest()
    out = tmp_path / "out"
    names = {layer: "%s.yaml" % layer for layer in ("platform", "person", "floor")}

    def run(*flags, out=out, max_boxes=2, sha256=pinned, clone=clone, cache="cache"):
        data = gen.load_data()
        data["source"] = {
            "repo": "example/chair",
            "commit": commit,
            "path": "chair.urdf",
            "sha256": sha256,
        }
        data["platform"]["max_boxes"] = max_boxes
        data["outputs"] = {layer: str(out / name) for layer, name in names.items()}
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
        "floor": "floor",
    }
    assert {layer: doc["excludable"] for layer, doc in loaded.items()} == {
        "platform": False,
        "person": True,
        "floor": False,
    }
    boxes = [o["name"] for o in loaded["platform"]["obstacles"]]
    assert boxes == ["chassis_0", "chassis_1"]
    # the report says where the URDF and the arm's spheres came from (on a
    # host with several checkouts, which one) and what was written
    assert str(clone) in said.out
    assert str(PACKAGED_CONFIG_DIR / "robot_gen3_2f85.yaml") in said.out
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

    # Regenerating puts them back.
    status, said = run()
    assert status == 0
    assert {layer: (out / name).read_text() for layer, name in names.items()} == written

    # A failed gate (one box is 99.7 % over, the budget is 15 %): non-zero,
    # and nothing is written, not even the directory.
    status, said = run(out=tmp_path / "refused", max_boxes=1)
    assert status != 0 and "99.7" in said.err
    assert not (tmp_path / "refused").exists()

    # --offline with no cache and no clone: non-zero, and GitHub is not asked
    # (the fixture's urlopen would raise).
    status, said = run("--offline", clone=tmp_path / "missing", cache="empty")
    assert status != 0 and "raw.githubusercontent.com" in said.err

    # A cache that cannot be written does not stop the run; the report says so.
    (tmp_path / "blocker").write_text("a file where the cache directory should go")
    status, said = run(cache="blocker/cad")
    assert status == 0 and "cache not written" in said.out

    # Without --data the inputs are scripts/mebot_layers.yaml: it is the real
    # pin that is asked for (and, isolated and offline, not found).
    status = gen.main(["--offline", "--check"])
    assert status != 0
    assert "rammp_prototype_base.urdf" in capsys.readouterr().err
