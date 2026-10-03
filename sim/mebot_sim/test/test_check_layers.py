"""S0 (mebot_sim/check_layers.py): the committed layers against the true
shapes, on the CPU. It needs what test_scene.py needs, and skips without it.
"""

import json
import math

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

from mebot_sim import check_layers, scene, stack  # noqa: E402

stack.core()
from rammp_curobo.scene import Layer, Obstacle, load_layer  # noqa: E402

POSTURES = ("upright_lap", "lean_lap", "forearms_forward")
# G1's bodies, config/scenarios.yaml's standin.sizes (spec 6.4): their sitting
# heights in inches, ANSUR II row 8 of the NCSU table
SIZES = {"woman_5th": 31.61, "man_50th": 36.14, "man_95th": 38.46}
# the humanoid's lower legs and feet
SHINS_AND_FEET = {
    "%s_%s" % (part, side)
    for part in ("shin", "foot1", "foot2")
    for side in ("right", "left")
}


def _model(*geoms):
    """A model of `geoms` (add_geom's keywords) on one body, moved and turned."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(pos=[0.1, -0.2, 0.3], quat=[0.9, 0.2, 0.3, 0.1])
    for g in geoms:
        body.add_geom(**g)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return m, d


def _local(d, g, points):
    """`points` in geom `g`'s frame."""
    return (points - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)


def test_boxes_are_sampled_on_their_faces_and_cylinders_on_their_rims():
    box = {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.1, 0.2, 0.3]}
    cylinder = {
        "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "size": [0.05, 0.1, 0],
        "pos": [0.3, 0, 0],
    }
    m, d = _model(box, cylinder)
    on_box = _local(d, 0, check_layers.surface_samples(m, d, [0]))
    # the outer layer of a 9 x 9 x 9 lattice, every point on a face
    assert len(on_box) == 9**3 - 7**3
    ratio = np.abs(on_box) / [0.1, 0.2, 0.3]
    assert ratio.max() <= 1 + 1e-12
    assert np.allclose(ratio.max(axis=1), 1.0)
    on_cylinder = _local(d, 1, check_layers.surface_samples(m, d, [1]))
    # 72 angles x 2 radii x 9 heights
    assert len(on_cylinder) == 72 * 2 * 9
    radius = np.linalg.norm(on_cylinder[:, :2], axis=1)
    assert sorted(set(np.round(radius, 12))) == [0.025, 0.05]
    heights = sorted(set(np.round(on_cylinder[:, 2], 12)))
    assert heights == pytest.approx(list(np.linspace(-0.1, 0.1, 9)))
    assert len(check_layers.surface_samples(m, d, [0, 1])) == 386 + 1296


def test_only_boxes_and_cylinders_are_sampled():
    m, d = _model({"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.1, 0, 0]})
    with pytest.raises(ValueError, match="neither a box nor a cylinder"):
        check_layers.surface_samples(m, d, [0])


def test_a_rotated_layer_box_is_refused():
    tilted = Obstacle(
        {
            "name": "tilted",
            "position": [0, 0, 0],
            "dims": [0.1] * 3,
            "rpy_deg": [0, 0, 5],
        }
    )
    with pytest.raises(ValueError, match="tilted is rotated"):
        check_layers.bounds([tilted])


@pytest.fixture(scope="module")
def empty_chair(assets):
    return scene.build(assets, None)


def test_the_mebot_registers_in_the_platform_and_a_1mm_shrink_is_caught(
    assets, empty_chair
):
    platform = assets.layers["platform"]
    kinds = [
        int(empty_chair.geom_type[g])
        for g in scene.contact_groups(empty_chair)[1]["mebot"]
    ]
    whole = check_layers.registration(empty_chair, platform)
    assert whole["geoms"] == len(kinds)
    assert whole["samples"] == 386 * kinds.count(
        mujoco.mjtGeom.mjGEOM_BOX
    ) + 1296 * kinds.count(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert whole["outside"] == 0
    # a check that could not fail would pass this too
    assert check_layers.registration(empty_chair, platform, shrink=0.001)["outside"]


def test_g0_fails_when_mujocos_parse_lacks_a_primitive_of_the_urdf(assets, monkeypatch):
    # what MuJoCo kept is still inside the platform, and the shrink still
    # caught: only the URDF's own count sees the wheel it lost
    parse = scene.mebot_spec

    def lossy(urdf):
        spec = parse(urdf)
        spec.delete(spec.body("drive_wheel_l").geoms[0])
        return spec

    monkeypatch.setattr(scene, "mebot_spec", lossy)
    case = check_layers.cases(assets)[0]
    assert case["outside"] == 0 and case["outside_shrunk"] > 0
    assert case["gates"] == {"G0": False} and not case["passed"]
    assert case["outcome"].endswith(
        "; the URDF has 46: missing drive_wheel_l (cylinder)"
    )
    assert case["missing"] == ["drive_wheel_l (cylinder)"] and case["geoms"] == 45


@pytest.mark.parametrize("defect", ["drop", "shift_1mm"])
def test_registration_catches_a_defective_platform(assets, empty_chair, defect):
    platform = assets.layers["platform"]
    first, rest = platform.obstacles[0], platform.obstacles[1:]
    if defect == "drop":
        boxes = rest
    else:
        moved = Obstacle(
            {
                "name": first.name,
                "position": np.add(first.position, [0.001, 0, 0]),
                "dims": first.dims,
            }
        )
        boxes = [moved] + rest
    broken = Layer("platform", False, boxes, platform.path)
    assert check_layers.registration(empty_chair, broken)["outside"] > 0


def _ball_and_rod():
    along_x = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    return [
        {
            "name": "ball",
            "type": int(mujoco.mjtGeom.mjGEOM_SPHERE),
            "size": np.array([0.05, 0, 0]),
            "pos": np.array([0.5, 0.5, 0.5]),
            "mat": np.eye(3).ravel(),
        },
        {
            "name": "rod",
            "type": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            "size": np.array([0.04, 0.1, 0]),
            "pos": np.array([-0.3, 0.2, 0.1]),
            "mat": along_x.ravel(),
        },
    ]


def test_a_body_is_sampled_through_its_volume():
    ball, rod = _ball_and_rod()
    grid = 0.004
    for g, volume in (
        (ball, 4 / 3 * math.pi * 0.05**3),
        (rod, math.pi * 0.04**2 * 0.2 + 4 / 3 * math.pi * 0.04**3),
    ):
        points = check_layers.volume_samples([g], grid)
        assert len(points) * grid**3 == pytest.approx(volume, rel=0.03)
        # inside the shape: in its frame, within its radius of its axis
        local = (points - g["pos"]) @ g["mat"].reshape(3, 3)
        local[:, 2] -= np.clip(local[:, 2], -g["size"][1], g["size"][1])
        assert np.linalg.norm(local, axis=1).max() <= g["size"][0] + 1e-12


def test_coverage_counts_each_point_as_covered_in_the_column_or_a_gap():
    ball, _ = _ball_and_rod()
    centre = ball["pos"]
    # a box over the half x > 0.5 of the ball, and the column over the half
    # z < 0.5: a quarter of the ball is in neither
    box = (centre + [0, -1, -1], centre + [1, 1, 1])
    column = (centre + [-1, -1, -1], centre + [1, 1, 0])
    c = check_layers.coverage([ball], [box], column, grid=0.002)
    volume = 4 / 3 * math.pi * 0.05**3
    assert c["body_m3"] == pytest.approx(volume, rel=0.02)
    assert c["covered_pct"] == pytest.approx(50, abs=1)
    assert c["column_m3"] == pytest.approx(volume / 4, rel=0.05)
    assert c["gap_m3"] == pytest.approx(volume / 4, rel=0.05)
    assert c["gap_points"] > 0 and set(c["gap_where"]) == {"ball"}
    # a point in a box is covered, whether or not it is in the column too
    other_half = (centre + [-1, -1, -1], centre + [0, 1, 1])
    whole = check_layers.coverage([ball], [box, other_half], column, grid=0.002)
    assert (whole["covered_pct"], whole["column_m3"], whole["gap_points"]) == (
        100.0,
        0.0,
        0,
    )


@pytest.fixture(scope="module")
def person(assets):
    """The person layer padded as the chair planner pads it, and the arm's
    column."""
    obstacles = assets.layers["person"].obstacles
    padded = check_layers.bounds(obstacles, check_layers.chair_padding())
    return padded, check_layers.column(assets)


@pytest.mark.parametrize("posture", POSTURES)
@pytest.mark.parametrize("back", ["front", "rear"])
@pytest.mark.parametrize("size", SIZES)
def test_the_standin_is_inside_the_padded_person_layer(
    assets, person, size, posture, back
):
    # spec 6.4: from the 5th-percentile woman to the 95th-percentile man, and
    # wherever the backrest face is in its CAD range, the body is in the
    # padded person layer, or in the arm's column where the arm's base is
    seat = assets.data["person"]["seat"]
    face = (max if back == "front" else min)(seat["backrest_face_x"])
    height = SIZES[size] * 0.0254
    padded, column = person
    geoms, scale = scene.standin(assets, posture, face, height)
    c = check_layers.coverage(geoms, padded, column)
    assert c["gap_points"] == 0, c["gap_where"]
    assert c["covered_pct"] > 99.0
    # the whole humanoid, at the scale that seats it at that size
    whole = check_layers.humanoid_m3(assets) * scale**3
    assert c["body_m3"] == pytest.approx(whole, rel=0.02)


def test_a_person_layer_without_its_lap_leaves_a_gap(assets, person):
    padded, column = person
    names = [o.name for o in assets.layers["person"].obstacles]
    lapless = [b for name, b in zip(names, padded) if name != "lap"]
    geoms, _ = scene.standin(assets, "upright_lap")
    c = check_layers.coverage(geoms, lapless, column)
    assert c["gap_points"] > 0
    assert {"thigh_right", "thigh_left"} <= set(c["gap_where"])


def test_s0_is_one_registration_and_eighteen_standins_each_gated(assets):
    cases = check_layers.cases(assets)
    faces = sorted(assets.data["person"]["seat"]["backrest_face_x"])
    # the three bodies of spec 6.4, at row 8's sitting heights: an edit to
    # config/scenarios.yaml that seats any other body fails here
    sizes = stack.scenarios()["standin"]["sizes"]
    assert {size: row["inches"] for size, row in sizes.items()} == SIZES
    assert [c["case"] for c in cases] == ["S0 registration"] + [
        "S0 %s %s %.4f" % (size, posture, face)
        for size in SIZES
        for posture in POSTURES
        for face in faces
    ]
    # each stand-in seated at its size's sitting height: the scale grows with
    # it, posture by posture
    heights = [c["sitting_height_in"] for c in cases[1:]]
    assert heights == [inches for inches in SIZES.values() for _ in range(6)]
    scales = np.reshape([c["scale"] for c in cases[1:]], (3, 6))
    heights = np.reshape(heights, (3, 6))
    assert scales / scales[-1] == pytest.approx(heights / heights[-1])
    assert cases[0]["gates"] == {"G0": True}
    assert cases[0]["outside"] == 0 and cases[0]["outside_shrunk"] > 0
    assert all(c["gates"] == {"G1": True} for c in cases[1:])
    assert all(c["passed"] for c in cases)
    # run_suite writes them to its result file as they are
    assert json.loads(json.dumps(cases))[1]["gates"] == {"G1": True}


def _layer_file(path, layer, obstacles):
    """A layer file at `path` holding `obstacles` (rammp_curobo Obstacles)."""
    entries = [
        {"name": o.name, "position": list(o.position), "dims": list(o.dims)}
        for o in obstacles
    ]
    path.write_text(
        yaml.safe_dump(
            {"layer": layer.name, "excludable": layer.excludable, "obstacles": entries}
        )
    )
    return path


def test_s0_fails_its_gates_on_broken_layer_files(assets, monkeypatch, tmp_path):
    files = stack.layer_files()
    platform, person = (load_layer(files[name]) for name in ("platform", "person"))
    lapless = [o for o in person.obstacles if o.name != "lap"]
    broken = dict(
        files,
        platform=_layer_file(tmp_path / "p.yaml", platform, platform.obstacles[1:]),
        person=_layer_file(tmp_path / "q.yaml", person, lapless),
    )
    monkeypatch.setattr(stack, "layer_files", lambda: broken)
    cases = check_layers.cases(assets)
    assert cases[0]["gates"] == {"G0": False} and not cases[0]["passed"]
    assert all(c["gates"] == {"G1": False} for c in cases[1:])


def test_g1_fails_when_the_column_excuses_what_the_person_layer_leaves_out(
    assets, monkeypatch, tmp_path
):
    # the column is the generator's, as the person layer is: carved too wide,
    # the layer leaves out a part of the body that the column then excuses,
    # and no gap is left to fail on
    files = stack.layer_files()
    person = load_layer(files["person"])
    lapless = [o for o in person.obstacles if o.name != "lap"]
    carved = dict(files, person=_layer_file(tmp_path / "q.yaml", person, lapless))
    monkeypatch.setattr(stack, "layer_files", lambda: carved)
    monkeypatch.setattr(check_layers, "column", lambda assets: ([-5.0] * 3, [5.0] * 3))
    standins = check_layers.cases(assets)[1:]
    assert all(c["gap_m3"] == 0 and c["covered_pct"] < 99 for c in standins)
    assert all(c["gates"] == {"G1": False} for c in standins)
    assert all("covered (under 99)" in c["outcome"] for c in standins)


def test_g1_fails_on_a_standin_thinner_than_the_humanoid_at_its_scale(
    assets, monkeypatch
):
    # every part of the body thinner by a fifth, where it was: the layer
    # covers it as well as ever, and it is half the body
    standin = scene.standin

    def thinner(*args):
        geoms, scale = standin(*args)
        return [dict(g, size=0.8 * g["size"]) for g in geoms], scale

    monkeypatch.setattr(scene, "standin", thinner)
    standins = check_layers.cases(assets)[1:]
    assert all(c["gap_m3"] == 0 and c["covered_pct"] >= 99 for c in standins)
    assert all(c["gates"] == {"G1": False} for c in standins)
    assert all(c["body_m3"] < 0.6 * c["humanoid_m3"] for c in standins)
    assert all("m3 (not the humanoid's " in c["outcome"] for c in standins)


def test_g1_seats_the_standin_at_the_scenarios_sizes(assets, monkeypatch):
    # the bodies are config/scenarios.yaml's, whatever occupant the person
    # layer was sized for: one size of 36.0 in, at which the feet hung out of
    # the legs sized by the 95th-percentile thigh, is six stand-ins at its
    # scale, and the layer holds them
    data = stack.scenarios()
    one = {"mid": {"row": 8, "sex": "men", "percentile": 50, "inches": 36.0}}
    sized = dict(data, standin=dict(data["standin"], sizes=one))
    monkeypatch.setattr(stack, "scenarios", lambda: sized)
    monkeypatch.setattr(assets, "sitting_height", 2.0)
    standins = check_layers.cases(assets)[1:]
    assert [c["case"].split()[1] for c in standins] == ["mid"] * 6
    assert all(c["sitting_height_in"] == 36.0 for c in standins)
    assert all(c["gap_m3"] == 0 and 0.05 < c["body_m3"] < 0.07 for c in standins)
    assert all(c["gates"] == {"G1": True} for c in standins)


def test_s0_with_no_size_to_seat_the_standin_at_is_an_error(
    assets, monkeypatch, capsys
):
    # with no body seated, G1 would judge nothing, and pass
    data = stack.scenarios()
    empty = dict(data, standin=dict(data["standin"], sizes={}))
    monkeypatch.setattr(stack, "scenarios", lambda: empty)
    said = "config/scenarios.yaml holds no standin.sizes"
    with pytest.raises(ValueError, match=said):
        check_layers.cases(assets)
    assert check_layers.main([]) == 1
    assert said in capsys.readouterr().err


def test_g1_fails_legs_sized_by_the_95th_percentile_thigh_alone(
    assets, monkeypatch, tmp_path
):
    # The person layer as it was generated for the 95th-percentile man alone,
    # its legs' back face at his thigh (21.57 in). With their back at the rear
    # of the backrest's range, the smaller bodies' shins and feet hang behind
    # it, under the lap: G1 fails them, and only them.
    generator = stack.generator()
    data = generator.load_data()
    thigh = data["person"]["ansur_in"]["buttock_popliteal_length"]
    thigh.update(sex="men", percentile=95, inches=21.57)
    old = tmp_path / "layer_mebot_person.yaml"
    old.write_text(
        generator.generate(data, assets.urdf)["files"][data["outputs"]["person"]]
    )
    files = dict(stack.layer_files(), person=old)
    monkeypatch.setattr(stack, "layer_files", lambda: files)
    standins = check_layers.cases(assets)[1:]
    rear = "%.4f" % min(assets.data["person"]["seat"]["backrest_face_x"])
    failed = [c for c in standins if not c["passed"]]
    assert [c["case"] for c in failed] == [
        "S0 %s %s %s" % (size, posture, rear)
        for size in ("woman_5th", "man_50th")
        for posture in POSTURES
    ]
    for c in failed:
        assert c["gates"] == {"G1": False} and c["gap_m3"] > 0
        assert set(c["gap_where"]) <= SHINS_AND_FEET, c["gap_where"]
    # (the committed layer, its legs' back face at the shortest thigh, holds
    # every one of them: test_s0_is_one_registration_and_eighteen_standins_...)


def test_g1_bounds_are_the_scenarios_thresholds(assets, monkeypatch):
    data = stack.scenarios()
    strict = dict(
        data,
        thresholds=dict(data["thresholds"], covered_pct=99.9, body_volume_pct=0.01),
    )
    monkeypatch.setattr(stack, "scenarios", lambda: strict)
    standins = check_layers.cases(assets)[1:]
    assert all(c["gates"] == {"G1": False} for c in standins)
    under = [c for c in standins if c["covered_pct"] < 99.9]
    assert under and all("covered (under 99.9)" in c["outcome"] for c in under)
    assert all("m3 (not the humanoid's " in c["outcome"] for c in standins)


def test_s0_checks_the_layer_files_whatever_the_chair_config_lists(assets, monkeypatch):
    # a chair config that drops the person is for the flying scenarios to
    # catch; S0 is about the files
    monkeypatch.delitem(assets.layers, "person")
    assert all(case["passed"] for case in check_layers.cases(assets))


def test_check_layers_prints_a_line_per_case_and_exits_with_the_verdict(assets, capsys):
    assert check_layers.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    cases = [line for line in lines if line.startswith("S0 ")]
    assert len(cases) == 19 and all(line.endswith("PASS") for line in cases)
