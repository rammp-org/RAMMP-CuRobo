"""The pure parts of find_pairs (mebot_sim/find_pairs.py): the search's
geometry in joint space, MuJoCo's measure of a path, and the scenarios file
it writes. The searches themselves need the planner on the GPU: they run in
find_pairs, and run_suite's positive controls check what they found.
"""

import math

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

from mebot_sim import find_pairs, scene, stack  # noqa: E402

TURNS = np.array([True, False, True, False, True, False, True])
# cuRobo's Gen3 limits: the turning joints a turn either way
LIMITS = np.array(
    [
        [-6.28, -2.41, -6.28, -2.66, -6.28, -2.23, -6.28],
        [6.28, 2.41, 6.28, 2.66, 6.28, 2.23, 6.28],
    ]
)
START = np.array(stack.scenarios()["start"])
# A state inside the upright stand-in (test_sim_arm.py): 21 mm inside it, and
# nothing else is touched.
IN_STANDIN = np.array([-0.968, -0.764, -2.324, 0.608, 1.785, -0.726, 2.177])


def test_turning_joints_are_the_ones_whose_limits_span_more_than_a_turn():
    assert find_pairs.turning(LIMITS).tolist() == TURNS.tolist()


def test_a_goal_moves_onto_the_starts_branch_by_whole_turns():
    start = np.array([3.0, 0.0, -3.0, 0.0, 0.1, 0.0, 0.0])
    goal = np.array([-3.0, 2.0, 3.0, -2.0, 0.2, 1.0, 6.0])
    wrapped = find_pairs.wrap(start, goal, TURNS)
    # the same arm pose: the turning joints by whole turns, the others not at all
    turns = (wrapped - goal) / (2 * math.pi)
    assert np.allclose(turns, np.round(turns))
    assert np.array_equal(wrapped[~TURNS], goal[~TURNS])
    # and within half a turn of the start
    assert (np.abs(wrapped - start)[TURNS] <= math.pi).all()
    assert wrapped[[0, 2, 4, 6]] == pytest.approx(
        [2 * math.pi - 3, 3 - 2 * math.pi, 0.2, 6 - 2 * math.pi]
    )
    # many at once
    many = find_pairs.wrap(np.stack([start] * 3), np.stack([goal] * 3), TURNS)
    assert np.array_equal(many, np.stack([wrapped] * 3))


def test_the_search_box_is_inside_the_sim_arms_ranges_and_the_planners_limits(
    assets,
):
    m = scene.build(assets, None)
    low, high = find_pairs.search_box(m, LIMITS)
    margin = find_pairs.JOINT_MARGIN_RAD
    for i, name in enumerate(stack.MUJOCO_JOINTS):
        j = m.joint(name).id
        if m.jnt_limited[j]:
            # menagerie's ranges are inside cuRobo's limits
            assert (low[i], high[i]) == pytest.approx(
                (m.jnt_range[j][0] + margin, m.jnt_range[j][1] - margin)
            )
        else:
            # as the driver reports a turning joint
            assert (low[i], high[i]) == (-math.pi, math.pi)
    tight = LIMITS * 0.5
    low, high = find_pairs.search_box(m, tight)
    assert (low >= tight[0] + margin - 1e-12).all()
    assert (high <= tight[1] - margin + 1e-12).all()


def test_a_straight_path_is_its_states_strictly_between_start_and_goal():
    start = np.array([[0.0] * 7, [1.0] * 7])
    goal = np.array([[0.5] * 7, [-1.0] * 7])
    paths = find_pairs.straight(start, goal, 4)
    assert paths.shape == (2, 4, 7)
    assert paths[0, :, 0] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert paths[1, :, 3] == pytest.approx([0.6, 0.2, -0.2, -0.6])


def test_a_path_is_densified_to_its_step():
    goal = START + [0.3, -0.2, 1.1, 0.0, 0.4, 0.2, -0.7]
    path = find_pairs.densify(START, goal, 0.005)
    assert np.array_equal(path[0], START) and path[-1] == pytest.approx(goal)
    assert np.abs(np.diff(path, axis=0)).max() <= 0.005 + 1e-12
    # and no more states than that takes
    assert (len(path) - 2) * 0.005 < np.abs(goal - START).max()
    assert len(find_pairs.densify(START, START, 0.005)) == 2


def test_pairs_are_drawn_apart_onto_their_starts_branch_and_within_limits():
    rng = np.random.default_rng(0)
    pool = rng.uniform(-1.0, 1.0, size=(500, 7))
    pool[:, TURNS] = rng.uniform(-math.pi, math.pi, size=(500, TURNS.sum()))
    start, goal = find_pairs.sample_pairs(
        np.random.default_rng(1), pool, 200, (0.8, 2.0), TURNS, LIMITS
    )
    assert len(start) == len(goal) == 200
    apart = np.abs(goal - start).max(axis=1)
    assert ((apart >= 0.8) & (apart <= 2.0)).all()
    assert (np.abs(goal - start)[:, TURNS] <= math.pi).all()
    # some goals were moved a turn: they lie outside the pool's [-pi, pi]
    assert (np.abs(goal[:, TURNS]) > math.pi).any()
    assert ((goal >= LIMITS[0]) & (goal <= LIMITS[1])).all()
    # each start is a state of the pool, and so is each goal, by whole turns
    assert all((pool == s).all(axis=1).any() for s in start)
    turns = goal[:, None] - pool[None]
    same = np.isclose(turns / (2 * math.pi), np.round(turns / (2 * math.pi)), atol=1e-9)
    assert same.all(axis=2).any(axis=1).all()
    # seeded
    again = find_pairs.sample_pairs(
        np.random.default_rng(1), pool, 200, (0.8, 2.0), TURNS, LIMITS
    )
    assert np.array_equal(again[0], start) and np.array_equal(again[1], goal)


@pytest.fixture(scope="module")
def seated(assets):
    return scene.build(assets, "upright_lap")


def test_a_path_is_measured_by_how_deep_the_sim_arm_goes_into_each_group(seated):
    at = find_pairs.deepest_contacts(seated, [IN_STANDIN])
    assert set(at) == {"standin"} and 0.01 < at["standin"] < 0.05
    along = find_pairs.deepest_contacts(seated, find_pairs.densify(START, IN_STANDIN))
    # the path ends at the state, to the last ulp of densify's arithmetic
    assert along["standin"] >= at["standin"] - 1e-12
    # where the sim starts, it touches nothing
    assert find_pairs.deepest_contacts(seated, [START]) == {}


DIGESTS = {
    "layer_mebot_floor.yaml": "a" * 64,
    "layer_mebot_platform.yaml": "b" * 64,
    "layer_mebot_person.yaml": "c" * 64,
    "layer_mebot_seat.yaml": "d" * 64,
}


def _found():
    """What the searches find, made up: find_pairs.generated's input."""
    rng = np.random.default_rng(5)

    def q():
        return rng.uniform(-3, 3, size=7)

    def pair(depth):
        return {"start": q(), "goal": q(), "crossing_mm": depth}

    return {
        "digests": DIGESTS,
        "start": q(),
        "start_tool": [0.249, -0.366, 0.253],
        "person_start": q(),
        "person_start_box": "person/thighs",
        "person_start_depth": 0.105,
        "reaches": {
            "seed": 33,
            "states": 1000,
            "valid": 22,
            "goals": [
                ("front" if k < 4 else "beside", q(), rng.uniform(size=3))
                for k in range(8)
            ],
        },
        "platform": {
            "seed": 11,
            "posture": None,
            "pairs": [pair(23.69), pair(18.6), pair(30.0)],
        },
        "person": {
            "seed": 22,
            "posture": "upright_lap",
            "pairs": [pair(47.3), pair(51.1), pair(10.8)],
        },
    }


def test_the_generated_part_holds_the_requests_exactly():
    found = _found()
    data = yaml.safe_load(find_pairs.generated(found))
    assert data["layers_sha256"] == DIGESTS
    # floats as written read back as the same numbers
    assert data["start"] == found["start"].tolist()
    assert data["person_start"] == found["person_start"].tolist()
    assert data["reaches"]["seed"] == 33
    assert data["reaches"]["goals"] == [
        g.tolist() for _, g, _ in found["reaches"]["goals"]
    ]
    for key in ("platform", "person"):
        out = data["%s_pairs" % key]
        assert out["seed"] == found[key]["seed"]
        assert out["posture"] == (found[key]["posture"] or "none")
        assert [p["start"] for p in out["pairs"]] == [
            p["start"].tolist() for p in found[key]["pairs"]
        ]
        assert [p["goal"] for p in out["pairs"]] == [
            p["goal"].tolist() for p in found[key]["pairs"]
        ]
        assert [p["crossing_mm"] for p in out["pairs"]] == [
            round(p["crossing_mm"], 1) for p in found[key]["pairs"]
        ]
    assert find_pairs.generated(found).isascii()


HAND = """# hand-written
standin:
  seated: {knee_left: -90}
thresholds: {end_error_rad: 0.05}
"""


def test_the_generated_part_is_replaced_and_the_hand_written_one_kept():
    first = find_pairs.compose(HAND, find_pairs.generated(_found()))
    assert first.startswith(HAND)
    assert first.count(find_pairs.MARKER) == 1
    found = _found()
    found["start"] = np.zeros(7)
    second = find_pairs.compose(first, find_pairs.generated(found))
    assert second.startswith(HAND) and second.count(find_pairs.MARKER) == 1
    data = yaml.safe_load(second)
    assert data["start"] == [0.0] * 7
    assert data["standin"] == {"seated": {"knee_left": -90}}
    assert data["thresholds"] == {"end_error_rad": 0.05}


def test_a_key_written_by_hand_and_generated_is_refused():
    # YAML keeps the last of two equal keys, silently
    with pytest.raises(ValueError, match="'start'"):
        find_pairs.compose(HAND + "start: [0, 0, 0, 0, 0, 0, 0]\n", "start: []\n")


def test_the_committed_scenarios_are_written_as_find_pairs_writes_them():
    path = stack.config("scenarios.yaml")
    text = path.read_text(encoding="utf-8")
    head, data = find_pairs.hand_written(text)
    assert text.startswith(head) and find_pairs.MARKER in text
    assert {"standin", "thresholds"} <= set(data)
    found = yaml.safe_load(text)
    assert set(found) == set(data) | set(find_pairs.GENERATED)
    assert set(found["layers_sha256"]) == set(stack.layer_digests())


class _AlongX:
    """A planner for inside_person: a sphere no joint moves, always inside a
    seat box as the arm's base column is beside the carved seat, and one
    that moves along x with the first joint. Every state is valid without
    the person."""

    fixed = np.array([[0.0, 0.0, 5.0, 0.05]])

    def __init__(self):
        self.pushed = []

    def push(self, world):
        self.pushed.append(world)
        return world

    def feasible(self, states):
        return np.ones(len(states), dtype=bool)

    def kinematics(self, states):
        moving = np.zeros((len(states), 4))
        moving[:, 0], moving[:, 3] = states[:, 0], 0.05
        fixed = np.repeat(self.fixed[None], len(states), axis=0)
        return np.zeros((len(states), 3)), np.concatenate([fixed, moving[:, None]], 1)


def _cube(x, z=0.0, side=0.2):
    return {"dims": [side] * 3, "pose": [x, 0.0, z, 1.0, 0.0, 0.0, 0.0]}


def test_the_s5_start_is_inside_the_person_and_nothing_else():
    present = {
        "seat/back": _cube(0.0, z=5.0),  # around the fixed sphere: ignored
        "person/thighs": _cube(1.0),
        "person/legs": _cube(2.0),
        "seat/frame": _cube(2.15),  # beside the legs: a state in both is not
    }
    absent = {"seat/back": present["seat/back"], "seat/frame": present["seat/frame"]}
    box = (np.r_[0.0, np.zeros(6)], np.r_[2.5, np.ones(6)])
    search = {"seed": 7, "states": 400, "min_depth_m": 0.02}
    planner = _AlongX()
    q, key, depth = find_pairs.inside_person(planner, search, present, absent, box)
    # planned in the chair without the person, judged with it
    assert planner.pushed == [absent, present]
    assert key.startswith("person/") and depth >= 0.02
    # its moving sphere reaches the person only: never the seat beside the legs
    x = q[0]
    assert 0.85 - 0.05 + 0.02 <= x <= 1.15 + 0.05 - 0.02 or (
        1.85 + 0.02 - 0.05 <= x <= 2.05 + 0.05 - 0.02
    )
    assert not 2.05 - 0.05 < x < 2.25 + 0.05
    # seeded: the same state again
    again = find_pairs.inside_person(_AlongX(), search, present, absent, box)
    assert np.array_equal(again[0], q)


def test_no_state_inside_the_person_fails_naming_what_it_tried():
    # the person is only ever reached shallower than the minimum, or
    # together with the seat
    present = {
        "seat/back": _cube(0.0, z=5.0),
        # its bottom 0.04 over the path: the 0.05 sphere reaches 10 mm in
        "person/thighs": _cube(1.0, z=0.14),
        "person/legs": _cube(2.0),
        "seat/frame": _cube(2.0, side=0.3),  # around the legs entirely
    }
    absent = {"seat/back": present["seat/back"], "seat/frame": present["seat/frame"]}
    box = (np.zeros(7), np.r_[2.5, np.ones(6)])
    search = {"seed": 7, "states": 400, "min_depth_m": 0.02}
    with pytest.raises(RuntimeError) as exc:
        find_pairs.inside_person(_AlongX(), search, present, absent, box)
    message = str(exc.value)
    assert "no state of 400 (seed 7), of the 400 valid without the person" in message
    assert "reaches 0.02 m into the person and nothing else" in message
    shallow = int(message.rsplit("(", 1)[1].split()[0])
    assert shallow > 0  # the thighs, reached but not deep enough
