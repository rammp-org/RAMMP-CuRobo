"""The suite's own logic (mebot_sim/run_suite.py), without a ROS graph: how
each case is judged, the gates, the stale-requests check, the result file,
and the scenarios' order, worlds and postures, run against a scripted
stand-in of client.Tester.
"""

import json
import re

import pytest

pytest.importorskip("mujoco")  # run_suite's S0

from mebot_sim import run_suite, stack  # noqa: E402

LIMIT = 0.05
WHEEL = {
    "group": "mebot",
    "link": "spherical_wrist_2_link",
    "part": "drive_wheel_r",
    "force_n": 254.0,
    "depth_mm": 0.8,
}
THIGH = dict(WHEEL, group="standin", link="half_arm_1_link", part="thigh_right")
WORLD = "world set to x (23 boxes: scene 0 | ...)"
IK_FAIL = "MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal"
INVALID_START = (
    "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state "
    "collides with the world (or exceeds limits) [start collides with "
    "person/lap (154 mm), person/torso_head_3 (134 mm)] [world: 23 boxes: ...]"
)


def flown(*contacts, end=0.004, tracking=0.012):
    return {
        "planned": True,
        "message": "planned 120 points, 2.40 s at full speed",
        "plan_s": 0.21,
        "world": WORLD,
        "executed": True,
        "exec_error": "",
        "contacts": list(contacts),
        "closest": {"mebot": {"mm": 31.5, "link": "bracelet_link", "part": "mebot"}},
        "end_error_rad": end,
        "tracking_rad": tracking,
    }


def refused(message=IK_FAIL):
    return {"planned": False, "message": message, "plan_s": 0.15, "world": WORLD}


def test_a_request_with_its_layers_on_is_refused_or_flown_clear():
    case = run_suite.with_layers("S1a on", "here", refused(), LIMIT)
    assert case["gates"] == {"G2": True} and case["outcome"].startswith("refused")
    case = run_suite.with_layers("S1a on", "here", flown(), LIMIT)
    assert case["gates"] == {"G2": True, "G6": True} and case["passed"]
    case = run_suite.with_layers("S1a on", "here", flown(WHEEL), LIMIT)
    assert case["gates"] == {"G2": False} and "the wheelchair" in case["outcome"]
    # G6: came to rest away from its last point, so the sim did not fly it
    case = run_suite.with_layers("S1a on", "here", flown(end=0.08), LIMIT)
    assert case["gates"] == {"G2": True, "G6": False}
    unflown = dict(flown(), executed=False, exec_error="PREEMPTED by a stop")
    case = run_suite.with_layers("S1a on", "here", unflown, LIMIT)
    assert case["gates"] == {"G2": False} and "PREEMPTED" in case["outcome"]
    # a start found valid that is not is no refusal of the path
    case = run_suite.with_layers("S1a on", "here", refused(INVALID_START), LIMIT)
    assert case["gates"] == {"G2": False}
    assert case["outcome"].startswith(run_suite.PREMISE)
    # nor is a world the planner refused, or a goal it would not take
    for message in ("set_world refused: no such file", "/x rejected the goal"):
        case = run_suite.with_layers("S1a on", "here", refused(message), LIMIT)
        assert case["gates"] == {"G2": False} and message in case["outcome"]
    # after a consumer's push, a request bears on G4 as well
    case = run_suite.with_layers("S4a", "here", flown(THIGH), LIMIT, ("G4", "G2"))
    assert case["gates"] == {"G4": False, "G2": False}
    case = run_suite.with_layers("S4a", "here", flown(), LIMIT, ("G4", "G2"))
    assert case["gates"] == {"G4": True, "G2": True, "G6": True}


def test_a_positive_control_passes_only_when_it_hits_what_its_layer_covers():
    case = run_suite.without_layer("S1a off", "here", flown(WHEEL), "mebot")
    assert case["gates"] == {"G3": True} and "the wheelchair" in case["outcome"]
    for report in (
        refused(),
        flown(),
        flown(dict(WHEEL, group="floor")),
        dict(flown(WHEEL), executed=False, exec_error="PREEMPTED by a stop"),
    ):
        case = run_suite.without_layer("S1a off", "here", report, "mebot")
        assert case["gates"] == {"G3": False}
        assert case["outcome"].startswith("premise broken; re-run find_pairs")


REPLY = (
    "world set to /c/consumer_world.yaml (30 boxes: scene 8 | config layers: "
    "platform 16, person 6 | scene layers: none)"
)


def test_the_push_reply_names_every_config_layer_with_its_boxes():
    expected = {"platform": 16, "person": 6}
    case = run_suite.pushed("here", True, REPLY, expected)
    assert case["gates"] == {"G4": True}
    for ok, reply in (
        (True, REPLY.replace("person 6", "person 0")),
        (True, REPLY.replace(", person 6", "")),
        (True, "world set to /c/platform 16 person 6.yaml (8 boxes: scene 8)"),
        (False, "collision world is empty; refusing it"),
    ):
        case = run_suite.pushed("here", ok, reply, expected)
        assert case["gates"] == {"G4": False}, reply


def test_a_plan_from_the_retract_fails_naming_the_person_and_a_depth():
    case = run_suite.from_retract("here", refused(INVALID_START))
    assert case["gates"] == {"G5": True}
    for report in (
        refused(INVALID_START.replace("person/", "platform/")),
        refused(
            "MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS: [start: no world box "
            "penetrated — self-collision or joint limits]"
        ),
        refused(INVALID_START.replace(" (154 mm)", "").replace(" (134 mm)", "")),
        flown(),
    ):
        assert run_suite.from_retract("here", report)["gates"] == {"G5": False}


def test_each_gate_is_the_verdict_of_the_cases_that_bear_on_it():
    cases = [
        stack.case("a", "x", "y", {"G2": True, "G6": True}),
        stack.case("b", "x", "y", {"G2": False}),
        stack.case("c", "x", "y", {"G3": True}),
    ]
    assert run_suite.gates(cases) == {
        "G0": None,
        "G1": None,
        "G2": False,
        "G3": True,
        "G4": None,
        "G5": None,
        "G6": True,
    }


# ------------------------------------------------------- the suite, scripted
DATA = stack.scenarios()


class Scripted:
    """client.Tester as run_suite uses it, answering from `script`, a function
    of (planner, world, joints, execute) -> a move report; it records every
    call."""

    made = []

    def __init__(self, name, script):
        self.calls, self.script = [], script
        self.planners = {stack.CHAIR: self}
        Scripted.made.append(self)

    def connect(self):
        self.calls.append(("connect",))

    def posture(self, name):
        self.calls.append(("posture", name))

    def teleport(self, q):
        self.calls.append(("teleport", list(q)))

    def set_world(self, world):
        self.calls.append(("set_world", str(world)))
        return True, REPLY

    def move(self, planner, world, start, joints=None, speed=0.5, execute=True):
        self.calls.append(("move", planner, world and str(world), list(start)))
        return self.script(planner, world and str(world), joints, execute)

    def close(self):
        self.calls.append(("close",))


def _answers(planner, world, joints, execute):
    """Every request with its layers refused or flown clear, every positive
    control a hit, and the retract refused naming the person."""
    if not execute:
        return refused(INVALID_START)
    floor = stack.worlds()["floor_only"][1]
    no_person = stack.worlds()["no_person"][1]
    if (planner, world) == (stack.CONTROL, floor):
        return flown(WHEEL, tracking=0.5)  # held by what it hit
    if (planner, world) == (stack.CONTROL, no_person):
        pairs = DATA["person_pairs"]["pairs"]
        if any(joints == pair["goal"] for pair in pairs):
            return flown(THIGH, tracking=0.6)
        return refused()
    return flown()


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """run_suite.main against a scripted Tester (answers: `script`), with S0
    stood in for, results under tmp_path. Returns a runner: argv -> (exit
    status, the Scripted made, if any, and the result file's contents)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    Scripted.made = []
    monkeypatch.setattr(
        run_suite.check_layers,
        "cases",
        lambda assets: [
            stack.case("S0 registration", "x", "y", {"G0": True}),
            stack.case("S0 upright_lap 0.1587", "x", "y", {"G1": True}),
        ],
    )
    monkeypatch.setattr(run_suite.scene, "Assets", lambda *args: None)

    def run(argv, script=_answers):
        monkeypatch.setattr(run_suite, "Tester", lambda name: Scripted(name, script))
        status = run_suite.main(argv)
        files = list((tmp_path / "rammp_curobo" / "sim").glob("*.json"))
        assert len(files) == 1
        result = json.loads(files[0].read_text())
        files[0].unlink()
        return status, (Scripted.made or [None])[-1], result

    return run


def test_each_scenario_plans_in_its_world_with_the_standin_as_found(scripted, capsys):
    status, tester, result = scripted([])
    out = capsys.readouterr().out
    assert status == 0, out
    worlds = stack.worlds()
    on_s1 = worlds["no_person"][:2]
    off_s1 = worlds["floor_only"][:2]
    chair = worlds["all"][:2]
    moves = [call[1:3] for call in tester.calls if call[0] == "move"]
    platform, person = DATA["platform_pairs"], DATA["person_pairs"]
    assert moves == (
        [on_s1, off_s1] * len(platform["pairs"])
        + [chair, worlds["no_person"][:2]] * len(person["pairs"])
        + [chair] * len(DATA["reaches"]["goals"])
        + [(stack.CHAIR, None)] * len(person["pairs"])
        + [chair]
    )
    # S1 with the chair empty, the rest with the stand-in as find_pairs had it
    postures = [call[1] for call in tester.calls if call[0] == "posture"]
    assert postures[0] == "none" and set(postures[1:]) == {person["posture"]}
    # S4 is pushed the consumer's world before its requests
    pushes = [call[1] for call in tester.calls if call[0] == "set_world"]
    assert pushes == [str(stack.config("consumer_world.yaml"))]
    # S5 plans from the bench retract, last, without flying (only a request
    # not flown is answered INVALID_START here)
    assert tester.calls[-2][0] == "move" and tester.calls[-1] == ("close",)
    assert tester.calls[-2][3] == run_suite.bench_retract()
    assert result["cases"][-1]["case"] == "S5 retract"
    # one line per case: S0 (as stood in for), 3 pairs on and off in S1 and
    # S2, 8 reaches, the push and the 3 requests after it, the retract
    lines = [line for line in out.splitlines() if re.match(r"S\d", line)]
    n = 2 + 2 * 3 + 2 * 3 + 8 + 1 + 3 + 1
    assert len(lines) == n and all(line.endswith("PASS") for line in lines)
    assert [c["case"] for c in result["cases"]][:4] == [
        "S0 registration",
        "S0 upright_lap 0.1587",
        "S1a on",
        "S1a off",
    ]
    assert result["passed"] and result["error"] is None
    assert result["gates"] == {gate: "PASS" for gate in run_suite.GATES}
    assert result["reported"]["s3_planned"] == "8 of 8"
    # how well the sim tracks, from the flights that touched nothing
    assert result["reported"]["tracking_rad_max"] == 0.012


def test_a_positive_control_that_misses_fails_the_suite(scripted, capsys):
    def miss(planner, world, joints, execute):
        if (planner, world) == (stack.CONTROL, stack.worlds()["floor_only"][1]):
            return flown()
        return _answers(planner, world, joints, execute)

    status, _, result = scripted(["--only", "S1"], script=miss)
    out = capsys.readouterr().out
    assert status == 1
    assert "premise broken; re-run find_pairs" in out
    assert result["gates"]["G3"] == "FAIL" and not result["passed"]
    assert result["gates"]["G0"] == "not run"


def test_requests_found_with_other_layers_fail_loudly_before_anything_runs(
    scripted, capsys, monkeypatch
):
    stale = dict(DATA, layers_sha256=dict(DATA["layers_sha256"]))
    stale["layers_sha256"]["layer_mebot_person.yaml"] = "0" * 64
    monkeypatch.setattr(stack, "scenarios", lambda: stale)
    status, tester, result = scripted([])
    out = capsys.readouterr()
    assert status == 1 and tester is None
    text = out.out + out.err
    assert "premise broken; re-run find_pairs" in text
    assert "layer_mebot_person.yaml" in text
    assert result["error"] and not result["passed"] and result["cases"] == []


def test_a_stack_that_stops_answering_ends_the_suite_failed(scripted, capsys):
    def stops(planner, world, joints, execute):
        raise TimeoutError("%s planning: no answer in 30 s" % planner)

    status, tester, result = scripted(["--only", "S3"], script=stops)
    assert status == 1 and tester.calls[-1] == ("close",)
    assert "no answer" in result["error"] and not result["passed"]


def test_the_result_is_named_by_the_time_and_the_commit(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = run_suite.result_path("20261003T141502Z", "0da044c")
    assert path == tmp_path / "rammp_curobo" / "sim" / "20261003T141502Z-0da044c.json"
    monkeypatch.delenv("XDG_CACHE_HOME")
    path = run_suite.result_path("20261003T141502Z", "0da044c")
    assert str(path).endswith("/.cache/rammp_curobo/sim/20261003T141502Z-0da044c.json")


def test_a_push_reply_without_the_person_fails_whatever_the_chair_config_lists(
    scripted, monkeypatch
):
    # a chair config that drops the person is the regression G4 is there for:
    # the reply is held to the layer files, not to what the config lists
    layers, world = stack.chair_layers()
    platform_only = [layer for layer in layers if layer.name == "platform"]
    monkeypatch.setattr(stack, "chair_layers", lambda: (platform_only, world))
    without = REPLY.replace(", person 6", "")
    monkeypatch.setattr(Scripted, "set_world", lambda self, world: (True, without))
    status, _, result = scripted(["--only", "S4"])
    assert status == 1 and result["gates"]["G4"] == "FAIL"
    assert "person 6" in result["cases"][0]["outcome"]
