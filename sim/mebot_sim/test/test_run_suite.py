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


# client.Tester.check_state's report: the planner's answer to a plan from a
# state to itself, in the world it holds
VALID = {
    "planned": True,
    "message": "planned 2 points, 0.00 s at full speed",
    "plan_s": 0.12,
    "world": None,
}


def checked(report, check=VALID):
    """`report` of a request planned with its layers, as Suite makes it: with
    the planner's `check` of its goal."""
    return dict(report, goal_as_start=check)


def test_a_request_with_its_layers_on_is_refused_or_flown_clear():
    case = run_suite.with_layers("S1a on", "here", checked(refused()), LIMIT)
    assert case["gates"] == {"G2": True} and case["outcome"].startswith("refused")
    case = run_suite.with_layers("S1a on", "here", checked(flown()), LIMIT)
    assert case["gates"] == {"G2": True, "G6": True} and case["passed"]
    case = run_suite.with_layers("S1a on", "here", checked(flown(WHEEL)), LIMIT)
    assert case["gates"] == {"G2": False} and "the wheelchair" in case["outcome"]
    # G6: came to rest away from its last point, so the sim did not fly it
    case = run_suite.with_layers("S1a on", "here", checked(flown(end=0.08)), LIMIT)
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
    case = run_suite.with_layers(
        "S4a", "here", checked(flown(THIGH)), LIMIT, ("G4", "G2")
    )
    assert case["gates"] == {"G4": False, "G2": False}
    case = run_suite.with_layers("S4a", "here", checked(flown()), LIMIT, ("G4", "G2"))
    assert case["gates"] == {"G4": True, "G2": True, "G6": True}


RECHECK = "LIBRARY_VALIDATION_FAILED: plan rejected by post-validation: "


@pytest.mark.parametrize("gates", [("G2",), ("G4", "G2")])
def test_only_a_planner_that_judged_a_request_and_found_no_plan_refuses_it(gates):
    # cuRobo found no plan, or the planner's re-check flagged the plan it found
    for message in (
        IK_FAIL,
        "MotionGenStatus.GRAPH_FAIL: planning failed (MotionGenStatus.GRAPH_FAIL)",
        "MotionGenStatus.TRAJOPT_FAIL: the goal is reachable but no collision-free "
        "PATH was found",
        "MotionGenStatus.FINETUNE_TRAJOPT_FAIL: a collision-free path WAS found but "
        "retiming failed",
        RECHECK + "constraint re-check flagged 3/120 states infeasible",
    ):
        report = checked(refused(message))
        case = run_suite.with_layers("S2a on", "here", report, LIMIT, gates)
        assert case["gates"] == dict.fromkeys(gates, True), message
        assert case["outcome"].startswith("refused: ")
    # a planner that did not judge it refused nothing: busy with another
    # client's plan, raising, its re-check unable to run, its action ended
    # with no answer, or ended with a status no plan of a valid request has
    for message in (
        "planner busy — one plan at a time",
        "",
        "EXCEPTION: cuRobo plan_single_js raised: CUDA error",
        RECHECK + "constraint re-check unavailable: CUDA error",
        "MotionGenStatus.DT_EXCEPTION: planning failed (MotionGenStatus.DT_EXCEPTION)",
        "no result: planning failed (no result)",
    ):
        case = run_suite.with_layers("S2a on", "here", refused(message), LIMIT, gates)
        assert case["gates"] == dict.fromkeys(gates, False), message
        assert case["outcome"] == "not a refusal: %s" % (message or "no message")
    # find_pairs found the start and the goal inside the limits: a planner
    # that refuses either was not asked what was found
    for message in (
        "START_OUTSIDE_LIMITS: start: joint_2 is 0.120 rad outside the model "
        "position limits [-2.250, 2.250] (max clamp 0.050)",
        "GOAL_OUTSIDE_LIMITS: goal: joint_4 is 0.120 rad outside the model "
        "position limits [-2.580, 2.580] (max clamp 0.050)",
    ):
        case = run_suite.with_layers("S2a on", "here", refused(message), LIMIT, gates)
        assert case["gates"] == dict.fromkeys(gates, False), message
        assert case["outcome"].startswith(run_suite.PREMISE)


# What the stack's planners answer a request whose joint goal is in a layer.
# They plan to joints by 'auto': the joint-space plan fails naming the boxes
# the goal is in, but that is only logged, and the fallback plans to the
# goal's pose, whose failure says nothing of the joint goal.
AUTO_IK_FAIL = (
    "MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal — move "
    "it away from obstacles or relax the orientation. [world: 25 boxes: scene 0 | "
    "config layers: platform 16, person 8 | scene layers: floor 1]"
)


@pytest.mark.parametrize("gates", [("G2",), ("G4", "G2")])
def test_a_request_whose_goal_the_planner_refuses_as_a_start_tests_nothing(gates):
    # find_pairs found the goal valid; asked from it, the planner refuses it
    in_person = dict(refused(INVALID_START), world=None)
    # refused, or flown to another arm at the goal's pose by the fallback
    for report in (refused(AUTO_IK_FAIL), flown()):
        case = run_suite.with_layers(
            "S3 reach 1", "here", checked(report, in_person), LIMIT, gates
        )
        assert case["gates"] == dict.fromkeys(gates, False)
        assert case["outcome"].startswith(run_suite.PREMISE + ": the goal is refused")
        assert "person/lap (154 mm)" in case["outcome"]
    # a goal it judged leaves the request's verdict as it was: one it plans
    # from, or one it plans from to no plan (it took the start)
    for check in (VALID, dict(refused(IK_FAIL), world=None)):
        case = run_suite.with_layers(
            "S3 reach 1", "here", checked(refused(AUTO_IK_FAIL), check), LIMIT, gates
        )
        assert case["gates"] == dict.fromkeys(gates, True)
        assert case["outcome"] == "refused: IK_FAIL"
    # a check that judged nothing checked nothing
    busy = dict(refused("planner busy — one plan at a time"), world=None)
    case = run_suite.with_layers(
        "S3 reach 1", "here", checked(flown(), busy), LIMIT, gates
    )
    assert case["gates"] == dict.fromkeys(gates, False)
    assert case["outcome"] == (
        "its goal's check judged nothing: planner busy — one plan at a time"
    )


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
    "world set to /c/consumer_world.yaml (32 boxes: scene 8 | config layers: "
    "platform 16, person 8 | scene layers: none)"
)


def test_the_push_reply_names_every_config_layer_with_its_boxes():
    expected = {"platform": 16, "person": 8}
    case = run_suite.pushed("here", True, REPLY, expected)
    assert case["gates"] == {"G4": True}
    for ok, reply in (
        (True, REPLY.replace("person 8", "person 0")),
        (True, REPLY.replace(", person 8", "")),
        (True, "world set to /c/platform 16 person 8.yaml (8 boxes: scene 8)"),
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
    of (planner, world, joints, execute) -> a move report, and checking a
    state with `check`, a function of (planner, q) -> a check_state report;
    it records every call."""

    made = []

    def __init__(self, name, script, check):
        self.calls, self.script, self.check = [], script, check
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
        self.calls.append(("move", planner, world and str(world), list(start), joints))
        return self.script(planner, world and str(world), joints, execute)

    def check_state(self, planner, q):
        self.calls.append(("check_state", planner, list(q)))
        return self.check(planner, list(q))

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
    """run_suite.main against a scripted Tester (answers: `script`; checks of
    a state: `check`, every one valid unless given), with S0 stood in for,
    results under tmp_path. Returns a runner: argv -> (exit status, the
    Scripted made, if any, and the result file's contents)."""
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

    def run(argv, script=_answers, check=lambda planner, q: VALID):
        monkeypatch.setattr(
            run_suite, "Tester", lambda name: Scripted(name, script, check)
        )
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
    # each request planned with its layers, refused or flown, is followed by
    # the planner's check of its goal, in the world it planned in; a
    # positive control's and the retract's are not checked
    checks = [call[1:] for call in tester.calls if call[0] == "check_state"]
    assert checks == (
        [(stack.CONTROL, pair["goal"]) for pair in platform["pairs"]]
        + [(stack.CHAIR, pair["goal"]) for pair in person["pairs"]]
        + [(stack.CHAIR, goal) for goal in DATA["reaches"]["goals"]]
        + [(stack.CHAIR, pair["goal"]) for pair in person["pairs"]]
    )
    for before, call in zip(tester.calls, tester.calls[1:]):
        if call[0] == "check_state":
            assert before[0] == "move" and (before[1], before[4]) == call[1:]
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
    # S2 on, S3 and S4: what a consumer on the chair asks
    assert result["reported"]["chair_flown"] == "14 of 14" and "WARNING" not in out
    # how well the sim tracks, from the flights that touched nothing
    assert result["reported"]["tracking_rad_max"] == 0.012


def test_each_line_names_the_world_its_planner_said_it_planned_in(
    scripted, monkeypatch
):
    # what the planner replied to the push, and not what the suite pushed: a
    # chair config that drops the person is shown as the planner holds it
    taken = (
        "world set to world_mebot.yaml (17 boxes: scene 0 | config layers: "
        "platform 16 | scene layers: floor 1)"
    )
    consumer = (
        "world set to /c/consumer_world.yaml (24 boxes: scene 8 | config layers: "
        "platform 16 | scene layers: none)"
    )
    floor = stack.worlds()["floor_only"][1]

    def answers(planner, world, joints, execute):
        if (planner, world) == (stack.CONTROL, floor):
            # a world the planner did not take is named as it was pushed
            return dict(refused("set_world refused: no such file"), world="no such")
        report = _answers(planner, world, joints, execute)
        return dict(report, world=taken) if planner == stack.CHAIR else report

    monkeypatch.setattr(Scripted, "set_world", lambda self, world: (True, consumer))
    _, _, result = scripted([], script=answers)
    where = {case["case"]: case["where"] for case in result["cases"]}
    chair = "/rammp_curobo (%s)" % taken[taken.index("(") + 1 : -1]
    assert where["S2a on"] == where["S3 reach 8"] == where["S5 retract"] == chair
    assert where["S4a"] == "/rammp_curobo (%s)" % consumer[consumer.index("(") + 1 : -1]
    assert where["S1a on"] == "/rammp_curobo_control (23 boxes: scene 0 | ...)"
    assert where["S1a off"] == "/rammp_curobo_control (control_floor.yaml)"


def test_a_whole_run_fails_a_gate_that_no_case_bore_on(scripted, capsys):
    # every request planned with its layers refused, every positive control a
    # hit: no flight clear of contact shows that the sim flies a plan (G6)
    def refusing(planner, world, joints, execute):
        report = _answers(planner, world, joints, execute)
        return refused() if execute and not report.get("contacts") else report

    status, _, result = scripted([], script=refusing)
    err = capsys.readouterr().err
    assert status == 1 and not result["passed"]
    assert result["gates"]["G2"] == "PASS" and result["gates"]["G6"] == "FAIL"
    assert "G6" in result["error"] and "G6" in err
    # a run of one scenario judges its own gates alone
    status, _, result = scripted(["--only", "S3"], script=refusing)
    assert status == 0 and result["gates"]["G6"] == "not run"


def test_a_run_says_loudly_when_the_chair_planner_flew_nothing(scripted, capsys):
    # G2 and G4 pass a refusal, and G6 the control planner's flights: a
    # chair planner that refuses everything passes every gate (spec 8 gates
    # no S3 success rate), so the run says so
    platform = [pair["goal"] for pair in DATA["platform_pairs"]["pairs"]]

    def chair_refuses(planner, world, joints, execute):
        if planner == stack.CHAIR and execute:
            return refused()
        if planner == stack.CONTROL and joints in platform:
            # S1, as on the Jetson: flown clear with the platform, a hit without
            return flown() if world == stack.worlds()["no_person"][1] else flown(WHEEL)
        return _answers(planner, world, joints, execute)

    status, _, result = scripted([], script=chair_refuses)
    out = capsys.readouterr().out
    assert status == 0 and result["passed"]
    assert result["gates"] == {gate: "PASS" for gate in run_suite.GATES}
    assert result["reported"]["chair_flown"] == "0 of 14"
    assert "WARNING: the chair planner, /rammp_curobo, flew none of the 14" in out


def test_a_goal_the_planner_no_longer_finds_valid_fails_the_suite(scripted, capsys):
    goal = DATA["reaches"]["goals"][0]

    def check(planner, q):
        return dict(refused(INVALID_START), world=None) if q == goal else VALID

    status, _, result = scripted(["--only", "S3"], check=check)
    out = capsys.readouterr().out
    assert status == 1 and result["gates"]["G2"] == "FAIL"
    first, *rest = result["cases"]
    assert first["case"] == "S3 reach 1" and not first["passed"]
    assert first["outcome"].startswith(run_suite.PREMISE + ": the goal is refused")
    assert "premise broken; re-run find_pairs" in out
    assert all(case["passed"] for case in rest)


@pytest.mark.parametrize(
    "key, listed",
    [("platform_pairs", "pairs"), ("person_pairs", "pairs"), ("reaches", "goals")],
)
def test_fewer_requests_than_find_pairs_finds_fail_loudly_before_anything_runs(
    scripted, monkeypatch, key, listed
):
    # each would leave its cases out, and the gates they bear on unjudged
    for short in (DATA[key][listed][:-1], [], None):
        data = dict(DATA, **{key: dict(DATA[key], **{listed: short})})
        monkeypatch.setattr(stack, "scenarios", lambda: data)
        status, tester, result = scripted([])
        assert status == 1 and tester is None and result["cases"] == []
        assert run_suite.PREMISE in result["error"], short
        assert "%d %s" % (len(short or []), key.split("_")[0]) in result["error"]


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
    without = REPLY.replace(", person 8", "")
    monkeypatch.setattr(Scripted, "set_world", lambda self, world: (True, without))
    status, _, result = scripted(["--only", "S4"])
    assert status == 1 and result["gates"]["G4"] == "FAIL"
    assert "person 8" in result["cases"][0]["outcome"]
