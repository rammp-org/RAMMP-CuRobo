"""The sim suite: the paired scenarios of the MEBot layers, planned by the
stack's planner nodes and flown on the sim arm, as a consumer would.

    ros2 launch mebot_sim suite.launch.py   # the stack and the suite (make sim)
    ros2 run mebot_sim run_suite            # against a stack already up
    ros2 run mebot_sim run_suite --only S2

The requests are the ones find_pairs wrote to config/scenarios.yaml, each
flown from its start, the sim arm put there first. One line per case: what
was run, the planner and its world as the planner summed it up when it took
it, what came of it, PASS or FAIL; then each gate's verdict and what is
reported without a gate. The same goes, as JSON, to
${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/sim/<UTC>-<git sha>.json. Exit
status 1 if anything failed, or if a run of every scenario left a gate with
no case to judge.

  S0  The layers against the true shapes (check_layers): G0 and G1.
  S1  Platform pairs, the chair empty. The control planner, with the floor
      and the platform, refuses each or plans a flight that touches nothing
      (G2). With the floor alone it plans it, and the flight hits the MEBot
      (G3): the sim sees the chair, and leaving the layer out is dangerous.
  S2  Person pairs, the stand-in seated: the same on the chair planner, and
      on the control planner with the floor and the platform, whose flights
      hit the stand-in.
  S3  Eight reaches from one start on the chair planner: every flight
      touches nothing (G2), and at least min_chair_reaches of them are
      planned and flown (G7).
  S4  A consumer's push, config/consumer_world.yaml: 8 boxes and no layers,
      on the chair planner. Its reply names each layer of its config with
      its boxes, and the S2 requests, asked again, are refused or flown
      touching nothing (G4).
  S5  A plan from the bench retract on the chair planner fails, naming the
      person box the start is in and how deep (G5).

G6: each planner flew a plan that touched nothing, and every flight that
touched nothing came to rest within end_error_rad of its plan's last point,
so the sim flew the plans of both. A run of every scenario fails it when
either planner flew none; a run of one judges the flights it made.

G7: the chair planner planned and flew at least min_chair_reaches of the
S3 reaches touching nothing (config/scenarios.yaml). G2 and G4 pass a
refusal; G6 and G7 fail a chair planner that refuses every request.

A refusal is the planner's answer to a request it judged and found no safe
plan for: cuRobo's IK, graph or trajectory optimisation failed, or the
planner's re-check flagged states of the plan cuRobo found. Any other
failure (a busy planner, an exception, a re-check that could not run, no
answer) judged nothing and fails. A positive control that does not hit, a
start or goal the planner refuses that find_pairs found valid, and requests
that find_pairs found with other layer files than the committed ones, or
fewer than it finds, fail with "premise broken; re-run find_pairs". The goal
of each request planned with its layers that the planner refused or flew is
checked as a start, in the world it planned in (client.Tester.check_state).
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mebot_sim import check_layers, find_pairs, scene, stack
from mebot_sim.client import WHO, Tester, contact_text

SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5")
GATES = {
    "G0": "MuJoCo's MEBot is inside the platform, and not when it shrinks 1 mm",
    "G1": "the stand-in, the humanoid at each size from the 5th-percentile woman "
    "to the 95th-percentile man, is in the padded person layer (covered_pct of "
    "it) or the arm's column",
    "G2": "no flight planned with the layers touches anything",
    "G3": "every positive control is planned, flown, and hits what its layer covers",
    "G4": "the layers survive a consumer's push",
    "G5": "a plan from the retract names person/ and a depth",
    "G6": "each planner flies a plan that touches nothing, and every such flight "
    "comes to rest at its last point",
    "G7": "the chair planner plans and flies at least min_chair_reaches of the S3 "
    "reaches, touching nothing",
}
PREMISE = "premise broken; re-run find_pairs"
# The layers the chair config composes into every world, by the layer files
# they are held to: a config that drops one is the regression S4 is for.
CHAIR_LAYERS = ("platform", "person")
# How cuRobo fails a valid request it finds no plan for.
REFUSALS = ("IK_FAIL", "GRAPH_FAIL", "TRAJOPT_FAIL", "FINETUNE_TRAJOPT_FAIL")


def _status(message):
    """The planner's status, at the head of a failed plan's message."""
    return message.split(":", 1)[0].replace("MotionGenStatus.", "")


def refusal(message):
    """Whether a failed plan's `message` is the planner's refusal: cuRobo
    found no plan, or the planner's post-validation found every problem it
    lists in the plan cuRobo found: states its re-check flagged, or a limit
    the plan exceeds. A re-check that could not run flagged nothing."""
    status = _status(message)
    if status in REFUSALS:
        return True
    head = "plan rejected by post-validation: "
    if status != "LIBRARY_VALIDATION_FAILED" or head not in message:
        return False
    problems = message.split(head, 1)[1].split("; ")
    return all(
        p.startswith("constraint re-check flagged") or " > limit " in p
        for p in problems
    )


def planned_in(planner, reply, world):
    """A case's planner and the world it planned in, as the planner summed
    it up in its `reply` to the push of `world`: "/rammp_curobo (25 boxes:
    scene 0 | config layers: platform 16, person 8 | scene layers: floor 1)".
    A world it did not take is named as it was pushed."""
    taken = re.fullmatch(r"world set to .* \(([^()]*)\)", reply or "")
    return "%s (%s)" % (planner, taken.group(1) if taken else Path(world).name)


def with_layers(name, where, report, limit, gates=("G2",)):
    """The case of a request planned with every layer that covers what it
    would hit: refused, or flown touching nothing and at rest within `limit`
    rad of the plan's last point (G6), its goal still a valid state.
    `report` is Suite._request's; `gates` are the gates its refusal or its
    flight bears on."""

    def verdict(passed, outcome, **more):
        return stack.case(
            name, where, outcome, dict({g: passed for g in gates}, **more), **report
        )

    message = report["message"]
    status = _status(message)
    if not report["planned"]:
        # find_pairs found the start valid, and the start and the goal
        # inside the limits: refusing any of that says nothing about the path
        if status.startswith("INVALID_START") or status == "START_OUTSIDE_LIMITS":
            return verdict(False, "%s: the start is refused: %s" % (PREMISE, message))
        if status == "GOAL_OUTSIDE_LIMITS":
            return verdict(False, "%s: the goal is refused: %s" % (PREMISE, message))
        if not refusal(message):
            return verdict(False, "not a refusal: %s" % (message or "no message"))
    elif not report["executed"]:
        return verdict(False, "not flown to its end: %s" % report["exec_error"])
    elif report["contacts"]:
        return verdict(False, contact_text(report["contacts"]))
    # find_pairs found the goal valid too. A goal the planner takes as a
    # start is still valid, whether or not it then finds a plan from it; one
    # it refuses makes the request's refusal, or its flight to another arm
    # at the goal's pose, say nothing about the path
    check = report["goal_as_start"]
    if not (check["planned"] or refusal(check["message"])):
        if _status(check["message"]).startswith("INVALID_START"):
            return verdict(
                False,
                "%s: the goal is refused as a start: %s" % (PREMISE, check["message"]),
            )
        return verdict(
            False,
            "its goal's check judged nothing: %s" % (check["message"] or "no message"),
        )
    if not report["planned"]:
        return verdict(True, "refused: %s" % status)
    rest = report["end_error_rad"]
    return verdict(
        True,
        "flown, no contact; at rest %.3f rad from its last point" % rest,
        G6=rest <= limit,
    )


def without_layer(name, where, report, group):
    """A positive control: the request planned without the layer that covers
    contact group `group`, which its flight must hit (G3)."""
    hit = report["planned"] and report["executed"]
    if hit and any(c["group"] == group for c in report["contacts"]):
        return stack.case(
            name, where, contact_text(report["contacts"]), {"G3": True}, **report
        )
    if not report["planned"]:
        what = "not planned: %s" % report["message"]
    elif not report["executed"]:
        what = "not flown to its end: %s" % report["exec_error"]
    else:
        what = "flown without touching %s (%s)" % (
            WHO[group],
            contact_text(report["contacts"]),
        )
    return stack.case(name, where, "%s: %s" % (PREMISE, what), {"G3": False}, **report)


def chair_reaches(where, cases, least):
    """S3's count (G7): of the reaches, `cases`, the chair planner planned at
    least `least` and flew them to their end touching nothing. A refusal
    passes G2: G7 holds the chair planner to plans of its own."""
    flown = sum(bool(case.get("executed")) and not case["contacts"] for case in cases)
    passed = flown >= least
    return stack.case(
        "S3 reaches",
        where,
        "%d of %d planned and flown, no contact%s"
        % (flown, len(cases), "" if passed else " (under %d)" % least),
        {"G7": passed},
        flown_clear=flown,
        min_chair_reaches=least,
    )


def pushed(where, ok, reply, expected):
    """S4's push: the chair planner took the consumer's world, and its reply
    names each layer of its config with its boxes, `expected` ({layer:
    boxes})."""
    named = re.search(r"config layers: ([^|)]*)", reply) if ok else None
    listed = dict(re.findall(r"(\w+) (\d+)", named.group(1))) if named else {}
    missing = [
        "%s %d" % item
        for item in expected.items()
        if listed.get(item[0]) != str(item[1])
    ]
    if not ok:
        outcome = "refused: %s" % reply
    elif missing:
        outcome = "the reply does not name %s: %s" % (", ".join(missing), reply)
    else:
        outcome = "taken: %s" % reply[reply.rfind("(") + 1 : reply.rfind(")")]
    return stack.case(
        "S4 push", where, outcome, {"G4": ok and not missing}, reply=reply
    )


def from_retract(where, report):
    """S5: planned from the bench retract, the request fails, naming a person
    box the start is in and how deep (G5)."""
    named = re.search(r"person/\w+ \(\d+ mm\)", report["message"])
    passed = not report["planned"] and named is not None
    outcome = (
        "refused (%s): the start in %s" % (_status(report["message"]), named.group(0))
        if passed
        else "planned"
        if report["planned"]
        else report["message"]
    )
    return stack.case("S5 retract", where, outcome, {"G5": passed}, **report)


def gates(cases):
    """Each gate's verdict: whether every case that bears on it passed it,
    None if none bears on it."""
    out = {}
    for gate in GATES:
        verdicts = [case["gates"][gate] for case in cases if gate in case["gates"]]
        out[gate] = all(verdicts) if verdicts else None
    return out


def unjudged(cases):
    """What a run of every scenario must judge that `cases` leave unjudged,
    {gate: in words}: each gate none of them bears on, and G6 while either
    planner flew none of them that bears on it. G6 judges each planner on
    its own flights: the other's show nothing of how the sim flies its
    plans."""
    flew = {case.get("planner") for case in cases if "G6" in case["gates"]}
    unflown = [
        planner for planner in (stack.CHAIR, stack.CONTROL) if planner not in flew
    ]
    out = {}
    for gate, verdict in gates(cases).items():
        if gate == "G6" and unflown:
            out[gate] = "G6 for %s" % " and ".join(unflown)
        elif verdict is None:
            out[gate] = gate
    return out


def result_path(utc, sha):
    """Where the result of a run at `utc` of commit `sha` goes."""
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache) / "rammp_curobo" / "sim" / ("%s-%s.json" % (utc, sha))


def _git():
    """(this checkout's commit, short; whether its tracked files differ from
    it), or ("unknown", None) where git cannot say."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(stack.checkout()), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        return sha, bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def bench_retract():
    """The bench retract: the robot config's retract_config, which is also
    cuRobo's IK seed, arm_joint_1..arm_joint_7."""
    stack.core()
    import yaml
    from rammp_curobo.config import load_planner_config, resolve_config

    cfg, where = load_planner_config(stack.CHAIR_CONFIG)
    # read as load_robot_config reads it, which refuses to load without the
    # arm's model; the seed is taken from the file as-is
    path = resolve_config(cfg["robot"], relative_to=where)
    with open(path, encoding="utf-8") as f:
        cspace = yaml.safe_load(f)["robot_cfg"]["kinematics"]["cspace"]
    by_name = dict(zip(cspace["joint_names"], cspace["retract_config"]))
    return [float(by_name[name]) for name in cfg["joint_names"]]


class Suite:
    """A run of the suite: the cases so far, the Tester flying them, and what
    failed the run beyond its cases, if anything did: what ended it early, or
    a gate that a run of every scenario gave no case to judge. `assets` makes
    S0's scene.Assets."""

    def __init__(self, speed, assets):
        self.speed, self.assets = speed, assets
        self.cases, self.error, self.tester = [], None, None
        self.limit = None  # G6's, from config/scenarios.yaml

    @property
    def passed(self):
        return self.error is None and all(case["passed"] for case in self.cases)

    def add(self, case):
        self.cases.append(case)
        print(stack.case_line(case), flush=True)

    def run(self, only):
        """The scenarios, `only` one of them if it is given."""
        selected = [only] if only else list(SCENARIOS)
        data = stack.scenarios()
        if selected != ["S0"]:
            # S1 to S5 fly what find_pairs found with the layers it was given
            found, now = data.get("layers_sha256") or {}, stack.layer_digests()
            changed = sorted(
                name
                for name in set(found) | set(now)
                if found.get(name) != now.get(name)
            )
            if changed:
                raise RuntimeError(
                    "%s: config/scenarios.yaml holds requests found with other "
                    "layer files than the committed ones (%s)"
                    % (PREMISE, ", ".join(changed))
                )
            # ... and all it found: a request fewer is a case left out unseen
            reaches = find_pairs.REACHES
            wanted = {
                "platform pairs": (data["platform_pairs"]["pairs"], find_pairs.PAIRS),
                "person pairs": (data["person_pairs"]["pairs"], find_pairs.PAIRS),
                "reaches": (
                    data["reaches"]["goals"],
                    reaches["per_region"] * len(reaches["regions"]),
                ),
            }
            short = [
                "%d %s, not %d" % (len(listed or []), what, n)
                for what, (listed, n) in wanted.items()
                if len(listed or []) != n
            ]
            if short:
                raise RuntimeError(
                    "%s: config/scenarios.yaml holds %s" % (PREMISE, "; ".join(short))
                )
            # held to no reach, G7 would pass a chair planner that refused them all
            least = data["thresholds"]["min_chair_reaches"]
            if least < 1:
                raise ValueError(
                    "config/scenarios.yaml holds thresholds.min_chair_reaches %s: "
                    "G7 would hold the chair planner to no reach, and judge nothing"
                    % least
                )
        if "S0" in selected:
            for case in check_layers.cases(self.assets()):
                self.add(case)
        flown = [s for s in selected if s != "S0"]
        if not flown:
            return
        self.limit = data["thresholds"]["end_error_rad"]
        self.tester = Tester("mebot_run_suite")
        self.tester.connect()
        for scenario in flown:
            getattr(self, "_" + scenario.lower())(data)

    def _request(self, planner, world, start, goal, layers=True):
        """client.Tester.move's report of a request to the joint goal `goal`,
        flown from `start`, the sim arm put there first; with its `planner`.
        Planned with its `layers`, a request the planner refused or planned
        holds `goal_as_start` as well: client.Tester.check_state's report of
        the goal, in the world the request was planned in.

        find_pairs found the goal valid. A failed plan names the boxes its
        joint goal is in only when the planner plans to joints by 'js'; the
        stack's planners plan by 'auto', whose fallback plans to the goal's
        pose and fails about that, or flies another arm there."""
        self.tester.teleport(start)
        report = self.tester.move(planner, world, start, joints=goal, speed=self.speed)
        report["planner"] = planner
        if layers and (report["planned"] or refusal(report["message"])):
            report["goal_as_start"] = self.tester.check_state(planner, goal)
        return report

    def _pairs(self, scenario, pairs, on, off, group):
        """Each pair planned with the layer that covers `group` (`on`: a key of
        stack.worlds) and without it (`off`), and flown."""
        worlds = stack.worlds()
        for k, pair in enumerate(pairs):
            name = "%s%s" % (scenario, chr(ord("a") + k))
            for key in (on, off):
                planner, world, _ = worlds[key]
                report = self._request(
                    planner, world, pair["start"], pair["goal"], layers=key == on
                )
                where = planned_in(planner, report["world"], world)
                if key == on:
                    self.add(with_layers(name + " on", where, report, self.limit))
                else:
                    self.add(without_layer(name + " off", where, report, group))

    def _s1(self, data):
        pairs = data["platform_pairs"]
        self.tester.posture(pairs["posture"])
        self._pairs("S1", pairs["pairs"], "no_person", "floor_only", "mebot")

    def _s2(self, data):
        pairs = data["person_pairs"]
        self.tester.posture(pairs["posture"])
        self._pairs("S2", pairs["pairs"], "all", "no_person", "standin")

    def _s3(self, data):
        self.tester.posture(data["person_pairs"]["posture"])
        planner, world, _ = stack.worlds()["all"]
        reaches = []
        for k, goal in enumerate(data["reaches"]["goals"]):
            report = self._request(planner, world, data["start"], goal)
            where = planned_in(planner, report["world"], world)
            reaches.append(
                with_layers("S3 reach %d" % (k + 1), where, report, self.limit)
            )
            self.add(reaches[-1])
        least = data["thresholds"]["min_chair_reaches"]
        self.add(chair_reaches(planner, reaches, least))

    def _s4(self, data):
        pairs = data["person_pairs"]
        self.tester.posture(pairs["posture"])
        world = stack.config("consumer_world.yaml")
        ok, reply = self.tester.planners[stack.CHAIR].set_world(world)
        layers = stack.generated_layers()
        expected = {name: len(layers[name].obstacles) for name in CHAIR_LAYERS}
        self.add(pushed("%s (%s)" % (stack.CHAIR, world.name), ok, reply, expected))
        # asked in the world pushed, with no push of their own
        where = planned_in(stack.CHAIR, reply, world)
        for k, pair in enumerate(pairs["pairs"]):
            name = "S4%s" % chr(ord("a") + k)
            if not ok:
                self.add(stack.case(name, where, "not asked: no push", {"G4": False}))
                continue
            report = self._request(stack.CHAIR, None, pair["start"], pair["goal"])
            self.add(with_layers(name, where, report, self.limit, ("G4", "G2")))

    def _s5(self, data):
        planner, world, _ = stack.worlds()["all"]
        report = self.tester.move(
            planner, world, bench_retract(), joints=data["start"], execute=False
        )
        self.add(from_retract(planned_in(planner, report["world"], world), report))

    def reported(self):
        """What the suite reports without a gate."""
        # of the requests asked of the chair planner with its layers, how
        # many it flew
        asked = [
            case
            for case in self.cases
            if case.get("planner") == stack.CHAIR and "G2" in case["gates"]
        ]
        flown = sum(bool(case.get("executed")) for case in asked)
        plans = sorted(case["plan_s"] for case in self.cases if "plan_s" in case)
        flights = [case for case in self.cases if case.get("executed")]
        closest = {}
        for case in flights:
            if "G2" in case["gates"]:  # planned with the layers
                for group, near in case["closest"].items():
                    closest[group] = min(closest.get(group, near["mm"]), near["mm"])
        # how well the sim tracks a plan: a flight held by what it hit says
        # nothing of that
        tracking = [
            c["tracking_rad"]
            for c in flights
            if not c["contacts"] and c["tracking_rad"] is not None
        ]
        return {
            "chair_flown": "%d of %d" % (flown, len(asked)) if asked else None,
            "plan_s": {
                "min": plans[0],
                "median": statistics.median(plans),
                "max": plans[-1],
            }
            if plans
            else None,
            "closest_mm": closest,
            "tracking_rad_max": max(tracking, default=None),
        }

    def finish(self, only):
        """Print the summary, write the result file, and return its path."""
        verdicts = gates(self.cases)
        idle = unjudged(self.cases)
        if idle and only is None and self.error is None:
            # nothing is skipped: what every scenario gave no case to judge is
            # not shown to hold
            self.error = (
                "no case bore on %s, which a run of every scenario must judge"
                % ", ".join(idle.values())
            )
            verdicts.update(dict.fromkeys(idle, False))
        failed = sum(not case["passed"] for case in self.cases)
        reported = self.reported()
        sha, dirty = _git()
        utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = result_path(utc, sha)
        result = {
            "utc": utc,
            "git": {"sha": sha, "dirty": dirty},
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "only": only,
            "speed": self.speed,
            "cases": self.cases,
            "gates": {
                gate: "not run" if v is None else "PASS" if v else "FAIL"
                for gate, v in verdicts.items()
            },
            "reported": reported,
            "error": self.error,
            "passed": self.passed,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

        print("\n%d cases, %d failed" % (len(self.cases), failed))
        for gate, about in GATES.items():
            print("%s %-7s %s" % (gate, result["gates"][gate], about))
        said = []
        if reported["chair_flown"]:
            said.append("%s chair planner requests flown" % reported["chair_flown"])
        if reported["plan_s"]:
            said.append(
                "plans in %(min).2f to %(max).2f s, median %(median).2f s"
                % reported["plan_s"]
            )
        if reported["closest_mm"]:
            said.append(
                "closest with the layers: "
                + ", ".join(
                    "%s %.1f mm" % (WHO[group].split()[-1], mm)
                    for group, mm in sorted(reported["closest_mm"].items())
                )
            )
        if reported["tracking_rad_max"] is not None:
            said.append(
                "off its last point as a flight without contact ended: up to "
                "%.3f rad" % reported["tracking_rad_max"]
            )
        if said:
            print("reported: " + "; ".join(said))
        if self.error:
            print("run_suite: %s" % self.error, file=sys.stderr)
        print("result: %s" % path)
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=SCENARIOS, help="run this scenario alone")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="fly plans at this fraction of their speed (default: 0.5)",
    )
    parser.add_argument(
        "--menagerie",
        default=scene.MENAGERIE,
        help="a mujoco_menagerie checkout, for S0 (default: %(default)s)",
    )
    parser.add_argument(
        "--humanoid",
        default=scene.HUMANOID,
        help="the DeepMind humanoid the stand-in is posed from, for S0",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not 0 < args.speed <= 1:
        parser.error("--speed must be in (0, 1], got %g" % args.speed)
    suite = Suite(args.speed, lambda: scene.Assets(args.menagerie, args.humanoid))
    try:
        try:
            suite.run(args.only)
        except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            suite.error = "%s: %s" % (type(exc).__name__, exc)
        except KeyboardInterrupt:
            suite.error = "interrupted"
        suite.finish(args.only)
    finally:
        if suite.tester is not None:
            suite.tester.close()
    return 0 if suite.passed else 1


if __name__ == "__main__":
    sys.exit(main())
