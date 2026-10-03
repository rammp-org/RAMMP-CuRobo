"""The sim suite: the paired scenarios of the MEBot layers, planned by the
stack's planner nodes and flown on the sim arm, as a consumer would.

    ros2 launch mebot_sim suite.launch.py   # the stack and the suite (make sim)
    ros2 run mebot_sim run_suite            # against a stack already up
    ros2 run mebot_sim run_suite --only S2

The requests are the ones find_pairs wrote to config/scenarios.yaml, each
flown from its start, the sim arm put there first. One line per case: what
was run, the planner and its world, what came of it, PASS or FAIL; then each
gate's verdict and what is reported without a gate. The same goes, as JSON,
to ${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/sim/<UTC>-<git sha>.json. Exit
status 1 if anything failed.

  S0  The layers against the true shapes (check_layers): G0 and G1.
  S1  Platform pairs, the chair empty. The control planner, with the floor
      and the platform, refuses each or plans a flight that touches nothing
      (G2). With the floor alone it plans it, and the flight hits the MEBot
      (G3): the sim sees the chair, and leaving the layer out is dangerous.
  S2  Person pairs, the stand-in seated: the same on the chair planner, and
      on the control planner with the floor and the platform, whose flights
      hit the stand-in.
  S3  Eight reaches from one start on the chair planner: how many plan is
      reported, and every flight touches nothing (G2).
  S4  A consumer's push, config/consumer_world.yaml: 8 boxes and no layers,
      on the chair planner. Its reply names each layer of its config with
      its boxes, and the S2 requests, asked again, are refused or flown
      touching nothing (G4).
  S5  A plan from the bench retract on the chair planner fails, naming the
      person box the start is in and how deep (G5).

G6: every flight that touched nothing came to rest within end_error_rad of
its plan's last point, so the sim flew the plan. A positive control that
does not hit, and requests that find_pairs found with other layer files than
the committed ones, fail with "premise broken; re-run find_pairs".
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

from mebot_sim import check_layers, scene, stack
from mebot_sim.client import WHO, Tester, contact_text

SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5")
GATES = {
    "G0": "MuJoCo's MEBot is inside the platform, and not when it shrinks 1 mm",
    "G1": "the stand-in is inside the padded person layer or the arm's column",
    "G2": "no flight planned with the layers touches anything",
    "G3": "every positive control is planned, flown, and hits what its layer covers",
    "G4": "the layers survive a consumer's push",
    "G5": "a plan from the retract names person/ and a depth",
    "G6": "every flight that touches nothing comes to rest at its last point",
}
PREMISE = "premise broken; re-run find_pairs"
# The layers the chair config composes into every world, by the layer files
# they are held to: a config that drops one is the regression S4 is for.
CHAIR_LAYERS = ("platform", "person")


def _status(message):
    """The planner's status, at the head of a failed plan's message."""
    return message.split(":", 1)[0].replace("MotionGenStatus.", "")


def with_layers(name, where, report, limit, gates=("G2",)):
    """The case of a request planned with every layer that covers what it
    would hit: refused, or flown touching nothing and at rest within `limit`
    rad of the plan's last point (G6). `report` is client.Tester.move's;
    `gates` are the gates its refusal or its flight bears on."""

    def verdict(passed, outcome, **more):
        return stack.case(
            name, where, outcome, dict({g: passed for g in gates}, **more), **report
        )

    if not report["planned"]:
        message = report["message"]
        if "INVALID_START" in message:
            # find_pairs found every start valid here: a refusal of the start
            # says nothing about the path
            return verdict(False, "%s: the start is refused: %s" % (PREMISE, message))
        if message.startswith("set_world refused") or message.endswith(
            "rejected the goal"
        ):
            return verdict(False, message)
        return verdict(True, "refused: %s" % _status(message))
    if not report["executed"]:
        return verdict(False, "not flown to its end: %s" % report["exec_error"])
    if report["contacts"]:
        return verdict(False, contact_text(report["contacts"]))
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
    cuRobo's IK seed, joint_1..joint_7."""
    stack.core()
    from rammp_curobo.config import load_planner_config, resolve_config
    from rammp_curobo.robot_config import load_robot_config

    cfg, where = load_planner_config(stack.CHAIR_CONFIG)
    robot = load_robot_config(resolve_config(cfg["robot"], relative_to=where))
    cspace = robot["kinematics"]["cspace"]
    by_name = dict(zip(cspace["joint_names"], cspace["retract_config"]))
    return [float(by_name[name]) for name in cfg["joint_names"]]


class Suite:
    """A run of the suite: the cases so far, the Tester flying them, and what
    ended the run early, if anything did. `assets` makes S0's scene.Assets."""

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

    def _pairs(self, scenario, pairs, on, off, group):
        """Each pair planned with the layer that covers `group` (`on`: a key of
        stack.worlds) and without it (`off`), and flown."""
        worlds = stack.worlds()
        for k, pair in enumerate(pairs):
            name = "%s%s" % (scenario, chr(ord("a") + k))
            for key in (on, off):
                planner, world, sees = worlds[key]
                self.tester.teleport(pair["start"])
                report = self.tester.move(
                    planner, world, pair["start"], joints=pair["goal"], speed=self.speed
                )
                where = "%s (%s)" % (planner, sees)
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
        planner, world, sees = stack.worlds()["all"]
        for k, goal in enumerate(data["reaches"]["goals"]):
            self.tester.teleport(data["start"])
            report = self.tester.move(
                planner, world, data["start"], joints=goal, speed=self.speed
            )
            where = "%s (%s)" % (planner, sees)
            self.add(with_layers("S3 reach %d" % (k + 1), where, report, self.limit))

    def _s4(self, data):
        pairs = data["person_pairs"]
        self.tester.posture(pairs["posture"])
        world = stack.config("consumer_world.yaml")
        ok, reply = self.tester.planners[stack.CHAIR].set_world(world)
        layers = stack.generated_layers()
        expected = {name: len(layers[name].obstacles) for name in CHAIR_LAYERS}
        self.add(pushed("%s (%s)" % (stack.CHAIR, world.name), ok, reply, expected))
        # asked in the world pushed, with no push of their own
        where = "%s (the push; %s)" % (stack.CHAIR, ", ".join(expected))
        for k, pair in enumerate(pairs["pairs"]):
            name = "S4%s" % chr(ord("a") + k)
            if not ok:
                self.add(stack.case(name, where, "not asked: no push", {"G4": False}))
                continue
            self.tester.teleport(pair["start"])
            report = self.tester.move(
                stack.CHAIR, None, pair["start"], joints=pair["goal"], speed=self.speed
            )
            self.add(with_layers(name, where, report, self.limit, ("G4", "G2")))

    def _s5(self, data):
        planner, world, sees = stack.worlds()["all"]
        report = self.tester.move(
            planner, world, bench_retract(), joints=data["start"], execute=False
        )
        self.add(from_retract("%s (%s)" % (planner, sees), report))

    def reported(self):
        """What the suite reports without a gate."""
        reaches = [case for case in self.cases if case["case"].startswith("S3 ")]
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
            "s3_planned": "%d of %d"
            % (sum(c["planned"] for c in reaches), len(reaches))
            if reaches
            else None,
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
        if reported["s3_planned"]:
            said.append("%s S3 reaches planned" % reported["s3_planned"])
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
