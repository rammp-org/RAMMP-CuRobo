#!/usr/bin/env python3
"""A behaviour snapshot of the planner, to compare before and after a refactor.

    PYTHONPATH=<repo>/core python3 scripts/behavior_snapshot.py record OUT.json
    PYTHONPATH=<repo>/core python3 scripts/behavior_snapshot.py compare BASE.json NEW.json [--atol X]

`record` builds the planner for three cases (kitchen, bench, mebot) and writes
what it computes: the robot's names, limits, composed world, FK, state checks
and a few plans, floats at full precision. `compare` prints IDENTICAL (exit 0)
or the first 20 differences as `case/field[index]: base -> new` (exit 1).

It uses only the public planner API, so it runs unchanged on the code before
and after the refactor; the two shims below absorb the one signature that
changes (prepare_urdf) and are the only version-dependent lines. Needs a GPU.
"""

import argparse
import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# case -> (candidate planner configs, first existing wins; URDF), repo-relative
CASES = {
    "kitchen": (["core/rammp_curobo/configs/gen3.yaml"], "core/tests/data/arm_level.urdf"),
    "bench": (
        ["core/rammp_curobo/configs/gen3_real.yaml", "core/tests/data/dojo/planner.yaml"],
        "core/tests/data/arm_level.urdf",
    ),
    "mebot": (["core/tests/data/mebot/planner.yaml"], "core/tests/data/arm_tilt30.urdf"),
}
N_FK, N_CHECK, N_PLAN = 50, 200, 8


def _plain(x):
    """numpy -> lists and Python scalars, so json stores full precision."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _urdf_text(config, robot_name_or_path, cfg_dir, xml):
    """prepare_urdf(xml) before the robot file feeds it, prepare_urdf(xml, spec) after."""
    from rammp_curobo.config import resolve_config
    from rammp_curobo.urdf_model import prepare_urdf

    if len(inspect.signature(prepare_urdf).parameters) == 1:
        return prepare_urdf(xml)
    from rammp_curobo.robot_config import load_robot_spec

    return prepare_urdf(xml, load_robot_spec(resolve_config(robot_name_or_path, relative_to=cfg_dir)))


def _guard(fn):
    """The call's result, or its exception as a result: a raise is behaviour."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"raised": type(e).__name__, "message": str(e)}


def _seeded(plan):
    """Run a plan call after reseeding torch: cuRobo's solver seeds draw from
    torch's generator, so without this two runs plan different trajectories."""
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    return plan()


def _result(r):
    out = {"success": bool(r.success), "status": r.status, "error": r.error}
    t = r.joint_traj
    if t is not None:
        out.update(
            joint_names=list(t.joint_names),
            positions=_plain(t.positions),
            velocities=_plain(t.velocities),
            dt=float(t.dt),
        )
    return out


def _valid_states(planner, rng, lim, count, max_draws=2000):
    states = []
    for _ in range(max_draws):
        q = rng.uniform(lim[0], lim[1])
        if planner.check_state_valid(q)[0]:
            states.append(q)
            if len(states) == count:
                break
    return states


def record_case(name):
    from rammp_curobo.config import load_planner_config, resolve_config
    from rammp_curobo.planner import CuRoboPlanner
    from rammp_curobo.urdf_model import write_urdf

    cands, urdf = CASES[name]
    config = next(REPO / c for c in cands if (REPO / c).is_file())
    cfg, cfg_dir = load_planner_config(str(config))
    xml = (REPO / urdf).read_text()
    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = write_urdf(_urdf_text(config, cfg["robot"], cfg_dir, xml), tmp)
        planner = CuRoboPlanner.from_config(str(config), urdf_path=str(urdf_path))
    world = str(resolve_config(cfg["world"], relative_to=cfg_dir)) if cfg.get("world") else None

    lim = planner.joint_limits()
    rec = {
        "config": str(config.relative_to(REPO)),
        "joint_names": list(planner.joint_names),
        "base_frame": planner.base_frame,
        "ee_link": planner.ee_link,
        "retract_pose": _plain(planner.retract_pose),
        "joint_limits": _plain(lim),
        "world_summary": planner.world_summary(),
    }
    cuboids, sources = planner._compose(planner._scene, allow_empty=True)
    rec["composed_boxes"] = [
        {"name": k, "source": sources[k], "pose": _plain(cuboids[k]["pose"]), "dims": _plain(cuboids[k]["dims"])}
        for k in sorted(cuboids)
    ]
    pos = lim["position"]
    rng = np.random.default_rng(1)
    rec["fk"] = [
        _guard(lambda q=rng.uniform(pos[0], pos[1]): _plain(planner.fk(q))) for _ in range(N_FK)
    ]
    rng = np.random.default_rng(2)
    states = [rng.uniform(pos[0], pos[1]) for _ in range(N_CHECK)]
    checks = [_guard(lambda q=q: _plain(planner.check_state_valid(q))) for q in states]
    rec["check_state_valid"] = checks
    start = next((q for q, c in zip(states, checks) if isinstance(c, list) and c[0]), None)
    rec["plan_start"] = _plain(start)
    if start is None:
        rec["plan_to_joints"] = rec["plan_to_pose"] = "no valid start state"
    else:
        goals = _valid_states(planner, np.random.default_rng(3), pos, N_PLAN)
        rec["plan_to_joints"] = [
            _guard(lambda g=g: _result(_seeded(lambda: planner.plan_to_joints(g, start)))) for g in goals
        ]
        poses = [
            planner.fk(q) for q in _valid_states(planner, np.random.default_rng(4), pos, N_PLAN)
        ]
        rec["plan_to_pose"] = [
            _guard(lambda p=p: _result(_seeded(lambda: planner.plan_to_pose(p[0], p[1], start)))) for p in poses
        ]
    if world:
        rec["update_world"] = _guard(lambda: (planner.update_world(world), "ok")[1])
        rec["world_summary_after_update"] = planner.world_summary()
    return rec


def record(out):
    snap = {}
    for name in CASES:
        print("recording %s ..." % name, flush=True)
        snap[name] = _guard(lambda: record_case(name))
    Path(out).write_text(json.dumps(snap, indent=1))
    print("wrote", out)


def _diffs(path, a, b, atol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield "%s/%s: %s -> %s" % (path, k, a.get(k, "<absent>"), b.get(k, "<absent>"))
            else:
                yield from _diffs("%s/%s" % (path, k) if path else k, a[k], b[k], atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield "%s: length %d -> %d" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _diffs("%s[%d]" % (path, i), x, y, atol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        same = a == b or (a != a and b != b) or abs(a - b) <= atol
        if not same:
            yield "%s: %r -> %r" % (path, a, b)
    elif a != b or type(a) is not type(b):
        yield "%s: %r -> %r" % (path, a, b)


def compare(base, new, atol):
    a, b = json.loads(Path(base).read_text()), json.loads(Path(new).read_text())
    found = []
    for d in _diffs("", a, b, atol):
        found.append(d)
        if len(found) == 20:
            break
    print("\n".join(found) if found else "IDENTICAL")
    return 1 if found else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record").add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.out)
        return 0
    return compare(args.base, args.new, args.atol)


if __name__ == "__main__":
    sys.exit(main())
