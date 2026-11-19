"""How far, and how repeatably, plan_to_pose swings the arm on the chair.

    run:    plan a fixed set of start -> goal-pose pairs `--reps` times each,
            under one planner condition, and write one JSON line per plan.
            Each rep jitters the start (`--start-noise`), as the arm's
            measured q never repeats exactly.
    replay: re-plan a planner node's `plan_log` (real requests, each in the
            box world it was planned in) under one condition.
    report: compare the conditions' JSONL files, per pair, against a baseline.

The pairs come from the MEBot sim's states (sim/mebot_sim/config/scenarios.yaml),
kept only if valid with every layer of the planner config on. A goal is the
tool POSE of another state, so IK chooses the goal configuration, as in use.

Per plan it records the swept volume (the union of the arm's collision spheres
over the trajectory, voxelised), the furthest any sphere strays from the
straight line between where it starts and ends, the tool's path length, and
the final joints (whose clusters across reps are the "families" a pair lands in).
See bench/README.md for the image and the conditions.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
VOXEL_M = 0.02
FAMILY_TOL_RAD = 0.3  # final joints further apart than this (any joint) differ


def _pool():
    s = yaml.safe_load((ROOT / "sim/mebot_sim/config/scenarios.yaml").read_text())
    states = [s["start"]] + s["reaches"]["goals"]
    for group in ("person_pairs", "platform_pairs"):
        for p in s[group]["pairs"]:
            states += [p["start"], p["goal"]]
    return states


def _sweep_metrics(planner, positions):
    """Swept volume (m^3), max sphere excursion from its start-end chord (m),
    and the tool's path and straight-line lengths (m)."""
    torch = planner._torch
    q = torch.tensor(
        [planner._to_curobo_order(p) for p in positions],
        device=planner._tensor_args.device,
        dtype=planner._tensor_args.dtype,
    )
    st = planner._motion_gen.kinematics.get_state(q)
    sph = st.link_spheres_tensor.detach().cpu().numpy().astype(float)  # (T, n, 4)
    ee = st.ee_position.detach().cpu().numpy().astype(float)
    live = sph[0, :, 3] > 0
    sph = sph[:, live]
    c, r = sph[..., :3], sph[..., 3]

    # union of the spheres over the trajectory, voxelised
    vox = []
    flat_c, flat_r = c.reshape(-1, 3), r.reshape(-1)
    span = int(math.ceil(flat_r.max() / VOXEL_M))
    g = np.arange(-span, span + 1)
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    dist = np.linalg.norm(off * VOXEL_M, axis=1)
    base = np.round(flat_c / VOXEL_M).astype(np.int64)
    for rad in np.unique(np.round(flat_r, 3)):
        sel = np.isclose(np.round(flat_r, 3), rad)
        o = off[dist <= rad]
        vox.append((base[sel][:, None, :] + o[None]).reshape(-1, 3))
    vox = np.concatenate(vox)
    n_vox = np.unique(vox, axis=0).shape[0]

    # furthest any sphere centre gets from its own start-end chord
    a, b = c[0], c[-1]
    ab = b - a
    t = np.einsum("tnk,nk->tn", c - a, ab) / np.maximum((ab * ab).sum(-1), 1e-12)
    proj = a + np.clip(t, 0, 1)[..., None] * ab
    excursion = float(np.linalg.norm(c - proj, axis=-1).max())

    ee_path = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
    ee_straight = float(np.linalg.norm(ee[-1] - ee[0]))
    return n_vox * VOXEL_M**3, excursion, ee_path, ee_straight


def _traj_metrics(planner, start, positions):
    p = np.vstack([np.array(start, dtype=float)[None], np.asarray(positions)])
    vol, exc, eep, ees = _sweep_metrics(planner, p)
    return dict(
        final=[float(v) for v in p[-1]],
        jdist=float(np.linalg.norm(p[-1] - p[0])),
        jpath=float(np.abs(np.diff(p, axis=0)).sum()),
        sweep_m3=vol,
        excursion_m=exc,
        ee_path_m=eep,
        ee_straight_m=ees,
    )


def _overrides(cfg, sets):
    out = {}
    for kv in sets:
        k, v = kv.split("=", 1)
        out[k] = json.loads(v)
        cfg["planner"][k] = out[k]
    return out


def _header(f, **kw):
    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    f.write(json.dumps(dict(header=True, git=sha, **kw)) + "\n")


def _planner(cfg, cfg_dir, urdf_text):
    """A planner on `urdf_text`, prepared as the node prepares the model."""
    import tempfile

    from rammp_curobo.contract import check_robot_dir
    from rammp_curobo.planner import CuRoboPlanner
    from rammp_curobo.urdf_model import write_urdf

    with tempfile.TemporaryDirectory() as tmp:
        path = write_urdf(check_robot_dir(cfg, cfg_dir, urdf_text), tmp)
        return CuRoboPlanner(cfg, config_dir=cfg_dir, urdf_path=path)


def replay(args):
    """Re-plan a node's plan_log (its pose requests, each in the box world it
    was planned in) under one condition. The logged world is the padded,
    composed one cuRobo was given, so the planner here pads nothing, lists no
    layers, and gets each request's boxes pushed straight to cuRobo."""
    from rammp_curobo.config import load_planner_config
    from rammp_curobo.world import to_world_config

    lines = [json.loads(x) for x in Path(args.log).read_text().splitlines() if x]
    heads = [x for x in lines if x.get("header")]
    if not heads:
        sys.exit("%s has no header line (the node writes one at start-up)" % args.log)
    if len({h["urdf"] for h in heads}) > 1:
        sys.exit(
            "%s holds sessions on different models: split it at its header lines"
            % args.log
        )
    head = heads[0]
    poses = [x for x in lines if x.get("kind") == "pose"]
    # planned without a hold or via here, so not comparable with the live plan
    reqs = [
        x for x in poses if not (x["goal"].get("constraint") or x["goal"].get("via"))
    ]
    print(
        "replaying %d pose requests; skipped %d with a hold or via, %d joint requests"
        % (
            len(reqs),
            len(poses) - len(reqs),
            sum(x.get("kind") == "joints" for x in lines),
        ),
        flush=True,
    )
    cfg, cfg_dir = load_planner_config(args.config)
    cfg["layers"] = []
    cfg["world"] = "example:world_empty.yaml"
    cfg["planner"]["world_padding"] = 0.0
    overrides = _overrides(cfg, args.set)
    planner = _planner(cfg, cfg_dir, head["urdf"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        _header(
            f,
            name=args.name,
            set=overrides,
            config=str(args.config),
            log=str(args.log),
            pool=0,
            pairs=len(reqs),
            reps=args.reps,
        )
        for rep in range(args.reps):
            for pid, r in enumerate(reqs):
                boxes = {
                    b["name"]: {"dims": b["dims"], "pose": b["pose"]}
                    for b in r["world"]
                }
                if boxes:
                    planner._motion_gen.update_world(to_world_config(boxes))
                else:  # an empty-world session: update_world would keep the last
                    planner._motion_gen.clear_world_cache()
                res = planner.plan_to_pose(
                    r["goal"]["position"], r["goal"]["quat_xyzw"], start=r["start"]
                )
                rec = {
                    "pair": pid,
                    "rep": rep,
                    "success": bool(res.success),
                    "status": res.status,
                    "time_s": res.timing,
                    "live_success": r["success"],
                }
                if res.success:
                    rec.update(
                        _traj_metrics(planner, r["start"], res.joint_traj.positions)
                    )
                if rep == 0 and r["success"]:
                    rec["live"] = _traj_metrics(planner, r["start"], r["positions"])
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print("%s: rep %d/%d done" % (args.name, rep + 1, args.reps), flush=True)


def run(args):
    from rammp_curobo.config import load_planner_config

    cfg, cfg_dir = load_planner_config(args.config)
    overrides = _overrides(cfg, args.set)
    t0 = time.monotonic()
    planner = _planner(cfg, cfg_dir, Path(args.urdf).read_text())
    build_s = time.monotonic() - t0

    lo, hi = (np.array(v) for v in planner.joint_limits()["position"])
    m = planner.joint_limit_margin_rad
    pool = [
        q
        for q in _pool()
        if planner.check_state_valid(q)[0]
        and np.all(np.array(q) > lo + m)
        and np.all(np.array(q) < hi - m)
    ]
    pairs = [(i, j) for i in range(len(pool)) for j in range(len(pool)) if i != j]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.pairs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        _header(
            f,
            name=args.name,
            set=overrides,
            config=str(args.config),
            pool=len(pool),
            pairs=len(pairs),
            reps=args.reps,
            build_s=build_s,
            start_noise=args.start_noise,
        )
        for rep in range(args.reps):
            for pid, (i, j) in enumerate(pairs):
                start = pool[i]
                if args.start_noise:  # the arm's measured q is never the same twice
                    rng = np.random.default_rng([args.seed, rep, pid])
                    start = list(np.add(start, rng.normal(0, args.start_noise, 7)))
                pos, quat = planner.fk(pool[j])
                res = planner.plan_to_pose(pos, quat, start=start)
                rec = {
                    "pair": pid,
                    "i": i,
                    "j": j,
                    "rep": rep,
                    "success": bool(res.success),
                    "status": res.status,
                    "time_s": res.timing,
                }
                if res.success:
                    rec.update(_traj_metrics(planner, start, res.joint_traj.positions))
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print("%s: rep %d/%d done" % (args.name, rep + 1, args.reps), flush=True)


def _families(finals):
    reps = []
    for q in finals:
        if not any(np.max(np.abs(np.array(q) - r)) < FAMILY_TOL_RAD for r in reps):
            reps.append(np.array(q))
    return len(reps)


def report(args):
    runs = {}
    for path in args.files:
        lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x]
        head, recs = lines[0], lines[1:]
        runs[head["name"]] = (head, recs)
        live = [
            dict(r["live"], pair=r["pair"], success=True, time_s=0.0)
            for r in recs
            if "live" in r
        ]
        if live and "live" not in runs:  # replay: the plans as executed
            runs["live"] = (dict(head, name="live", set={}), live)
    base = runs[args.baseline][1]

    def per_pair(recs, key):
        d = {}
        for r in recs:
            if r["success"]:
                d.setdefault(r["pair"], []).append(r[key])
        return {k: float(np.median(v)) for k, v in d.items()}

    base_sweep = per_pair(base, "sweep_m3")
    rows = []
    for name, (head, recs) in runs.items():
        ok = [r for r in recs if r["success"]]
        sweep = per_pair(recs, "sweep_m3")
        ratio = [sweep[p] / base_sweep[p] for p in sweep if p in base_sweep]
        fams = {}
        for r in ok:
            fams.setdefault(r["pair"], []).append(r["final"])
        nfam = [_families(v) for v in fams.values()]
        spread = [
            np.median(
                [
                    np.linalg.norm(np.subtract(a, b))
                    for k, a in enumerate(v)
                    for b in v[:k]
                ]
            )
            for v in fams.values()
            if len(v) > 1
        ]
        times = [r["time_s"] for r in recs]
        rows.append(
            (
                name,
                json.dumps(head["set"]) if head["set"] else "-",
                "%d/%d" % (len(ok), len(recs)),
                "%.2f" % np.mean(nfam),
                "%d" % sum(n > 1 for n in nfam),
                "%.2f" % np.median(spread),
                "%.3f" % np.median([r["sweep_m3"] for r in ok]),
                "%.3f" % np.percentile([r["sweep_m3"] for r in ok], 95),
                "%.2f" % np.median(ratio),
                "%.2f" % np.percentile(ratio, 95),
                "%.2f" % np.median([r["excursion_m"] for r in ok]),
                "%.2f" % np.percentile([r["excursion_m"] for r in ok], 95),
                "%.2f" % np.median([r["jdist"] for r in ok]),
                "%.2f/%.2f" % (np.median(times), np.percentile(times, 95)),
            )
        )
    cols = (
        "condition",
        "set",
        "success",
        "families/pair",
        "pairs>1fam",
        "final spread rad p50",
        "sweep m3 p50",
        "p95",
        "vs base p50",
        "p95",
        "excursion m p50",
        "p95",
        "|dq| p50",
        "time s p50/p95",
    )
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    head = runs[args.baseline][0]
    print(
        "\n%d pairs x %d reps, pool %d states, git %s; families: final joints "
        "within %.1f rad on every joint; sweep: %g m voxels"
        % (
            head["pairs"],
            head["reps"],
            head["pool"],
            head["git"],
            FAMILY_TOL_RAD,
            VOXEL_M,
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=str(ROOT / "core/tests/data/mebot/planner.yaml"))
    r.add_argument("--urdf", default=str(ROOT / "core/tests/data/arm_tilt30.urdf"))
    r.add_argument("--set", action="append", default=[], help="planner key=JSON")
    r.add_argument("--pairs", type=int, default=80)
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--start-noise", type=float, default=0.003, help="rad, per joint")
    y = sub.add_parser("replay")
    y.add_argument("log", help="a planner node's plan_log JSONL")
    y.add_argument("--config", required=True, help="the robot's planner.yaml")
    y.add_argument("--name", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("--set", action="append", default=[], help="planner key=JSON")
    y.add_argument("--reps", type=int, default=5)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--baseline", required=True)
    args = ap.parse_args()
    {"run": run, "replay": replay, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
