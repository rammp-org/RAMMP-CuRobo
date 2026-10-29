"""Tracking analysis of track.py's npz runs, against the PLAN: the driver's
feedback carries only `actual` (desired is empty), so the reference is the
planned trajectory, started at the goal's acceptance plus a fitted offset.

    uv run --with numpy --with scipy --with matplotlib analyze.py <run dir> <urdf> <figure dir>

One JSON row per run on stdout; a figure per stop/blend pair.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fk import FK

CONTINUOUS = [0, 2, 4, 6]


def wrap(e):
    """J1/J3/J5/J7 spin freely: their errors the short way."""
    e = np.array(e, dtype=float)
    e[..., CONTINUOUS] = (e[..., CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return e


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    d["summary"] = json.loads(str(d["summary"]))
    return d


def planned_at(d, tau):
    """The plan's joints at `tau` s after its start (before: its first point's start, after: its end)."""
    t = np.r_[0.0, d["t"]]  # point k is at (k + 1) dt; at 0 the arm is where the plan starts
    q = np.vstack([d["q"][:1], d["q"]])
    return np.stack([np.interp(tau, t, q[:, j]) for j in range(7)], axis=1)


def fit_start(d):
    """The offset (s, after acceptance) that best aligns the measured joints with the plan."""
    tau0 = d["fb_t"] - d["times"][1]
    best, best_e = 0.0, np.inf
    for lag in np.arange(-0.05, 0.3, 0.001):
        e = np.sqrt((wrap(planned_at(d, tau0 - lag) - d["fb_act"]) ** 2).mean())
        if e < best_e:
            best, best_e = lag, e
    return best


def speed(t, x, window=0.02):
    """|dx/dt| (N,), from x (N, k) sampled at t, over a `window` s central difference."""
    out = np.zeros(len(t))
    lo = np.searchsorted(t, t - window / 2)
    hi = np.clip(np.searchsorted(t, t + window / 2), 0, len(t) - 1)
    ok = hi > lo
    out[ok] = np.linalg.norm(x[hi[ok]] - x[lo[ok]], axis=1) / (t[hi[ok]] - t[lo[ok]])
    return out


def path_deviation(actual, planned):
    """Per actual point, the distance (m) to the nearest point of the planned path (densified)."""
    seg = np.linspace(0, 1, 8)[:, None, None]
    dense = (planned[:-1][None] * (1 - seg) + planned[1:][None] * seg).reshape(-1, 3)
    out = np.empty(len(actual))
    for i in range(0, len(actual), 512):
        a = actual[i : i + 512]
        out[i : i + 512] = np.sqrt(((a[:, None, :] - dense[None]) ** 2).sum(-1)).min(axis=1)
    return out


FIXED_LAG_S = 0.012  # the start lag position mode showed (10-15 ms): impedance lag is error, not offset


def analyze(d, fk):
    s = d["summary"]
    lag = FIXED_LAG_S
    tau = d["fb_t"] - d["times"][1] - lag
    ref = planned_at(d, tau)
    err = wrap(ref - d["fb_act"])
    tool_ref, _ = fk.tool(ref)
    tool_act, _ = fk.tool(d["fb_act"])
    terr = np.linalg.norm(tool_ref - tool_act, axis=1)
    tool_plan, _ = fk.tool(d["q"])
    dev = path_deviation(tool_act, tool_plan)
    v_act = speed(d["fb_t"], tool_act)
    v_ref = speed(d["fb_t"], tool_ref)
    moving = np.flatnonzero(v_act > 0.01)  # m/s
    # the motion: first to last instant the tool moves faster than 1 cm/s
    motion_s = float(d["fb_t"][moving[-1]] - d["fb_t"][moving[0]]) if moving.size else 0.0
    inside = slice(moving[0], moving[-1] + 1) if moving.size else slice(0, 0)
    dt = np.diff(d["fb_t"][inside], append=d["fb_t"][inside][-1:])
    still_s = float(dt[v_act[inside] < 0.01].sum()) if moving.size else 0.0
    final = wrap(d["q"][-1] - d["fb_act"][-1])
    return {
        "run": s["label"],
        "planned_s": round(float(d["t"][-1]), 2),
        "motion_s": round(motion_s, 2),
        "still_inside_s": round(still_s, 2),
        "peak_tool_m_s": round(float(v_ref.max()), 3),
        "joint_max_deg": round(float(np.degrees(np.abs(err).max())), 3),
        "joint_rms_deg": round(float(np.degrees(np.sqrt((err**2).mean()))), 3),
        "worst_joint": int(np.abs(err).max(axis=0).argmax()) + 1,
        "tool_max_mm": round(float(terr.max() * 1000), 2),
        "tool_rms_mm": round(float(np.sqrt((terr**2).mean()) * 1000), 2),
        "path_dev_max_mm": round(float(dev.max() * 1000), 2),
        "final_deg": round(float(np.degrees(np.abs(final).max())), 3),
        "joint_max_deg_each": np.degrees(np.abs(err).max(axis=0)).round(1).tolist(),
        "ran_s": round(float(tau[-1]), 2),
        "corner_cm": s.get("corner_cm"),
        "error_code": s["error_code"],
    }, (tau, v_ref, v_act, terr, dev)


def main():
    src, urdf, figs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    fk = FK(urdf.read_text())
    figs.mkdir(parents=True, exist_ok=True)
    runs = {p.stem: load(p) for p in sorted(src.glob("*.npz"))}
    out = {}
    for name, d in runs.items():
        row, series = analyze(d, fk)
        out[name] = (row, series)
        print(json.dumps(row))
    for name, (row, _) in out.items():
        if not name.endswith("_stop") or name[: -len("stop")] + "blend" not in out:
            continue
        case = name[: -len("_stop")]
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for label in ("stop", "blend"):
            r, (tau, v_ref, v_act, terr, dev) = out[f"{case}_{label}"]
            line = ax[0].plot(tau, v_act, lw=1.2, label=f"{label} measured ({r['motion_s']} s)")[0]
            ax[0].plot(tau, v_ref, "--", lw=0.8, color=line.get_color(), label=f"{label} planned")
            ax[1].plot(tau, terr * 1000, lw=1, color=line.get_color(), label=label)
            ax[2].plot(tau, dev * 1000, lw=1, color=line.get_color(), label=label)
        ax[0].set_ylabel("tool speed (m/s)")
        ax[1].set_ylabel("tool error vs plan (mm)")
        ax[2].set_ylabel("distance from\nplanned path (mm)")
        ax[2].set_xlabel("s from the plan's start")
        for a in ax:
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
        fig.suptitle(case)
        fig.tight_layout()
        fig.savefig(figs / f"{case}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
