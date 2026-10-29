"""Summary: measured motion time per case, stopped vs blended, per mode."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1])
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STOP, BLEND = "#2a78d6", "#eb6834"
CASES = ["square", "mixed_hold", "joint_sweep", "grab_like"]
PANELS = [("video10.jsonl", "1", "Position, speed 1.0"), ("s08.jsonl", "0.8", "Position, speed 0.8"), ("stiff.jsonl", "0.3", "STIFF impedance, speed 0.3")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
for ax, (f, tag, title) in zip(axes, PANELS):
    rows = {json.loads(l)["run"]: json.loads(l) for l in open(R / f)}
    x = np.arange(len(CASES))
    w = 0.38
    for k, (kind, color) in enumerate((("stop", STOP), ("blend", BLEND))):
        vals = [rows[f"{c}_{tag}_{kind}"]["motion_s"] for c in CASES]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label="stopped at each waypoint" if kind == "stop" else "blended", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    saved = [1 - rows[f"{c}_{tag}_blend"]["motion_s"] / rows[f"{c}_{tag}_stop"]["motion_s"] for c in CASES]
    top = max(rows[f"{c}_{tag}_stop"]["motion_s"] for c in CASES)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks(x, [f"{c.replace('_', ' ')}\n−{v * 100:.0f}%" for c, v in zip(CASES, saved)], fontsize=8.5, color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_ylim(0, top * 1.15)
axes[0].set_ylabel("measured motion time (s)", color=INK2, fontsize=9)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper right", ncol=2, labelcolor=INK)
fig.suptitle("Measured motion time, stopped vs blended (abra, 2026-10-29); under each case: time saved", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(R / "summary.png", dpi=130, facecolor=SURFACE)
