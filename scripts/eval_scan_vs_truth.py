#!/usr/bin/env python3
"""Score a scanned world against the sim's ground truth (scene.yaml).

For every ground-truth body that the sweep should have seen (inside the
sweep's reach/bearing envelope and taller than the table band), find the
detected box that claims it and report the center error. Also flag
detections that match nothing (false positives — usually fusion noise).

    python3 scripts/eval_scan_vs_truth.py \
        ~/.ros/rammp_curobo/scanned_world.yaml \
        ~/RAMMP-Kinova/ros2_ws/src/curobo_planner/config/scene.yaml
"""

import math
import sys

import numpy as np

from rammp_curobo.scene import load_scene


def expected_bodies(scene, max_reach=1.05, min_top_z=0.06, bearing_deg=100.0):
    """Ground-truth bodies the +/-90 deg sweep should have seen."""
    out = []
    for o in list(scene.obstacles) + list(scene.objects):
        dims = o.bounding_dims() if hasattr(o, "bounding_dims") else o.dims
        c = np.asarray(o.position, dtype=float)
        half = np.asarray(dims, dtype=float) / 2.0
        closest_xy = np.maximum(np.abs(c[:2]) - half[:2], 0.0)
        if np.linalg.norm(closest_xy) > max_reach:
            continue
        if c[2] + half[2] < min_top_z:
            continue  # below the table band the scan deliberately drops
        if abs(math.degrees(math.atan2(c[1], c[0]))) > bearing_deg:
            continue  # behind the sweep arc
        if o.name == "pedestal":
            continue  # inside the self-filter / base cutout by design
        out.append((o.name, c, half))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    scanned = load_scene(sys.argv[1])
    truth = load_scene(sys.argv[2])
    dets = [
        (o.name, np.asarray(o.position), np.asarray(o.dims) / 2.0)
        for o in scanned.obstacles
        if o.name.startswith("det_")
    ]

    expected = expected_bodies(truth)
    hits, misses = [], []
    claimed = set()
    for name, c, half in expected:
        best, best_d = None, 1e9
        for dname, dc, dhalf in dets:
            # detection claims the body if the boxes overlap generously
            gap = np.abs(dc - c) - (dhalf + half + 0.12)
            if (gap < 0).all():
                d = float(np.linalg.norm(dc - c))
                if d < best_d:
                    best, best_d = dname, d
        if best is None:
            misses.append(name)
        else:
            hits.append((name, best, best_d))
            claimed.add(best)

    print("=== expected bodies: %d ===" % len(expected))
    for name, det, d in sorted(hits, key=lambda h: h[0]):
        print("  HIT  %-18s <- %-8s center offset %.3f m" % (name, det, d))
    for name in misses:
        print("  MISS %s" % name)
    ghosts = [d for d, _c, _h in dets if d not in claimed]
    print(
        "detections: %d | matched: %d | unmatched (extra/split): %d"
        % (len(dets), len(claimed), len(ghosts))
    )
    if ghosts:
        print("  extras: %s" % ", ".join(ghosts))
    recall = 1.0 - len(misses) / max(1, len(expected))
    mean_err = float(np.mean([d for _n, _b, d in hits])) if hits else float("nan")
    print("RECALL %.0f%% | mean center offset %.3f m" % (recall * 100, mean_err))
    if misses:
        sys.exit(1)


if __name__ == "__main__":
    main()
