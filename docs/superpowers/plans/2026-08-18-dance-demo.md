# Dance Demo (easter egg) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ros2 run rammp_curobo_ros dance_demo --execute` makes the arm
dance — bobbing up/down, swaying left/right, random circles, wrist
twists, nods, shimmies — as randomized rounds of fluid motion, with every
safety gate of the existing stack intact.

**Architecture:** A pure `choreograph()` function turns an RNG into a
move list (all fingertip waypoints confined to the bench-proven safe
box). `dance_demo.py` reuses tour_demo's machinery wholesale — the
`TourDemo` action client, chained pre-planning via `start_joints`,
`merge_trajectories` into ONE controller goal per round (the
no-motion-fault lesson), the homing gate, dry-run default, typed
confirmation, Ctrl-C = cancel + hold. Wrist stays flat (Rz(bearing) ⊗
home attitude) except deliberate twist/nod moves done as joint-space
segments on j7/j6.

**Tech Stack:** Python 3.10, existing rammp_curobo_ros actions; no new
dependencies, no planner/core changes.

**Spec:** this header (bounded demo; design approved in chat 2026-08-18).

## Global Constraints

- Execution safety unchanged: `--execute` flag + typed `dance` + every
  ExecuteTrajectory gate; default speed 0.4, clamp [0.1, 1.0]; homing
  gate identical to tour_demo; NEVER run by an agent — the human runs it.
- Cartesian waypoints hard-clamped to the proven volume:
  x ∈ [0.35, 0.60], y ∈ [-0.30, 0.30], z ∈ [0.18, 0.60] (today's
  IK-verified calibration box).
- Joint flourishes small: |Δj7| ≤ 0.7 rad, |Δj6| ≤ 0.4 rad, always
  emitted as a there-and-back pair so the chain returns to a flat wrist.
- One merged trajectory per ROUND (no mid-round goal transitions).
- Style: Ruff defaults; tests offline (`-p no:anyio`). Commit per task,
  **NO git push** (user instruction 2026-08-18).

---

### Task 1: choreography generator (pure, TDD)

**Files:**
- Create: `rammp_curobo_ros/rammp_curobo_ros/dance_demo.py` (generator part)
- Test: `rammp_curobo_ros/test/test_dance_moves.py`

**Interfaces:**
- Produces: `choreograph(rng, n_moves=8, center=(0.45, 0.0, 0.40)) ->
  list[tuple]` where each element is `("pose", [x, y, z])` or
  `("twist", dj7: float)` or `("nod", dj6: float)`; twist/nod always
  appear as +d followed by -d; every pose is inside SAFE_BOX; the list
  ends with a `("pose", center)` settle move. Constants `SAFE_BOX`,
  `DANCE_CENTER` exported for tests and main.

- [ ] **Step 1: Write the failing tests**

`rammp_curobo_ros/test/test_dance_moves.py`:

```python
"""The dance choreography generator: pure, seeded, safe-boxed."""

import random

from rammp_curobo_ros.dance_demo import SAFE_BOX, choreograph


def _poses(moves):
    return [m[1] for m in moves if m[0] == "pose"]


def test_seeded_choreography_is_repeatable():
    a = choreograph(random.Random(7), n_moves=8)
    b = choreograph(random.Random(7), n_moves=8)
    assert a == b
    c = choreograph(random.Random(8), n_moves=8)
    assert c != a


def test_every_pose_stays_in_the_safe_box():
    for seed in range(20):
        for p in _poses(choreograph(random.Random(seed), n_moves=10)):
            for v, (lo, hi) in zip(p, SAFE_BOX):
                assert lo - 1e-9 <= v <= hi + 1e-9


def test_flourishes_come_in_cancelling_pairs():
    for seed in range(20):
        moves = choreograph(random.Random(seed), n_moves=10)
        for i, m in enumerate(moves):
            if m[0] in ("twist", "nod") and m[1] > 0:
                assert moves[i + 1] == (m[0], -m[1])  # un-twist follows
        total = {"twist": 0.0, "nod": 0.0}
        for kind, val in moves:
            if kind in total:
                total[kind] += val
        assert abs(total["twist"]) < 1e-9 and abs(total["nod"]) < 1e-9


def test_flourish_amplitudes_are_bounded():
    for seed in range(20):
        for kind, val in choreograph(random.Random(seed), n_moves=10):
            if kind == "twist":
                assert abs(val) <= 0.7
            if kind == "nod":
                assert abs(val) <= 0.4


def test_ends_settled_at_center():
    moves = choreograph(random.Random(3), n_moves=6)
    assert moves[-1][0] == "pose"


def test_circles_produce_multiple_round_waypoints():
    # with enough moves and seeds, at least one circle occurs and shows
    # up as >= 5 consecutive poses at (roughly) constant distance from
    # their own centroid
    import numpy as np

    found = False
    for seed in range(30):
        poses = _poses(choreograph(random.Random(seed), n_moves=12))
        for i in range(len(poses) - 5):
            arc = np.array(poses[i : i + 6])
            c = arc.mean(axis=0)
            r = np.linalg.norm(arc - c, axis=1)
            if r.std() < 0.01 and r.mean() > 0.04:
                found = True
    assert found
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd ~/RAMMP-CuRobo && source install/setup.zsh && \
python3 -m pytest rammp_curobo_ros/test/test_dance_moves.py -q -p no:anyio
```

Expected: ImportError (`dance_demo` missing).

- [ ] **Step 3: Implement the generator (top of dance_demo.py)**

```python
#!/usr/bin/env python3
"""Dance easter egg: the arm bobs, sways, circles, twists — safely.

Randomized rounds of goofy-but-gated motion: every waypoint lives in the
bench-proven safe box, the wrist stays flat except deliberate twist/nod
flourishes, and each round pre-plans as chained segments merged into ONE
trajectory (zero controller goal transitions — the no-motion-fault
lesson), executed through every ExecuteTrajectory gate.

    ros2 run rammp_curobo_ros dance_demo               # dry-run: plan + print
    ros2 run rammp_curobo_ros dance_demo --execute     # needs typed 'dance'

SAFETY: clear workspace, human on the physical e-stop, planner launched
with execute:=true. Ctrl+C cancels; the arm holds position.
"""

import argparse
import math
import random
import sys
import time

import rclpy

from rammp_curobo.geometry import ang_diff, yaw_about_world_z
from rammp_curobo_ros.tour_demo import (
    HOME,
    HOME_QUAT_XYZW,
    TourDemo,
    merge_trajectories,
    traj_end,
    traj_time,
)

# x / y / z bounds — the IK-verified volume from the calibration session
SAFE_BOX = ((0.35, 0.60), (-0.30, 0.30), (0.18, 0.60))
DANCE_CENTER = (0.45, 0.0, 0.40)


def _clamp(p):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(p, SAFE_BOX)]


def choreograph(rng, n_moves=8, center=DANCE_CENTER):
    """A random dance as ('pose', [xyz]) / ('twist', dj7) / ('nod', dj6).

    Flourishes always emit +d then -d so the chain lands back on a flat
    wrist; every pose is clamped into SAFE_BOX; the dance settles at
    `center` at the end.
    """
    cx, cy, cz = center
    moves = []
    for _ in range(n_moves):
        kind = rng.choice(["bob", "sway", "circle", "twist", "nod", "shimmy"])
        if kind == "bob":
            dz = rng.uniform(0.06, 0.12)
            moves += [("pose", _clamp([cx, cy, cz - dz])),
                      ("pose", _clamp([cx, cy, cz + dz]))]
        elif kind == "sway":
            dy = rng.uniform(0.10, 0.20)
            moves += [("pose", _clamp([cx, cy - dy, cz])),
                      ("pose", _clamp([cx, cy + dy, cz]))]
        elif kind == "circle":
            r = rng.uniform(0.05, 0.09)
            direction = rng.choice([-1.0, 1.0])
            for k in range(6):
                a = direction * 2.0 * math.pi * k / 6.0
                moves.append(
                    ("pose",
                     _clamp([cx, cy + r * math.sin(a), cz + r * math.cos(a)]))
                )
        elif kind == "twist":
            d = rng.uniform(0.4, 0.7)
            moves += [("twist", d), ("twist", -d)]
        elif kind == "nod":
            d = rng.uniform(0.25, 0.4)
            moves += [("nod", d), ("nod", -d)]
        elif kind == "shimmy":
            d = rng.uniform(0.03, 0.05)
            for s in (-1, 1, -1):
                moves.append(("pose", _clamp([cx, cy + s * d, cz])))
    moves.append(("pose", list(center)))
    return moves
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest rammp_curobo_ros/test/test_dance_moves.py -q -p no:anyio
```

Expected: 6 passed.

- [ ] **Step 5: Commit (no push)**

```bash
git add rammp_curobo_ros/rammp_curobo_ros/dance_demo.py rammp_curobo_ros/test/test_dance_moves.py && \
git commit -m "dance demo: seeded safe-box choreography generator"
```

---

### Task 2: the dance runner

**Files:**
- Modify: `rammp_curobo_ros/rammp_curobo_ros/dance_demo.py` (append main)
- Modify: `rammp_curobo_ros/setup.py` (console script `dance_demo`)

**Interfaces:**
- Consumes: `TourDemo` client (`plan_pose_from`, `plan_home_from`, `run`,
  `joints`), `merge_trajectories`, `traj_end`, `traj_time` — all from
  tour_demo; `PlanToJoints` for twist/nod segments.
- Produces: `ros2 run rammp_curobo_ros dance_demo [--execute] [--speed]
  [--rounds] [--moves] [--seed]`.

- [ ] **Step 1: Append the runner**

```python
JOINT_IDX = {"twist": 6, "nod": 5}  # j7 / j6, 0-based in the 7-vector


def plan_move(demo, move, start):
    """One choreography move -> a chained plan (or None)."""
    kind, val = move
    if kind == "pose":
        quat = list(
            yaw_about_world_z(HOME_QUAT_XYZW, math.atan2(val[1], val[0]))
        )
        return demo.plan_pose_from(val, quat, start)
    from rammp_curobo_interfaces.action import PlanToJoints

    q = [float(v) for v in start]
    q[JOINT_IDX[kind]] += float(val)
    g = PlanToJoints.Goal(target_joints=q)
    g.start_joints = [float(v) for v in start]
    return demo._call(demo.plan_joints, g)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--execute", action="store_true", help="allow motion")
    ap.add_argument("--speed", type=float, default=0.4,
                    help="execution scale (dance default 0.4; 1.0 = rated)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="dance rounds (each is one merged trajectory)")
    ap.add_argument("--moves", type=int, default=8, help="moves per round")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    scale = min(max(args.speed, 0.1), 1.0)
    rng = random.Random(args.seed)

    rclpy.init()
    node = rclpy.create_node("rammp_curobo_dance")
    demo = TourDemo(node)

    q_now = demo.joints()
    if max(abs(ang_diff(a, b)) for a, b in zip(q_now, HOME)) > 0.1:
        if not args.execute:
            sys.exit("arm is not at home — rerun with --execute to home it")
        if input("arm is away from home — type 'go' to home it at 25% "
                 "(hand on e-stop): ").strip() != "go":
            sys.exit("aborted — nothing moved")
        plan = demo.plan_home_from(None)
        if plan is None or not plan.success:
            sys.exit("cannot plan home")
        if not demo.run(plan.trajectory, 0.25):
            sys.exit("homing failed — see planner log")
        print("homed.")

    print("choreographing %d round(s) of %d moves..." % (args.rounds, args.moves))
    rounds = []
    start = None  # live state for round 1 segment 1
    for r in range(args.rounds):
        plans, skipped = [], 0
        for move in choreograph(rng, n_moves=args.moves):
            plan = plan_move(demo, move, start if start else demo.joints())
            if plan is None or not plan.success:
                skipped += 1
                continue
            plans.append(plan)
            start = traj_end(plan)
        home_plan = demo.plan_home_from(start)
        if home_plan is None or not home_plan.success:
            sys.exit("cannot plan the return home for round %d" % (r + 1))
        plans.append(home_plan)
        start = traj_end(home_plan)
        rounds.append(merge_trajectories(plans))
        total = sum(traj_time(p, scale) for p in plans)
        print("  round %d: %d segments (%d unplannable skipped), %.1f s "
              "at speed %.2f" % (r + 1, len(plans), skipped, total, scale))

    if not args.execute:
        print("dry-run complete — nothing moved (add --execute)")
        return

    print("\n*** DANCE TIME: workspace COMPLETELY CLEAR, human on the "
          "physical e-stop. Ctrl+C stops (arm holds). ***")
    if input("type 'dance' to start: ").strip() != "dance":
        print("aborted — nothing moved")
        return

    for r, merged in enumerate(rounds):
        print("round %d/%d (%d points)..." % (r + 1, len(rounds),
                                              len(merged.points)))
        ok = False
        for attempt in range(3):
            if demo.run(merged, scale):
                ok = True
                break
            moved = max(abs(ang_diff(a, b)) for a, b in
                        zip(demo.joints(), merged.points[0].positions))
            if moved > 0.05 or attempt == 2:
                sys.exit("dance stopped (arm %.3f rad from round start) — "
                         "arm holds; see the planner log" % moved)
            print("  no-motion fault at start, recovered — retrying "
                  "(%d/2)" % (attempt + 1))
            time.sleep(3.0)
        if not ok:
            sys.exit("dance failed")
        time.sleep(0.5)
    print("\nDANCE COMPLETE — take a bow (the arm already did).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndance stopped — arm holds")
```

- [ ] **Step 2: Entry point**

In `rammp_curobo_ros/setup.py` `console_scripts`, after the tour line:

```python
            "dance_demo = rammp_curobo_ros.dance_demo:main",
```

- [ ] **Step 3: Tests + build + smoke**

```bash
python3 -m pytest rammp_curobo_ros/test -q -p no:anyio && \
source /opt/ros/humble/setup.zsh && \
colcon build --symlink-install --packages-select rammp_curobo_ros && \
source install/setup.zsh && ros2 run rammp_curobo_ros dance_demo --help | head -5
```

Expected: all tests pass; help text prints.

- [ ] **Step 4: Commit (no push)**

```bash
git add rammp_curobo_ros && git commit -m "dance demo: randomized gated dance rounds (easter egg)"
```

---

### Task 3: docs

**Files:**
- Modify: `README.md` (one line near the tour mention), `CLAUDE.md`
  (parts list mention: "+ dance_demo easter egg, same gates as tour")

- [ ] **Step 1: Add the lines, run ruff + mdformat if configured**
- [ ] **Step 2: Commit (no push)**

```bash
git add README.md CLAUDE.md && git commit -m "docs: dance_demo easter egg"
```

---

## Attended follow-up (the human runs it)

Dry-run first (`ros2 run rammp_curobo_ros dance_demo`), then
`--execute` at the default 0.4 speed with the workspace clear and a hand
on the e-stop. First hardware run: `--rounds 1 --moves 4 --speed 0.25`.
