# Real-arm bring-up runbook (Kinova Gen3 @ 192.168.1.10)

Follow this in order. **A human holds the physical e-stop from the moment
the kortex bringup starts until the last motion ends. No exceptions.**

The execution code path is byte-identical to the sim-verified one — same
controller names, same action, same gates. What changes on hardware is the
world model, the timing source, and the consequences.

## 0. Coordination — who has the arm?

Mutually exclusive stacks on this bench can claim the Gen3:

| stack | how it talks to the arm |
|---|---|
| **this repo via kinova_arm_ros2** (`kinova_arm_node`) | kinova-gen3-driver 1 kHz RT core, /execute_joint_trajectory |
| ros2_kortex bringup (RAMMP-Kinova ws) | ros2_control @ 1 kHz, joint_trajectory_controller — RETIRED for this repo 2026-08-14 |
| Demo-Software `arm_driver` | Kortex Python API, high-level servoing |

Only one may run. Before starting, confirm with the team nobody else is
mid-session, and check locally:

```bash
pgrep -af "arm_driver|teleop_socket|trajectory_run|kortex|kinova_arm_node" ; ls /tmp/kinova.lock 2>/dev/null
```

Note: `arm_driver` persists joint speed soft-limits (MEDIUM: 25 deg/s) and a
10-degree following-error threshold on the arm's controller. If a previous
`arm_driver` session ran, those limits still apply — harmless for our slow
first runs (an extra speed cap on top of ours), but know they're there.

## 1. Measure the bench, edit the world

(The bench: see `Dojo_pic_1.jpg` / `Dojo_pic_2.jpg` in this directory —
the arm is bolted nearly flush to a caster-mounted table, e-stop clamped
at the front edge.)

Edit `core/rammp_curobo/configs/world_real_bench.yaml` (or a copy):
measure from the arm's base_link origin (+x forward, +z up) and set at
minimum the real table top height/extents plus anything within reach.
**Err tall on the table** — modeled-too-tall costs workspace, never safety.
The sim kitchen world does NOT exist on the bench; never plan against it
on hardware (the node warns if you try).

## 2. Network + bringup

```bash
ping -c 2 192.168.1.10        # arm reachable?

source /opt/ros/humble/setup.zsh
source /tmp/kinova-ros2-ws/install/setup.zsh        # the driver workspace
source ~/RAMMP-CuRobo/install/setup.zsh

# terminal 1 — the arm driver (kinova_arm_ros2; this repo deliberately
# does not launch it — the planner never owns the arm). Run from the core
# checkout so models/ resolves:
cd /tmp/kinova-ros2-ws/src/kinova-gen3-driver
ros2 run kinova_arm_ros2 kinova_arm_node --ip 192.168.1.10 \
    --urdf models/gen3_7dof_2f85.urdf
```

This must be the ONLY arm stack running. Real-arm gotchas (from the
driver's README + docs/on-robot-runbook.md in its repo — read that
runbook too, it is the driver's own attended procedure):
- The node must be a KORTEX-linked build (`-DKINOVA_ENABLE_KORTEX=ON`);
  a sim-only build errors out in real mode. Sanity check: the binary is
  ~10 MB (sim-only ~1.5 MB).
- The workspace lives in **/tmp** (labmate's rsync dev loop; abra has no
  GitHub key) — a reboot wipes it; re-sync/build before hardware days.
- No launch files, no controller_manager, no fault_controller — the node
  IS the whole arm stack, and it enters low-level servoing itself.

Verify before going on:

```bash
ros2 action list | grep execute_joint_trajectory
ros2 topic hz --qos-reliability best_effort /joint_states   # ~100 Hz;
                                # plain `ros2 topic hz` shows NOTHING
                                # (best-effort publisher)
```

## 3. Planner node — dry-run first

```bash
source ~/RAMMP-CuRobo/install/setup.zsh    # on top of the two above
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
```

No `execute:=true` yet: this node plans but refuses motion. Sanity-check a
dry-run plan of a tiny move from wherever the arm is:

```bash
python3 examples/plan_and_execute.py --joints-relative 0.15 0 0 0 0 0 0
```

Read the excursion table. It should show ~0.15 rad on joint_1 and ~0
elsewhere (`--joints-relative` + native joint-space planning = the motion
you asked for, nothing else). If the plan fails or looks wrong, stop here.

## 4. First motion: small joint move near home, then the abort drill

Restart the node with execution armed, at a crawl:

```bash
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml execute:=true
python3 examples/plan_and_execute.py --joints-relative 0.15 0 0 0 0 0 0 \
    --execute --speed-scale 0.15
```

Type `yes` only when the e-stop hand is ready. The move is a slow base-yaw
sweep away from the table.

**Abort drill (required before anything else):** run the reverse move and
press Ctrl+C mid-motion:

```bash
python3 examples/plan_and_execute.py --joints-relative -0.15 0 0 0 0 0 0 \
    --execute --speed-scale 0.15
# Ctrl+C while it moves -> goal cancel -> controller stops and holds
```

Confirm the arm freezes and holds. This is the software abort you'll reach
for before the e-stop; prove it works while the motion is trivial.

## 5. Only after 3 & 4 are clean

- Repeat with other single-joint deltas; then 25% speed (`--speed-scale 0.25`).
- Keep `world_real_bench.yaml` matching reality (step 1) — obstacles are
  measured and edited by hand, or pushed live from another module via
  `/rammp_curobo/set_world`.
- Cartesian goals (`--pos ... --quat ...`) only AFTER the world file has
  been validated against reality, and never near surfaces on the first day.
- **Speed ceiling with this driver:** its position mode rate-limits
  commanded references to 0.5 rad/s (deliberately conservative until it
  has hardware data). cuRobo full-speed plans peak ~1.39 rad/s, so keep
  `speed_scale` ≤ ~0.35 — above that the arm lags its timestamps, the
  driver still "succeeds" on its timer, and our arrival check fails the
  run. Full-tilt tours return when the driver's `max_ref_speed` is raised.
- Gripper control is NOT available through this driver yet (the kortex-era
  `/rammp_curobo/open_gripper` services were removed with it).

## Faults / recovery

- Software abort: Ctrl+C on the example (goal cancel), or
  `ros2 action send_goal` cancel — the driver's Supervisor resets its
  trajectory executor and the arm holds its position.
- **No-motion "success"** (goals complete, arm doesn't move): the driver
  completes on its TIMER and has no goal-tolerance check yet, so our
  executor's wrap-aware arrival check is what catches this (verified
  against the static sim). Recovery: restart `kinova_arm_node` — it
  re-enters low-level servoing on startup (there is no fault_controller
  or controller_manager in this stack). If it repeats, e-stop,
  power-cycle the arm, restart the node.
- **PATH_TOLERANCE_VIOLATED (-4)** mid-motion: the driver's divergence
  guard tripped — physical contact/blockage, or the plan out-ran the
  0.5 rad/s reference cap. Our executor arms the guard only on the
  bounded joints (joint_2/4/6) and only under the cap, because the
  driver's guard is not wrap-aware (raw |q_meas − q_desired| explodes
  near ±pi, and joint_3 lives at +pi in the home family).
- The kortex-era fault taxonomy (servoing drop at goal transitions,
  reset_fault + JTC bounce, block_write wedge) applies to the RETIRED
  ros2_kortex stack only — see git history (pre-2026-08-14) if that
  stack is ever revived.
- After any e-stop or fault, RE-RUN the dry-run step before arming again
  (the arm may have been moved by hand; stale plans are refused, but check
  the world still matches reality too).
