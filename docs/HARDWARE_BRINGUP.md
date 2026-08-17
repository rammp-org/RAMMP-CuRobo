# Real-arm bring-up runbook (Kinova Gen3 @ 192.168.1.10)

Follow this in order. **A human holds the physical e-stop from the moment
the kortex bringup starts until the last motion ends. No exceptions.**

The execution code path is byte-identical to the sim-verified one — same
controller names, same action, same gates. What changes on hardware is the
world model, the timing source, and the consequences.

## 0. Coordination — who has the arm?

Three mutually exclusive stacks on this bench can claim the Gen3:

| stack | how it talks to the arm |
|---|---|
| **this repo via ros2_kortex** (RAMMP-Kinova ws) | ros2_control @ 1 kHz, joint_trajectory_controller |
| Demo-Software `arm_driver` | Kortex Python API, high-level servoing |
| `~/atdev/kinova-gen3-driver` | custom C++ 1 kHz low-level driver |

Only one may run. Before starting, confirm with the team nobody else is
mid-session, and check locally:

```bash
pgrep -af "arm_driver|teleop_socket|trajectory_run|kortex" ; ls /tmp/kinova.lock 2>/dev/null
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
source ~/RAMMP-Kinova/ros2_ws/install/setup.zsh     # ros2_kortex lives here
source ~/RAMMP-CuRobo/install/setup.zsh

# terminal 1 — the arm driver + controllers (RAMMP-Kinova workspace; this
# repo deliberately does not launch it — the planner never owns the arm):
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 \
    dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false
```

This must be the ONLY bringup: never combine it with the MuJoCo sim or a
second kortex bringup (one /controller_manager per arm).

Gotchas (from the ros2_kortex source, all defaults):
- `robot_ip` is REQUIRED (no default) — the launch fails without it.
- `gripper:=robotiq_2f_85` must be passed or the gripper controller is
  simply not spawned.
- `launch_rviz` defaults **true** — pass false on the headless Jetson.
- Do NOT have the MuJoCo sim running — both bringups claim
  `/controller_manager`.

Verify before going on:

```bash
ros2 control list_controllers   # joint_state_broadcaster, joint_trajectory_controller,
                                # robotiq_gripper_controller — all active
ros2 topic hz /joint_states     # streaming
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
- Gripper: `ros2 service call /rammp_curobo/close_gripper std_srvs/srv/Trigger`
  (and open) — the arm doesn't move, but keep clear of the fingers.
- Cartesian goals (`--pos ... --quat ...`) only AFTER the world file has
  been validated against reality, and never near surfaces on the first day.
- Raise `speed_scale` gradually; 1.0 means "as planned", which is full
  cuRobo time-parameterization — do not go there this week.

## Faults / recovery

- Software abort: Ctrl+C on the example (goal cancel), or
  `ros2 action send_goal` cancel; the JTC holds position.
- Controller/arm fault: check `ros2 control list_controllers`; the kortex
  bringup spawns a fault controller — `ros2 service list | grep -i fault`
  for the reset interface. If in doubt: e-stop, then power-cycle the arm
  and restart the bringup.
- **Arm ignores every command but reports success** (seen 2026-08-13):
  every trajectory "succeeds" with zero motion and the driver log spams
  "combination of Control Mode and Active State are not supported" — the
  arm dropped out of low-level servoing (states still stream, writes go
  nowhere; the stock JTC config's disabled goal tolerances mask it).
  Fix, verified live (reset_fault ALONE is not enough — on this driver
  build it restores single-level mode, and the JTC's stale hold position
  is a jump hazard):
  1. `ros2 service call /fault_controller/reset_fault example_interfaces/srv/Trigger`
  2. `ros2 control switch_controllers --deactivate joint_trajectory_controller`
  3. `ros2 control switch_controllers --activate joint_trajectory_controller`
  (the planner node also runs this sequence automatically on the
  no-motion signature). Verify low-level servoing (mode 3) is back via a
  parallel Kortex query before commanding motion. The transient form fires
  at controller goal TRANSITIONS (a new goal right after a completed one)
  — which is why tour_demo merges its whole tour into ONE trajectory and
  retries from a standstill only.
- **Reset succeeded, no fault spam, arm STILL ignores everything** (also
  2026-08-13): the arm itself reports SERVOING_READY (verifiable with a
  parallel Kortex session) but nothing moves — the DRIVER is wedged, not
  the arm. kortex_driver sets a `block_write` flag when preparing a
  controller-mode switch and clears it only when the switch completes; a
  fault mid-switch strands it set, and every write is silently discarded
  forever after. Fix: restart the kortex bringup (the arm holds its pose
  through the restart). Symptoms identifying this case: goals "succeed"
  with zero motion + NO fault log spam + the arm itself reporting
  READY/SERVOING via a direct Kortex query. (Do not rely on
  /fault_controller/internal_fault — it read `true` on this stack even
  while the arm was healthy in low-level servoing.)
- After any e-stop or fault, RE-RUN the dry-run step before arming again
  (the arm may have been moved by hand; stale plans are refused, but check
  the world still matches reality too).

## 6. Perceived world, stage 1 — Orbbec (attended)

Prereqs: steps 1-5 clean; the Orbbec Gemini 336L aimed at the bench with
an unobstructed view of the arm's workspace; an ArUco tag (DICT_4X4_50
id 0, side measured with calipers) mounted RIGIDLY on the gripper.

1. **Calibrate (one-time, redo if the camera moves):**
   ```bash
   # terminal 1: kortex bringup (step 2 above).  terminal 2:
   export ROS_LOCALHOST_ONLY=1
   ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true
   # terminal 3 (repo root, sourced):
   python3 scripts/calibrate_camera_extrinsics.py --marker-id 0 --marker-size <measured m>
   ```
   YOU jog the arm between recordings (~10 poses spread across the view,
   varying the wrist rotation axis — tilt AND twist). The script refuses
   to write above 1 cm RMS residual. Then rebuild:
   `colcon build --symlink-install --packages-select rammp_curobo_ros`.
   If the driver's topic names differ from the defaults, fix the
   `depth_topic`/`info_topic` fields in the written
   `rammp_curobo_ros/config/camera_orbbec_bench.yaml` (check with
   `ros2 topic list | grep camera`).
2. **RViz acceptance (no arm motion):** planner up, then
   `ros2 run rammp_curobo_ros cameras`; RViz displaying
   `/cameras/world_markers`. Place a box on the bench → a marker appears
   within ~2 s, within ±3 cm of reality; remove it → gone within ~3 s.
   Wave a hand through the view → NO persistent marker. With the arm
   bringup running, the ARM must not appear (self-filter); if it does,
   fix the mount calibration before proceeding.
3. **Service loop:** `python3 scripts/cameras_checks.py` — all three
   checks green (box round-trip, markers, plan-under-churn).
4. **First plan around a real obstacle:** put an object between home and
   a target; plan (dry-run) and eyeball the trajectory clearing the
   marker in RViz; only then execute at ≤0.25 speed, e-stop in hand.
   Known v1 limitation: an obstacle the arm OCCLUDES decays out of the
   world after a few seconds — keep the baseline world honest and don't
   rely on perception for objects the arm is hiding.
