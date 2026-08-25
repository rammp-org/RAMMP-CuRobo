# Real-arm bring-up runbook (Kinova Gen3 @ 192.168.1.10)

Follow this in order. **A human holds the physical e-stop from the moment
the kortex bringup starts until the last motion ends. No exceptions.**

**This repo cannot move the arm.** It plans; something else executes. So
this runbook covers the planning side of a hardware session — bench, world,
network, and proving the plans are sane — and hands off to the arm owner
(`kinova_arm_ros2`) for first motion and the abort drill. What changes on
hardware is the world model, the timing source, and the consequences.

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

## 3. Planner node — plans only, always

```bash
source ~/RAMMP-CuRobo/install/setup.zsh    # on top of the two above
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
```

There is no arming step, because there is nothing to arm: this node has no
execution surface. Sanity-check a plan of a tiny move from wherever the arm
is. Read the arm's configuration yourself (the bringup publishes it) and
hand it to the planner as `start_joints`:

```bash
ros2 topic echo /joint_states --once          # note joint_1..7 positions
ros2 action send_goal /rammp_curobo/plan_to_joints \
    rammp_curobo_interfaces/action/PlanToJoints \
    "{target_joints: [<q1+0.15>, <q2>, <q3>, <q4>, <q5>, <q6>, <q7>],
      start_joints:  [<q1>, <q2>, <q3>, <q4>, <q5>, <q6>, <q7>]}"
```

Expect `success: true`, a low `goal_mismatch_rad`, and a trajectory whose
last point is ~0.15 rad from the start on joint_1 and ~0 elsewhere. If the
plan fails or looks wrong, stop here — a bad plan is a bad plan whoever
runs it.

`ros2 run rammp_curobo_ros tour_demo` is the broader check: it chain-plans
a whole tour from `HOME` and moves nothing.

## 4. First motion — the arm owner's runbook

First motion, speed ramping and the abort drill are **not this repo's** and
are not performed with this repo's tools. Use `kinova_arm_ros2`'s `GoToEEPose`
/ `GoToJointConfig` actions and follow its runbook, with a human on the
physical e-stop. Prove the software abort (goal cancel → arm stops and
holds) on a trivial move before anything larger.

The planning-side rules still apply throughout:

- Keep `world_real_bench.yaml` matching reality (step 1) — obstacles are
  measured and edited by hand, or pushed live from another module via
  `/rammp_curobo/set_world`. A plan is only as safe as the world it dodged.
- Cartesian goals only AFTER the world file has been validated against
  reality, and never near surfaces on the first day.
- Plans come back at full cuRobo time-parameterization. Anything slower is
  the executor's business — cuRobo's `velocity_scale` stays 1.0.
- Re-plan after any manual repositioning of the arm: `start_joints` is the
  configuration you measured, so a stale one plans from a fiction.

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
  Verify low-level servoing (mode 3) is back via a parallel Kortex query
  before commanding motion. The transient form fires at controller goal
  TRANSITIONS (a new goal right after a completed one) — which is why a
  chained tour is best executed as ONE merged trajectory, retrying from a
  standstill only. (This repo's planner used to run the recovery sequence
  itself; it no longer touches the arm, so recovery belongs to whoever
  owns the driver.)
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
- After any e-stop or fault, RE-RUN step 3 before commanding motion again:
  the arm may have been moved by hand, so re-read `/joint_states` and plan
  from the configuration it is actually in — and check the world still
  matches reality too.
