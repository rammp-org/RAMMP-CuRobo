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
ros2 launch kortex_bringup gen3.launch.py \
    robot_ip:=192.168.1.10 dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false
```

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
- Build the collision world automatically: `ros2 run rammp_curobo_ros
  sweep_scan --camera camera_d405_wrist.yaml --apply` (see the README's
  "Automatic obstacle scanning" section — the camera mount YAML must be
  measured and dry-capture-verified first). The static table plane still
  comes from this runbook's step 1 either way.
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
  Fix, verified live:
  `ros2 service call /fault_controller/reset_fault example_interfaces/srv/Trigger`
  then confirm the log spam stopped before commanding motion again. A
  single isolated no-motion "success" is the milder intermittent form —
  sweep_scan retries it automatically.
- After any e-stop or fault, RE-RUN the dry-run step before arming again
  (the arm may have been moved by hand; stale plans are refused, but check
  the world still matches reality too).
