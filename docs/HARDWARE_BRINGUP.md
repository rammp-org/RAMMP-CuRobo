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

## 6. Perceived world — wrist D405 (attended)

Prereqs: steps 1-5 clean; the D405 plugged into the Jetson on its wrist
bracket. NO extrinsic calibration exists or is needed — the camera rides
the arm's TF; its bracket mount lives in
`rammp_curobo_ros/config/camera_d405_wrist.yaml` (photo-estimated; step 3
below validates it against reality).

1. **Drivers up** (every terminal: `export ROS_LOCALHOST_ONLY=1`):
   ```bash
   # terminal 1: kortex bringup (step 2 above).  terminal 2:
   ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 camera_name:=d405
   # terminal 3: planner.  terminal 4:
   ros2 run rammp_curobo_ros cameras
   ```
   The cameras node defaults to the wrist config. Sanity: its log shows
   the baseline obstacle count and NO "no fresh depth frames" warning.
2. **See what it sees:** open `http://192.168.1.11:8766/` in any
   browser on the lab network — the node streams the camera feed with
   every perceived box drawn on it (no monitor needed; `view:=false`
   disables). RViz + `/cameras/world_markers` shows the same boxes in
   3D. The bench in front of the gripper (0.07-0.9 m from the camera)
   populates as the wrist looks at it; the table itself is stripped by
   the baseline; the ARM must not appear (self-filter).
3. **Mount validation via acceptance:** place a box on the bench in
   front of the gripper → marker within ~2 s, within ±3 cm of the real
   box (tape measure). More than 3 cm off in a consistent direction =
   the bracket moved since the photos — re-measure the mount offsets in
   the YAML before trusting anything.
4. **Look-away memory (the wrist-camera contract):** with the box's
   marker up, move the arm so the camera points elsewhere — the marker
   MUST SURVIVE indefinitely (frustum-scoped decay: out of view is
   remembered, not forgotten). Remove the box, look back at the empty
   spot — the marker fades within ~3 s. Wave a hand through the view —
   no persistent marker.
5. **Service loop:** `python3 scripts/cameras_checks.py` — all three
   checks green (box round-trip, markers, plan-under-churn).
6. **First plan around a real obstacle:** put an object between home and
   a target, let the wrist SEE it (markers confirm), plan (dry-run) and
   eyeball the trajectory clearing the marker in RViz; only then execute
   at ≤0.25 speed, e-stop in hand.

Honest limits: the wrist camera only maps where it has looked — plans
through never-seen space rely on the baseline world (keep
`world_real_bench.yaml` honest); range is 0.9 m, so the map builds up
close-in, which matches manipulation. The D405 is PASSIVE stereo (no
projector): on textureless surfaces its permissive default settings
invent depth rather than admit ignorance (field 2026-08-19: the blank
white bench read 0.3 m at a true 0.7 m — 20 phantom boxes, sightings
floating 25 cm above the bench). The wrist YAML therefore declares
`depth_module.visual_preset: 3` (High Accuracy) as a `sensor_params`
contract; the cameras node and the seek preflight push it to the driver
at startup and keep retrying (log: `sensor_params: /d405/d405 <- ...`),
so driver restarts and start order don't matter. The flip side is
physics, not configuration: a truly featureless surface returns HOLES —
unknown space, not obstacles. A blank white box is seen only by its
edges and shading; anything like that in the workspace belongs in the
baseline world. Frames captured while the wrist
moves (or rotates — rotation sweeps points far faster than it moves the
lens) are dropped by design: the "depth frames gated" log line is normal
during motion. Thin objects (cables, rods) near the silhouette can
flicker — pad them in the baseline if they matter. A mapped obstacle
whose location the camera can no longer see through (blocked, depth
hole) persists by design: clear it by looking at the spot, setting an
ignore region over it (which also purges what's already mapped there),
or restarting the node.

(The retired fixed-Orbbec path — browser-click fingertip calibration via
`scripts/calibrate_camera_extrinsics.py` — still works if a bench camera
returns; it writes `camera_orbbec_bench.yaml` and the node takes it via
the `cameras` parameter list. Note it now shares the frustum-scoped
decay: unlike the original Orbbec build, a removed object backed by a
depth hole or occlusion persists until provably seen through.)

## 7. Seeker — "go to the bottle" (attended)

Prereqs: §6 acceptance passed; the D405 driver running with
`align_depth.enable:=true`; planner with `execute:=true`; cameras node
up. Then:

```bash
ros2 run rammp_curobo_ros seeker --ros-args -p target:="go to the bottle"
```

It moves autonomously once targeted (owner decision 2026-08-19) —
e-stop in hand the whole time, Ctrl+C stops and holds. Behavior, all
one loop: 3 agreeing frames acquire the object (watch `:8767` for the
detections, `:8766` for the obstacle map) → short cuRobo-planned hops
close in through the perceived world, wrist flat at the object's
height, stopping 0.18 m out → move the object and it follows; hide it
>4 s and it returns to the survey pose; lift it >15 cm and it HOLDS
(the ignore region follows the target, so chasing a hand-held object
would blind collision checking exactly where the hand is — put it
down to resume). No camera data = no motion. Retarget live:
`ros2 service call /seeker/set_target rammp_curobo_interfaces/srv/SetTarget
"{text: 'go to the cup'}"` ("" idles the arm).

First run: one object alone, 0.45-0.65 m in front of the arm. Second
run: add a box between arm and object — the hops must bow around its
cuboid. If detections land on the wrong thing, `:8767` shows exactly
what YOLO claims; the fix is the model or the scene, never the map.

### Grasping (GraspGenX)

With the grasp server up (see README), inside 0.35 m the seeker asks
GraspGenX for 6-DoF grasps and runs pre-grasp → grasp → close.
**Two things MUST be settled before the first hardware grasp:**

1. **Gripper mount twist.** `gen3.yaml`'s `spin_deg` (90 deg in sim)
   ships DISABLED pending re-measurement on the real arm. GraspGenX's
   grasp frame matches the URDF's `end_effector_link`; if the physical
   gripper is bolted 90 deg off the URDF, every grasp closes across the
   wrong axis. Verify with a tape measure / photo before trusting one.
2. **Grasp depth.** `tool_offset` defaults to 0.120 (our `tool_frame`);
   GraspGenX calls the 2F-85 fingertip 0.136. Dry-run first
   (`execute:=false` on the planner) and eyeball the pre-grasp pose in
   RViz before arming.

Dry-run the whole pipeline first, then a first live grasp on something
light and forgiving, e-stop in hand.
