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

(A fixed bench camera — the Orbbec §8 uses — is calibrated by
`scripts/calibrate_orbbec.py`: print a 60 mm tag with `make_tag.py`,
tape it to the gripper, run with `--execute` (moves the arm; e-stop in
hand). It solves eye-to-hand and writes `camera_orbbec_bench.yaml`,
which the node takes via the `cameras` parameter list.
`scripts/calibrate_camera_extrinsics.py` — browser fingertip-click, no
fiducials — remains the fallback when no tag can be printed. Residual
translation error is corrected at startup by the node's
self-registration (§8, "How the arm stays out of its own map"); never
hand-edit mounts. A fixed camera shares the frustum-scoped decay: a
removed object backed by a depth hole or occlusion persists until
provably seen through.)

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
detections, `:8766` for the obstacle map) → cuRobo plans straight to
the object through the perceived world (pre-grasp → grasp with
`grasp:=true`, or a standoff with `grasp:=false`) → move the object and it follows; hide it
>4 s and it returns to the survey pose; lift it >15 cm and it HOLDS
(the ignore region follows the target, so chasing a hand-held object
would blind collision checking exactly where the hand is — put it
down to resume). No camera data = no motion. Retarget live:
`ros2 service call /seeker/set_target rammp_curobo_interfaces/srv/SetTarget
"{text: 'go to the cup'}"` ("" idles the arm).

First run: one object alone, 0.45-0.65 m in front of the arm. Second
run: add a box between arm and object — the approach must bow around its
cuboid. If detections land on the wrong thing, `:8767` shows exactly
what YOLO claims; the fix is the model or the scene, never the map.

### Grasping (GraspGenX)

With the grasp server up (see README), the seeker asks
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

## 8. Sweep-and-avoid demo (attended)

The arm sweeps left and right; the fixed Orbbec watches; anything that
gets in the way is planned around. Three outcomes, all deliberate:

| what happens | why |
|---|---|
| stroke completes | nothing in the corridor |
| stroke is cancelled, arm arcs around | the watchdog found the remaining path blocked |
| arm stops and waits — **HOLD** | the obstacle overlaps the arm's own body, so `INVALID_START_STATE_WORLD_COLLISION`: there is no path to plan. Measured on this arm, that begins around **15 cm**. Not a fault |

**The watchdog is the safety-relevant part.** `validate_goal_msg` checks
names, limits, velocity, timing, continuity and the start match — it
never re-checks *collision*. A trajectory planned before an obstacle
appeared passes every gate and drives through it. `~/check_trajectory`
is what closes that hole, so if it is not running, the demo is not safe.

### How the arm stays out of its own map

The cameras node erases the arm from the depth cloud with **cuRobo's own
47 collision spheres** (`self_model_gen3_2f85.yaml`, baked by
`scripts/bake_self_model.py`), placed by TF at each depth frame's stamp.
The arm-link frames are identical between cuRobo's URDF and
ros2_kortex's (verified, all eight joints); the gripper is expressed in
`end_effector_link` because the two URDFs attach it with a different yaw.
`self_radius` is now the MARGIN added to every sphere radius (default
0.08 m), not a capsule radius — it only has to absorb camera-pose error
and TF/depth skew.

At startup the node also **registers each fixed camera off the arm**
(`auto_register`, default on): eight still depth frames are fitted to
the sphere model — coarse grid search, then point-to-plane ICP,
translation only, camera-facing surfaces only — and the solved shift is
applied to the mount in memory for the session, with the corrected
`mount_xyz` line logged for you to paste into the config. Gates: at
least 300 arm points, residual under 2 cm, shift under 20 cm; otherwise
it warns and leaves the calibration alone. Synthetic accuracy: 0.2 mm
with obstacles touching the arm, errors up to 19 cm.

Why: the tag calibration left the camera 7-9 cm off, and the first
fix — `perception_debug --apply`, which measured the arm's BOXES — made
it worse. Those boxes had already been through the self-filter, so only
the far fringe of the displaced arm survived, and it measured the
fringe. The registration reads the raw frames before any masking.

### Prerequisites (do not skip)

1. **Measure the bench** — now one command, from the registered camera:
   `python3 scripts/measure_bench.py --apply` (Orbbec running, nothing
   moves). Done 2026-08-24: tabletop at z=-0.027, 4.3 cm HIGHER than the
   old placeholder — the "scraping the table" margin never existed.
   Re-run after moving the arm or the bench. The written world is a
   true-height `table` (no_pad, the base spheres sit 12 mm above it)
   plus four padded `floor_*` guards: a hard 2 cm keep-out over the real
   surface everywhere the arm can swing.

   The old advice stands only as history: **Measure the bench.** `world_real_bench.yaml` is still placeholder
   geometry. The cameras node *subtracts* baseline boxes from the depth
   cloud, so a wrong table does not just mis-model collision — the real
   table surface becomes one enormous perceived obstacle.
2. **Prove the self-filter.** With the arm, camera and cameras node up
   and the workspace EMPTY:

   ```bash
   python3 scripts/perception_debug.py --seconds 20
   ```

   Requirement: **zero boxes overlap the arm**, and the camera
   registration section reports a shift (ideally under 1 cm once the
   config has been corrected; `--apply` writes it). If the arm grows
   phantom obstacles it will spend the demo dodging itself — on the
   bench that looked like the shoulder arcing 190 deg over the top and
   the goal reporting IK_FAIL, because the map had filled with the
   arm's own image.
3. **Check the reactive chain**, no arm and no camera needed:

   ```bash
   ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
   python3 scripts/sweep_demo_checks.py     # 12 checks, must all PASS
   ```

4. **Check that Ctrl+C stops a moving arm** (stub controller, no hardware):

   ```bash
   python3 scripts/abort_checks.py          # must print PASS
   ```

   This one is not ceremony. rclpy's default SIGINT handler tears the
   context down synchronously, and `executor.run()` delivers a cancel by
   polling `goal_handle.is_cancel_requested` — which only advances while
   the node's executor is spinning. Before this was fixed, Ctrl+C during
   motion did not stop the arm, it stopped *watching* it, and the
   controller drove the trajectory to its end. `planner_node` now owns
   SIGINT: it cancels the controller goal (arm stops and holds), waits
   for the execution to end, and only then exits.

### Running it

Arm bringup and the Orbbec driver first, in their own terminals (§2),
the camera calibrated (§6's fixed-camera note: `calibrate_orbbec.py`),
then:

```bash
ros2 launch rammp_curobo_ros sweep_demo.launch.py                 # dry run
ros2 launch rammp_curobo_ros sweep_demo.launch.py execute:=true \
    speed_scale:=0.25                                             # it moves
```

Dry run never sends an execution goal. It plans a stroke and polls the
same watchdog, printing CLEAR/BLOCKED — hold something into the corridor
and watch it flip, then watch the replan route around it. **Do the dry
run first every session.**

The reaction budget, at the launch file's 5 Hz / `occupied_at` 2:

```
perception 0.28-0.48 + watchdog notice <=0.20 + check 0.04
          + cancel & settle (MEASURED AND LOGGED EACH TIME) + replan 0.26
```

**Workspace sector.** `max_left_deg` / `max_right_deg` bound the BASE
yaw so the arm cannot swing through 360 to reach something behind it —
the launch defaults are 75 left / 90 right. The arm's azimuth is
`-joint_1` (verified on hardware: `joint_1 = -90 deg` puts the tool at
azimuth +90), so those become `joint_1 in [-75, +90]`.

It has to be applied before warmup, and that is why it is a launch
argument rather than a service: cuRobo's solvers read the joint-limit
tensors once and cache them. Truncate after the first solve and the
planner keeps using the old range — our validator sees the new one, so
every plan comes back `LIBRARY_VALIDATION_FAILED` instead of being
routed inside the sector.

What it does and does not guarantee, measured:

- the base never yaws outside the sector, and goals behind the arm
  (azimuth 180, -140) are refused outright with IK_FAIL;
- the A<->B sweep stays at whole-arm azimuth -49..+49;
- but it bounds the BASE, not the fingertip. For a goal just outside the
  sector the arm can still bend to reach it. A true workspace wedge
  needs collision walls, and those leak: the arm folds inside the gap
  they must leave around the base and rotates through it.

**Standoff is bimodal — do not tune it as if it were a dial.** Measured
on this bench against a raised slab across the sweep corridor:

| `activation_distance` | route | gap | tool z |
|---|---|---|---|
| 0.03 (default) | under | 0.053 | 0.08–0.35 |
| 0.06 | under | 0.067 | 0.07–0.35 |
| **0.07** | **under** | **0.084** | 0.08–0.35 |
| 0.08 | OVER | 0.143 | 0.35–**1.12** |

Between 0.07 and 0.08 the planner stops squeezing underneath and starts
lifting over the top, and the tool excursion jumps to 1.12 m — a
completely different, much larger manoeuvre, not a wider version of the
same one. `world_padding` >= 0.04 flips it the same way. So 0.07 with
padding left at 0.02 is the working point: the widest clearance that
still takes the tight route, and the highest the arm stays above the
table (0.084 m) of any of the under-routes.

Clearance otherwise tracks the knob linearly (`gap ~= activation +
0.027`). `activation_distance` is safe to raise — a cost term, so no
state becomes infeasible. `world_padding` hard-inflates every box except
`no_pad_names`, so 0.08 puts the arm inside the modelled table and
nothing plans at all.

**Where the watchdog actually trips.** cuRobo's hard verdict flips when
the planned path comes within **0.020 m** of a perceived box — that is
`world_padding`, measured by bisection, NOT `collision_activation_distance`
(0.03), which feeds the IK/trajopt cost terms and never reaches the
constraint checker that `check_trajectory` uses. Raising that config knob
buys no standoff here. What does is `clearance_margin`, which is measured
against the *unpadded* boxes. It defaults to 0 because only 0 is
livelock-free: a margin wider than a legitimate plan's own clearance
(typically ~0.055 m) would trip the instant every fresh stroke began. The
demo detects that case, warns, and ignores the margin for that stroke
rather than stuttering forever — but pick the value from what you see in
the dry run.

Every cancel logs `arm still after N s`. That term could not be measured
off-hardware; watch it on the first run. If it exceeds ~1 s the demo will
look frozen rather than reactive — shorten the stroke (lower `sweep_deg`)
rather than raising the speed.

### Parking it

```bash
python3 scripts/go_home.py --execute        # --speed 0.25 by default
```

Plans home from the arm's LIVE state through the normal gate chain, so
it is safe from wherever the demo left it. Without `--execute` it plans
and prints, and nothing moves.

### Contortion guards in the demo

- **Pinned joint goals.** A and B are planned to by POSE the first time
  and the joints they land on are remembered; every later stroke plans
  to those JOINTS, so cuRobo's IK cannot pick the other elbow/wrist
  family for the same pose and connect them with a half-turn.
- **`max_joint_span_deg`** (default 120): a plan in which one joint's
  travel EXCEEDS its direct end-to-start move by more than that is a
  contortion and is refused and re-planned. (Excess, not absolute span:
  a long first approach from a far pose is direct and legitimate; the
  bench contortion swept the shoulder 193 deg on a near-zero net move.)
  Every plan logs its per-joint spans.
- The core planner refuses any plan with a joint span over 270 deg
  (`WINDING`) regardless of caller.

### Behaviour notes worth knowing before you demo it

- **Depth only.** No object detection is involved; a hand, a box and a
  mug are identical to it. Test with a box before testing with fingers,
  so the reaction time is a measured number first.
- **Hold the prop still.** A voxel needs `occupied_at` confirmations, so
  something waved quickly may never confirm. Keep it above the `min_z`
  crop (not resting on the bench) and *ahead* of the arm — anything
  within (sphere radius + the `self_radius` margin) of the arm's
  collision-sphere model is erased as part of the arm.
- **Obstacles can linger.** Decay only forgets a voxel the camera can
  prove it sees through, so a withdrawn prop can persist while the arm
  occludes that spot. Call `~/set_ignore_region` to purge if it sticks.
