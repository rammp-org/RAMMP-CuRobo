# Perceived collision world (cameras node) — design

Approved 2026-08-17 (chat review). Goal: the planner's collision world
continuously reflects what the bench cameras actually see, so plans avoid
real obstacles without anyone hand-editing world YAMLs.

## Decision summary

- **Approach A** (chat options A/B/C): revive the proven scan pipeline
  (removed in the 2026-08-14 cleanup, recovered from `0e0ec5b~1`) and run
  it continuously in a new node. cuRobo-native nvblox rejected (install
  risk vs pinned stack, 3.3 GB disk free, world stops being inspectable
  cuboids). Occupancy-grid boxes live *inside* A as the accumulator.
- **Continuous updates** (~2 Hz), not scan-then-plan — user's call.
- **Lives in `rammp_curobo_ros`** beside the planner — user's call.
  The planner remains the sole owner of the cuRobo world; perception is
  just another client of its services.
- Node name: **`cameras`** (user renamed from `world_watcher`).

## Architecture

```
Orbbec Gemini 336L ─┐                                  ┌ /rammp_curobo/plan_to_pose
 (fixed bench view) ├─> cameras node ─UpdateWorldBoxes─> planner_node ─ cuRobo
D405 (wrist, stage2)┘   2 Hz loop                      └ existing gates untouched
```

- `cameras` subscribes depth + camera_info (sensor-data QoS — best-effort
  subscriber is compatible with both reliable and best-effort publishers),
  processes at 2 Hz, pushes boxes to the planner, publishes a
  `visualization_msgs/MarkerArray` on `~/world_markers` for RViz truth.
- Perceived boxes are always merged **on top of the static baseline world**
  (`world_real_bench.yaml`): updates are never empty (v0.7.8 empty-world
  trap) and the table survives when cameras see nothing.
- cuRobo v0.7.8 world = axis-aligned cuboids only, `collision_cache_obb=60`
  cap. Everything ends life as bounded AABBs in `base_link`.

## New interfaces (`rammp_curobo_interfaces`)

`srv/UpdateWorldBoxes.srv` — served by **planner_node**:

```
string[] names                  # stable IDs from the perception tracker
geometry_msgs/Point[] centers   # base_link; boxes are axis-aligned
geometry_msgs/Vector3[] dims    # full extents (m)
string baseline                 # world YAML merged underneath ("" = keep current)
---
bool success
string message                  # includes box count vs the obb cache cap
```

`srv/SetIgnoreRegion.srv` — served by **cameras**: one axis-aligned box in
base_link whose points are dropped before clustering (the manipulation
target is not an obstacle). Zero dims clears it.

```
geometry_msgs/Point center
geometry_msgs/Vector3 dims
---
bool success
string message
```

Existing `SetWorld` (YAML path) is unchanged.

## Perception pipeline (per 2 Hz tick)

Pure stages (numpy/scipy, no ROS — live in `core/rammp_curobo/perception.py`
so they unit-test offline; the node is I/O glue):

1. Deproject the latest depth frame with camera_info intrinsics,
   range-clip per camera config (revived `capture_points` math).
2. Transform to base_link (camera extrinsics: fixed mount from the
   calibrated YAML for the Orbbec; TF chain for the wrist D405).
3. Workspace crop + **capsule self-filter** over the live TF arm chain
   (revived `robot_mask`) — erases the moving arm from the scene.
4. Drop points inside the ignore region and within 1 cm of baseline-world
   box surfaces (the table must not re-cluster as an obstacle).
5. **Voxel accumulator with hysteresis** (3 cm voxels): +1 when hit,
   −1 when not, occupied at score ≥3 (~1.5 s to appear), gone after ~2 s
   of absence. Kills flicker and transients. Known limitation, accepted
   for v1: no free-space raycasting, so an obstacle occluded (e.g. by the
   arm) decays away while still present — the baseline world and the slow
   decay are the mitigations.
6. Occupied voxels → connected components → tight AABBs (revived
   `cluster_boxes` / `_split_cells` logic, nearest-first cap ordering),
   capped at 20 boxes (headroom under obb cache 60 minus baseline).
7. **Stable naming**: match clusters to the previous tick by AABB overlap;
   IDs persist (`obs_3`) for sane logs/RViz.
8. Call `UpdateWorldBoxes` only when boxes changed materially (>1 cm
   center/dims movement, or count change) to avoid pointless world churn.

## Planner-side changes (small)

- New handler for `UpdateWorldBoxes`: builds the obstacle-dict list, calls
  a new `Planner.update_world_boxes(boxes, baseline=None)` (core) that
  loads/merges the baseline scene, all under the existing `_plan_lock`.
- `_plan()` lock acquisition changes from `acquire(blocking=False)` to a
  short blocking acquire (0.5 s timeout) so a 2 Hz updater holding the
  lock for milliseconds doesn't bounce plan requests.

## Calibration (Orbbec → base_link)

`scripts/calibrate_camera_extrinsics.py`, one-time, re-runnable, attended:

- ArUco tag rigidly on the gripper; **the human jogs the arm** to ~10
  poses spanning the camera view; the script only observes.
- Run the Orbbec with `depth_registration:=true` so depth is aligned into
  the color optical frame — one extrinsic covers both streams; ArUco runs
  on the color image.
- Eye-to-hand solve via `cv2.calibrateHandEye` (base→tool poses from TF
  inverted per the eye-to-hand trick; tag-in-camera from PnP). Neither the
  tag-in-tool offset nor the camera pose needs pre-measurement.
- Writes `rammp_curobo_ros/config/camera_orbbec_bench.yaml` in the schema
  `perception` reads (`parent_frame: base_link` + mount transform).
  Refuses to write if RMS residual > 1 cm.
- D405 stage 2 reuses the recovered `camera_d405_wrist.yaml` mount
  (re-verify the bracket before trusting it).

## Staging

1. **Stage 1 — Orbbec live world** (this plan): calibrate; run `cameras`
   Orbbec-only; RViz acceptance: box placed on bench appears ≤2 s within
   ±3 cm, disappears ≤3 s after removal; then plan around a real obstacle.
   No arm motion until RViz agrees with reality.
2. **Stage 2 — D405 fusion**: wrist camera as a second accumulator input
   (points trusted only when the wrist moves slowly; 0.07–0.9 m clip).
3. **Stage 3 — reactive stop on world change**: separate future design.

## Testing

- Offline pytest (`core/tests/test_perception.py`): synthetic depth →
  deproject/crop/cluster round-trips; accumulator appear/decay timing;
  cap-overflow behavior; ignore-region and baseline masking; tracker ID
  stability. Runs with the normal suite, no hardware.
- ROS-side pytest: `UpdateWorldBoxes` handler validation (mocked planner),
  lock-timeout behavior.
- Live: `scripts/cameras_checks.py` — topic liveness, RViz loop check,
  update_world-under-planning-load soak. Attended bench procedure appended
  to docs/HARDWARE_BRINGUP.md.
- Safety posture unchanged: `cameras` never executes anything; all
  execution gates untouched.

## Non-goals (v1)

Mid-motion replanning / reactive stop; mesh or non-AABB obstacles;
free-space raycasting; tag-free object recognition; any arm driver
involvement.
