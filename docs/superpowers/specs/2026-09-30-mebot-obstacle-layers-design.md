# MEBot obstacle layers: the wheelchair and the person

- **Status:** Design approved by the owner 2026-09-30. Revised 2026-10-02 after
  a sim spike: the sim is a ROS 2 stack (decision 1), and the spike's findings
  are folded in (§8, §13). The owner reviews this spec before the implementation
  plan is written.
- **Issue:** <https://github.com/rammp-org/RAMMP-CuRobo/issues/19>
- **Branch:** `feature/19-platform-layers`, off `origin/dev` (b734928, the
  v1.0.0 content), in the worktree `~/RAMMP-CuRobo-19`. Local commits only,
  never pushed.

**Summary.**

- `platform`: the MEBot drive base, 16 boxes generated from the pinned URDF.
- `person`: 95th-percentile occupant + 5 cm, 6 boxes in 3 segments, excludable.
- Config layers join every world, set_world pushes included; no IDL change.
- Gated: under-coverage 0, over-coverage ≤ 15 % (10.9 % measured), padded boxes
  ≥ 4 mm clear of the arm's fixed spheres.
- Every claim has a test that fails if cuRobo ignores the geometry.

**Conventions.** Measured numbers come from a design-phase CPU prototype of the
generator and from CPU prototypes of the arm-sphere and MuJoCo checks; the
implementation's generator, tests and sim recompute them, and §5.5, §8 (G0 to
G6) and §9 say which are gated. A file:line without a repo name is this repo at
`origin/dev` (b734928): `planner.py`, `world.py`, `scene.py` and `config.py` are
in `core/rammp_curobo/`, the tests in `core/tests/`, `planner_node.py` in
`rammp_curobo_ros/rammp_curobo_ros/`, and `build.yml` in `.github/workflows/`.
Other repositories are cited at RAMMP-box-opening b20c820 (branch
`no-fiducial`), RAMMP-generalized 79687a1 (`main`) and RAMMP-Kinova d7522ca
(`main`). cuRobo references are to v0.7.8 (d64c4b0); §16 gives their full paths.
Coordinates are metres in the arm's `base_link` frame unless the mebot frame is
named. Layer boxes are listed unpadded; the planner adds `world_padding` (0.02
m) per side.

## 1. Goal

The planner learns about the wheelchair the arm is mounted on (the **platform**:
the MEBot drive base) and the person in it (the **person**). The geometry is:

- generated from CAD;
- committed as box layers;
- composed into every collision world the planner builds.

The person stays a separate layer, so a later mouth approach can exclude it.
Every claim has a test that fails if cuRobo ignores the geometry. The layers are
also checked against the exact MEBot and an independent seated body in a MuJoCo
sim driven over ROS.

### 1.1 The ask, the planner today, and the review bar

The original ask: "cuRobo should support avoiding static obstacles (the chair).
Ideally, it can load a URDF to handle this." The issue turns that into geometry
generated from CAD that arrives as layers, with the person kept separable for a
later mouth approach. Perception is the long-term answer but is not deployable
yet: none of it exists, the camera calibration is unmeasured, depth dropouts on
a wheelchair frame read as free space, and it cannot be verified in time.
Nothing here is thrown away when it arrives; the person layer can become live
later.

What the planner does today:

- **Boxes only.** `make_world_config` emits `{"cuboid": ...}` (world.py:66);
  props become bounding boxes (scene.py:65-71).
- **cuRobo v0.7.8 drops geometry silently.** Cylinders and spheres vanish from a
  WorldConfig, and `update_world` with zero cuboids keeps the previous world
  (world.py:5-8).
- **Guards.** An empty world is refused (world.py:56-60); more boxes than the
  cache raises (world.py:61-65; the cache is 60, config.py:36).
- **`ignore=`** (planner.py:277) drops the props being reached for. It is the
  hook a mouth approach will use.
- **A field on an existing action or service is a MAJOR change**
  (CONTRIBUTING.md:132-140).
- Anything touching `CuRoboPlanner` needs CUDA. Pure-Python logic is covered in
  CI (`test_offline.py`); GPU work runs on the Jetson (`test_smoke.py`).

The issue's review bar, and how this design meets it:

| Fails review                                                       | This design                                                                                                                                                                                            |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| One bounding box for the wheelchair                                | 16 boxes. One box is 158.2 % over-covered and its padded box intrudes 7.9 mm into the arm's base sphere (§5.5); a test refuses a single box (§9)                                                       |
| Hand-measured numbers when CAD exists; geometry literals in code   | The platform comes from the pinned Demo-Software URDF; every other number is data with provenance in `scripts/mebot_layers.yaml` (§5.1, §6.1)                                                          |
| A loader with no test that would fail if cuRobo ignored its output | Paired GPU tests (§9, Task 1), a cache read-back test, a sphere-consistency test, and the sim's positive controls (§8)                                                                                 |
| Silent truncation or clamping when a scene exceeds the cache       | Overflow is refused, naming every source; nothing is truncated; the cache stays 60 (§4.6)                                                                                                              |
| Merging the person into the platform layer                         | The person is its own file and layer, `excludable: true`; excluding it leaves the platform byte-identical (§6.3)                                                                                       |
| New fields on `PlanToPose`, `PlanToJoints` or `SetWorld`           | None; `rammp_curobo_interfaces/` is untouched (§7.5, §11)                                                                                                                                              |
| An approximation with no coverage number                           | Under-coverage (containment certificate), over-coverage (≤ 15 %) and keep-out clearance (≥ 4 mm) are exact and gated; overlap and max stack are exact and reported; all are committed in `meta` (§5.5) |

## 2. Decisions (owner, 2026-09-30)

1. **"Test it in sim" means a ROS 2 sim stack** (§8), run with `ros2 launch`
   and `ros2 run`. Revised 2026-10-02: the first answer was a Python harness
   without ROS; after trying a spike of each, the owner chose ROS commands and a
   live view over rendered clips.
   - A MuJoCo arm stands in for the arm driver, with the driver's ROS names,
     beside the real planner node, on an isolated ROS domain.
   - The scene holds:
     - the menagerie Gen3 + 2F-85;
     - the MEBot, through MuJoCo's own URDF importer, with all joints deleted;
     - the floor;
     - a seated stand-in.
   - Plans are flown with physics; any contact with the true shapes is reported.
   - A live view streams the sim to a browser.
   - Nothing touches the live stack or the real arm.
1. **Mount = the RAMMS model pose.** Gen3 `base_link` is at mebot (0.505,
   -0.265, **0.376**) with identity rotation. The RAMMS chain sum 0.376381 and
   the unused Demo-Software TF (-0.033657, -0.262443, 0.416116; a UE component
   frame) are recorded as provenance in `scripts/mebot_layers.yaml`.
1. **Platform = the URDF drive base only.** Nothing models the seat, backrest,
   armrest, footplate or mount plate.
1. **Pinned fetch; no URDF copy.** The Demo-Software URDF is fetched by commit
   and sha256 from a cache, the local clone (`~/Demo-Software`) or GitHub raw.
   No copy is committed. The generated layer files are committed.
1. **Workflow and scope.**
   - A new local branch `feature/19-platform-layers` off `origin/dev` (b734928),
     in a separate worktree `~/RAMMP-CuRobo-19`. Local commits only, never
     pushed.
   - The rewritten issue is followed: box layers, a CPU generator with coverage,
     a separate excludable person layer, paired GPU tests, cache 60, no
     interface change.
   - The deviation the owner approved explicitly: **layers named in the planner
     config are composed into every world**, including set_world pushes. The
     issue's scene-level `layers:` are kept as well. The other deviations in §12
     are part of the design approved in decision 8.
   - The MEBot is static with all joints at zero.
1. **Person size: ANSUR II 95th percentile + 0.05 m on every face**, kept
   despite its workspace cost: about 2.2 % of uniform joint samples are free
   with the person layer, and the bench retract is 154.4 mm inside it (§13, risk
   1). Sizes are data: tuning later is one edit to `scripts/mebot_layers.yaml`
   plus `make layers`.
1. **The person layer includes the lower legs and feet**: a third segment,
   `legs`, below the seat pan and in front of the seat (§6.2).
1. **The design is approved**, with 6 and 7. The owner reviews this written spec
   before the implementation plan is written.

## 3. Architecture in one page

```
scripts/mebot_layers.yaml --> scripts/generate_layers.py --> core/rammp_curobo/configs/
  (pin, mount, budgets,        (CPU: numpy + PyYAML + stdlib;    layer_mebot_platform.yaml  16 boxes, excludable: false
   keep-out, ANSUR II,          pinned sha256 fetch; pure-Python  layer_mebot_person.yaml     6 boxes, excludable: true
   seat anchors, floor)         FK; 0.1 mm integer grid)          layer_mebot_floor.yaml      1 box
                                                                  (committed; `make layers` + CI diff-assert)

gen3_mebot.yaml: world: world_mebot.yaml   (a scene whose only content is `layers: [layer_mebot_floor.yaml]`)
                 layers: [layer_mebot_platform.yaml, layer_mebot_person.yaml]   (CONFIG layers)

every world the planner builds = scene boxes     (bare names, obj_ props)
                               + config layers   (<layer>/<box>, persist across set_world)
                               + scene layers    (<layer>/<box>, live and die with the scene)
  `- compose_cuboids(): pure Python, all guards BEFORE cuRobo is imported
       dedupe, validate numbers, refuse duplicates, excludable-only ignore, empty rule, overflow per source
  `- fixed-sphere check (layer boxes vs the arm's joint-independent spheres) --> cuRobo update_world
```

Entry points that compose config layers:

- construction (planner.py:113-118);
- the node's `world` param (planner_node.py:100-103);
- SetWorld (planner_node.py:229-238);
- `update_world(path | Scene | list)` (planner.py:277-305).

A refused push leaves the previous (composed) world loaded, because every guard
runs before `MotionGen.update_world`.

## 4. Layers and composition

### 4.1 Layer file format

Layer files sit flat in `core/rammp_curobo/configs/`. The package-data glob
`configs/*.yaml` does not recurse (core/pyproject.toml:25-26): a probe wheel
built with setuptools 79.0.1 held a flat `configs/*.yaml` file but not one under
`configs/layers/`. The image installs `core` non-editable
(docker/Dockerfile:96-98), so a subdirectory would be missing from its packaged
configs.

A layer file uses the scene obstacle schema (scene.py:21-31) plus three keys:

```yaml
# GENERATED by scripts/generate_layers.py from scripts/mebot_layers.yaml -- do not hand-edit; `make layers`.
layer: platform          # required; ^[a-z][a-z0-9_]*$ ; 'scene' is reserved
excludable: false        # optional YAML bool, default false; only true layers may be dropped by ignore=
meta: {...}              # optional mapping; informational (pin, mount, budgets, coverage); planner ignores it
obstacles:               # required, non-empty list
  - {name: drive_wheel_r, position: [-0.01925, -0.04165, -0.27515], dims: [0.3681, 0.0751, 0.3681]}
```

`scene.load_layer(path)` is strict. Layer files are new, so the strictness
breaks no existing file.

- The file is read as UTF-8 with a SafeLoader subclass that refuses **duplicate
  mapping keys**. Plain PyYAML keeps the last duplicate, so a second
  `obstacles:` would silently drop the first.
- Allowed top-level keys: `layer`, `excludable`, `meta`, `obstacles`. Any other
  key is refused, which also stops layers nesting (`layers:`).
- Allowed entry keys: `name`, `position`, `dims`, `rpy_deg`, `color`.
- `name` is a non-empty string without `/`, unique within the file.
- `position` and `rpy_deg` are 3 finite numbers. `dims` are 3 finite numbers
  greater than 0.

Every refusal is a `ValueError` naming the file, the layer and
`obstacles[i] (name)`. This closes cuRobo's silent-invisible-box cases: a NaN
position, a zero-norm quaternion or a negative dim loads as an enabled box that
can never collide. cuRobo's `normalize_quaternion` divides by the norm with no
epsilon (types/math.py:536), and a CPU probe of v0.7.8 stored all three cases
enabled, with no error or log.

`Layer` has `__slots__` `name, excludable, obstacles, path` and reuses
`Obstacle`.

A layer file is also a valid v1.0.0 scene: v1.0.0's `load_scene` ignores the
extra keys, which makes it a debugging fallback only.

### 4.2 Scene-level `layers:` (the issue's design)

- `load_scene(path)` reads an optional `layers: [ref, ...]`. A value that is not
  a list of strings is a `ValueError`.
- Each ref resolves as an absolute path, then relative to the **scene file's
  directory**, then from the packaged configs. It is **never resolved from
  CWD**, which closes the shadowing risk of config.py:73.
- This uses `resolve_config(ref, relative_to=scene_dir, search_cwd=False)`, a
  new keyword on an internal function.
- A missing ref raises
  `FileNotFoundError("layer 'x.yaml' listed in <scene> not found; tried: ...")`.
- `Scene.__init__(..., layers=())` and `Scene.layers` hold the loaded `Layer`s.
  This is an additive keyword on public API.
- Scene-level layers belong to that scene. A set_world push replaces them with
  the pushed scene's own.
- `load_scene` now opens files with `encoding="utf-8"`. scene.py:125 has no
  encoding, and the kitchen world has non-ASCII comments; the image sets
  `PYTHONUTF8=1` for exactly that reason (docker/Dockerfile:105-111).
- Malformed obstacle and object entries become
  `ValueError("<file>: obstacles[3] (table): missing 'dims'")`. Today a missing
  `dims` is a bare `KeyError` (scene.py:30), and SetWorld replies with
  `str(exc)` (planner_node.py:235-237), so the caller sees only `'dims'`.

### 4.3 Config-level layers (decision 5)

- `PLANNER_DEFAULTS["layers"] = []` at the top level. `gen3.yaml` restates
  `layers: []`, and the drift test (test_offline.py:48-58) is extended.
- `load_planner_config` checks that `layers` is a list of strings.
- It also refuses the two misplacements that would silently drop the chair: a
  top-level `layer:` and `planner: {layers: ...}`. The error is
  `ValueError("<path>: 'layer' is not a planner-config key; did you mean the top-level 'layers:'?")`.
  v1.0.0's `_merge` accepts any key (config.py:85-108). No other unknown-key
  strictness is added.
- `CuRoboPlanner.__init__` resolves each entry with
  `resolve_config(ref, relative_to=config_dir, search_cwd=False)` and loads it
  **before** `load_scene` and any CUDA work. It then composes once (pure) so
  that a bad layer, an overflow or a duplicate fails fast.
- The loaded layers are kept in memory (`self._layers`). Editing a file after
  start-up changes nothing until restart.
- **Composition order:** scene obstacles, scene props, config layers (in config
  order), scene layers (in scene order).
  - Layers are **deduplicated by resolved path**, so a file listed in both the
    config and the scene composes once.
  - The **same layer name from two different files** is refused, naming both
    files.

### 4.4 Naming, padding, `no_pad_names`

- **Keys.** Scene obstacles keep their bare names and props keep `obj_<name>`
  (world.py:30,39). Layer boxes become `<layer>/<box>`. The pedestal no-pad
  exemption, `ignore` of props, and box-opening's `PADDED_TRUE_SURFACES`
  compensation (RAMMP-box-opening
  `src/rammp_box_opening/rammp_box_opening/worlds.py:22-45`) therefore work
  unchanged.
- **Attribution.** A side map `sources = {key: 'scene' | <layer name>}` carries
  attribution. Names are never parsed.
- **Duplicates.** Any duplicate key after namespacing is refused, naming both
  owners. This covers scene-scene, scene-layer (a scene obstacle literally named
  `platform/x`), prop-prop and layer-layer. v1.0.0 kept the last duplicate
  silently (world.py:26-42).
- **Padding.** Every layer box is padded by `world_padding` (0.02 m per side).
  `no_pad_names` keeps matching **scene obstacles only**, exactly as today. No
  layer box needs a no-pad exemption: the generator guarantees the padded layers
  clear the arm's fixed spheres (§5.5).

### 4.5 `ignore=`

`update_world(world, ignore=...)` keeps its signature. For each name:

1. It matches a layer in the composed world (config or scene):
   - if a scene prop has the same raw name, raise
     `ValueError("ignore='person' names both a prop and a layer; rename the prop")`;
   - otherwise, if the layer is `excludable: false`, raise
     `ValueError("ignore='platform': layer 'platform' is not excludable")`;
   - otherwise drop every box of that layer, for this world only.
1. It matches only a prop: drop the prop (unchanged, world.py:34-36).
1. It matches nothing: silently accepted, as in v1.0.0. That is the safe
   direction, since the obstacle stays.

To exclude the person, call `planner.update_world(w, ignore={'person'})`. The
platform boxes stay byte-identical (tested), and the next push without `ignore`
brings the person back.

This is Python-only. SetWorld is `string world` (SetWorld.srv), and adding a
field is MAJOR (CONTRIBUTING.md:132-140).

### 4.6 Guards

All of them run before `from curobo.geom.types import WorldConfig`; today that
import comes first (world.py:50).

- **Numbers.** Every composed box (scene and layers) must have a finite
  position, finite rpy and finite dims greater than 0. The check runs after prop
  bounding and before padding.
- **Empty rule.** The scene's own content must be non-empty whatever layers are
  configured. The content is its obstacles, plus props left after `ignore`, plus
  scene layers left after `ignore`.
  - Without config layers the v1.0.0 text is kept: "collision world is empty;
    refusing to build it — cuRobo would silently keep the previous world on
    update" (test_smoke.py:124-126 matches `empty`).
  - With config layers the message is "collision world is empty apart from the
    configured layers (platform, person): the pushed scene has no obstacles,
    objects or layers — refusing it".
  - This keeps v1.0.0's consumer-typo protection. Otherwise a truncated
    box-opening world would silently become "chair only".
  - The chair's start-up scene is non-empty because it lists the floor layer.
- **Overflow.**
  `"%d collision boxes > collision_cache_obb=%d (scene 41, platform 16, person 6, floor 1); raise the cache setting in the planner config or shrink a layer"`.
  It still contains `collision_cache_obb` (test_smoke.py:127-132). Nothing is
  truncated. The cache stays at **60**.
- **Fixed spheres** (layer boxes only; §7.4). A padded layer box that penetrates
  the arm's joint-independent spheres is refused, because every state would be
  infeasible.
- `make_world_config`'s own default `cache_obb=40` (world.py:47) becomes 60.
  After `WorldConfig.from_dict` the planner asserts exactly `len(cuboids)`
  cuboids and empty `sphere`, `cylinder`, `capsule` and `mesh` lists. The shape
  check mirrors world.py:131-137 of 689eb2d, a local-only commit on the earlier
  branch feature/19-obstacle-avoidance in `~/RAMMP-CuRobo`; the count check is
  new.

**Box budget:**

| World                                                                                   | Boxes               |
| --------------------------------------------------------------------------------------- | ------------------- |
| Chair start-up: floor + platform + person                                               | 1 + 16 + 6 = **23** |
| box-opening worst case: 8 + 22 (the push replaces the floor)                            | 30                  |
| ADL: 2 + 2k (pedestal, placeholder table, up to 2 boxes for each of k moved parts) + 22 | 24 + 2k             |
| Sim kitchen: 26 + 22                                                                    | **48**              |

All fit in the 60-box cache (ADL up to k = 18, which is exactly 60). A
design-phase scan of 566 consumer worlds found a maximum of 26 boxes (the
kitchen). The 566 are the 563 worlds on this Jetson in
`~/.ros/rammp_box_opening/worlds/` (562 box-opening, 1 ADL moved-parts; 3 to 8
boxes), the kitchen, the bench (`world_real_bench.yaml`) and box-opening's
`world_bench.yaml`.

### 4.7 set_world semantics, old images, consumers

- **Pushes.** A push still **replaces the scene** (planner.py:277-305). Config
  layers persist. The pushed file's own `layers:` apply to that world only. So
  box-opening's push before every plan (RAMMP-box-opening
  `src/rammp_box_opening/rammp_box_opening/primitives/core.py:169`) and ADL's
  reset to `world_real_bench.yaml` at every task start (RAMMP-generalized
  `rammp_adl/sheppy_backend.py:501-505`) cannot drop the chair.
- **Consumers on `gen3.yaml` / `gen3_real.yaml` (`layers: []`)** get
  byte-identical worlds. This is tested against the unchanged v1.0.0
  `world_cuboids`. Only the message strings grow: failure details (§7.4) and the
  SetWorld success message (§7.5).
- **New refusals** (duplicates, non-finite or non-positive boxes, malformed
  entries) reject none of the 566 known consumer worlds (the same scan: 0
  duplicates, 0 bad numbers, 0 `/` names, 0 props named like a layer).
- **1.0.0 image.**
  - `gen3_mebot.yaml` and the layer files exist only in images ≥ 1.1.0, so
    `config:=gen3_mebot.yaml` on 1.0.0 fails with "config not found".
  - If someone mounts the files into a 1.0.0 container anyway, its `load_scene`
    sees `world_mebot.yaml` with **no inline obstacles or objects** (the only
    keys v1.0.0 turns into collision boxes; it ignores `layers:`,
    scene.py:127-132) and refuses the empty world at construction
    (world.py:56-60). It does not plan without the chair.
- **Deployment** (`config:=gen3_mebot.yaml` in a manifest) lands in the consumer
  repos. This repo ships no sheppy fragment.

## 5. The generator (`scripts/generate_layers.py`)

It is a build tool, like `scripts/bake_robot_config.py`, and is not public
surface (CONTRIBUTING.md:127-128). It keeps network-fetch code out of the
runtime wheel. Tests and the sim load it with `importlib` from
`<repo>/scripts/`, which is also present in the image
(`COPY . /opt/rammp_curobo_src`, docker/Dockerfile:96).

Dependencies: numpy, PyYAML, stdlib (`xml.etree`, `hashlib`, `subprocess`,
`urllib.request`) and `rammp_curobo.config` (for the packaged robot YAML path).
No yourdfpy, trimesh, cuRobo or MuJoCo, because CI installs only numpy, PyYAML
and pytest (build.yml:59-60).

### 5.1 Inputs: `scripts/mebot_layers.yaml` (all geometry is data, with provenance)

- **source**
  - `repo: rammp-org/Demo-Software`
  - `commit: 80908b9f73f02e094106c0b82640a23c3cd6f41c`. This is the commit that
    added the file on `origin/dev`, which is `origin/HEAD`.
  - `path: core/rammp_prototype_description/urdf/rammp_prototype_base.urdf`
  - `sha256: 201ec24aeeff868557b13211eaa4571b8f11cd6709795537098bafaef032c0ec`
  - `git_blob: 8f571f54`
  - `same_blob_at: [59d5aef (authoring commit; only on origin/feature/106-drive-base-publisher), 0a94cce (origin/main, Release 1.0)]`
  - Verified: `git show <c>:<path> | sha256sum` gives the same hash at all three
    commits and for the local copy `~/mebot.urdf`.
- **mount**
  - `arm_base_in_mebot: {position: [0.505, -0.265, 0.376], rpy_deg: [0, 0, 0]}`
  - `chain`, from the RAMMS `BP_Mebot_Ramms` blueprint: seat_base (0.345, 0,
    0.256381) + rail_r (0.020, -0.265, 0.085) + mount_arm (0.140, 0, 0.020) +
    ArmSkMesh (0, 0, 0.015) = (0.505, -0.265, 0.376381). Decision 2 uses z
    0.376.
  - `alt_chain`, from the RAMMS `BP_Mebot_Mujoco` blueprint: the arm sits at the
    mount_arm origin, (0.505, -0.265, 0.361381), without ArmSkMesh's +0.015.
  - `not_used: demo_software_tf (-0.033657, -0.262443, 0.416116)`. It matches
    the chain in x/y to 1e-6 only in the UE VehicleMesh component frame (root
    bone at (-0.538656, 0.002557, 0.028235)). Applied to this URDF it would put
    the arm 0.539 m further back. In the URDF frame it is (0.505, -0.265,
    0.388). Its +1.15 cm in z is unexplained.
  - A non-zero `rpy_deg` is refused, because the decomposition is axis-aligned
    in `base_link`.
- **grid_m** `0.0001`. **padding_assumed_m** `0.02`, which must equal the chair
  config's `world_padding` (tested). **keep_out_margin_m** `0.005`.
- **keep_out**
  - `robot_config: robot_gen3_2f85.yaml` (packaged; spheres and
    `collision_sphere_buffer` are read from it).
  - `shoulder_link_origin: {xyz: [0, 0, 0.15643], rpy: [3.1416, 0, 0]}`, from
    cuRobo's `kinova_gen3_7dof.urdf:55` (joint_1).
- **platform**: `max_boxes: 16`, `max_over_pct: 15.0`.
- **person**: the ANSUR II values, the margin and the seat anchors (§6.1).
- **floor**: `half_extent_m: 1.5`, `thickness_m: 0.1`.
- **outputs**: the three file paths.

### 5.2 Pinned fetch

Order of sources:

1. The cache `${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/cad/<sha256>.urdf`,
   re-hashed on every read. A mismatch is ignored and noted.
1. The local clone (`--clone`, `$RAMMP_DEMO_SOFTWARE`, or `~/Demo-Software`),
   via `git -C <clone> cat-file blob <commit>:<path>`.
1. `https://raw.githubusercontent.com/<repo>/<commit>/<path>` (urllib, 30 s
   timeout). Forbidden by `--offline`. Demo-Software is public, so no token is
   needed.

The sha256 is verified on the **raw bytes**. The file is CRLF and never enters
git here, so pre-commit's mixed-line-ending hook can never alter the hashed
bytes. A mismatch from a clone or URL is an error naming the origin and both
hashes. Verified bytes are written to the cache atomically (temp file +
`os.replace`). When every source fails, the error lists every source tried.

### 5.3 URDF parsing and FK (no yourdfpy)

- `xml.etree`. Exactly one root link. Joint types fixed, revolute, continuous
  and prismatic.
- `<mimic>` is refused. A movable joint whose limits exclude 0 is refused: all
  joints are held at zero.
- Geometry `<box>` and `<cylinder>` only. Mesh, sphere, capsule or empty
  geometry is refused, naming the link ("mesh is the stretch goal"). Sizes must
  be finite and > 0.
- URDF fixed-axis rpy is `Rz * Ry * Rx`. FK uses pure-Python floats (no BLAS).
  `math.sin` and `math.cos` come from the platform libm and can differ by an ulp
  between amd64 and arm64. The 1e-9 outward snap bias (§5.4) absorbs such
  differences; the ±1e-12 noise test (§9) and the CI regeneration on both
  architectures (§5.7) verify it.
- Primitives are placed in `base_link` by translating by -mount.
- **Bounding boxes:**
  - box: `sum_j |R_ij| * s_j / 2`, which absorbs the -1.6292068e-07 rad CAD yaw
    on every origin;
  - cylinder: `|a_i| * L / 2 + r * sqrt(1 - a_i^2)` per axis `i`, where `a` is
    the cylinder axis. That is exact circumscription, and for the y-axis wheels
    it is `[2r, L, 2r]`.
  - cuRobo's `get_obb_world` (geom/types.py:1096) is never used. It boxes each
    shape's trimesh (`get_cuboid`, geom/types.py:99-132), and a cylinder's
    trimesh is a 32-gon (geom/types.py:341), so its box is measured across the
    flats: at r 0.184 that is 0.36623 m against 0.368 m, 0.89 mm short per side.
    It also needs cuRobo, with CUDA as its default tensor device.

### 5.4 Grid and decomposition

1. **Snap once, outward**, onto the 0.1 mm grid:
   `lo = floor((v - 1e-9) / 1e-4)`, `hi = ceil((v + 1e-9) / 1e-4)`. Everything
   after this is integer "ticks".
1. **Absorb** boxes contained in another box. For the MEBot: 46 primitives
   become 35 boxes, with 0 over-coverage.
1. **Greedy pairwise merge.**
   - While the count exceeds `max_boxes`, merge the pair minimising
     `cost = vol(bbox(a, b)) - (vol(a) + vol(b) - vol(a & b))` (exact integers).
   - Ties are broken by `(cost, vol(bbox), i, j)` over a list sorted by
     primitive name.
   - A candidate whose padded box comes within `keep_out_margin_m` of a fixed
     arm sphere is skipped.
   - After each merge, absorb again. If the budget cannot be met without
     touching the keep-out, raise.
   - The **pairwise** objective is chosen over one that minimises the empty
     volume a merge adds to the whole layer, where overlap with other boxes
     costs nothing. Measured in the design phase at 16 boxes (same absorb,
     tie-break and keep-out skip), pairwise gives self-overlap 0.026 m³ with
     max stack 3 and 10.9 % over-coverage; the whole-layer objective gives 0.137
     m³ with max stack 5 and 6.6 % over-coverage. Pairwise is chosen because
     its overlap, which cuRobo sums in cost (`sum_collisions_ = true`,
     sphere_obb_kernel.cu:2711), is about 5× smaller, and its 10.9 % is within
     the 15 % budget.
1. **Naming.** Each box is named after its largest-volume member link (ties
   broken by name). The other members are listed in a trailing YAML comment. A
   `members:` key would be refused by the strict loader.

### 5.5 Coverage metrics and keep-out

- **under_m3 = 0 by certificate.** Every primitive's exact solid lies inside its
  exact AABB, which lies inside its outward-snapped AABB, which lies inside the
  box that absorbed it. The last containment is asserted in integers. Tests
  re-verify independently (§9).
- **over_m3** =
  `vol(union of the output boxes) - vol(union of the snapped primitive AABBs)`.
  It is computed exactly, by coordinate compression on integer coordinates with
  int64 cell volumes. **over_pct** = `100 * over_m3 / vol(union of the AABBs)`,
  formatted by integer round-half-up to 0.1. This is the gated metric.
- **cylinder_boxing_m3** = `sum (4 - pi) r^2 L` over the 6 wheels, analytic, as
  information (the gap between the AABB union and the true shapes). No
  lattice-derived number enters a diff-asserted file.
- **overlap_m3** = the sum of the box volumes minus the volume of their union.
  **max_stack** = the maximum number of boxes covering one point.
- **keep_out_clearance_m** = the minimum over boxes and fixed spheres of
  (distance from sphere centre to the padded box) - r. A centre inside the box
  counts as minus the depth to the nearest face.
  - The **fixed spheres** are the `base_link` spheres plus the `shoulder_link`
    spheres mapped through joint_1's origin, all with the 0.005 buffer.
  - The generator refuses a `shoulder_link` sphere with a non-zero local x or y,
    because it would not be joint-independent.
  - Column: z from -0.015 to 0.383 in `base_link`, radius + buffer up to 0.080.

**Gates.** On any failure the generator exits non-zero and writes nothing:

- sha256;
- unsupported input;
- containment;
- more boxes than `max_boxes`;
- `over_pct` above `max_over_pct`;
- any layer's keep-out clearance below 4 mm;
- duplicate names;
- non-ASCII output.

**Numbers** (measured with a design-phase CPU prototype of the generator):

- 46 primitives (40 boxes, 6 cylinders). `base_link` AABB x -0.8475 .. 0.3581, y
  -0.0792 .. 0.6041, z -0.4594 .. -0.0272.
- Union of the primitives' AABBs: 0.137922 m³. Chosen **16 boxes**:
  - layer 0.152908 m³;
  - **over 0.014985 m³ = 10.9 %** against the bounding boxes, about 15.0 %
    against the true shapes (0.133001 m³ on a 1 mm lattice);
  - cylinder boxing 0.005440 m³;
  - overlap 0.026491 m³, max stack 3;
  - keep-out clearance **53.4 mm**.
- Budget curve (over-coverage against the bounding boxes):

| boxes  | over-coverage |
| ------ | ------------- |
| 35     | 0 %           |
| 30     | 0.2 %         |
| 24     | 1.8 %         |
| 20     | 4.8 %         |
| 18     | 7.7 %         |
| **16** | **10.9 %**    |
| 14     | 16.0 %        |
| 12     | 19.5 %        |
| 10     | 27.2 %        |
| 8      | 32.5 %        |

- **One bounding box**: 158.2 % over, and its padded box intrudes on the base
  sphere by 7.9 mm, so every state would be infeasible.
- Build: 0.1 s after the fetch.
- **Determinism**: the rerun is byte-identical, and so are runs with every FK
  bound perturbed by ±1e-12 relative.

### 5.6 Outputs and determinism

- `layer_mebot_platform.yaml` (16 boxes, `excludable: false`),
  `layer_mebot_person.yaml` (6 boxes, `excludable: true`),
  `layer_mebot_floor.yaml` (1 box).
- The floor box is a 3.0 × 3.0 × 0.1 m slab under the arm base. Its top is the
  lowest drive-wheel point (-0.459359) **rounded up** (outward for the floor
  solid) to -0.4593. It is not part of the platform. It is the chair's cell
  stand-in until #21 and is used only as a scene-level layer.
- Text is written by hand, not with `yaml.dump`: fixed key order, obstacles
  sorted by name, LF endings, ASCII, a GENERATED header.
- Positions are printed exactly from integers: centre = (lo + hi) in 0.05 mm
  units, 5 decimals. Dims are hi - lo in 0.1 mm units, 4 decimals.
- `meta` holds the pin, mount, grid, assumed padding, budgets, coverage numbers
  and keep-out clearance (all integer-derived or closed-form). The diff-assert
  therefore also freezes the numbers.

### 5.7 CLI, make, CI

- `python3 scripts/generate_layers.py [--data scripts/mebot_layers.yaml] [--check] [--offline] [--clone PATH] [--cache DIR]`.
  - It prints the report: source, per-box members, budget point, coverage,
    keep-out, and person volumes.
  - It writes the three files atomically only after every gate passes.
  - `--check` regenerates in memory and exits 1 with a unified diff on any
    difference.
- `make layers` runs the generator, then `git diff --exit-code` on the three
  outputs. This mirrors `make bake` (Makefile:47-52).
- CI:
  - `core/tests/test_layergen.py::test_committed_layers_regenerate_byte_identical`
    runs inside the existing `pytest core/tests` step on amd64 and arm64
    (build.yml:40-75). It fetches from GitHub raw on the runner.
  - It **skips locally** when no source is available, and **fails** when
    `CI=true` (GitHub sets it), so CI can never go green without regenerating.
  - The existing guard step (build.yml:69-75) relies on pytest exiting 5 when
    the file it names collects nothing. One invocation naming several files
    exits 5 only if all of them are empty, so the step runs one unpiped
    `python3 -m pytest core/tests/<file> -q` per offline test file
    (`test_offline.py`, `test_layers.py`, `test_layergen.py`); GitHub's default
    `bash -e` stops the step at the first failure.

## 6. The person layer

### 6.1 Parameters and anchors (data, with sources)

**Anthropometry.** ANSUR II 2012 (Gordon et al. 2014), via The Ergonomics Center
(NCSU), "Anthropometric Data for U.S. Adults", 11/2024. Values are the 95th
percentile of the larger sex, stored in inches with their table row and
converted at 0.0254 m/in:

| Parameter                                       | Table row | Larger sex        | Inches | Metres | Sets                       |
| ----------------------------------------------- | --------- | ----------------- | ------ | ------ | -------------------------- |
| sitting height                                  | 8         | men               | 38.46  | 0.977  | torso_head top             |
| sitting elbow height                            | 11        | men               | 11.46  | 0.291  | lap top                    |
| thigh clearance                                 | 12        | men               | 8.15   | 0.207  | lap top                    |
| buttock-knee length                             | 13        | men               | 26.34  | 0.669  | lap front, legs front      |
| buttock-popliteal length                        | 14        | men               | 21.57  | 0.548  | legs back                  |
| popliteal height                                | 16        | men               | 18.54  | 0.471  | legs bottom                |
| shoulder breadth (bideltoid)                    | 17        | men               | 22.32  | 0.567  | torso_head and lap width   |
| hip breadth                                     | 19        | women (men 15.24) | 15.75  | 0.400  | legs width                 |
| chest depth                                     | 20        | men               | 11.73  | 0.298  | torso_head front, lap back |
| abdominal depth                                 | 21        | men               | 12.60  | 0.320  | torso_head front, lap back |
| hand breadth (forearm and hand thickness proxy) | 29        | men               | 3.78   | 0.096  | lap top                    |
| foot length                                     | 30        | men               | 11.54  | 0.293  | legs front                 |

**Margin.** `margin_m: 0.05` on every face, a deliberate over-size (decision 6).
The planner adds 0.02 of padding, so about 7 cm is effective.

**Seat anchors.** These come from the RAMMS `rammp_seat` simple-collision hulls
(ramms-assets), in the mebot frame. The asset's mesh payload is
Oodle-compressed; only its bounds, sockets and collision hulls can be decoded.
The anchors are transcribed with provenance: the generator cannot read uassets,
and seat modelling is out of scope (decision 3).

| Anchor                | Value                                                                              | Source                     |
| --------------------- | ---------------------------------------------------------------------------------- | -------------------------- |
| pan top z             | 0.4705                                                                             | hull1 max z                |
| backrest face x       | range **0.1587 .. 0.2733**; the true face is unknown, so the layer spans the range | hull1 min x to hull0 max x |
| backrest rear x       | -0.1502                                                                            | hull3 min x                |
| occupant centreline y | 0                                                                                  | seat_base socket y         |

In `base_link` the occupant centreline is at y +0.265 and the pan top at z
+0.0945. The occupant faces +x.

### 6.2 Geometry: three segments, carved around the arm's column → 6 boxes

Segments are built in the mebot frame (m is the margin), moved into `base_link`
by -mount, and snapped outward:

- **torso_head**
  - x from (backrest rear - m) to chest_front = (backrest-face max +
    max(abdominal depth, chest depth) + m);
  - y: centreline ± (bideltoid/2 + m);
  - z from (pan - m) to (pan + sitting height + m).
- **lap** (thighs, and forearms in the lap or on the armrests)
  - x from chest_front to (backrest-face max + buttock-knee + m);
  - y: centreline ± (bideltoid/2 + m);
  - z from (pan - m) to (pan + max(thigh clearance, sitting elbow height + hand
    breadth) + m).
- **legs** (lower legs and feet, below the pan and in front of the seat;
  decision 7)
  - x from (backrest-face min + buttock-popliteal - m), the back of the knee
    with the backrest at its rearmost, to (backrest-face max + buttock-knee +
    foot length + m), the front of the knee plus a foot with the backrest at its
    foremost;
  - y: centreline ± (hip breadth/2 + m);
  - z from (pan - popliteal height - m) to (pan - m), the level of the
    torso_head and lap bottom faces. It abuts lap's bottom face; it starts in
    front of torso_head, and in front of the lap (x > 0.4874) its top is open.
  - Its bottom is mebot z -0.0505 (`base_link` -0.4265), above the ground at
    mebot -0.0834: it ends 3.3 cm above the floor layer's top (-0.4593).

**Carve.** The column box is the union of the fixed spheres' AABBs, plus
padding, plus the 5 mm margin, snapped outward: x, y ±0.1051, z -0.0401 ..
0.4085 in `base_link`. It is subtracted from each segment in integer ticks, as
box-minus-box slabs in x, then y, then z. A segment left in several pieces names
them `<segment>_<k>`.

- This is physically forced. A person cannot occupy the arm's base, and any
  padded box there makes every state INVALID_START.
- The carve stops at the column top. A full-height corridor would under-size the
  person exactly where the arm works.
- The margin is 5 mm (`keep_out_margin_m`). It leaves the padded person 5.1 mm
  clear of the fixed spheres; a 1 mm margin would fail the 4 mm keep-out gate.
- Only torso_head meets the column, and it becomes 4 boxes. lap starts at x
  0.1383 and legs at x 0.1515, both in front of the column (x ≤ 0.1051), so the
  carve, which runs on every segment, leaves them whole.

**Result** (`base_link`, unpadded):

| Box          | x                  | y                 | z                 |
| ------------ | ------------------ | ----------------- | ----------------- |
| lap          | 0.1383 .. 0.4874   | -0.0685 .. 0.5985 | 0.0444 .. 0.5316  |
| legs         | 0.1515 .. 0.7805   | 0.0149 .. 0.5151  | -0.4265 .. 0.0446 |
| torso_head_0 | -0.7053 .. -0.1051 | -0.0685 .. 0.5985 | 0.0444 .. 1.1214  |
| torso_head_1 | 0.1051 .. 0.1384   | -0.0685 .. 0.5985 | 0.0444 .. 1.1214  |
| torso_head_2 | -0.1051 .. 0.1051  | 0.1051 .. 0.5985  | 0.0444 .. 1.1214  |
| torso_head_3 | -0.1051 .. 0.1051  | -0.0685 .. 0.1051 | 0.4085 .. 1.1214  |

The same boxes as the generator writes them (header and `meta` omitted):

```yaml
layer: person
excludable: true
obstacles:
  - {name: lap, position: [0.31285, 0.26500, 0.28800], dims: [0.3491, 0.6670, 0.4872]}
  - {name: legs, position: [0.46600, 0.26500, -0.19095], dims: [0.6290, 0.5002, 0.4711]}
  - {name: torso_head_0, position: [-0.40520, 0.26500, 0.58290], dims: [0.6002, 0.6670, 1.0770]}
  - {name: torso_head_1, position: [0.12175, 0.26500, 0.58290], dims: [0.0333, 0.6670, 1.0770]}
  - {name: torso_head_2, position: [0.00000, 0.35180, 0.58290], dims: [0.2102, 0.4934, 1.0770]}
  - {name: torso_head_3, position: [0.00000, 0.01830, 0.76495], dims: [0.2102, 0.1736, 0.7129]}
```

- Envelope 0.867678 m³, layer 0.854392 m³, carved 0.013286 m³.
- Padded clearance to the fixed spheres **5.1 mm**.
- Self-overlap 0.000066 m³: the outward snap overlaps lap and torso_head by 0.1
  mm, and legs and lap by 0.2 mm.
- Deterministic under ±1e-12 relative FK noise.

**Why not the issue's "one conservative volume".** The occupant's centreline is
0.265 m from the arm axis, and a 95th-percentile half-bideltoid is 0.283 m. Any
single box covering the occupant therefore contains the arm's base spheres,
which makes every state infeasible. A single box would also swallow the space
the arm works in: in front of the chest above the lap, and above the lower legs
in front of the knees. So the design uses three segments that follow the seated
body, plus the carve.

### 6.3 Exclusion

The person is its own file and layer (`excludable: true`) and is never merged
into the platform. `update_world(w, ignore={'person'})` drops exactly the
`person/*` boxes. `ignore={'platform'}` raises.

- Offline test: the platform keys, dims and poses are identical with and without
  the ignore.
- GPU test: the retract is invalid naming `person/`, valid after the ignore, and
  a platform-colliding state still fails naming `platform/`.

### 6.4 Coverage against an independent stand-in (sim, and the numbers that ground it)

The stand-in is the stock DeepMind humanoid, consumed by path from
`/opt/ros/humble/opt/mujoco_vendor/model/humanoid/humanoid.xml`. Its shape is
independent of the layer parameters. Only its overall scale comes from ANSUR.

- Posed seated: hips and knees at -90°.
- Scaled uniformly to the 95th-percentile sitting height: ×1.229, or ×1.254 when
  leaning.
- Placed with the buttock on the pan top, against the backrest face, centred.
- Postures (joint angles as data in `sim/mebot_sim/config/scenarios.yaml`):
  - `upright_lap`: hands on the thighs;
  - `lean_lap`: 15° forward lean;
  - `forearms_forward`: upper arm vertical, forearm forward.
- The `upright_lap` and `forearms_forward` arm angles come from seeded posture
  searches.
- Sampled volumetrically on a 6 mm grid. Each point of the body, lower legs and
  feet included, is classified as covered, inside the arm's carved column, or
  **gap**.

Measured with the design-phase prototypes, 3 postures × backrest face 0.1587 and
0.2733:

- **Gap against the padded person layer (what the planner uses): 0 m³ in all 6
  cases.** 99.11 to 99.86 % of the 0.076 to 0.0805 m³ body is covered; the rest
  (about 0.0001 to 0.0007 m³) is inside the arm's column, where the arm's own
  base is.
- Unpadded: 0, except slivers of the feet (0.00001 to 0.00007 m³; backrest face
  0.1587, all three postures) and `lean_lap` at 0.2733 (0.0001 m³; the head
  passes the chest-front face). Both lie inside the 0.02 m padding. Against the
  unpadded layer, 0.0006 to 0.0018 m³ lies inside the arm's column.

**Physical finding.** At the bench retract, the menagerie Gen3 meshes clear the
stand-in as follows:

- backrest face at 0.1587: 15.5 to 28.5 mm;
- backrest face at 0.2733: from -7.0 mm (`upright_lap`: the arm's `half_arm_1`
  interpenetrates the stand-in's right forearm) and -2.6 mm (`lean_lap`) up to
  +25.1 mm.

So a 95th-percentile occupant and the arm's bench retract are physically
incompatible if the backrest face is at the front of its CAD range. The sim
scene therefore places the stand-in at 0.1587. Measuring the backrest face on
the real chair is a follow-up (§15).

## 7. Planner changes (`core/rammp_curobo`)

### 7.1 `scene.py`

- `_StrictLoader`: a SafeLoader that refuses duplicate keys.
- `class Layer`.
- `load_layer(path)` (§4.1).
- `load_layers(refs, relative_to, listed_in)`: list type check, resolution via
  `resolve_config(ref, relative_to, search_cwd=False)`, missing-file message,
  dedupe by path, same name from two paths refused.
- `Scene(..., layers=())` and `Scene.layers`.
- `load_scene` reads `layers:` relative to the scene file, uses UTF-8, and wraps
  entry errors with file and entry.
- `scene_from_obstacles`, `Obstacle`, `SceneObject`, `Target` and `__all__` are
  unchanged.
- Still stdlib + PyYAML only.

### 7.2 `config.py`

- `PLANNER_DEFAULTS["layers"] = []`.
- `resolve_config(name_or_path, relative_to=None, search_cwd=True)`.
- `load_planner_config`: `layers` type check, and refusal of a top-level `layer`
  and of `planner.layers`.

### 7.3 `world.py` (pure except `to_world_config`)

- `world_cuboids(...)` is **unchanged**. It is v1.0.0 behaviour, used by
  `test_offline.py` and as the equivalence oracle.
- `compose_cuboids(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset(), cache_obb=60, *, layers=()) -> (cuboids, sources)`.
  It implements §4.3 to §4.6.
- `to_world_config(cuboids)` imports cuRobo, calls
  `WorldConfig.from_dict({"cuboid": ...})`, and asserts a boxes-only result with
  the right count.
- `make_world_config(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset(), cache_obb=60, *, layers=())`
  keeps its v1.0.0 return type (`WorldConfig`) and is compose +
  `to_world_config`. Its default is now 60. `layers` is keyword-only in both,
  so a v1.0.0-style positional call such as `make_world_config(scene, 0.03)`
  keeps its meaning.
- `sphere_box_penetration(spheres, cuboids) -> {key: depth_m}` (numpy): spheres
  are (N, 4) with radii including the buffer; boxes are oriented
  `[x, y, z, qw, qx, qy, qz]` with padded dims. It is the same test as cuRobo's
  feasibility check at activation 0.
- `point_box_distance(point, cuboids) -> {key: signed_m}`.
- `summarize(sources, config_layer_names) -> "23 boxes: scene 0 | config layers: platform 16, person 6 | scene layers: floor 1"`.
- The module docstring's trap list gains: duplicates collapse, NaN,
  zero-quaternion or negative dims are invisible, and an unknown top-level key
  gives an empty world.

### 7.4 `planner.py`

**`__init__`:**

1. Resolve robot and world.
1. `self._layers = load_layers(config["layers"], relative_to=config_dir, listed_in="planner config in %s" % config_dir)`.
   Refs resolve without CWD. Log each: name, boxes, excludable, path.
1. `load_scene`.
1. `compose_cuboids(self._scene, ..., layers=self._layers)`: a pure fail-fast
   before any GPU work.
1. `_init_curobo(warmup)`.

**`_init_curobo`**, in this order:

1. `full = compose_cuboids(scene, ..., layers=self._layers)`.
1. `warm_first = bool(self._layers) and warmup`.
1. `init = compose_cuboids(scene, ..., layers=())` if `warm_first`, else
   `full`. The scene-only world still contains the scene's own layers (the
   floor).
1. Build `MotionGen` with `to_world_config(init)`, keeping the existing checker,
   cache, activation and joint-name checks.
1. `self._fixed_spheres`: FK spheres
   (`kinematics.get_state(...).link_spheres_tensor`; radii include the buffer,
   cuda_robot_generator.py:759-771) at the retract and two seeded in-limit
   configurations. Keep the spheres with r > 0 whose centres agree within 1e-6
   m: the base and on-axis shoulder spheres.
1. `self._check_fixed_spheres(full)`. Any padded **layer** box that penetrates
   them raises a ValueError naming the deepest intrusion, for example (at
   `world_padding` 0.030 all four torso_head boxes intrude, torso_head_3
   deepest):
   `ValueError("layer box person/torso_head_3 intrudes 4.9 mm into the arm's fixed base/shoulder spheres (world_padding 0.030 m): every start state would be infeasible; the layers were generated for 0.02 m — regenerate them or restore world_padding")`.
   Scene boxes are not checked, so v1.0.0 behaviour is unchanged.
1. `warmup()` (logged: "warming up in the scene-only world (1 box); config
   layers load after warm-up").
1. If `warm_first`: `self._motion_gen.update_world(to_world_config(full))`.

This ordering is safe in v0.7.8 because:

- the OBB cache is allocated at `collision_cache["obb"]` (60) at construction by
  `_init_cache` (geom/sdf/world.py:716, 724-727; the mesh checker delegates,
  world_mesh.py:72-81);
- `update_world` writes in place with `fix_cache_reference=use_cuda_graph=True`
  (motion_gen.py:1825-1837);
- the kernels read the box count from device memory (sphere_obb_kernel.cu:2453),
  so graphs captured during warm-up stay valid.

It is needed because warm-up plans from the retract (motion_gen.py:1896-1945),
and `plan_single_js` / `plan_single` return INVALID_START before any solver runs
(motion_gen.py:2105-2113, 3007-3015). With the person layer the retract is 154
mm inside, so warm-up would exercise nothing.

Afterwards:

- store `self._world_cuboids`, `self._world_sources` and
  `self._warm_boxes = len(init)`;
- log `world_summary()`;
- `check_state_valid(self.retract_pose)`. If it is invalid, log a WARNING: "the
  IK seed (retract_config) is not a valid state in this world: ... plans still
  work from valid starts; a chair seed is #20".

**`update_world(world, ignore=())`:**

1. Parse as today (planner.py:286-291).
1. `cuboids, sources = compose_cuboids(scene, ..., ignore=..., layers=self._layers)`.
1. `_check_fixed_spheres`.
1. `self._motion_gen.update_world(to_world_config(cuboids))`.
1. Only then commit `self._scene`, `self._world_cuboids` and
   `self._world_sources`.
1. Log the summary and any excluded layers (INFO).

**New read-only API (additive, MINOR):**

- `world_boxes`: a list of
  `{name, layer, dims (padded), pose [x, y, z, qw, qx, qy, qz]}` for the
  installed world, used by the node, the tests and the sim overlays;
- `world_summary()`.

**`_colliding_boxes(q)`:**

1. FK the buffered spheres of one state.
1. Drop spheres with r ≤ 0.
1. `sphere_box_penetration` against `self._world_cuboids`.
1. Return the hits sorted by depth: `[(key, layer, depth_m)]`.

It never touches cuRobo's world state. Attributing by toggling obstacles in
cuRobo (`enable_obstacle`, geom/sdf/world.py:978-1014) would mutate the live
cache and miss collisions that involve several layers.

**`check_state_valid(q)`.** When infeasible, the detail becomes, for the bench
retract on the chair,
`"constraint violation at state; collides with person/lap (154 mm), person/torso_head_3 (134 mm), person/torso_head_1 (89 mm)"`.
The v1.0.0 prefix is kept, and the suffix is added only when boxes are hit.

**`_finish(result, t0, q_goal=None, start=None, goal_point=None)`.**
`plan_to_pose` passes `start` and the corrected goal `xyz`. `plan_to_joints(js)`
passes `start` and `q_goal`. On failure the error is `_explain(status)`
(unchanged) plus a note that **never raises**; any exception becomes
`"(attribution unavailable: ...)"`. The note:

- **INVALID_START:** `" [start collides with person/lap (154 mm), ...]"`, or
  `" [start: no world box penetrated — self-collision or joint limits]"`.
- **Joint goal:** `" [goal collides with ...]"` when the goal state penetrates
  boxes.
- **IK failure on a pose goal:**
  `" [goal point inside person/lap; within 5 cm of platform/...] (heuristic: tool point only)"`.
- **Always:**
  `" [world: 23 boxes: scene 0 | config layers: platform 16, person 6 | scene layers: floor 1]"`.

`PlanResult.error` reaches ROS clients as `"%s: %s"` (planner_node.py:222-225),
so no IDL change is needed.

### 7.5 `rammp_curobo_ros/rammp_curobo_ros/planner_node.py` (strings only)

- After construction and after the optional `world` override, log
  `Collision world: <world_summary()>` through the ROS logger. Core INFO logs
  are not routed there.
- The SetWorld success message becomes
  `"world set to %s (%s)" % (request.world, planner.world_summary())`.
  box-opening ignores it on success (RAMMP-box-opening
  `src/rammp_box_opening/rammp_box_opening/primitives/core.py:169-171`) and ADL
  only records it.
- The ready log includes the summary. For an operator this is the visible proof
  that the chair config is active.
- No parameter, topic, service or IDL change. `rammp_curobo_interfaces/` is
  untouched.

### 7.6 Configs

- `gen3.yaml`: `layers: []`, with a comment.
- New `gen3_mebot.yaml`: `robot: robot_gen3_2f85.yaml`,
  `world: world_mebot.yaml`,
  `layers: [layer_mebot_platform.yaml, layer_mebot_person.yaml]`. Every knob
  stays default, in particular `world_padding 0.02` and
  `no_pad_names [pedestal]`. The header explains persistence, the invalid bench
  retract, and `ignore={'person'}`.
- New `world_mebot.yaml` (hand-written, no geometry): `base_frame: base_link`,
  `layers: [layer_mebot_floor.yaml]`, `obstacles: []`. Its comment says:
  - the chair and person are config layers;
  - this scene carries only the floor, which a push replaces;
  - it has no inline obstacles, so a v1.0.0 planner refuses it instead of
    planning without the chair.
- Three generated layer files (§5.6).

## 8. The sim: a ROS 2 stack on its own domain (`sim/mebot_sim/`, decision 1)

The sim is tested the way a consumer uses the planner: over ROS. A MuJoCo arm
stands in for the arm driver, the real planner node plans, and test commands are
ordinary `ros2 launch` and `ros2 run`.

**What runs.**

| Node                    | What it is                                                                                                                                                                                                                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/mebot_sim`            | The MuJoCo arm with physics, the exact MEBot, the seated stand-in and the floor. It serves the driver's names: `/joint_states` (best effort, 50 Hz) and `/execute_joint_trajectory` (`rammp_arm_interfaces/ExecuteJointTrajectory`; cancel returns PREEMPTED and the arm holds). It also serves the live view. |
| `/rammp_curobo`         | The planner node of this repo with `config:=gen3_mebot.yaml`: what a consumer gets on the chair.                                                                                                                                                                                                               |
| `/rammp_curobo_control` | A second planner node with `config:=gen3.yaml` (`layers: []`). The tests push it scenes that list only some layers (`[floor, platform]`, `[floor]`), which is the positive control: the same request planned without a layer.                                                                                  |

`/mebot_sim` also offers `/mebot_sim/contacts` (`std_msgs/String`, JSON: contacts
now and the worst per group since the last move), `/mebot_sim/teleport`
(`std_msgs/Float64MultiArray`, seven joints), `/mebot_sim/posture`
(`std_msgs/String`) and `/mebot_sim/reset` (`std_srvs/Trigger`).

**Isolation.** `/mebot_sim` serves the real driver's names, so the stack must
never share the bench's ROS domain. The launch files and every node switch to
`ROS_DOMAIN_ID=88` when the domain is unset or 0, and say so. No node here
imports the Kortex API or opens a connection to an arm. Agents never drive the
arm (CONTRIBUTING.md:173-177); this stack cannot.

**Where.** `sim/` carries a `COLCON_IGNORE`, like `core/`, so CI's colcon job
(build.yml:80-101) does not need `rammp_arm_interfaces`. `make sim` builds the
package by naming it as a base path.

| File                                      | Contents                                                                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `sim/mebot_sim/mebot_sim/scene.py`        | MjSpec model builder                                                                                                               |
| `sim/mebot_sim/mebot_sim/sim_arm.py`      | the `/mebot_sim` node: physics loop, driver interface, contacts, clearance, live view                                              |
| `sim/mebot_sim/mebot_sim/client.py`       | test client of the planner and driver interfaces                                                                                   |
| `sim/mebot_sim/mebot_sim/try_goal.py`     | `ros2 run mebot_sim try_goal`: one goal, planned and flown, contacts reported                                                      |
| `sim/mebot_sim/mebot_sim/run_suite.py`    | `ros2 run mebot_sim run_suite`: the scenarios and gates below                                                                      |
| `sim/mebot_sim/mebot_sim/check_layers.py` | `ros2 run mebot_sim check_layers`: registration and stand-in coverage (CPU, no ROS graph)                                          |
| `sim/mebot_sim/mebot_sim/find_pairs.py`   | `ros2 run mebot_sim find_pairs`: seeded GPU search that writes the start/goal pairs (in-process planner)                           |
| `sim/mebot_sim/launch/sim.launch.py`      | the stack                                                                                                                          |
| `sim/mebot_sim/launch/suite.launch.py`    | the stack plus `run_suite`; shuts down with its exit status                                                                        |
| `sim/mebot_sim/config/scenarios.yaml`     | start/goal pairs (generated by `find_pairs`), stand-in postures, thresholds                                                        |
| `sim/mebot_sim/config/control_*.yaml`     | the control planner's scenes: `layers: [layer_mebot_floor.yaml, layer_mebot_platform.yaml]` and `layers: [layer_mebot_floor.yaml]` |
| `sim/mebot_sim/README.md`                 | how to run it and read the result                                                                                                  |

**Dependencies.** None is added to `core/pyproject.toml`; the sim is not part of
the package.

- ROS 2 Humble with `rmw_cyclonedds_cpp`.
- An overlay that provides `rammp_arm_interfaces` (rammp-interfaces-ros2 v1.0.0),
  consumed as built, never copied. On this Jetson that is `~/rammp_deps_ws`.
- Python `mujoco` (3.10.0 installed; MjSpec attach needs ≥ 3.2), Pillow, and
  `MUJOCO_GL=egl`.
- `~/mujoco_menagerie` (`kinova_gen3/gen3.xml`, `robotiq_2f85/2f85.xml`) and the
  humanoid XML, both by path (launch arguments).
- The pinned URDF through the generator's `fetch_pinned`, so the sim uses the
  same bytes as the generator.
- Jetson CUDA with cuRobo v0.7.8 for the two planner nodes. CUDA and EGL stay in
  separate processes: the planners hold CUDA, `/mebot_sim` holds EGL.

**Import guard.** The launch puts this checkout's `core/` first on the planner
nodes' `PYTHONPATH`, and the planner refuses to start if `rammp_curobo` was
imported from another checkout's `core/`. On this Jetson the editable install
maps `rammp_curobo` to `~/RAMMP-CuRobo/core`, which is on the older branch
feature/19-obstacle-avoidance.

**Scene build.** This re-implements the recipe of RAMMP-Kinova
`ros2_ws/src/mujoco_sim/mujoco_sim/build_scene.py:49-56, 71-104, 121-163` by
reference, without copying it:

1. Menagerie `gen3.xml`: mesh paths made absolute, keyframes deleted,
   `option.impratio=10` and elliptic cone set before attach, and the
   `base_link`/`shoulder_link` contact exclude.
1. The 2F-85 `base` attached into `bracelet_link` at pos `[0, 0, -0.06149039]`,
   quat `[0, 0, 1, 0]`, prefix `g_`.
1. The MEBot through **MuJoCo's own URDF importer**:
   - the pinned bytes with
     `<mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>`
     injected in memory;
   - all 11 joints deleted;
   - body `mebot` attached at -mount;
   - 46 exact primitives, with true cylinders.
1. A floor plane at the floor layer top.
1. The stand-in (§6.4) frozen as static geoms. `/mebot_sim/posture` switches it
   between the three postures and an empty chair.
1. The committed layer files, padded by 0.02 m, as visual-only overlays:
   `contype=conaffinity=0`, group 2, alpha 0.2; blue for platform, orange for
   person, grey for floor.

The design-phase prototype of this scene has nbody 69, ngeom 108 (without
overlays), njnt 15, nu 8.

**The simulated arm.**

- Physics steps at 2 ms in real time. Menagerie position actuators track the
  reference: the trajectory being executed, or the last commanded state.
- Like the real driver, `/execute_joint_trajectory` flies what it is sent on the
  trajectory's own time base. The client dilates time (`try_goal --speed`), as
  consumers do.
- **A contact is any MuJoCo contact between an arm or gripper geom and a group
  (`mebot`, `standin`, `floor`) with a normal force above 1 N**, sampled every 10
  ms of sim time. Penetration depth is not used: in the design-phase spike the
  arm pressed into the stand-in for 4.8 s at 744 N while the deepest penetration
  stayed under 1 mm.
- **Closest approach** per group is reported, not gated. It uses
  `mj_geomDistance`, which in MuJoCo 3.10 returns false zeros and sign-flipped
  values for some pairs of an arm mesh and a MEBot primitive (237 to 584 false
  zeros per trajectory in the spike, and -346 mm for a pair that was 346 mm
  apart). A value at or below zero counts only when the pair is also in contact
  or a hull vertex is inside the primitive.

**Live view.** `/mebot_sim` serves a page (default `http://127.0.0.1:8765`;
`host:=` adds addresses) that streams the EGL render as MJPEG at 20 fps. The
viewer can orbit, pan and zoom, send a tool goal to either planner, push the
joints by hand, switch the stand-in's posture, and see contacts and forces. It
is how a person watches the sim on a headless Jetson; no test depends on it.

**Commands.**

```sh
make sim            # build .sim_ws/, launch suite.launch.py, exit with run_suite's status

# or by hand, in two shells, each after:  source .sim_ws/install/setup.bash
ros2 launch mebot_sim sim.launch.py [posture:=none] [host:=127.0.0.1,<address>]
ros2 run mebot_sim try_goal --pos 0.45 0.20 0.25 --control      # the same goal on both planners
ros2 run mebot_sim run_suite [--only S2]
ros2 run mebot_sim check_layers
```

`make sim` sources ROS and the interfaces overlay (`SIM_OVERLAY`, default
`~/rammp_deps_ws/install/setup.bash`), builds `rammp_curobo_interfaces`,
`rammp_curobo_ros` and `sim/mebot_sim` into the git-ignored `.sim_ws/`, and runs
with `PYTHONPATH=$(CURDIR)/core` prepended.

**Scenarios** (`run_suite`). Start/goal pairs are data in `scenarios.yaml`,
written by `find_pairs` with fixed seeds and regenerated whenever the layers
change. A pair whose premise no longer holds fails loudly; nothing is skipped.

- **S0 layers against the true shapes** (`check_layers`, no planning)
  - MuJoCo's parse of the URDF has **0** surface samples outside the platform
    boxes, and the same layer shrunk by 1 mm leaves >0 outside. Design-phase
    prototype: 0 of 23,216, and 4,321 after the shrink.
  - Stand-in gap against the padded person layer is 0 for the three postures at
    both backrest positions.
- **S1 platform pairs** (empty chair). The straight path between start and goal
  crosses the true MEBot.
  - *Layer on*: `/rammp_curobo_control` with `[floor, platform]`. Refused, or
    flown with no MEBot contact.
  - *Layer off*: `/rammp_curobo_control` with `[floor]`. Planned and flown, and
    it contacts the MEBot. This is the positive control: the sim sees the chair,
    and dropping the layer is dangerous.
- **S2 person pairs.** The straight path crosses the stand-in.
  - *Layer on*: `/rammp_curobo`. Refused, or flown with no stand-in contact.
  - *Layer off*: `/rammp_curobo_control` with `[floor, platform]`. Planned and
    flown, and it contacts the stand-in.
- **S3 ordinary reaches.** Eight joint goals from one valid start on
  `/rammp_curobo`. The success rate is reported. Every flown plan is
  contact-free.
- **S4 a consumer's push.** `set_world` on `/rammp_curobo` with an 8-box scene
  that lists no layers. The reply must list platform 16 and person 6 (§7.5), and
  the S2 requests are then run again and must still be refused or avoid the
  stand-in.
- **S5 diagnostics.** A plan from the bench retract on `/rammp_curobo` fails
  with `person/` and a depth in the message.

**Gates** (`run_suite` exits 1 on any failure).

- **G0**: S0 registration and non-vacuity.
- **G1**: stand-in gap = 0.
- **G2**: no contact with `mebot`, `standin` or `floor` in any flown plan made
  with the relevant layers.
- **G3**: every positive control is planned, flown and makes contact with the
  shape its missing layer covers.
- **G4**: the layers survive a push (S4).
- **G5**: diagnostics name `person/` (S5).
- **G6**: every contact-free flight ends within 0.05 rad of its last waypoint,
  so the sim really flew the plan.

Reported, not gated: closest approach per group, plan times, the S3 success
rate, and tracking error. MuJoCo mesh hulls and cuRobo spheres differ by design:
a design-phase probe found the arm's hulls up to 4.7 mm outside cuRobo's spheres

- buffer.

What the earlier Python-only design checked inside the sim process moves to the
GPU tests (§9), where the planner is in-process: the installed world against the
layer files, cuRobo's cache read-back, and cuRobo's own spheres against every
padded box along a plan.

**Result.** `run_suite` prints one line per case (case, planner and world,
outcome, PASS or FAIL) and a summary, and writes the same as JSON to
`${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/sim/<UTC>-<git-sha>.json`, outside the
repo. No video is rendered; the live view replaces it.

**Design-phase evidence.** A throwaway prototype of this stack ran on the Jetson
on 2026-10-02 with the v1.0.0 planner node and the layer boxes loaded as plain
world files: 6 platform cases, 6 person cases and 8 reaches, 20 of 20 as
expected (7 of 8 reaches planned), with `/joint_states` at 50 Hz and plans in
0.2 to 0.3 s. Nodes on domain 88 were not visible from domain 0.

## 9. Tests (mapped to the issue's tasks)

Offline tests run in CI on amd64 and arm64 (numpy, PyYAML, pytest). GPU tests
run on the Jetson (`test_smoke.py`, which skips without CUDA). The sim runs
through `make sim` (§8).

`core/tests/conftest.py` (new, 10 lines) guards against testing the wrong tree.
It raises `pytest.UsageError` only when `rammp_curobo` is imported from a
`core/` directory **other than this checkout's**, which is the host's editable
install of `~/RAMMP-CuRobo`. It cannot trip in CI (editable install of the same
checkout) or in the image (non-editable, site-packages).

`pytest.ini` addopts become `-p no:anyio -p no:launch_testing -p no:launch_ros`.
That is required on this Jetson with ROS on PYTHONPATH: pytest 9.1.1 then fails
at plugin load with launch_testing's `PluginValidationError`, and the offline
suite passes with the two extra flags. The flags are harmless where the plugins
are absent (checked with pytest 9.1.1).

### Task 1: what v0.7.8 honours (`test_smoke.py`, GPU; gen3 planner; kitchen restored in `finally`)

- **`test_box_on_the_path_forces_a_detour_and_absent_goes_straight`**
  - Free world (one far box):
    `plan_to_joints(retract with joint_1 +1.0, retract, method='js')` succeeds
    with `dev < 0.05`.
  - A 0.08 m obstacle (tmp scene file) is placed at the FK tool point of the
    free plan's middle waypoint.
  - **Premise:** start and goal are still valid, and some free-path state is now
    invalid, so cuRobo sees the box.
  - Replanning fails, or succeeds with `dev > 0.10`.
- **`test_rotated_box_is_read_as_oriented`**
  - The same pair with a 0.30 × 0.02 × 0.30 slab yawed so its thin axis lies
    along the tool's path tangent: blocked against straight.
  - An OBB-not-AABB state check: a 0.40 × 0.02 × 0.40 slab at 45° yaw is offset
    0.15 m along its normal from a tool point. That tool point lies inside the
    slab's AABB.
  - The state is valid with the rotated slab and invalid with an axis-aligned
    box of the AABB's size.
- **`test_a_layer_box_on_the_path_is_honoured_and_ignore_removes_it`**
  - The blocking box is delivered through a tmp layer file (`layer: block`,
    `excludable: true`) listed by a tmp scene. It blocks.
  - The same scene with `ignore={'block'}` plans straight.
- **`test_update_world_guards_and_round_trip`** (existing, extended): the
  overflow message also contains "scene 61", and a refused push leaves
  `world_boxes` unchanged.
- **Mesh answer.** Meshes were not attempted. The answer is written in README
  (§11).

### Task 2: scene layers (`core/tests/test_layers.py`, offline, new)

| Test                                                                             | Proves                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_layer_boxes_are_namespaced_padded_and_attributed`                          | `<layer>/<box>` keys; scene names bare and `obj_` props unchanged; pedestal still unpadded; every layer box +0.04 per dim; sources map                                                                                                                                                                                                                       |
| `test_layer_free_worlds_compose_exactly_as_v100[kitchen, bench]`                 | `compose_cuboids(scene) == world_cuboids(scene)` (the unchanged v1.0.0 function): gen3 and gen3_real are unaffected                                                                                                                                                                                                                                          |
| `test_scene_layers_resolve_next_to_the_scene_and_never_from_cwd`                 | a tmp scene dir's layer is found; a same-named file in CWD is ignored; packaged fallback                                                                                                                                                                                                                                                                     |
| `test_config_layers_reach_every_world_input`                                     | Real `from_config` / `__init__` / `update_world` code. `_init_curobo` is patched to a recorder, and a fake `curobo.geom.types` module records `from_dict` input. `update_world(path)`, `(list)` and `(Scene)` all carry `platform/*` and `person/*`                                                                                                          |
| `test_missing_layer_is_named_with_every_path_tried`                              | from a scene and from a config: FileNotFoundError naming the ref, the listing file and the paths                                                                                                                                                                                                                                                             |
| `test_malformed_layer_is_refused[...]`                                           | parametrized: not a mapping, duplicate YAML key, unknown key or nested `layers:`, bad or reserved name, non-bool `excludable`, missing, empty or non-list `obstacles`, entry not a mapping, unknown entry key (`rpy:`), missing `dims`, `/` in name, duplicate box, NaN, dims ≤ 0, 2-element vector. Each ValueError names the file, the layer and the entry |
| `test_malformed_scene_entry_names_file_and_entry`                                | a scene whose `obstacles[i]` lacks `dims`, or whose `objects[i]` lacks `position`, raises ValueError naming the file and the entry (`obstacles[3] (table): missing 'dims'`), not a bare KeyError                                                                                                                                                             |
| `test_one_layer_name_from_two_files_is_refused_and_one_file_twice_composes_once` | no two sources of truth                                                                                                                                                                                                                                                                                                                                      |
| `test_duplicate_names_are_refused_not_collapsed`                                 | scene-scene, scene-layer (`platform/x`), prop-prop, layer-layer                                                                                                                                                                                                                                                                                              |
| `test_boxes_curobo_would_silently_ignore_are_refused`                            | NaN or inf position, zero or negative dims or radius, non-finite rpy, in scenes and layers                                                                                                                                                                                                                                                                   |
| `test_cache_overflow_names_every_source`                                         | `collision_cache_obb` plus per-source counts; nothing returned                                                                                                                                                                                                                                                                                               |
| `test_empty_scene_is_refused_even_with_config_layers`                            | `empty` with and without config layers; a scene whose only content is a scene layer is accepted                                                                                                                                                                                                                                                              |
| `test_guards_run_without_curobo`                                                 | with `sys.modules['curobo'] = None`, empty, overflow, duplicate and malformed inputs raise ValueError, not ImportError                                                                                                                                                                                                                                       |
| `test_misplaced_layers_key_is_refused`                                           | top-level `layer:`, `planner.layers`, and `layers` that is not a list of strings                                                                                                                                                                                                                                                                             |
| `test_chair_config_composes_within_the_cache`                                    | `gen3_mebot.yaml` resolves platform (not excludable) and person (excludable); the start-up world is 23 boxes; kitchen + chair layers = 48 ≤ 60; `world_padding` equals the files' `padding_assumed_m`                                                                                                                                                        |
| `test_chair_startup_world_is_empty_to_a_v100_loader`                             | `world_mebot.yaml` has no inline obstacles or objects (the only keys v1.0.0 `load_scene` turns into collision boxes)                                                                                                                                                                                                                                         |
| `test_world_files_load_as_utf8_under_a_c_locale`                                 | a subprocess with `PYTHONUTF8=0 LC_ALL=C` loads `world_sim_kitchen.yaml` (non-ASCII comments) and a layer file; v1.0.0's `open()` without an encoding raises UnicodeDecodeError there                                                                                                                                                                        |
| `test_sphere_box_penetration_matches_brute_force`                                | 300 random rotated OBBs × 50 spheres against dense point sampling; the right box is named; a rotated case where AABB and OBB answers differ                                                                                                                                                                                                                  |
| `test_world_summary_distinguishes_config_and_scene_layers`                       | format                                                                                                                                                                                                                                                                                                                                                       |

`test_offline.py` (extended):

- `test_packaged_configs_exist` adds the five new files;
- the drift test covers `layers`.

### Task 3: the generator (`core/tests/test_layergen.py`, offline, new; loads `scripts/generate_layers.py` via importlib)

| Test                                                                                     | Proves                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_fk_and_mount_on_a_synthetic_urdf`                                                  | fixed, revolute, prismatic and continuous joints at zero; missing `<axis>`; rpy convention; hand-computed AABBs                                                                                                                                                                                                                           |
| `test_cylinder_bounding_box_is_exact`                                                    | a y-axis cylinder gives `[2r, L, 2r]`; a tilted cylinder's formula box contains dense samples of the exact solid and is tight to 1e-9                                                                                                                                                                                                     |
| `test_unsupported_input_is_refused`                                                      | mesh, sphere and capsule geometry, mimic, planar or floating joints, 0 outside the limits, two roots, non-finite or non-positive size, rotated mount: each names the link or joint                                                                                                                                                        |
| `test_pinned_fetch_verifies_sha256_and_uses_the_cache`                                   | tmp git repo as clone; a correct hash is cached; a wrong hash raises naming the origin; a cache hit makes no subprocess or urllib call; a corrupt cache file is ignored                                                                                                                                                                   |
| `test_merge_meets_the_budget_and_never_touches_the_keep_out`                             | synthetic boxes near a synthetic sphere; budget met; an unmeetable budget raises                                                                                                                                                                                                                                                          |
| `test_over_coverage_is_exact_on_known_cases`                                             | two boxes with a known gap; overlaps counted once; max stack                                                                                                                                                                                                                                                                              |
| `test_containment_certificate_catches_a_defective_layer[drop, shrink_1_tick, shift_1mm]` | the under-coverage metric is not vacuous                                                                                                                                                                                                                                                                                                  |
| `test_person_boxes_are_the_envelope_minus_the_arm_column`                                | union of the boxes = envelope - column exactly (integers), over all three segments; padded clearance ≥ 4 mm; each parameter moves exactly its face (sitting height +0.1 raises the top by 0.1; popliteal height +0.1 lowers the legs' bottom by 0.1; the backrest-face range moves the chest and knee faces and both x faces of the legs) |
| `test_output_is_deterministic_under_float_noise`                                         | same bytes twice, and under ±1e-12 relative FK noise; grid-aligned; sorted                                                                                                                                                                                                                                                                |
| `test_committed_layers_regenerate_byte_identical`                                        | **Task 3's diff-assert**: pinned fetch, then all three committed files byte-for-byte. Skips locally without a source; fails when `CI=true`                                                                                                                                                                                                |
| `test_committed_platform_covers_every_urdf_primitive`                                    | an **independent sampler**: parse the COMMITTED platform file, sample the true URDF shapes (box lattices; cylinder rims at 72 angles × 2 radii × 9 heights), 0 points outside; each primitive's exact AABB is inside a box                                                                                                                |
| `test_committed_layers_meet_budgets_and_clear_the_arm`                                   | no network: ≤ 16 boxes and `over_pct` ≤ 15.0 (and > 1 box, so no single box); every padded box of every layer clears the keep-out spheres re-derived from the **packaged** robot YAML by ≥ 4 mm; flags and unique names                                                                                                                   |

### Task 4: the person layer

- Offline: `test_layers.py::test_ignore_drops_only_an_excludable_layer`.
  `ignore={'person'}` removes exactly `person/*`, with the platform
  byte-identical. `ignore={'platform'}` raises. A prop named `person` plus
  `ignore={'person'}` raises (ambiguous). An unknown name is a no-op.
- Offline:
  `test_layergen.py::test_person_boxes_are_the_envelope_minus_the_arm_column`.
- GPU: `test_person_is_excluded_without_touching_the_platform` (below).
- Sim: G1 (independent stand-in coverage) and S2.

### GPU tests on the generated layers (`test_smoke.py`)

The `chair_planner` session fixture (`gen3_mebot.yaml`) is built lazily, only by
the tests that need it.

| Test                                                          | Proves                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_platform_layer_pair` (gen3 planner, scene-level layers) | Seeded batched search for a retract-neighbourhood pair whose straight path is invalid only because of `platform/*`. With `[floor, platform]` it is blocked (fail or `dev > 0.10`); with `[floor]` it is straight. The generated geometry reaches cuRobo                                                                                                                                                                                                                                         |
| `test_world_read_back_matches_composition`                    | after construction and after `update_world` (both planners), cuRobo's `_env_n_obbs`, `_env_obbs_names[:n]`, enable bytes and dims (atol 1e-6) equal `world_boxes`, and `_cube_tensor_list[1][0, :n, :7]` equals the inverse of each `world_boxes` pose (atol 1e-5; quaternions up to sign; cuRobo stores inverse poses, geom/types.py:1309-1312). These are v0.7.8 private attributes (geom/sdf/world.py:712-715, 856-865), used **in tests only**; the read-back is a test, not a runtime gate |
| `test_chair_config_composes_the_layers_and_warms_up_first`    | `world_boxes` holds floor 1, platform 16, person 6; `planner._warm_boxes == 1` (white-box: warm-up ran in the scene-only world); the retract is invalid naming `person/`                                                                                                                                                                                                                                                                                                                        |
| `test_chair_layers_survive_every_push`                        | `update_world(list)`, `(path)` and `(Scene)` without layers keep platform 16 and person 6; a state found to penetrate only `platform/*` stays invalid, naming `platform/`                                                                                                                                                                                                                                                                                                                       |
| `test_person_is_excluded_without_touching_the_platform`       | retract invalid naming `person/`; after `update_world('world_mebot.yaml', ignore={'person'})` the retract is valid and the platform entries of `world_boxes` are identical; a platform-colliding state still fails; `ignore={'platform'}` raises and keeps the previous world                                                                                                                                                                                                                   |
| `test_invalid_start_names_the_blocking_layer`                 | `plan_to_joints` from the retract on the chair returns INVALID_START with `person/` and "mm" in the error                                                                                                                                                                                                                                                                                                                                                                                       |
| `test_a_layer_box_over_the_arm_base_is_refused`               | a scene layer with a box at (0, 0, 0.2) is refused naming the box and the fixed spheres; the previous world is kept                                                                                                                                                                                                                                                                                                                                                                             |
| `test_plans_keep_curobo_spheres_outside_every_padded_box`     | on the chair planner, seeded valid goals are planned and cuRobo's FK spheres at every waypoint are tested on the CPU with `world.sphere_box_penetration` against every padded composed box: ≤ 0.5 mm. A larger value means cuRobo did not honour a box, which `_recheck_constraints` cannot detect because it asks the same cuRobo world                                                                                                                                                        |

### Task 5: sim (decision 1)

`make sim` launches the ROS sim stack and runs `run_suite`, gating G0 to G6
(§8). The hardware pair on the real chair stays open
(deviation 8, §15).

## 10. File changes

| Path                                                                          | Action          | Purpose                                                                                                                                                                             |
| ----------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/rammp_curobo/scene.py`                                                  | modify          | strict layer loader, `Layer`, `load_layer(s)`, `Scene(layers=)`, scene `layers:`, UTF-8, entry errors naming file and entry                                                         |
| `core/rammp_curobo/world.py`                                                  | modify          | `compose_cuboids`, `to_world_config`, `make_world_config(layers=)` (cache default 60), `sphere_box_penetration`, `point_box_distance`, `summarize`; guards before the cuRobo import |
| `core/rammp_curobo/config.py`                                                 | modify          | `layers: []`, `resolve_config(search_cwd=)`, misplaced-key and type checks                                                                                                          |
| `core/rammp_curobo/planner.py`                                                | modify          | config layers loaded, composed everywhere; warm-up first; fixed-sphere check; `world_boxes`, `world_summary()`; CPU attribution in `check_state_valid` and `_finish`; seed warning  |
| `core/rammp_curobo/configs/gen3.yaml`                                         | modify          | `layers: []`                                                                                                                                                                        |
| `core/rammp_curobo/configs/gen3_mebot.yaml`                                   | add             | chair planner config                                                                                                                                                                |
| `core/rammp_curobo/configs/world_mebot.yaml`                                  | add             | chair start-up scene (floor as a scene layer; no inline obstacles)                                                                                                                  |
| `core/rammp_curobo/configs/layer_mebot_platform.yaml`                         | add (generated) | 16 boxes, coverage meta                                                                                                                                                             |
| `core/rammp_curobo/configs/layer_mebot_person.yaml`                           | add (generated) | 6 boxes in three segments, excludable, parameters and volumes in meta                                                                                                               |
| `core/rammp_curobo/configs/layer_mebot_floor.yaml`                            | add (generated) | floor slab at the URDF ground                                                                                                                                                       |
| `scripts/generate_layers.py`                                                  | add             | the CPU generator and CLI (`--check`, `--offline`, `--clone`, `--cache`)                                                                                                            |
| `scripts/mebot_layers.yaml`                                                   | add             | generator inputs with provenance                                                                                                                                                    |
| `core/tests/conftest.py`                                                      | add             | wrong-tree import guard                                                                                                                                                             |
| `core/tests/test_layers.py`                                                   | add             | composition and loading tests (Task 2 and 4)                                                                                                                                        |
| `core/tests/test_layergen.py`                                                 | add             | generator tests including regen diff-assert and independent sampler (Task 3 and 4)                                                                                                  |
| `core/tests/test_offline.py`                                                  | modify          | new packaged files; drift test covers `layers`                                                                                                                                      |
| `core/tests/test_smoke.py`                                                    | modify          | Task-1 pairs; platform pair; read-back; chair fixture tests                                                                                                                         |
| `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`                           | modify          | summary in the start-up log, the ready log and the SetWorld success message                                                                                                         |
| `sim/mebot_sim/` (package, launch files, config, README), `sim/COLCON_IGNORE` | add             | the ROS sim stack (§8)                                                                                                                                                              |
| `pytest.ini`                                                                  | modify          | `-p no:launch_testing -p no:launch_ros`                                                                                                                                             |
| `Makefile`                                                                    | modify          | `layers` and `sim` targets (+ `.PHONY`)                                                                                                                                             |
| `.github/workflows/build.yml`                                                 | modify          | guard step: one unpiped pytest invocation per offline test file (`test_offline.py`, `test_layers.py`, `test_layergen.py`)                                                           |
| `README.md`                                                                   | modify          | "Obstacle layers: the wheelchair and the person" section (includes the Task-1 mesh answer); tree listing adds `scripts/generate_layers.py` and `sim/`                               |
| `INTEGRATION.md`                                                              | modify          | "Planning on the MEBot (≥ 1.1.0)"                                                                                                                                                   |
| `CLAUDE.md`                                                                   | modify          | invariants (§11)                                                                                                                                                                    |
| `CONTRIBUTING.md`                                                             | modify          | building and testing: `make layers` (clone or network, must be a no-op), `make sim` (Jetson); `sim/` added to the "Not covered" line (:127)                                         |
| `CHANGELOG.md`                                                                | modify          | `[Unreleased]`: Added / Changed / Fixed                                                                                                                                             |
| `docker/README.md`                                                            | modify          | `config:=gen3_mebot.yaml` example                                                                                                                                                   |

`rammp_curobo_interfaces/` and `core/pyproject.toml` are **unchanged**.
`.gitignore` and `.dockerignore` gain `.sim_ws/`. The sim's results live outside
the repo, and the new configs are flat, so the existing package-data glob ships
them.

## 11. Docs and versioning

**README**, new section "Obstacle layers: the wheelchair and the person":

- config and scene layers, and persistence across set_world;
- `excludable` and `ignore={'person'}`;
- the chair config;
- `make layers` and the coverage table: platform 16 boxes, under 0, over 10.9 %
  of the bounding boxes, about 15 % of the true shapes; person 6 boxes in three
  segments (torso and head, lap, lower legs and feet), +0.05 m on every face;
- the headline workspace numbers (§13, risk 1);
- `make sim`.

**Task-1 mesh answer:** meshes were not attempted. v0.7.8's MESH checker has
three silent traps:

- a zero-mesh update keeps the old meshes (world_mesh.py:101-117);
- a reused name keeps the old geometry (world_mesh.py:167-173);
- mesh-cache overflow reallocates despite `fix_cache_reference`
  (world_mesh.py:103-105), so recorded CUDA graphs keep stale pointers
  (inferred, from the docstring at world_mesh.py:96-99).

cuRobo's docs also put the cuboid checker at 4× faster than the mesh checker
(<https://curobo.org/get_started/2c_world_collision.html>). So the layers are
boxes only, and the paired cuboid tests are the Task-1 acceptance.

**INTEGRATION.md**:

- `config:=gen3_mebot.yaml`;
- pushed scenes replace only task geometry, including the start-up floor, so
  carry your own floor;
- the bench retract is not a valid start on the chair; start from the measured
  state;
- excluding the person is Python-only;
- the SetWorld reply lists the layers;
- this needs ≥ 1.1.0, and a 1.0.0 image refuses the chair config.

**CLAUDE.md invariants:**

- config `layers:` are composed into every world; scene `layers:` last only
  until the next push, so safety layers belong in the planner config;
- `layer_mebot_*.yaml` are GENERATED: run `make layers` and never hand-edit
  them;
- nothing padded may touch the arm's joint-independent sphere column (z -0.015
  .. 0.383 m);
- worktree tests need `PYTHONPATH=<worktree>/core`.

**Versioning (CONTRIBUTING.md:108-171):**

- No IDL change: PlanToPose, PlanToJoints and SetWorld are untouched.
- MINOR additions: the new optional config key `layers`, the scene key
  `layers:`, `Scene(layers=)`, `CuRoboPlanner.world_boxes` and
  `world_summary()`.
- `world.py` / `scene.py` internals, `scripts/` and `sim/` are not covered
  (CONTRIBUTING.md:127-128; this branch adds `sim/` to that line).
- The new refusals only reject inputs that were silently broken. They are listed
  under **Fixed**, and none of the 566 known consumer worlds is affected.
- This branch **does not bump** the version. The release PR bumps it to 1.1.0 in
  the five places (CONTRIBUTING.md:150-154; bump at release time, :169).

**CHANGELOG, under `[Unreleased]`:**

- **Added**: layers (config and scene); generated MEBot platform, person and
  floor layers; `gen3_mebot.yaml` and `world_mebot.yaml`;
  `scripts/generate_layers.py` and `make layers`; `world_boxes` and
  `world_summary()`; which-layer attribution in failures and
  `check_state_valid`; the ROS sim stack `sim/mebot_sim` and `make sim`.
- **Changed**: warm-up runs before config layers load; SetWorld success messages
  and plan-failure errors carry the world summary.
- **Fixed**: duplicate box names refused instead of collapsed; non-finite or
  non-positive boxes refused; guards run without cuRobo; malformed scene entries
  name file and entry; world files read as UTF-8.

**Workflow.** Branch `feature/19-platform-layers` off `origin/dev` (b734928,
equal to the v1.0.0 content) in the worktree `~/RAMMP-CuRobo-19`.
`~/RAMMP-CuRobo` stays on feature/19-obstacle-avoidance, which the editable
install follows. Local commits only, never pushed. Run tests with
`PYTHONPATH=<worktree>/core`.

## 12. Deviations from the issue (and why)

1. **Planner-config layers are composed into every world, alongside the issue's
   scene-level `layers:`.** This is decision 5. Consumers push whole worlds
   (box-opening before every plan; ADL at every task start, §4.7), and v1.0.0
   ignores unknown keys, so scene-only layers would drop the chair silently. The
   issue's mechanism is kept for replaceable geometry: the chair's floor uses
   it.
1. **The platform is the URDF drive base only.** There is no seat, armrests,
   footplate or mount plate (decision 3). The only machine-readable CAD is the
   URDF. The seat exists only as Unreal Engine assets: ramms-assets
   `rammp_seat`, whose mesh payload is Oodle-compressed so that only its bounds,
   sockets and collision hulls decode, and `SM_seat_2` inside the packaged RAMMP
   UI, whose Oodle-compressed container cannot be extracted offline.
1. **The generator reads URDF primitives, not "mesh files".** Mesh input is
   refused. The CAD is a primitives URDF, and the three v0.7.8 mesh traps are in
   §11.
1. **The person is 6 boxes in three segments, not one volume.** One box covering
   the occupant contains the arm's fixed base spheres, because the occupant's
   centreline is 0.265 m from the arm axis and a 95th-percentile half-bideltoid
   is 0.283 m: every state would be infeasible. One box would also swallow the
   workspace in front of the chest and above the lower legs. Three segments
   follow the seated body instead: torso and head, lap, and lower legs and feet
   (decision 7). The carve around the arm's column splits torso_head into 4
   boxes.
1. **The person layer is carved around the arm's joint-independent sphere
   column**, and every layer is gated against that column. This is not in the
   issue, but it is forced: any padded box there makes every state INVALID_START
   (base spheres reach z -0.015).
1. **Warm-up runs in the scene-only world before config layers load.** cuRobo
   warms up from the retract and exits early on an invalid start
   (motion_gen.py:1896-1945, 2105-2113, 3007-3015). The person layer puts the
   retract 154 mm inside.
1. **A generated floor is added as a scene-level layer of the chair start-up
   world.** `world:` is required (config.py:105-107) and must be non-empty. The
   ground height comes from the same CAD. Keeping the start-up scene free of
   inline obstacles makes a 1.0.0 image refuse the chair config. This is a
   sliver of #21.
1. **Task 5 is the ROS sim stack, not the attended hardware run** (decision 1). The hardware pair stays open (§15). Agents never drive the arm
   (CONTRIBUTING.md:173-177).
1. **The release is MINOR (1.1.0), not "no version bump".** New optional config
   and new API are MINOR (CONTRIBUTING.md:113-115). The branch itself bumps
   nothing.
1. **Silently-broken consumer inputs are refused**: duplicate names, and
   non-finite or non-positive boxes. The issue's core fear is geometry cuRobo
   silently ignores. 0 of 566 known worlds are affected.
1. **A narrow `layer:` / `planner.layers` misplacement guard.** A typo in the
   one key that switches the chair on would otherwise plan without the chair.
   General unknown-key strictness for planner configs is not added.
1. **The URDF is pinned at 80908b9, not 59d5aef.** 59d5aef is only on a feature
   branch. The blob and sha256 are identical.

## 13. Risks

1. **The person layer confines the arm at the RAMMS mount.** This is a known
   consequence of the approved size (decision 6), not an open question. The arm
   axis is 0.265 m from the occupant centreline, and a 95th-percentile
   half-bideltoid is 0.283 m. Uniform joint samples free of world collision (a
   design-phase CPU sphere model of the arm, N = 4000, world collision only; the
   legs segment does not change these):

   | World      | All samples | Tool high and in front (z > 0.3, x > 0.2) |
   | ---------- | ----------- | ----------------------------------------- |
   | floor      | 98.4 %      | n/a                                       |
   | + platform | 87.9 %      | 99.7 %                                    |
   | + person   | **2.2 %**   | **1.2 %**                                 |

   The bench retract is **154.4 mm inside** the padded person layer
   (`person/lap` vs `spherical_wrist_1_link`; retract +0.1 rad on joint_1:
   -134.4 mm). It clears the platform by 53.4 mm. Consumer tasks near the
   occupant will IK_FAIL with this envelope. In the spike, none of 22,869 valid
   states (of 1,000,000 sampled; 2.3 %) put the tool more than 0.1 m onto the
   person's side or in front of the knees: anything handed to the person needs
   the person layer excluded. The sizes are data, so tuning them
   later is one edit to `scripts/mebot_layers.yaml` plus `make layers`.

1. **The mount is unverified** (§15).

   - Arm z is 0.361 (`BP_Mebot_Mujoco`), 0.376 (`BP_Mebot_Ramms`, used) or
     0.388 (Demo-Software TF), depending on the source (§5.1).
   - The Demo-Software TF differs by frame.
   - The physical chair may be a newer build.
   - The sim uses the same URDF as the generator, so it cannot measure
     URDF-vs-reality error.

1. **The backrest face is uncertain (0.1587 .. 0.2733).** At 0.2733 a
   95th-percentile stand-in intersects the Gen3 at retract (-7.0 mm). The layer
   spans the range; the sim places the stand-in at 0.1587.

1. **Unmodelled regions** (decision 3):

   - Seat, armrests, footplate and mount plate are unmodelled.
   - Excluding the person also removes the only coverage of the backrest region.

1. **The MEBot is modelled at all joints zero.** The elevator and carriage move
   the envelope by up to about 0.2 m in x and 0.3 m in z. Under-coverage is zero
   only at q = 0.

1. **The retract is also cuRobo's IK seed and null-space reference.** An
   infeasible seed may lower plan success on the chair. The sim measures it; a
   chair seed is #20.

1. **A 1.0.0 image ignores `layers:`.** Mitigated: the chair config ships only
   in ≥ 1.1.0, and its start-up world is empty to 1.0.0.

1. **Consumer worlds are bench-shaped.** box-opening's `floor_*` and table, and
   ADL's placeholder table, become phantom geometry on the chair (consumer
   follow-ups, #21). Motions cuRobo never plans are not protected by any layer:
   box-opening's straight descent, lift and recoil, and ADL's back-out.

1. **Per-box plan-time cost on the Orin at 23 to 48 boxes is unmeasured.** It is
   linear in loaded boxes (sphere_obb_kernel.cu:2453). The sim reports plan
   times.

1. **SUM_COLLISIONS stacks cost in overlaps.** Platform self-overlap is 0.026 m³
   (max stack 3; kept low by the pairwise objective, §5.4) and person 0.00007
   m³; the legs box also has 0.005 m³ inside the platform's front-caster boxes.
   The composed chair world's max stack stays 3.

1. **The CI regeneration test depends on GitHub raw.** It fails, never skips, in
   CI.

1. **Host import path.** The editable install follows `~/RAMMP-CuRobo`.
   Mitigated by the conftest guard, the sim guard and `PYTHONPATH`.

1. **Rare-event searches.** About 2 % of states are valid with the person layer,
   and about 1 % of goals near the retract cross the platform. Searches are
   seeded and capped, and fail loudly, never skip.

1. **The stand-in is indicative, not anthropometric truth.** It uses DeepMind
   proportions scaled to one ANSUR dimension.

1. **The fixed-sphere check and the read-back test use cuRobo v0.7.8
   internals.** `link_spheres_tensor` is public; the cache tensors are private
   and used in tests only. They must be revisited if cuRobo is ever changed (it
   is pinned).

1. **Joint-space goals suffer most under the person layer.** In the spike,
   native joint-space planning reached 1 of 8 valid joint goals with the person
   layer and 7 of 8 without it. The `fk_pose` fallback rescued 6, each ending in
   a different joint configuration, 3 to 6 rad from the requested one. Consumers
   that plan to fixed joint poses will see more failures and different postures
   on the chair config.

1. **MuJoCo's distance query is unreliable for mesh-against-primitive pairs.**
   `mj_geomDistance` in 3.10 returns false zeros and sign-flipped values (§8).
   Clearance is therefore reported with a guard and never gated; the gates use
   contact forces.

1. **The sim serves the real driver's names and runs two planners.** Domain
   isolation (§8) is what keeps it away from the bench; the launch files and
   nodes enforce it. Two cuRobo instances share the Orin's GPU with whatever
   else is running.

## 14. Explicitly out of scope

- perception and dynamic obstacles;
- the robot's own model, the wrist camera and a chair IK seed (#20);
- the cell layer: rooms, tables, walls and a persistent floor (#21);
- speed or deceleration near the person;
- EE constraints (#16);
- reaching the mouth, and a ROS path for excluding the person (a future new
  service would be MINOR);
- meshes (the stretch goal, answered in §11);
- the MEBot's joint motion;
- seat, backrest, armrest, footplate and mount-plate modelling;
- a sheppy fragment or deployment changes (consumer repos);
- a hardware session;
- raising the collision cache (stays 60).

## 15. Follow-ups and known unknowns

Everything else is decided (§2). These are not:

- **Mount and backrest on the real chair.** The arm's z is 0.361
  (`BP_Mebot_Mujoco`), 0.376 (`BP_Mebot_Ramms`, used) or 0.388 (Demo-Software
  TF), depending on the source (§5.1), and the physical chair may be a newer
  build than the URDF. The backrest face lies somewhere in 0.1587 .. 0.2733; at
  the front of that range a 95th-percentile occupant meets the arm's bench
  retract (§6.4). Both are data: a measurement is one edit to
  `scripts/mebot_layers.yaml` plus `make layers`.
- **A chair IK seed (#20).** The retract is 154.4 mm inside the person layer, so
  it is not a valid start on the chair, and it is also cuRobo's IK seed and
  null-space reference (risk 6). The planner warns at start-up.
- **Whether the floor should persist (#21).** The floor is a scene-level layer
  of the chair's start-up world, so the first push replaces it. A persistent
  floor belongs to the cell layer.
- **The hardware session for Task 5.** The paired test on the real chair,
  attended, with a human on the e-stop (docs/HARDWARE_BRINGUP.md).
- **A ROS path for excluding the person.** Exclusion is Python-only. A new
  service would be MINOR; a field on SetWorld would be MAJOR.

## 16. Evidence index

The generator, sphere-model and coverage numbers come from design-phase CPU
prototypes. The sim rows come from a throwaway spike that ran the v1.0.0 planner
on the Jetson GPU with the layer boxes loaded as plain scene obstacles, first
in-process and then as a ROS stack on an isolated domain. No repository was
changed while measuring them. The implementation recomputes them; §5.5, §8 (G0 to G6) and §9 say which
are gated.

| Evidence                                       | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Design-phase CPU prototype of the generator    | platform, person and floor text; 16 platform boxes (a second, independent prototype produced the same 16, box for box); over 10.9 %; overlap 0.026491 m³; keep-out 53.4 mm; person 6 boxes at 5.1 mm, and the torso_head intrusions at `world_padding` 0.030 (§7.4); floor top -0.4593; determinism under ±1e-12. Budget sweep in §5.5; the whole-layer objective at 16 boxes (§5.4); legs-platform overlap and the composed stack (§13, risk 10)                                                                                                                                                                                                                                                                                                                                        |
| Design-phase CPU sphere model of the arm       | retract clearance per layer and per box (§7.4); joint-space fractions (yourdfpy FK on cuRobo's Gen3 URDF with the packaged robot config's spheres + the 0.005 buffer; prototype only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Design-phase MuJoCo prototype                  | registration 0 of 23,216 samples, and 4,321 after a 1 mm shrink; stand-in coverage (§6.4); MuJoCo retract distances; scene counts; the `upright_lap` and `forearms_forward` posture searches                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Sim spike with the v1.0.0 planner (2026-10-01) | 3 platform pairs and 3 person pairs behaved as designed (detours 0.15 to 0.62 rad with no contact; without the layer, straight paths 10 to 52 mm into the true shape); 7 of 8 reaches planned, all contact-free; an independent re-check confirmed it; joint-space freedom, joint-goal and `mj_geomDistance` findings (§13)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ROS prototype of the sim stack (2026-10-02)    | 20 of 20 cases as expected over `/rammp_curobo` and `/execute_joint_trajectory` on domain 88 (§8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Consumer world scan                            | 566 consumer worlds (the 563 in `~/.ros/rammp_box_opening/worlds/` on this Jetson, the kitchen, the bench, box-opening's `world_bench.yaml`; §4.6): 0 files newly refused; max 26 boxes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Git                                            | sha256 201ec24a... for the blob at 59d5aef, 80908b9 and 0a94cce and for `~/mebot.urdf`; `branch -r --contains`: 59d5aef only on `origin/feature/106-drive-base-publisher`, 80908b9 on `origin/dev` (= `origin/HEAD`), 0a94cce on `origin/main`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| cuRobo v0.7.8 source read                      | warm-up and early exits (motion_gen.py:1868-1960, 2105-2113, 3007-3015); `update_world` (motion_gen.py:1825-1837); cache init and load (geom/sdf/world.py:716, 724-727, 827-866); mesh `_init_cache` delegation (world_mesh.py:72-81); sphere buffer (cuda_robot_generator.py:759-771); `sum_collisions_` (sphere_obb_kernel.cu:2711); box count from device memory (sphere_obb_kernel.cu:2453); `enable_obstacle` / `enable_obb` / `get_obb_idx` (geom/sdf/world.py:978-1014, 1094-1110); `get_cuboid` and the cylinder trimesh (geom/types.py:99-132, 341); `get_obb_world` (geom/types.py:1096); inverse cube poses in `batch_tensor_cube` (geom/types.py:1309-1312); `normalize_quaternion` (types/math.py:536); MESH-checker traps (world_mesh.py:96-99, 101-117, 103-105, 167-173) |

cuRobo paths are relative to `src/curobo/` of v0.7.8 (d64c4b0).
`geom/sdf/world.py`, `geom/types.py` and `types/math.py` are written in full;
the short names are:

| Short name                | Path                                                |
| ------------------------- | --------------------------------------------------- |
| `motion_gen.py`           | `wrap/reacher/motion_gen.py`                        |
| `world_mesh.py`           | `geom/sdf/world_mesh.py`                            |
| `sphere_obb_kernel.cu`    | `curobolib/cpp/sphere_obb_kernel.cu`                |
| `cuda_robot_generator.py` | `cuda_robot_model/cuda_robot_generator.py`          |
| `kinova_gen3_7dof.urdf`   | `content/assets/robot/kinova/kinova_gen3_7dof.urdf` |
