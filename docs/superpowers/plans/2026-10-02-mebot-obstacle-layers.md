# MEBot Obstacle Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The planner composes a generated wheelchair layer and a separate, excludable person layer into every collision world, proven by offline tests, paired GPU tests and a ROS sim stack (issue #19).

**Architecture:** Pure-Python loading and composition (`scene.py`, `config.py`, `world.py`) runs every guard before cuRobo is imported. A CPU generator in `scripts/` turns the pinned MEBot URDF into committed layer files. `planner.py` composes config layers into every world and attributes failures to layers. A ROS 2 package under `sim/` stands a MuJoCo arm in for the driver beside two planner nodes.

**Tech Stack:** Python 3.10, numpy, PyYAML, cuRobo v0.7.8 (GPU tests only), pytest, ROS 2 Humble, MuJoCo 3.10 (sim only).

**Spec:** `docs/superpowers/specs/2026-09-30-mebot-obstacle-layers-design.md`. The spec is authoritative. It gives exact signatures, error strings, test names and file lists; this plan gives the order, the task boundaries and the interfaces between tasks. Where they disagree, follow the spec and report it.

**Implementation code is specified by contract, not transcribed here.** Each step names the spec section that fixes the signatures and messages, and the design-phase prototype to port where one exists. Prototypes (read-only, outside the repo; re-implement, never import from them):

| Prototype                                                             | Use for                                                                                      |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `~/mebot-sim-spike/proto/layergen.py`                                 | Task 4: fetch, URDF FK, snap, merge, coverage, person, floor, text output                    |
| `~/mebot-sim-spike/proto/check_mj.py`, `~/mebot-sim-spike/mjscene.py` | Tasks 8 and 9: scene build, stand-in, contact groups, distance guard, registration, coverage |
| `~/mebot-sim-spike/live_sim.py`                                       | Task 8: physics loop, live view page, trajectory following                                   |
| `~/mebot-sim-spike/ros/src/mebot_sim/`                                | Tasks 8 and 9: ROS node, client, `try_goal`, `run_suite`, launch file                        |
| `~/mebot-sim-spike/plan_gpu.py`                                       | Tasks 7 and 9: seeded pair search with batched validity checks                               |

## Global Constraints

- Work only in the worktree `~/RAMMP-CuRobo-19`, branch `feature/19-platform-layers`. Local commits only. Never push, never open or comment on anything on GitHub.
- Never modify another repository or checkout (`~/RAMMP-CuRobo`, `~/RAMMP-box-opening`, `~/Demo-Software`, `~/mebot-sim-spike`, ...).
- Run tests with this checkout first on the path: `PYTHONPATH=$PWD/core python3 -m pytest ... -p no:launch_testing -p no:launch_ros` (the two flags move into `pytest.ini` in Task 1). The host's editable install points at another checkout.
- Before every commit: `pre-commit run --files <the files you changed>` until it passes (ruff 0.3.0, ruff-format, mdformat, end-of-file, LF). Commit only named files, never `git add -A`.
- Commit style follows the repo (`feat(core): ...`, `test: ...`, `docs: ...`), ends with `(#19)`, and carries the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No IDL change: `rammp_curobo_interfaces/` is untouched. No new field on `PlanToPose`, `PlanToJoints` or `SetWorld`.
- The collision cache stays at 60. `world_padding` stays 0.02 m. `no_pad_names` keeps matching scene obstacles only.
- Core runtime dependencies stay numpy and PyYAML. `scripts/generate_layers.py` uses numpy, PyYAML and the standard library only. No yourdfpy, trimesh, cuRobo or MuJoCo outside `core/tests/test_smoke.py` and `sim/`.
- Layer files sit flat in `core/rammp_curobo/configs/` (`configs/*.yaml` is the package-data glob; subdirectories are not shipped). Generated files are ASCII with LF endings.
- Geometry is data: mount `[0.505, -0.265, 0.376]`, identity rotation; pin `rammp-org/Demo-Software` commit `80908b9f73f02e094106c0b82640a23c3cd6f41c`, path `core/rammp_prototype_description/urdf/rammp_prototype_base.urdf`, sha256 `201ec24aeeff868557b13211eaa4571b8f11cd6709795537098bafaef032c0ec`; `grid_m 0.0001`; `padding_assumed_m 0.02`; `keep_out_margin_m 0.005`; keep-out gate 4 mm; platform `max_boxes 16`, `max_over_pct 15.0`; person `margin_m 0.05`.
- No code is copied from a sibling repository; menagerie models, the humanoid XML and the URDF are consumed by path or pinned fetch.
- No AI or cloud API anywhere.
- GPU: one cuRobo planner per test process; other workspaces share the GPU, so never kill foreign processes. Nothing may launch ROS on domain 0 or touch the real arm; the sim stack runs on `ROS_DOMAIN_ID=88`.

## Review Focus

Inputs the spec implies but its test tables do not name. Each has a test in the task that owns the code.

1. **A planner config that lives outside the package** (a deployment mounts `/config/gen3_mebot.yaml`) and lists layers by packaged name: the layers resolve from the packaged configs, not from CWD. Test in Task 6.
1. **`ignore` given as a bare string** (`ignore='person'`): it means that one name, not its characters. Test in Task 3.
1. **Numbers that are strings or booleans in a layer entry** (`dims: ["0.1", "0.2", "0.3"]`, `position: [true, 0, 0]`): refused, naming file, layer and entry. Test in Task 2.
1. **`update_world([])` and a list world under config layers**: the empty rule applies to the list path too, and a non-empty list keeps every config layer. Test in Task 6.
1. **An unwritable cache directory** during the pinned fetch (a read-only HOME in a container): the bytes are still returned from the clone or URL, and the generator says the cache was not written. Test in Task 4.

## File Structure

| File                                                                                              | Responsibility                                                                                |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `core/rammp_curobo/config.py`                                                                     | `layers` key, `resolve_config(search_cwd=)`, misplaced-key guard                              |
| `core/rammp_curobo/scene.py`                                                                      | layer files: `Layer`, `load_layer`, `load_layers`; scene `layers:`; UTF-8; entry errors       |
| `core/rammp_curobo/world.py`                                                                      | `compose_cuboids` and every guard; `to_world_config`; attribution maths; `summarize`          |
| `core/rammp_curobo/planner.py`                                                                    | config layers in every world; warm-up order; fixed-sphere check; `world_boxes`; failure notes |
| `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`                                               | world summary in logs and the SetWorld reply                                                  |
| `scripts/generate_layers.py`, `scripts/mebot_layers.yaml`                                         | the generator and its inputs                                                                  |
| `core/rammp_curobo/configs/layer_mebot_{platform,person,floor}.yaml`                              | generated layers                                                                              |
| `core/rammp_curobo/configs/gen3_mebot.yaml`, `world_mebot.yaml`                                   | the chair config and its start-up scene                                                       |
| `core/tests/conftest.py`, `test_layers.py`, `test_layergen.py`                                    | offline tests                                                                                 |
| `core/tests/test_offline.py`, `test_smoke.py`                                                     | extended offline and GPU tests                                                                |
| `sim/mebot_sim/`                                                                                  | the ROS sim stack                                                                             |
| `Makefile`, `pytest.ini`, `.github/workflows/build.yml`, `.gitignore`, `.dockerignore`            | targets, flags, CI guard, ignores                                                             |
| `README.md`, `INTEGRATION.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docker/README.md` | docs                                                                                          |

______________________________________________________________________

### Task 1: Test hygiene and the `layers` config key

**Files:**

- Modify: `pytest.ini`, `core/rammp_curobo/config.py`, `core/rammp_curobo/configs/gen3.yaml`, `core/tests/test_offline.py`
- Create: `core/tests/conftest.py`, `core/tests/test_layers.py`

**Interfaces:**

- Produces: `PLANNER_DEFAULTS["layers"] == []`; `resolve_config(name_or_path, relative_to=None, search_cwd=True)`; `load_planner_config` raises `ValueError("<path>: 'layer' is not a planner-config key; did you mean the top-level 'layers:'?")` for a top-level `layer:` and for `planner: {layers: ...}`, and `ValueError` when `layers` is not a list of strings.

- [ ] **Step 1: pytest flags and the wrong-tree guard.** `pytest.ini` addopts become `-p no:anyio -p no:launch_testing -p no:launch_ros` (keep the existing comment, add one line on the ROS plugins). Create `core/tests/conftest.py`:

```python
"""Refuse to test another checkout's rammp_curobo (this host has an editable install of one)."""

import os

import pytest


def pytest_configure(config):
    import rammp_curobo

    here = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.realpath(os.path.dirname(os.path.dirname(rammp_curobo.__file__)))
    if os.path.basename(src) == "core" and src != here:
        raise pytest.UsageError(
            "rammp_curobo is imported from %s, not this checkout (%s); "
            "run with PYTHONPATH=%s" % (src, here, here)
        )
```

- [ ] **Step 2: Verify.** `python3 -m pytest core/tests/test_offline.py -q` without PYTHONPATH fails with the UsageError on this host; `PYTHONPATH=$PWD/core python3 -m pytest core/tests/test_offline.py -q` passes 14.
- [ ] **Step 3: Write failing tests** in `core/tests/test_layers.py`: `test_misplaced_layers_key_is_refused` (top-level `layer:`, `planner: {layers: [...]}`, `layers: "x.yaml"`, `layers: [1]`), and a `resolve_config` test that a file in CWD is found with `search_cwd=True` and not with `search_cwd=False` (use `monkeypatch.chdir(tmp_path)`). Extend the drift test in `test_offline.py` (lines 48-58) to cover `layers`.
- [ ] **Step 4: Run them, expect failures.**
- [ ] **Step 5: Implement** per spec §4.3 and §7.2; add `layers: []` with a comment to `gen3.yaml`.
- [ ] **Step 6: Run** `PYTHONPATH=$PWD/core python3 -m pytest core/tests -q` (offline pass, smoke may run on the Jetson: all pass).
- [ ] **Step 7: pre-commit, commit** `feat(core): layers config key, CWD-free resolution, wrong-tree test guard (#19)`.

### Task 2: Layer files and scene-level `layers:`

**Files:**

- Modify: `core/rammp_curobo/scene.py`, `core/tests/test_layers.py`

**Interfaces:**

- Consumes: `resolve_config(ref, relative_to=..., search_cwd=False)`.

- Produces:

  - `class Layer` with `__slots__ = ("name", "excludable", "obstacles", "path")`; `obstacles` is a list of `Obstacle`; `path` is the resolved absolute path.
  - `load_layer(path) -> Layer` (strict, spec §4.1).
  - `load_layers(refs, relative_to, listed_in) -> list[Layer]`: type check, resolution without CWD, `FileNotFoundError("layer 'x.yaml' listed in <listed_in> not found; tried: ...")`, dedupe by resolved path, same name from two paths refused naming both files.
  - `Scene(base_frame, obstacles, targets, objects=(), layers=())` and `Scene.layers`.
  - `load_scene(path)` reads optional `layers:` relative to the scene file's directory, opens files as UTF-8, and turns malformed entries into `ValueError("<file>: obstacles[3] (table): missing 'dims'")`.

- [ ] **Step 1: Write failing tests** (spec §9 Task 2 table): `test_malformed_layer_is_refused[...]` (every case in the table, plus Review Focus 3: numeric strings and booleans), `test_missing_layer_is_named_with_every_path_tried` (scene half), `test_scene_layers_resolve_next_to_the_scene_and_never_from_cwd`, `test_one_layer_name_from_two_files_is_refused_and_one_file_twice_composes_once` (loading half), `test_malformed_scene_entry_names_file_and_entry`, `test_world_files_load_as_utf8_under_a_c_locale` (subprocess with `PYTHONUTF8=0 LC_ALL=C`, `PYTHONPATH` set to this checkout's `core`).

- [ ] **Step 2: Run, expect failures.**

- [ ] **Step 3: Implement** per spec §4.1, §4.2, §7.1. `_StrictLoader` is a `yaml.SafeLoader` subclass whose mapping constructor raises on a duplicate key. A number is `int` or `float`, not `bool`, and finite. `scene_from_obstacles`, `Obstacle`, `SceneObject`, `Target` and `__all__` keep their behaviour.

- [ ] **Step 4: Run** the whole offline suite; the kitchen and bench worlds still load.

- [ ] **Step 5: pre-commit, commit** `feat(core): strict layer files and scene-level layers (#19)`.

### Task 3: Composition and guards

**Files:**

- Modify: `core/rammp_curobo/world.py`, `core/tests/test_layers.py`

**Interfaces:**

- Consumes: `Layer`, `Scene.layers`.

- Produces (spec §7.3):

  - `world_cuboids(...)` unchanged.
  - `compose_cuboids(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset(), cache_obb=60, *, layers=()) -> (cuboids, sources)`. `cuboids` has the shape `world_cuboids` returns today: `{key: {"dims": [dx, dy, dz], "pose": [x, y, z, qw, qx, qy, qz]}}`. `sources` is `{key: "scene" | <layer name>}`. `layers=` are config layers; scene layers come from `scene.layers`. Order: scene obstacles, scene props, config layers, scene layers. Layers are deduplicated by `Layer.path`.
  - `to_world_config(cuboids) -> WorldConfig` (the only function that imports cuRobo).
  - `make_world_config(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset(), cache_obb=60, *, layers=()) -> WorldConfig`.
  - `sphere_box_penetration(spheres, cuboids) -> {key: depth_m}`: `spheres` is an (N, 4) array of x, y, z, r; only boxes with positive penetration appear.
  - `point_box_distance(point, cuboids) -> {key: signed_m}`.
  - `summarize(sources, config_layer_names) -> str`, e.g. `"23 boxes: scene 0 | config layers: platform 16, person 6 | scene layers: floor 1"`.

- [ ] **Step 1: Write failing tests** (spec §9 Task 2 and Task 4 offline rows): `test_layer_boxes_are_namespaced_padded_and_attributed`, `test_layer_free_worlds_compose_exactly_as_v100[kitchen, bench]`, `test_duplicate_names_are_refused_not_collapsed`, `test_boxes_curobo_would_silently_ignore_are_refused`, `test_cache_overflow_names_every_source`, `test_empty_scene_is_refused_even_with_config_layers`, `test_guards_run_without_curobo` (set `sys.modules['curobo'] = None` with `monkeypatch.setitem`), `test_ignore_drops_only_an_excludable_layer` (plus Review Focus 2: `ignore='person'`), `test_sphere_box_penetration_matches_brute_force`, `test_world_summary_distinguishes_config_and_scene_layers`, and the composition half of `test_one_layer_name_from_two_files...`. Use small layer files written to `tmp_path`; the real layer files do not exist until Task 5.

- [ ] **Step 2: Run, expect failures.**

- [ ] **Step 3: Implement** per spec §4.3 to §4.6. Message texts are in §4.5 and §4.6; the v1.0.0 empty-world text and the `collision_cache_obb` substring are kept because `test_smoke.py:124-132` matches them. `make_world_config`'s default cache becomes 60. The module docstring's trap list gains the three new traps (§7.3).

- [ ] **Step 4: Run** the offline suite and, on the Jetson, `core/tests/test_smoke.py` (9 pass: `make_world_config` still serves the planner).

- [ ] **Step 5: pre-commit, commit** `feat(core): compose layers into the collision world, guards before cuRobo (#19)`.

### Task 4: The generator

**Files:**

- Create: `scripts/generate_layers.py`, `scripts/mebot_layers.yaml`, `core/tests/test_layergen.py`

**Interfaces:**

- Consumes: nothing from Tasks 1 to 3 (it may run in a sibling worktree).

- Produces, as module-level names tests and the sim load with `importlib` from `<repo>/scripts/generate_layers.py`:

  - `load_data(path=None) -> dict` (default `scripts/mebot_layers.yaml`).
  - `fetch_pinned(source, cache_dir=None, clone=None, offline=False) -> (bytes, origin_str)`.
  - `keep_out_spheres(data) -> list[(centre_xyz, radius_with_buffer, link_name)]`, read from the packaged `robot_gen3_2f85.yaml`.
  - `generate(data, urdf_bytes, perturb=0.0) -> dict` with keys `files` (`{relative_output_path: text}`), `platform`, `person`, `floor` (the numbers behind the report).
  - `main(argv=None) -> int` with `--data`, `--check`, `--offline`, `--clone`, `--cache`.

- [ ] **Step 1: Write the data file** `scripts/mebot_layers.yaml` per spec §5.1 and §6.1 (pin, mount with `chain`, `alt_chain` and `not_used`, grid, padding, keep-out, platform budget, the twelve ANSUR II rows with row numbers, margin, seat anchors, floor, outputs).

- [ ] **Step 2: Write failing tests** (spec §9 Task 3 table, the rows that need no committed layer files): `test_fk_and_mount_on_a_synthetic_urdf`, `test_cylinder_bounding_box_is_exact`, `test_unsupported_input_is_refused`, `test_pinned_fetch_verifies_sha256_and_uses_the_cache` (plus Review Focus 5: unwritable cache), `test_merge_meets_the_budget_and_never_touches_the_keep_out`, `test_over_coverage_is_exact_on_known_cases`, `test_containment_certificate_catches_a_defective_layer[drop, shrink_1_tick, shift_1mm]`, `test_person_boxes_are_the_envelope_minus_the_arm_column`, `test_output_is_deterministic_under_float_noise`.

- [ ] **Step 3: Run, expect failures.**

- [ ] **Step 4: Implement** per spec §5.2 to §5.7 and §6.2, porting the prototype. Differences from the prototype: inputs come from the data file, not a `DATA` literal; the person has three segments (the prototype in `~/mebot-sim-spike/proto/layergen.py` already has `legs`); `meta` is written in full (§5.6); gates exit non-zero and write nothing (§5.5); `--check` prints a unified diff; non-ASCII output is refused.

- [ ] **Step 5: Run** `python3 scripts/generate_layers.py --offline --clone ~/Demo-Software` into a temporary output directory and compare with the spec's numbers: 16 platform boxes, over 10.9 %, keep-out 53.4 mm; 6 person boxes at 5.1 mm, with the box table of §6.2; floor top -0.4593.

- [ ] **Step 6: pre-commit, commit** `feat(scripts): generate the MEBot platform, person and floor layers from the pinned URDF (#19)`.

### Task 5: Committed layers, the chair config, `make layers`

**Files:**

- Create: `core/rammp_curobo/configs/layer_mebot_platform.yaml`, `layer_mebot_person.yaml`, `layer_mebot_floor.yaml` (generated), `gen3_mebot.yaml`, `world_mebot.yaml`
- Modify: `Makefile`, `core/tests/test_offline.py`, `core/tests/test_layers.py`, `core/tests/test_layergen.py`, `.github/workflows/build.yml`

**Interfaces:**

- Consumes: `generate_layers.py`, `load_layer`, `load_scene`, `compose_cuboids`, `load_planner_config`.

- Produces: the packaged names `gen3_mebot.yaml`, `world_mebot.yaml`, `layer_mebot_platform.yaml` (layer `platform`, not excludable, 16 boxes), `layer_mebot_person.yaml` (layer `person`, excludable, 6 boxes), `layer_mebot_floor.yaml` (layer `floor`, 1 box).

- [ ] **Step 1: Generate** the three files with `python3 scripts/generate_layers.py`. Write `gen3_mebot.yaml` and `world_mebot.yaml` by hand per spec §7.6.

- [ ] **Step 2: Write tests** (they pass once the files exist; run each once against a deliberately broken copy to see it fail): `test_committed_layers_regenerate_byte_identical` (skips locally without a source, fails when `CI=true`), `test_committed_platform_covers_every_urdf_primitive`, `test_committed_layers_meet_budgets_and_clear_the_arm` in `test_layergen.py`; `test_chair_config_composes_within_the_cache`, `test_chair_startup_world_is_empty_to_a_v100_loader` in `test_layers.py`; `test_packaged_configs_exist` in `test_offline.py` gains the five files.

- [ ] **Step 3: Makefile** `layers` target mirroring `bake` (Makefile:47-52): run the generator, then `git diff --exit-code` on the three outputs; add to `.PHONY` and the header comment.

- [ ] **Step 4: CI guard** (`build.yml:69-75`): one unpiped `python3 -m pytest core/tests/<file> -q` per offline test file (`test_offline.py`, `test_layers.py`, `test_layergen.py`), with the comment explaining why one invocation per file.

- [ ] **Step 5: Run** the offline suite and `make layers` (a no-op diff).

- [ ] **Step 6: pre-commit, commit** `feat(config): the generated MEBot layers and the chair planner config (#19)`.

### Task 6: Planner integration

**Files:**

- Modify: `core/rammp_curobo/planner.py`, `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`, `core/tests/test_layers.py`

**Interfaces:**

- Consumes: everything from Tasks 1 to 5.

- Produces (spec §7.4, §7.5): `CuRoboPlanner._layers`; `world_boxes` (list of `{"name", "layer", "dims", "pose"}`, padded dims, pose `[x, y, z, qw, qx, qy, qz]`); `world_summary() -> str`; `_warm_boxes`; `check_state_valid` detail naming boxes with depths; plan-failure notes; `update_world(world, ignore=())` with layers, the fixed-sphere check and commit-after-success; the planner node's start-up, ready and SetWorld messages carrying the summary.

- [ ] **Step 1: Write the failing offline test** `test_config_layers_reach_every_world_input` (spec §9): real `from_config` / `__init__` / `update_world` code with `_init_curobo` patched to a recorder and a fake `curobo.geom.types` module recording `from_dict`; `update_world(path)`, `(list)` and `(Scene)` all carry `platform/*` and `person/*`. Add Review Focus 1 (a config in `tmp_path` listing packaged layer names resolves them; a same-named file in CWD is ignored) and Review Focus 4 (`update_world([])` refused with the config-layers empty text; a non-empty list keeps the layers).

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `__init__`, `_init_curobo`, `update_world`, `world_boxes`, `world_summary`** per spec §7.4, including the warm-up order (compose `full`; if config layers exist and warm-up is on, build MotionGen with the scene-only world, run the fixed-sphere check on `full`, warm up, then `update_world` to `full`) and the retract WARNING.

- [ ] **Step 4: Implement attribution**: `_colliding_boxes(q)`, the `check_state_valid` suffix, and `_finish(result, t0, q_goal=None, start=None, goal_point=None)` notes that never raise (§7.4).

- [ ] **Step 5: planner_node.py** strings (§7.5). No parameter, topic, service or IDL change.

- [ ] **Step 6: Run** the offline suite, then on the Jetson `core/tests/test_smoke.py` (the existing 9 pass on `gen3.yaml`), then a manual check: `CuRoboPlanner.from_config("gen3_mebot.yaml")` logs warm-up in the scene-only world, `world_summary()` reads `23 boxes: scene 0 | config layers: platform 16, person 6 | scene layers: floor 1`, and `check_state_valid(retract)` names `person/lap`.

- [ ] **Step 7: pre-commit, commit** `feat(planner): config layers in every world, warm-up first, which-layer failure notes (#19)`.

### Task 7: GPU tests

**Files:**

- Modify: `core/tests/test_smoke.py`

**Interfaces:**

- Consumes: the planner API of Task 6 and the packaged chair files of Task 5.

- [ ] **Step 1: Task-1 pairs** (spec §9 "Task 1"): `test_box_on_the_path_forces_a_detour_and_absent_goes_straight`, `test_rotated_box_is_read_as_oriented`, `test_a_layer_box_on_the_path_is_honoured_and_ignore_removes_it`, and the extension of `test_update_world_guards_and_round_trip` ("scene 61"; a refused push leaves `world_boxes` unchanged). Every test restores the kitchen in `finally`.

- [ ] **Step 2: Generated-layer tests** (spec §9 "GPU tests on the generated layers"): a lazily built `chair_planner` session fixture; `test_platform_layer_pair`, `test_world_read_back_matches_composition`, `test_chair_config_composes_the_layers_and_warms_up_first`, `test_chair_layers_survive_every_push`, `test_person_is_excluded_without_touching_the_platform`, `test_invalid_start_names_the_blocking_layer`, `test_a_layer_box_over_the_arm_base_is_refused`, `test_plans_keep_curobo_spheres_outside_every_padded_box`. Searches are seeded and capped and fail loudly when they find nothing.

- [ ] **Step 3: Run** `PYTHONPATH=$PWD/core python3 -m pytest core/tests/test_smoke.py -q` on the Jetson until green, three times in a row (seeded searches must be stable).

- [ ] **Step 4: pre-commit, commit** `test: paired GPU tests for boxes, layers and the generated chair (#19)`.

### Task 8: The sim stack

**Files:**

- Create: `sim/COLCON_IGNORE`, `sim/mebot_sim/package.xml`, `setup.py`, `setup.cfg`, `resource/mebot_sim`, `mebot_sim/__init__.py`, `mebot_sim/stack.py`, `mebot_sim/scene.py`, `mebot_sim/sim_arm.py`, `mebot_sim/view.py`, `mebot_sim/client.py`, `mebot_sim/try_goal.py`, `launch/sim.launch.py`, `config/control_floor.yaml`, `config/control_floor_platform.yaml`
- Modify: `Makefile`, `.gitignore`, `.dockerignore`

**Interfaces:**

- Consumes: the packaged chair config and layer files; `generate_layers.fetch_pinned`; `rammp_arm_interfaces` and `rammp_curobo_interfaces` from the overlay.

- Produces (spec §8): nodes `/mebot_sim`, `/rammp_curobo` (`config:=gen3_mebot.yaml`), `/rammp_curobo_control` (`config:=gen3.yaml`); `/joint_states`, `/execute_joint_trajectory`, `/mebot_sim/contacts`, `/mebot_sim/teleport`, `/mebot_sim/posture`, `/mebot_sim/reset`; `ros2 run mebot_sim try_goal`; `client.Tester` with `connect()`, `teleport(q)`, `posture(name)`, `move(planner, world, start, pos=None, rpy_deg=None, joints=None, speed=0.5, execute=True)`; `stack.isolate()`.

- [ ] **Step 1: Package and isolation.** `package.xml` (a valid maintainer e-mail, `ament_python`), `stack.isolate()` switching domain 0 or unset to 88, and the same rule in the launch file.

- [ ] **Step 2: Scene** (`scene.py`): the build recipe of spec §8, with the committed layer files (padded 0.02) as overlays, the stand-in postures, and the contact groups.

- [ ] **Step 3: The sim arm** (`sim_arm.py`, `view.py`): physics at 2 ms in real time, the driver interface, contacts with force above 1 N sampled every 10 ms, guarded closest approach, the live view page.

- [ ] **Step 4: Client and `try_goal`**, including `--control` (the same goal on both planners) and time dilation by `--speed`.

- [ ] **Step 5: Launch** `sim.launch.py` with both planner nodes, this checkout's `core/` first on their `PYTHONPATH`, and the launch arguments of spec §8. `make sim-build` builds `rammp_curobo_interfaces`, `rammp_curobo_ros` and `sim/mebot_sim` into `.sim_ws/` with the overlay `SIM_OVERLAY ?= ~/rammp_deps_ws/install/setup.bash`. `.gitignore` and `.dockerignore` gain `.sim_ws/`.

- [ ] **Step 6: Verify on the Jetson**: launch; both planners ready; `/joint_states` at 50 Hz; `try_goal --pos 0.50 -0.40 0.30` flies with no contact; `try_goal --pos 0.45 0.20 0.25 --control` is refused on `/rammp_curobo` and hits the stand-in through the control planner; no node is visible on domain 0.

- [ ] **Step 7: pre-commit, commit** `feat(sim): a ROS sim stack: MuJoCo arm, chair and seated body beside two planners (#19)`.

### Task 9: The sim suite and `make sim`

**Files:**

- Create: `sim/mebot_sim/mebot_sim/check_layers.py`, `find_pairs.py`, `run_suite.py`, `launch/suite.launch.py`, `config/scenarios.yaml`, `sim/mebot_sim/README.md`
- Modify: `sim/mebot_sim/setup.py`, `Makefile`

**Interfaces:**

- Consumes: Task 8's `client.Tester`, `scene.py`, and the in-process planner (for `find_pairs` only).

- Produces: `ros2 run mebot_sim check_layers | find_pairs | run_suite`; `ros2 launch mebot_sim suite.launch.py` exiting with `run_suite`'s status; `make sim`.

- [ ] **Step 1: `check_layers`** (S0, gates G0 and G1): registration against MuJoCo's URDF parse with the 1 mm shrink control, and stand-in coverage for three postures at both backrest positions.

- [ ] **Step 2: `find_pairs`**: seeded batched searches for three platform pairs, three person pairs, one valid start and eight reach goals; writes `config/scenarios.yaml` with the seeds and the layer files' sha256.

- [ ] **Step 3: `run_suite`**: S0 to S5 and gates G0 to G6 of spec §8, one line per case, JSON result under `${XDG_CACHE_HOME:-~/.cache}/rammp_curobo/sim/`, exit 1 on any failure. A pair whose positive control does not hit fails with "premise broken; re-run find_pairs".

- [ ] **Step 4: `suite.launch.py` and `make sim`.**

- [ ] **Step 5: Verify on the Jetson**: `make sim` exits 0 twice in a row.

- [ ] **Step 6: README, pre-commit, commit** `feat(sim): the paired sim suite and make sim (#19)`.

### Task 10: Docs and versioning

**Files:**

- Modify: `README.md`, `INTEGRATION.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docker/README.md`

- [ ] **Step 1: Write** each document per spec §11, with the numbers the committed layer files and the sim actually produce (read them from the files' `meta` and the last `make sim` result, not from the spec).

- [ ] **Step 2: pre-commit, commit** `docs: obstacle layers, the chair config, make layers and make sim (#19)`.

### Task 11: Whole-branch verification

- [ ] **Step 1:** `PYTHONPATH=$PWD/core python3 -m pytest core/tests -q` on the Jetson (offline and GPU): all pass.
- [ ] **Step 2:** `make layers`: no diff. `make sim`: exit 0.
- [ ] **Step 3:** `pre-commit run --all-files`: clean.
- [ ] **Step 4:** `git diff origin/dev --stat -- rammp_curobo_interfaces/` is empty.
- [ ] **Step 5:** A whole-branch review against the spec, section by section, and against the issue's "What would make this fail review" list.
