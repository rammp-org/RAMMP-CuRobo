# Robot Data Outside the Image — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One rammp-curobo image for every arm, base and site: the planner reads robot and world data only from the config it is given (or from an example it is explicitly asked for), checks it all at startup, and runs standalone with no config.

**Architecture:** File resolution loses its silent packaged fallback and gains an explicit `example:` prefix. The Gen3 names hard-coded in `urdf_model.py` come from the robot file instead, and one pure-Python check (`contract.check_robot_dir`) collects every problem with a robot directory and its URDF before cuRobo is imported. The node with no `config` runs the shipped example (Gen3 robot file, kitchen world, level URDF) and says so everywhere. The bench moves to rammp-deployments' dojo, flatbot gets a complete `curobo/` directory, and the image copies code only.

**Tech Stack:** Python 3.10, cuRobo v0.7.8 (pinned), ROS 2 Humble (rclpy), PyYAML, pytest, Docker (Jetson L4T base), GNU make.

**Spec:** `docs/superpowers/specs/2026-10-28-robot-data-outside-image-design.md`

## Global Constraints

- cuRobo stays pinned at v0.7.8; `core/` imports nothing from ROS; `rammp_curobo_interfaces` stays dependency-free.
- No action interface (`.action`, `.srv`) field changes; comments only.
- Nothing here may move the arm. Agents never run against the real arm; hardware bring-up is a human at the e-stop (docs/HARDWARE_BRINGUP.md).
- The sim never runs on ROS domain 0.
- A named config's references resolve next to that config, or absolute, or via `example:`. Never from the image by bare name.
- Robot data (robot file, world) is whole or refused; only planner settings default field by field (`PLANNER_DEFAULTS`).
- Every existing CLAUDE.md invariant stays true: the URDF-vs-spheres check (now for any arm), the fixed-sphere guard, padding certification, trimmed plans, `check_start_joints`.
- Worktree tests and `make layers` need `PYTHONPATH=<worktree>/core`.
- Breaking change: CHANGELOG `[Unreleased]` gets a **BREAKING** section; the version bump to 2.0.0 happens at release, not in this branch.

## Review Focus

1. **Empty `/config/robot`** (Docker started the planner before `set_robot.py`): a `config` path that does not exist is a FATAL naming that path — never a fall-through to example mode. Test in Task 4.
2. **An old deployment config saying `robot: robot_gen3_2f85.yaml`** with no copy next to it: the error names the directory searched and suggests `example:robot_gen3_2f85.yaml`. Test in Task 1.
3. **A robot file copied before this change** (no `model_check` block): the startup error names the missing block, in the same message as every other problem found. Test in Task 3.
4. **The tilted mount:** `arm_tilt30.urdf` still passes the kinematics check against the robot file (the mount joint may differ; joints below the arm's base may not). Test in Task 2.
5. **A malformed `example:` reference** (`example:../gen3.yaml`, `example:nope.yaml`): refused, listing the examples that exist. Test in Task 1.

## Order and branches

- **RAMMP-CuRobo**: branch `feature/robot-data-outside-image` from `dev` AFTER PR #46 merges (this plan assumes the seat layer and `scripts/generate_layers.py --data/--outdir` from #46). Tasks 1–8, one PR.
- **rammp-deployments**: branch `feature/curobo-robot-dirs` from `dev`. Tasks 9–10, one PR. It MUST merge before the RAMMP-CuRobo PR: today's nightly reads it unchanged (a local `robot.yaml` resolves next to the config; cuRobo ignores the extra top-level `model_check` key; layers are just boxes), and the RAMMP-CuRobo PR's nightly refuses `config: gen3_real.yaml`, which is what dojo runs until Task 10 lands.
- Task 9 runs the generator from the RAMMP-CuRobo branch (Task 6) before that branch merges; that is fine, the generator is deterministic.

---

### Task 1: Explicit `example:` resolution, no packaged fallback

**Files:**
- Modify: `core/rammp_curobo/config.py:18,67-98` (`resolve_config`, new constants)
- Modify: `core/rammp_curobo/robot_config.py:46` (it calls `resolve_config(path)` again; pass the already-resolved path through)
- Modify (mechanical): `core/tests/test_offline.py`, `test_layers.py`, `test_smoke.py`, `test_planner_cpu.py`, `test_model_parity.py`, `test_layergen.py`, `rammp_curobo_ros/test/test_robot_description.py`, `test_node_wiring.py`, `sim/mebot_sim/mebot_sim/stack.py`, `sim/mebot_sim/test/test_stack.py`, `scripts/bake_robot_config.py`, `core/rammp_curobo/configs/gen3.yaml` comments
- Test: `core/tests/test_offline.py`

**Interfaces:**
- Produces: `EXAMPLE_PREFIX = "example:"`; `EXAMPLE_DIR = PACKAGED_CONFIG_DIR` (same directory, new name for its role); `resolve_config(name_or_path, relative_to=None, search_cwd=True) -> Path` with the new rule; `is_example(path: Path) -> bool` (true iff the resolved path is inside `EXAMPLE_DIR`). Every later task resolves files only through `resolve_config`.

Rule: `example:<name>` → `EXAMPLE_DIR / <name>`, where `<name>` is a plain file name (no `/`, no `..`), must exist, else `FileNotFoundError` listing `sorted(EXAMPLE_DIR)` yaml/json/urdf names. Otherwise: absolute → as is; relative → `relative_to`, then cwd (if `search_cwd`). `PACKAGED_CONFIG_DIR` is never a candidate. A relative name not found raises `FileNotFoundError` naming every location tried, and, when `EXAMPLE_DIR / name` exists, ends with `; to use the image's copy, write example:<name>`. A config inside `EXAMPLE_DIR` resolves its own bare references next to itself, so `example:gen3.yaml` naming `robot_gen3_2f85.yaml` and `world_sim_kitchen.yaml` works unchanged.

- [ ] **Step 1: Write the failing tests** in `core/tests/test_offline.py`:

```python
def test_example_prefix_resolves_into_the_image():
    assert resolve_config("example:gen3.yaml") == EXAMPLE_DIR / "gen3.yaml"
    assert is_example(resolve_config("example:robot_gen3_2f85.yaml"))

def test_bare_name_never_falls_back_to_the_image(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        resolve_config("robot_gen3_2f85.yaml", relative_to=tmp_path, search_cwd=False)
    assert str(tmp_path) in str(e.value)
    assert "example:robot_gen3_2f85.yaml" in str(e.value)

@pytest.mark.parametrize("ref", ["example:../gen3.yaml", "example:sub/gen3.yaml", "example:nope.yaml"])
def test_bad_example_reference_lists_the_examples(ref):
    with pytest.raises(FileNotFoundError) as e:
        resolve_config(ref)
    assert "gen3.yaml" in str(e.value) and "robot_gen3_2f85.yaml" in str(e.value)

def test_example_config_resolves_its_own_references():
    cfg, cfg_dir = load_planner_config("example:gen3.yaml")
    assert resolve_config(cfg["robot"], relative_to=cfg_dir) == EXAMPLE_DIR / "robot_gen3_2f85.yaml"
```

- [ ] **Step 2: Run** `PYTHONPATH=core python3 -m pytest core/tests/test_offline.py -q -k "example or fall_back"` — Expected: FAIL (`EXAMPLE_DIR` not importable).
- [ ] **Step 3: Implement** the rule above in `config.py`.
- [ ] **Step 4: Rewrite every bare packaged reference** in the files listed above to `example:<name>` (tests, sim `stack.py`, `bake_robot_config.py`). `test_the_chair_is_not_packaged_and_resolves_to_no_stale_copy` keeps passing unchanged in meaning. Candidate for the local model (`local-code`): purely mechanical — ask the user before offloading, review its diff.
- [ ] **Step 5: Run** `PYTHONPATH=core python3 -m pytest core/tests -q` and `python3 -m pytest rammp_curobo_ros/test -q -p no:anyio` — Expected: all pass (GPU tests skip off-Jetson; on abra run with `RAMMP_CUROBO_GPU=1`).
- [ ] **Step 6: Commit** `feat(config)!: resolve only next to the config or by example:, never the image by bare name`

### Task 2: Robot names come from the robot file

**Files:**
- Modify: `core/rammp_curobo/configs/robot_gen3_2f85.yaml` (add top-level `model_check`), `scripts/bake_robot_config.py` (writes it)
- Modify: `core/rammp_curobo/robot_config.py` (add `RobotSpec`, `load_robot_spec`)
- Modify: `core/rammp_curobo/urdf_model.py:41-53,60-98` (delete `ROOT_LINK`, `EE_LINK`, `ARM_JOINTS`, `GRIPPER_JOINTS`, `BASE_LINK`, `KINEMATICS_REFERENCE`; `prepare_urdf(xml, spec)`; `kinematic_reference(root, base_link)`)
- Modify: `core/rammp_curobo/config.py:24-32` (drop `joint_names` from `PLANNER_DEFAULTS`), `core/rammp_curobo/planner.py:72,100-122` (joint names default to the robot file's `cspace.joint_names`; a config `joint_names` must be the same set; the `ROOT_LINK` check goes — Task 3 checks it against the URDF)
- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py` (callers of `prepare_urdf`), `core/tests/conftest.py:21` users
- Test: `core/tests/test_offline.py`, `core/tests/data/two_link/` (new fixture: a 2-joint URDF and its robot file)

**Interfaces:**
- Consumes: Task 1's `resolve_config`.
- Produces:
  - `RobotSpec` (frozen dataclass): `root_link: str` (= `kinematics.base_link`), `ee_link: str`, `joint_names: tuple[str, ...]` (= `cspace.joint_names`), `locked_joints: tuple[str, ...]` (= keys of `lock_joints`), `arm_base_link: str`, `kinematics_reference: Path`, `path: Path`.
  - `load_robot_spec(path: Path) -> RobotSpec` — `ValueError` naming the file and each missing field.
  - `prepare_urdf(xml: str, spec: RobotSpec) -> str` — same refusals as today, by the spec's names: root must be `spec.root_link`; `spec.ee_link` must exist; `spec.joint_names + spec.locked_joints` must exist; joints below `spec.arm_base_link` must match `spec.kinematics_reference` (1 mm / 0.1°, structure, joint set; the mount joint may differ).

Robot-file block (top level, beside `robot_cfg`, so cuRobo never sees it):

```yaml
model_check:
  arm_base_link: arm_base_link
  kinematics_reference: arm_kinematics.json   # resolved next to this file (or example:)
```

- [ ] **Step 1: Write the failing tests:**

```python
def test_spec_reads_the_names_from_the_robot_file():
    spec = load_robot_spec(resolve_config("example:robot_gen3_2f85.yaml"))
    assert spec.root_link == "arm_mount" and spec.ee_link == "arm_tool_frame"
    assert spec.joint_names == tuple("arm_joint_%d" % i for i in range(1, 8))
    assert spec.locked_joints == ("arm_robotiq_85_left_knuckle_joint",)
    assert spec.kinematics_reference == EXAMPLE_DIR / "arm_kinematics.json"

def test_a_non_gen3_arm_passes_with_its_own_robot_file():
    d = DATA / "two_link"
    spec = load_robot_spec(d / "robot.yaml")
    assert "<visual" not in prepare_urdf((d / "two_link.urdf").read_text(), spec)

@pytest.mark.parametrize("urdf", ["arm_level.urdf", "arm_tilt30.urdf"])
def test_gen3_models_pass_on_either_mount(urdf):
    prepare_urdf((DATA / urdf).read_text(), gen3_spec())

def test_gen3_urdf_refused_against_another_arms_spec():
    with pytest.raises(ValueError, match="root link"):
        prepare_urdf((DATA / "arm_level.urdf").read_text(), load_robot_spec(DATA / "two_link" / "robot.yaml"))
```

  Keep every existing `prepare_urdf` refusal test, now passing `gen3_spec()` (a conftest helper returning the example spec).
- [ ] **Step 2: Run** `PYTHONPATH=core python3 -m pytest core/tests/test_offline.py -q -k "spec or non_gen3 or either_mount"` — Expected: FAIL (`load_robot_spec` undefined).
- [ ] **Step 3: Implement** `RobotSpec`/`load_robot_spec`, the `prepare_urdf` signature, the planner's joint-name default, and the `model_check` block; build the `two_link` fixture (URDF root `base`, joints `j1`,`j2`, tool `tool`; its `arm_kinematics.json` written by `kinematic_reference`).
- [ ] **Step 4: Run** the core and ROS suites — Expected: all pass. On abra: `make bake` leaves no diff (the bake script now emits `model_check`), and `test_model_parity.py` passes.
- [ ] **Step 5: Commit** `refactor!: the robot file names the arm; no Gen3 names in the planner's code`

### Task 3: One startup check that lists every problem

**Files:**
- Create: `core/rammp_curobo/contract.py`
- Modify: `core/rammp_curobo/planner.py:100-147` (call the check before `load_robot_config`), `rammp_curobo_ros/rammp_curobo_ros/planner_node.py:183-197` (call it instead of bare `prepare_urdf`; FATAL + exit 1 on failure)
- Test: `core/tests/test_contract.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `class RobotDataError(ValueError)` with `.problems: list[str]`; `str()` is `"robot data in <config_dir>: N problems:\n  - …"`. `check_robot_dir(cfg: dict, cfg_dir: Path, urdf_xml: str) -> str` returns the sanitised URDF or raises `RobotDataError` with ALL of: robot file missing/unparseable/missing a field (incl. `model_check`); kinematics reference missing; world missing, unparseable, or `base_frame` ≠ `spec.root_link`; a listed layer missing or refused by `load_layers` (padding shortfall included); every `prepare_urdf` refusal. Pure Python; imports no cuRobo. The planner constructor calls it with `Path(urdf_path).read_text()`; the node calls it on the model text and writes the returned XML.

- [ ] **Step 1: Write the failing tests** in `core/tests/test_contract.py`:

```python
def test_every_problem_in_one_error(tmp_path):
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nworld: world.yaml\nlayers: [missing_layer.yaml]\n")
    shutil.copy(EXAMPLE_DIR / "robot_gen3_2f85.yaml", tmp_path / "robot.yaml")   # no arm_kinematics.json next to it
    (tmp_path / "world.yaml").write_text("base_frame: base\nobstacles: [{name: b, position: [0,0,0], dims: [1,1,1]}]\n")
    cfg, d = load_planner_config(tmp_path / "planner.yaml")
    with pytest.raises(RobotDataError) as e:
        check_robot_dir(cfg, d, (DATA / "arm_level.urdf").read_text())
    msg = str(e.value)
    assert len(e.value.problems) == 3
    assert "arm_kinematics.json" in msg and "missing_layer.yaml" in msg and "'base'" in msg

def test_a_robot_file_without_model_check_is_named(tmp_path): ...
    # robot.yaml copied with the model_check block deleted -> one problem naming 'model_check'

def test_the_example_passes():
    cfg, d = load_planner_config("example:gen3.yaml")
    assert "<ros2_control" not in check_robot_dir(cfg, d, (DATA / "arm_level.urdf").read_text())
```

- [ ] **Step 2: Run** `PYTHONPATH=core python3 -m pytest core/tests/test_contract.py -q` — Expected: FAIL (no module).
- [ ] **Step 3: Implement** `contract.py` (run each loader under `try`, append its message, continue); wire the planner and the node.
- [ ] **Step 4: Run** core + ROS suites — Expected: all pass; the node's existing "model refused" test now sees the aggregated message.
- [ ] **Step 5: Commit** `feat(contract): check a robot directory and its URDF at startup, every problem at once`

### Task 4: Example mode in the node

**Files:**
- Create: `core/rammp_curobo/configs/arm_level.urdf` (copy of `core/tests/data/arm_level.urdf`)
- Modify: `core/pyproject.toml:27` (package data adds `"configs/*.urdf"`)
- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py` (`config` default `""`; example branch; banner; topic WARN; markers), `rammp_curobo_ros/launch/planner.launch.py:31` (default `""`)
- Test: `rammp_curobo_ros/test/test_node_wiring.py` (or a new `test_example_mode.py` in the same style)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `EXAMPLE_CONFIG = "example:gen3.yaml"`, `EXAMPLE_URDF = "example:arm_level.urdf"`, `EXAMPLE_MARKER = "(example robot)"` in `planner_node.py`; `node.example_mode: bool`.

Behaviour (exact copy):
- `config == ""` → example mode: no wait on `/arm/robot_description`; the model is `EXAMPLE_URDF`. Startup WARN, before anything else: `"=== NO CONFIG GIVEN: running the EXAMPLE robot from the image (Gen3 + 2F-85, level mount, world_sim_kitchen). Not your robot: pass config:=/config/robot/curobo/planner.yaml. ==="`
- Example mode keeps a subscription on `/arm/robot_description`; the first model received logs WARN once: `"a robot model is being published on /arm/robot_description, and this planner is IGNORING it: no config was given, so it plans the example robot."`
- Every PlanToPose / PlanToJoints result `message` and every SetWorld reply ends with `" " + EXAMPLE_MARKER` in example mode.
- In every mode, one startup INFO names where each piece came from: `robot data: config <path>, robot file <path>, world <path>, layers [<path>, …], model <topic|example>`, each path that `is_example` marks followed by `(example, image)`. A pushed world in `update_world` resolves by the same Task 1 rule (it already goes through `resolve_config` with `relative_to=config_dir`); the SetWorld reply names its path the same way.
- `config` non-empty and not resolvable → FATAL naming the path, exit 1 — never example mode.
- Named config: unchanged path (wait for the topic model; `EXAMPLE_URDF` never read).

- [ ] **Step 1: Write the failing tests:** example mode builds with no publisher on the topic and logs the banner; a latched model in example mode produces the WARN once; results and SetWorld replies carry the marker; `config:=/nonexistent/planner.yaml` exits 1 with that path in the FATAL and no banner.
- [ ] **Step 2: Run** `python3 -m pytest rammp_curobo_ros/test -q -p no:anyio -k example` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the ROS suite — Expected: all pass. On abra: `docker run --rm --runtime nvidia --network host -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=88 <image>` with no args reaches "rammp_curobo ready" and `ros2 run rammp_curobo_ros tour_demo` (same domain) completes.
- [ ] **Step 5: Commit** `feat(node): no config runs the example robot standalone, and says so everywhere`

### Task 5: The bench leaves the image

**Files:**
- Delete: `core/rammp_curobo/configs/gen3_real.yaml`, `core/rammp_curobo/configs/world_real_bench.yaml`
- Create: `core/tests/data/dojo/planner.yaml`, `core/tests/data/dojo/world.yaml` (fixture copies; `robot: example:robot_gen3_2f85.yaml`; the bench's planner settings and boxes verbatim)
- Modify: tests that used them (`test_offline.py:135-136`, `test_layers.py:763,1266`, `core/tests/data/curobo_at_limit.json` source note stays), `planner_node.py` sim-world tripwire text (no `world_real_bench.yaml` hint: say "pass your robot's config"), README:139, `rammp_curobo_ros/launch/planner.launch.py:10`
- Test: existing suites

**Interfaces:** Consumes Task 1. Produces the `core/tests/data/dojo/` fixture that Task 10 copies.

- [ ] **Step 1:** Move the files, rewrite references to `core/tests/data/dojo/...`.
- [ ] **Step 2: Run** core + ROS suites — Expected: all pass; `git grep -n "gen3_real\|world_real_bench" -- core/rammp_curobo rammp_curobo_ros` prints nothing.
- [ ] **Step 3: Commit** `refactor!: the dojo bench is dojo's data, not the image's`

### Task 6: The generator reads the robot file next to its input

**Files:**
- Modify: `scripts/generate_layers.py:458` (`keep_out.robot_config` resolved with `resolve_config(ref, relative_to=<--data file's dir>, search_cwd=False)`), its generated-file header (name the `--data` file actually used, relative to the repo when inside it)
- Modify: `scripts/mebot_layers.yaml` (`keep_out.robot_config: example:robot_gen3_2f85.yaml`), `core/tests/data/mebot/planner.yaml` (`robot: example:robot_gen3_2f85.yaml`)
- Modify: `Makefile:93-100` (`DATA ?= scripts/mebot_layers.yaml`, passed as `--data`; the diff-assert covers platform, seat, person, floor)
- Test: `core/tests/test_layergen.py`

**Interfaces:** Consumes Task 1. Produces `make layers DATA=<file> OUTDIR=<dir>` (used by Task 9).

- [ ] **Step 1: Write the failing test:** `--data` pointing at a copy of `mebot_layers.yaml` in `tmp_path` whose `keep_out.robot_config: robot.yaml`, with a copy of the example robot file there, generates byte-identical layers to the in-repo ones (headers aside); without the `robot.yaml` copy, it fails naming `tmp_path`.
- [ ] **Step 2: Run** `PYTHONPATH=core python3 -m pytest core/tests/test_layergen.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/core make layers` — Expected: no diff on the four layers (headers regenerate identically for the default `DATA`); `make sim-test` and `make sim` (abra) pass G0–G7.
- [ ] **Step 5: Commit** `feat(layers): the generator takes its robot file from next to its input`

### Task 7: The image holds code and examples only

**Files:**
- Modify: `docker/Dockerfile:96-103` (copy `core/pyproject.toml`, `core/rammp_curobo/`, `rammp_curobo_interfaces/`, `rammp_curobo_ros/` only; no `COPY .`)
- Modify: `Makefile:51-53` (`verify` bind-mounts the checkout read-only, e.g. `-v $(CURDIR):/src:ro`, and runs `python3 -m pytest /src/core/tests -q -p no:cacheprovider`); new target `image-contents` asserting `/opt/rammp_curobo_src` has no `core/tests`, `scripts`, `sim`, `docs`
- Test: `make verify`, `make image-contents`

**Interfaces:** none new.

- [ ] **Step 1: Add** the `image-contents` target and run it against the current image — Expected: FAIL (`core/tests` present).
- [ ] **Step 2: Change** the Dockerfile and `verify`.
- [ ] **Step 3: Run** on abra: `make image && make image-contents && make verify` — Expected: contents check passes; verify passes the same count as `PYTHONPATH=core RAMMP_CUROBO_GPU=1 pytest core/tests` in the checkout (conftest's checkout guard must accept the installed package; adjust it only if it refuses).
- [ ] **Step 4: Commit** `build(docker)!: copy code, not the repo; verify mounts the tests`

### Task 8: The contract, written down

**Files:**
- Create: `docs/ROBOT_DATA.md` — the contract: the robot directory table and the resolution rule from the spec, every robot-file field `load_robot_spec` and `check_robot_dir` require (with the example's values), `example:`, example mode and its three signals, and "swapping an arm/base/site" as a deployments checklist.
- Modify: `CLAUDE.md` (replace the "chair is NOT package data" invariant with the general rule; robot names from the robot file; example mode; the bench is dojo's), `INTEGRATION.md` (`from_config("example:gen3.yaml")`, robot directories), `README.md`, `docker/README.md`, `CHANGELOG.md` `[Unreleased]` **BREAKING** (bare packaged names refused; `gen3_real.yaml`/`world_real_bench.yaml` removed; `prepare_urdf` signature; `config` default `""`)
- Test: `mdformat --check` via pre-commit

- [ ] **Step 1: Write** the docs. **Step 2: Run** `pre-commit run --all-files` — Expected: pass. **Step 3: Commit** `docs: the robot-data contract`
- [ ] **Step 4: Open the RAMMP-CuRobo PR (draft)** against `dev`, linking the spec and noting it must merge after the deployments PR (Tasks 9–10). Ask the user before marking ready.

### Task 9 (rammp-deployments): flatbot's complete robot directory

**Files** (in `december_2026/config/robots/flatbot/curobo/`):
- Create: `robot.yaml` (copy of the example `robot_gen3_2f85.yaml` with `model_check`), `arm_kinematics.json` (copy), `mebot_layers.yaml` (copy of RAMMP-CuRobo's, `keep_out.robot_config: robot.yaml`, `outputs` unchanged — `--outdir` keeps only file names), `layer_mebot_seat.yaml`, `layer_mebot_person.yaml` (generated)
- Modify: `planner.yaml` (`robot: robot.yaml`; `layers: [layer_mebot_platform.yaml, layer_mebot_seat.yaml, layer_mebot_person.yaml]`; comments: these are flatbot's unmeasured best guesses from RAMMP-CuRobo #46, and the rest pose sits 25 mm inside `person/torso`, so plans from rest are refused), `layer_mebot_platform.yaml`, `layer_mebot_floor.yaml` (regenerated), `world.yaml` unchanged
- Modify: `profiles/flatbot.yaml` override comment ("no occupant for now" goes)

- [ ] **Step 1: Generate** from the RAMMP-CuRobo branch: `PYTHONPATH=$PWD/core make layers DATA=<deployments>/…/flatbot/curobo/mebot_layers.yaml OUTDIR=<deployments>/…/flatbot/curobo` — Expected: four layer files written; the in-repo diff assert still passes.
- [ ] **Step 2: Verify with today's nightly** on abra (GPU, no arm): build `CuRoboPlanner.from_config("/cfg/planner.yaml", urdf_path=<arm_tilt30.urdf>)` inside `ghcr.io/rammp-org/rammp-curobo:nightly` with the directory mounted at `/cfg` — Expected: world summary `30 boxes: scene 0 | config layers: platform 16, seat 5, person 8 | scene layers: floor 1`. Repeat with the RAMMP-CuRobo branch image — Expected: same summary, no `RobotDataError`.
- [ ] **Step 3: Commit** `flatbot: its own robot file and chair measurements; seat and person layers`

### Task 10 (rammp-deployments): dojo's robot directory; the manifest stops defaulting to the image

**Files:**
- Create: `december_2026/config/robots/dojo/curobo/planner.yaml` (RAMMP-CuRobo's `core/tests/data/dojo/planner.yaml` with `robot: robot.yaml`), `world.yaml` (the bench), `robot.yaml`, `arm_kinematics.json`
- Modify: `december_2026/sheppy-manifest.yaml` `curobo_planner_nightly.params` → `config: /config/robot/curobo/planner.yaml`, `world: ""` (comment: every robot that runs the nightly planner carries `curobo/`). Leave the `curobo_planner` (1.1.0) release entry alone: that image still ships the bench and predates the rule.
- Modify: `profiles/flatbot.yaml` — its `config`/`world` overrides now equal the default; keep them (explicit is fine) or drop them, but say which in the PR.

- [ ] **Step 1: Verify** on abra with both images, as Task 9 Step 2, using `arm_level.urdf` — Expected: summary shows the bench's pedestal and table, no layers.
- [ ] **Step 2: Commit** `dojo: its own planner directory; the nightly planner reads /config/robot/curobo`
- [ ] **Step 3: Open the deployments PR (draft)** against `dev` — the user may have to open it themselves (an earlier attempt was blocked). It merges FIRST.

## After both merge (human, not an agent)

- The next nightly carries the RAMMP-CuRobo change. Bring-up on snorlax (flatbot) and dojo follows docs/HARDWARE_BRINGUP.md with a person at the e-stop; first check the startup log names `/config/robot/curobo/...` for every piece and shows no `(example robot)`.
- Snorlax still runs from the local worktree `~/rammp-deployments-curobo`; returning it to `~/rammp-deployments` is the user's call.
- kinova_arm_ros2's docs (`docs/guide-goto-actions.md:31`, `guide-goto-ee-pose.md:32`) say `config:=gen3_real.yaml`; they need updating in that repo (the user's).
