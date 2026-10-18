# Plan Against the Arm's Own Model, in `arm_mount` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The planner builds its robot model from the URDF the robot publishes on `/arm/robot_description`, plans in `arm_mount` with the `arm_` joint names and the URDF's joint limits, keeps its audited collision spheres (moved onto the upstream links, not re-tuned), stamps every trajectory with `header.frame_id: arm_mount`, and can run with an empty collision world.

**Architecture:** `core` stays file-based and ROS-free: a new `urdf_model` module validates and sanitises the URDF text, and `CuRoboPlanner` takes the path of the file it was written to. The baked robot config switches to the upstream link names, and the bake script moves each sphere centre through forward kinematics of both models, so NVIDIA's model remains the source the spheres were audited on. The ROS node waits for the latched topic, writes the XML to a temp file, and hands the path to `core`. `Trajectory` gains a `frame_id`, which the conversion copies into the message header.

**Tech Stack:** Python 3.10, cuRobo **v0.7.8** (pinned), yourdfpy (ships with cuRobo), ROS 2 Humble (`rclpy`, `std_msgs`, `trajectory_msgs`), pytest, Docker on the Jetson (`rammp-curobo:jp6`).

**Spec:** `rammp-deployments` branch `feature/tilted-arm-mount`, `docs/superpowers/specs/2026-10-15-tilted-arm-mount-design.md`, section **RAMMP-CuRobo** (read with `git -C ~/atdev/rammp-deployments fetch && git -C ~/atdev/rammp-deployments show origin/feature/tilted-arm-mount:docs/superpowers/specs/2026-10-15-tilted-arm-mount-design.md`). The consumer contract is in kinova-gen3-ros2 PR #47: on a tilted mount, `GoToEEPose` refuses a pose plan whose `header.frame_id` is not exactly `arm_mount`.

## Where work runs

- **Code** is written on `muk` (x86, no cuRobo): offline tests (`core/tests/test_offline.py`, `rammp_curobo_ros/test`) run there if their imports allow; everything else runs on `abra`.
- **GPU tests, bake and the image** run on `abra` (Jetson AGX Orin, `ssh abra` from muk) inside the Jetson image. Build a branch image there with `docker build -f docker/Dockerfile -t rammp-curobo:arm-mount .` from a checkout of this branch (rsync or `git fetch` on abra). It rebuilds only from the source `COPY` layer once cuRobo's layer is cached. Run tests as `docker run --rm --runtime nvidia rammp-curobo:arm-mount python3 -m pytest /opt/rammp_curobo_src/core/tests -q`.
- **Never** start, stop or command an arm stack on abra (`sheppy-*` containers belong to the lab). This repo only plans; keep `--network none` on test containers except Task 6's live check, which uses `ROS_DOMAIN_ID=77` and `ROS_LOCALHOST_ONLY=1`.

## Global Constraints

- cuRobo **v0.7.8** only (v0.8 is an API rewrite — never upgrade); `core/` has **no ROS imports**.
- This repo never touches the arm: no executor, no `/joint_states`, no controller clients (CLAUDE.md, issue #6).
- Frame for all Cartesian I/O: `arm_mount` (Z against gravity when the chair is level, X forward, origin at the arm's base).
- Kinematics: `base_link: arm_mount`, `ee_link: arm_tool_frame`. Joint names `arm_joint_1 … arm_joint_7`. The retract seed's **values** are unchanged: `[0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]`.
- The gripper is locked open: `lock_joints` holds the actuated `arm_robotiq_85_left_knuckle_joint` at `0.0`, and cuRobo's mimic handling holds the other five (cuRobo v0.7.8 refuses mimic joints in `lock_joints`).
- Joint limits come from the URDF — the values the driver enforces (j2/j4/j6 ±2.24/±2.57/±2.09; continuous j1/3/5/7 become ±6.28 in cuRobo's parser).
- **Spheres are moved, not re-tuned.** Same count, same radii; only centres are re-expressed.
- Verification (spec): at a set of configurations every sphere's world position in `arm_mount` matches the old model's within **1 mm**; `arm_tool_frame` matches NVIDIA's `tool_frame` within **1 mm and 0.1°**; the existing self-collision checks pass.
- Every returned trajectory: `header.frame_id == "arm_mount"` (pose and joint plans alike).
- Topic: `/arm/robot_description`, `std_msgs/String`, QoS RELIABLE + TRANSIENT_LOCAL depth 1; wait 30 s then fail with an error naming `arm_state_publisher`.
- Not in this change: per-robot collision worlds, wrist-camera spheres, re-measuring tool corrections.

## Rulings made writing this plan

- **Ruling: the URDF is sanitised before cuRobo reads it** — `<visual>`, `<collision>` and `<ros2_control>` elements are stripped, and `mesh_link_names` becomes `[]`. The rendered model's mesh paths point at `/opt/ros/humble/share/...`, which this image does not contain; the spheres are the collision model, so meshes are never needed. *Cost if wrong:* nothing that reads meshes from this config (visualisation helpers) would work — none do today.
- **Ruling: the empty world is allowed only at construction.** `make_world_config(..., allow_empty=True)` at `CuRoboPlanner` init; `update_world` keeps refusing an empty world (cuRobo silently keeps the old one there). The node passes its `world` parameter into the config **before** construction instead of via `update_world`. Deployments choose `world_empty.yaml` per robot (flatbot) with the existing `world` parameter; defaults stay `world_real_bench.yaml` / `world_sim_kitchen.yaml`, so dojo keeps its table. If cuRobo v0.7.8 refuses an empty `WorldConfig` at `MotionGen` init, stop and report rather than inventing a dummy obstacle.
- **Ruling: this lands before the four open PRs (#28, #30, #31, #33)** -- it changes the model they all build on, so they adapt to it. Keep each change minimal where a PR rewrites the same code (`make_world_config`, the node's construction block, the bake script) so their rebases stay small; do not pre-port their features. *Cost if wrong:* their authors absorb the rebases listed in [Downstream](#downstream-the-open-prs-after-this-lands).
- **Ruling: `limit_clamp_rad` stays 0.05.** Tracked in kinova-gen3-ros2#49 (limits will match the arm firmware, set in our own xacro, #48); until then: the Gen3's park pose (`joint_4 ≈ -2.674`) is 0.10 rad outside the new ±2.57 limit, so a plan from park is **refused** with `arm_joint_4` named. Do not widen the clamp or the limit to hide this.

## Review Focus

1. A rendered URDF whose mesh files do not exist in the image must load (Task 1: `test_meshes_are_stripped_and_missing_files_load`).
1. Mimic tags and articulated gripper joints in the URDF must leave cuRobo with exactly the 7 arm joints in chain order (Task 3: `test_kinematics_joint_names_are_the_seven_arm_joints`).
1. A tilted model (`rpy_deg [0, 30, 0]`) must plan in `arm_mount`: the same arm-relative goal, rotated by `Ry(30°)`, gives the same joint solution (Task 3: `test_tilted_model_plans_in_arm_mount`).
1. A start pose outside the new limits by more than 0.05 rad is refused naming the joint, not silently clamped (Task 3: `test_park_pose_beyond_clamp_is_refused_by_name`).
1. A model that is not ours (unprefixed names, root not `arm_mount`, no `arm_tool_frame`) is refused by name before cuRobo sees it (Task 1: `test_foreign_model_is_refused_by_name`).

______________________________________________________________________

### Task 1: Test fixtures and URDF intake in `core`

**Files:**

- Create: `core/tests/data/arm_level.urdf`, `core/tests/data/arm_tilt30.urdf`, `core/tests/data/README.md`
- Create: `core/rammp_curobo/urdf_model.py`
- Modify: `core/rammp_curobo/robot_config.py` (`load_robot_config`)
- Test: `core/tests/test_offline.py`

**Interfaces:**

- Produces: `prepare_urdf(xml: str) -> str`, `write_urdf(xml: str, directory: str | Path) -> Path`, constants `ROOT_LINK = "arm_mount"`, `EE_LINK = "arm_tool_frame"`, `ARM_JOINTS: tuple[str, ...]` (`arm_joint_1..7`), `GRIPPER_JOINTS: tuple[str, ...]` (the six 2F-85 joints `prepare_urdf` requires), all in `rammp_curobo.urdf_model`. `load_robot_config(path, urdf_path: str | Path | None = None) -> dict`.

- [ ] **Step 1: Render the fixtures.** On abra, from an image built from kinova-gen3-ros2 `feat/tilted-arm-mount` (or `dev` once PR #47 merges), render two models with `python3 -m kinova_gen3_render --mount <file> --out <out>` (source `/module_ws/install/setup.bash`): `robot: fixture_level`, `rpy_deg: [0, 0, 0]` → `arm_level.urdf`; `robot: fixture_tilt30`, `rpy_deg: [0, 30, 0]` → `arm_tilt30.urdf`. Commit them unmodified. `data/README.md` records the image, commit and both commands. Check the bracelet is the no-vision one: `grep -c bracelet_no_vision_link arm_level.urdf` ≥ 1.

- [ ] **Step 2: Write the failing tests** in `core/tests/test_offline.py` (pure Python, `xml.etree` only):

```python
def test_meshes_are_stripped_and_missing_files_load():
    xml = prepare_urdf(read("arm_level.urdf"))
    root = ET.fromstring(xml)
    assert not root.findall(".//visual") and not root.findall(".//collision")
    assert not root.findall("ros2_control")
    assert root.find("link[@name='arm_mount']") is not None

def test_foreign_model_is_refused_by_name():
    for broken, why in [
        (read("arm_level.urdf").replace('"arm_mount"', '"base_link"'), "arm_mount"),
        (read("arm_level.urdf").replace("arm_tool_frame", "tool_frame"), "arm_tool_frame"),
        (read("arm_level.urdf").replace("arm_joint_4", "joint_4"), "arm_joint_4"),
        ("", "empty"), ("<robot", "parse"),
    ]:
        with pytest.raises(ValueError, match=why):
            prepare_urdf(broken)

def test_robot_config_takes_the_urdf_path(tmp_path):
    p = write_urdf(read("arm_level.urdf"), tmp_path)
    kin = load_robot_config("robot_gen3_2f85.yaml", urdf_path=p)["kinematics"]
    assert kin["urdf_path"] == str(p) and kin["asset_root_path"] == str(tmp_path)

def test_robot_config_without_a_urdf_is_refused():
    with pytest.raises(ValueError, match="/arm/robot_description"):
        load_robot_config("robot_gen3_2f85.yaml")
```

- [ ] **Step 3: Run them to see them fail** — `python3 -m pytest core/tests/test_offline.py -k "urdf or foreign or meshes" -q` → FAIL (`ModuleNotFoundError: rammp_curobo.urdf_model`).

- [ ] **Step 4: Implement `urdf_model.py` and the `load_robot_config` change.** `prepare_urdf` parses with `xml.etree.ElementTree`, refuses (ValueError naming the missing thing) an empty string, unparseable XML, a root link other than `arm_mount`, a missing `arm_tool_frame` link or any missing `ARM_JOINTS`, then strips `visual`, `collision` and top-level `ros2_control` and returns the serialised XML. `write_urdf` writes `robot_description.urdf` into `directory` and returns its path. `load_robot_config` gets `urdf_path`: when given, set `kinematics.urdf_path` to its absolute path and `kinematics.asset_root_path` to its directory; when absent and the YAML's `urdf_path` is the sentinel `from_robot_description` (Task 2 writes it), raise `ValueError("robot model required: pass the URDF from /arm/robot_description ...")`. Until Task 2 lands, make the tests that need the sentinel pass by writing it into the YAML in this task.

- [ ] **Step 5: Run them to see them pass** — same command → PASS; full offline suite still passes.

- [ ] **Step 6: Commit** — `feat(core): take the arm's model from a URDF file; fixtures rendered from kinova-gen3-ros2`.

### Task 2: Bake the robot config onto the upstream links

**Files:**

- Modify: `scripts/bake_robot_config.py`, `core/rammp_curobo/configs/robot_gen3_2f85.yaml` (regenerated), `Makefile` (`bake` runs in the image), `core/rammp_curobo/config.py` (`PLANNER_DEFAULTS["joint_names"]`), `core/rammp_curobo/configs/gen3.yaml` (`joint_names`)
- Create: `core/tests/data/robot_gen3_2f85_nvidia.yaml` (today's baked config, copied verbatim, the parity reference)
- Test: `core/tests/test_offline.py`

**Interfaces:**

- Consumes: `ARM_JOINTS`, `GRIPPER_JOINTS`, `ROOT_LINK`, `EE_LINK` (Task 1); fixture `arm_level.urdf`.
- Produces: `LINK_MAP: dict[str, str]` in `bake_robot_config.py` (NVIDIA link → upstream link); the regenerated `robot_gen3_2f85.yaml` with `base_link: arm_mount`, `ee_link: arm_tool_frame`, `urdf_path: from_robot_description`, `asset_root_path: ""`, `mesh_link_names: []`, `lock_joints: {arm_robotiq_85_left_knuckle_joint: 0.0}`, cspace `joint_names: [arm_joint_1..7]`.

`LINK_MAP` (arm links keep their frames, gain the prefix; gripper links move to the upstream 2F-85):

| NVIDIA                                             | upstream                                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `base_link` … `bracelet_link` (8 arm links)        | `arm_` + same name                                                             |
| `robotiq_arg2f_base_link`                          | `arm_robotiq_85_base_link`                                                     |
| `left_outer_finger` / `right_outer_finger`         | `arm_robotiq_85_left_finger_link` / `arm_robotiq_85_right_finger_link`         |
| `left_inner_finger_pad` / `right_inner_finger_pad` | `arm_robotiq_85_left_finger_tip_link` / `arm_robotiq_85_right_finger_tip_link` |
| `left_outer_knuckle` (buffer entry only)           | `arm_robotiq_85_left_knuckle_link`                                             |

- [ ] **Step 1: Copy today's baked config** to `core/tests/data/robot_gen3_2f85_nvidia.yaml` before changing anything.

- [ ] **Step 2: Write the failing offline test** (replaces `test_baked_robot_config_carries_the_patches`):

```python
def test_baked_robot_config_is_the_upstream_model():
    kin = load_yaml_robot("robot_gen3_2f85.yaml")["kinematics"]
    old = load_yaml_robot(DATA / "robot_gen3_2f85_nvidia.yaml")["kinematics"]
    assert kin["base_link"] == "arm_mount" and kin["ee_link"] == "arm_tool_frame"
    assert kin["urdf_path"] == "from_robot_description" and kin["mesh_link_names"] == []
    assert kin["lock_joints"] == {j: 0.0 for j in GRIPPER_JOINTS}
    assert kin["cspace"]["joint_names"] == list(ARM_JOINTS)
    assert kin["cspace"]["retract_config"] == old["cspace"]["retract_config"]
    for nv, up in LINK_MAP.items():   # moved, not re-tuned
        if nv in old["collision_spheres"]:
            assert [s["radius"] for s in kin["collision_spheres"][up]] == \
                   [s["radius"] for s in old["collision_spheres"][nv]]
    assert set(kin["collision_link_names"]) == {LINK_MAP[n] for n in old["collision_link_names"]}
```

and `test_planner_config_loads_and_merges` / `test_gen3_yaml_matches_planner_defaults` now expect `["arm_joint_%d" % i for i in range(1, 8)]`.

- [ ] **Step 3: Run to see it fail** (offline, muk or abra) → FAIL on `base_link`.

- [ ] **Step 4: Change `bake_robot_config.py`** to emit the upstream model. Sphere centres are authored in NVIDIA link frames (`ARM_SPHERES`, `GRIPPER_SHELL`, the stock gripper spheres) and moved by FK: at `q = 0` (gripper rigid in NVIDIA's model, locked open at 0 in ours) compute each link's pose in both models with yourdfpy — NVIDIA's URDF from cuRobo's content tree, ours from `core/tests/data/arm_level.urdf` (or `--urdf <path>`) — and set `c_up = T_up⁻¹ · T_nv · c_nv`. Rename `self_collision_ignore`/`self_collision_buffer` keys and entries through `LINK_MAP`. Write the keys listed in Produces. Keep the `make bake` no-diff rule; make the target run the script inside the image (`docker run --rm -v $(PWD):/w -w /w $(IMAGE) python3 scripts/bake_robot_config.py`). Update `PLANNER_DEFAULTS["joint_names"]` and `gen3.yaml`'s `joint_names` to `arm_joint_1..7`.

- [ ] **Step 5: Regenerate on abra and run the offline suite** — `make bake` (in the image) then `python3 -m pytest core/tests/test_offline.py -q` → PASS; `git diff` shows the regenerated YAML.

- [ ] **Step 6: Commit** — `feat(config): bake the robot config onto the upstream links; spheres moved by FK, not re-tuned`.

### Task 3: Planner on the URDF; parity and limits verified on the GPU

**Files:**

- Modify: `core/rammp_curobo/planner.py` (`__init__`, `from_config`, `_finish`, `_explain`, `plan_to_pose` docstring), `core/rammp_curobo/types.py` (`Trajectory`), `core/rammp_curobo/configs/gen3_real.yaml` (header comment)
- Create: `core/tests/test_model_parity.py` (GPU), `core/tests/conftest.py` (`urdf_level`, `urdf_tilt30` fixtures returning written paths)
- Modify: `core/tests/test_smoke.py` (fixture passes `urdf_path`; names and limits)

**Interfaces:**

- Consumes: `load_robot_config(path, urdf_path)` (Task 1); the regenerated config (Task 2).

- Produces: `CuRoboPlanner(config: dict, config_dir=None, *, urdf_path: str | Path | None = None)`, `CuRoboPlanner.from_config(name_or_path, *, urdf_path=None)` -- **keyword-only**, so open PR #31's positional `from_config(name, overrides)` cannot land a dict in `urdf_path`; `Trajectory.frame_id: str` (new field, default `""`); `CuRoboPlanner.base_frame -> str` (the robot config's `base_link`, i.e. `"arm_mount"`).

- [ ] **Step 1: Write the failing GPU tests** in `test_model_parity.py` (module-skip without CUDA like `test_smoke.py`). The reference is NVIDIA's model built from `data/robot_gen3_2f85_nvidia.yaml` (urdf resolved in cuRobo's content tree):

```python
CONFIGS = [retract, zeros, *seeded_random(20, seed=0)]   # within both models' limits

def test_spheres_match_the_old_model_within_1mm(new_kin, old_kin):
    for q in CONFIGS:   # world positions, arm_mount (new) vs base_link (old), level mount
        assert max_pair_distance(new_kin.spheres(q), old_kin.spheres(q)) < 1e-3

def test_tool_frame_matches_nvidia_within_1mm_and_0_1deg(new_kin, old_kin):
    for q in CONFIGS:
        dp, dang = pose_error(new_kin.ee_pose(q), old_kin.ee_pose(q))
        assert dp < 1e-3 and dang < math.radians(0.1)

def test_kinematics_joint_names_are_the_seven_arm_joints(planner):
    assert planner._motion_gen.kinematics.joint_names == list(ARM_JOINTS)

def test_joint_limits_are_the_urdfs(planner):
    lo, hi = planner.joint_limits()["position"]
    assert hi[1] == pytest.approx(2.24) and hi[3] == pytest.approx(2.57) and hi[5] == pytest.approx(2.09)

def test_every_plan_is_stamped_arm_mount(planner, start):
    assert planner.plan_to_joints(goal_near(start), start).joint_traj.frame_id == "arm_mount"
    assert planner.plan_to_pose(*reachable_pose(planner, start), start).joint_traj.frame_id == "arm_mount"

def test_tilted_model_plans_in_arm_mount(urdf_level, urdf_tilt30):
    level, tilt = planner_for(urdf_level), planner_for(urdf_tilt30)
    p, quat = level.fk(start)                      # a pose the level arm reaches at `start`
    R = Ry(30°)
    res = tilt.plan_to_pose(R @ p, rotate(R, quat), start)
    assert res.success and max_abs(res.final_joints - level.plan_to_pose(p, quat, start).final_joints) < 1e-2

def test_park_pose_beyond_clamp_is_refused_by_name(planner):
    park = list(planner.retract_pose); park[3] = -2.674          # real Gen3 park; limit ±2.57
    res = planner.plan_to_joints(planner.retract_pose, park)
    assert not res.success and "arm_joint_4" in res.error
```

`max_pair_distance` pairs spheres through `LINK_MAP` (same order per link). Self-collision: `planner.check_state_valid(retract)` and the existing `test_check_state_valid` stay green. Update `test_smoke.py`'s `planner` fixture to `from_config("gen3.yaml", urdf_path=urdf_level)`, its `joint_N` strings to `arm_joint_N`, the joint_2 limit comment to ±2.24, and replace `test_park_pose_outside_model_limits_is_clamped` with a clamp case **inside** 0.05 rad (`park[3] = -2.60` → clamped to -2.57, error-free).

- [ ] **Step 2: Build the branch image on abra and run them to see them fail** — `docker build ... -t rammp-curobo:arm-mount .` then `docker run --rm --runtime nvidia --network none rammp-curobo:arm-mount python3 -m pytest /opt/rammp_curobo_src/core/tests -q` → FAIL (`from_config() got an unexpected keyword argument 'urdf_path'`).

- [ ] **Step 3: Implement.** `CuRoboPlanner.__init__` passes `urdf_path` to `load_robot_config`; `from_config` forwards it. `_finish` sets `Trajectory(frame_id=self.base_frame, ...)`. `_explain`'s IK message and the `plan_to_pose` docstring say `arm_tool_frame` / `arm_mount`. Nothing else in the planner changes: limits already come from `get_joint_limits()`.

- [ ] **Step 4: Rebuild and run the full core suite** — same command → all PASS (record the count).

- [ ] **Step 5: Commit** — `feat(planner): plan on the arm's own URDF in arm_mount; parity with NVIDIA's model pinned at 1 mm`.

### Task 4: An empty collision world, at construction only

**Files:**

- Modify: `core/rammp_curobo/world.py` (`make_world_config`), `core/rammp_curobo/planner.py` (`_init_curobo`)
- Create: `core/rammp_curobo/configs/world_empty.yaml`
- Test: `core/tests/test_offline.py`, `core/tests/test_smoke.py`

**Interfaces:**

- Produces: `make_world_config(scene, padding=0.02, ignore=frozenset(), no_pad_names=frozenset(), cache_obb=40, allow_empty: bool = False)`; packaged `world_empty.yaml` (`base_frame: arm_mount`, `obstacles: []`, `objects: []`, `targets: []`).

- [ ] **Step 1: Write the failing tests** — offline: `make_world_config(load_scene("world_empty.yaml"), allow_empty=True)` returns a config with no cuboids, and without `allow_empty` still raises `match="empty"`; `test_packaged_configs_exist` includes `world_empty.yaml`. GPU (`test_smoke.py`): `CuRoboPlanner.from_config(cfg_with_world("world_empty.yaml"), urdf_path=urdf_level)` constructs and `plan_to_joints` from retract to a nearby goal succeeds; `update_world([])` on it still raises `match="empty"`.

- [ ] **Step 2: Run to see them fail** → FAIL (`unexpected keyword argument 'allow_empty'`).

- [ ] **Step 3: Implement** `allow_empty` (skip the `n < 1` refusal and return an empty `WorldConfig`); `_init_curobo` passes `allow_empty=True`; `update_world` keeps the default. If `MotionGen` refuses the empty config, stop and report (Rulings).

- [ ] **Step 4: Run offline + rebuilt GPU suites** → PASS.

- [ ] **Step 5: Commit** — `feat(world): an empty collision world at construction; update_world still refuses one`.

### Task 5: The ROS node reads `/arm/robot_description` and stamps the frame

**Files:**

- Modify: `rammp_curobo_ros/rammp_curobo_ros/planner_node.py`, `rammp_curobo_ros/rammp_curobo_ros/conversions.py`, `rammp_curobo_ros/package.xml` (`std_msgs`), `rammp_curobo_ros/launch/planner.launch.py`
- Test: `rammp_curobo_ros/test/test_robot_description.py`, `rammp_curobo_ros/test/test_start_joints_contract.py`

**Interfaces:**

- Consumes: `prepare_urdf`, `write_urdf` (Task 1); `CuRoboPlanner(config, config_dir, *, urdf_path=...)` (Task 3); `Trajectory.frame_id` (Task 3).

- Produces: node parameter `robot_description_timeout_s` (float, default `30.0`); `trajectory_to_msg(traj)` sets `msg.header.frame_id = traj.frame_id`; a module-level `planner_config(config: str, world: str) -> tuple[dict, Path]` that applies the `world` parameter to the config **before** construction.

- [ ] **Step 1: Write the failing tests** (offline with ROS messages on `PYTHONPATH`, the existing pattern): `trajectory_to_msg(Trajectory(..., frame_id="arm_mount")).header.frame_id == "arm_mount"`; `planner_config("gen3.yaml", "world_empty.yaml")[0]["world"] == "world_empty.yaml"` and `planner_config("gen3.yaml", "")` keeps the config's world; `JOINTS` in the start-joints contract test become `arm_joint_N`.

- [ ] **Step 2: Run to see them fail** — `python3 -m pytest rammp_curobo_ros/test -q -p no:anyio` → FAIL.

- [ ] **Step 3: Implement.** Before building the planner, the node subscribes to `/arm/robot_description` (`std_msgs/String`, `QoS(depth=1, reliability=RELIABLE, durability=TRANSIENT_LOCAL)`), spins until a message or `robot_description_timeout_s`, and on timeout logs FATAL `"no model on /arm/robot_description after N s. It is published by arm_state_publisher: ros2 launch kinova_gen3_description description.launch.py mount_file:=<robot>/mount.yaml"` and exits 1. It passes the XML through `prepare_urdf` (a `ValueError` is FATAL, message prefixed `model on /arm/robot_description:`), writes it with `write_urdf` into a `tempfile.mkdtemp()`, and builds `CuRoboPlanner(*planner_config(config, world), urdf_path=...)`. It logs `model from /arm/robot_description: base arm_mount, ee arm_tool_frame, joints [...]`. The old `update_world(self.world)` call after construction is removed. The launch file passes `robot_description_timeout_s` through unchanged (no new launch argument). SIGINT during the wait exits 0.

- [ ] **Step 4: Run the ROS tests** → PASS; `colcon build` of both packages in the rebuilt image succeeds.

- [ ] **Step 5: Commit** — `feat(node): the model comes from /arm/robot_description; every plan says arm_mount`.

### Task 6: Contract text, docs, and the live check on abra

**Files:**

- Modify: `rammp_curobo_interfaces/action/PlanToPose.action` (lines 1, 4, 28-31, 40-46, 66-68), `rammp_curobo_interfaces/action/PlanToJoints.action` (lines 1, 8), `core/rammp_curobo/constraints.py` (`PoseConstraint` docstring, lines 50-61), `CLAUDE.md`, `README.md`, `INTEGRATION.md`, `docker/README.md`, `CHANGELOG.md`, `rammp_curobo_ros/rammp_curobo_ros/tour_demo.py` (comments only)

- [ ] **Step 1: Contract text.** `PlanToPose.action` line 1: "a tool pose (`arm_tool_frame`) in `arm_mount`"; joint lists say `arm_joint_1..7`; the result field says its `header.frame_id` is `arm_mount`. Replace the "LEVEL WITH THE WORLD" paragraph (40-46) and the `constraints.py` caveat with the actual limitation: *HOLD_LEVEL holds level relative to `arm_mount`, which is level with the chair, not with gravity: on a ramp it is off by the ramp angle.* The hold maths and the verifier do not change.

- [ ] **Step 2: Docs.** CLAUDE.md and README: the model now comes from `/arm/robot_description` (no bundled URDF; `core` takes a path), `arm_` names, the empty world option, the fixtures and how they were rendered, the bake running in the image, the park-pose refusal (Rulings). INTEGRATION.md: callers send goals in `arm_mount` and read `header.frame_id`. CHANGELOG `Unreleased`: breaking (joint names, frame, required topic), the limits change (old ±2.41/±2.66/±2.23 → ±2.24/±2.57/±2.09), the frame stamp, `world_empty.yaml`.

- [ ] **Step 3: Live check on abra** (plan only — no arm stack touched). In one domain (`ROS_DOMAIN_ID=77`, `ROS_LOCALHOST_ONLY=1`), start `arm_state_publisher` from the kinova-gen3-ros2 image with a tilt-30 mount file, then this branch's image with `config:=gen3_real.yaml world:=world_empty.yaml`. Confirm the node logs the model line, then send one `plan_to_joints` and one `plan_to_pose` (`ros2 action send_goal`) and check both results have `header.frame_id: arm_mount` and `success: true`. Stop the planner, and confirm a restart without `arm_state_publisher` exits 1 after 30 s with the FATAL naming it.

- [ ] **Step 4: Full verification** — rebuilt image: `make verify` (core suite) and the ROS tests → PASS; record counts in the PR.

- [ ] **Step 5: Commit** — `docs: arm_mount contract, HOLD_LEVEL's real limitation, the model from /arm/robot_description`.

## Downstream: the open PRs after this lands

Not work for this branch. Post it on each PR when this merges, so the rebases are planned, not discovered.

- **#28 platform layers.** `allow_empty` must move into `compose_cuboids` (including its "empty apart from the configured layers" rule), `to_world_config` and the scene-only warm-up world, since #28 replaces `make_world_config` in the planner. `_curobo_stand_in` (`test_layers.py`) and `generate_layers.keep_out_spheres` use `joint_N` / `base_link` / `shoulder_link` -> `arm_` names. `sim/mebot_sim`: `stack.py` refuses non-`joint_N` trajectories, and the sim must publish a level `/arm/robot_description` or the planner node exits after 30 s. Regenerated layers change `layers_sha256` in `scenarios.yaml`. The generator's keep-out is not tilt-aware (fine on level mounts; follow-up before gen_1_5 layers).
- **#33 runtime layers.** `Obstacle.msg` says "planner's base frame (base_link)" with no frame field: after this the frame is `arm_mount`, and on flatbot a box sent in the arm's own base frame is silently 30° off. Recommend adding `string frame_id` to `SetLayer` (refuse anything but `arm_mount`) before it merges. Decide whether `set_layer` may sit on `world_empty.yaml`.
- **#30 attached object.** `ATTACHED_PARENT = "tool_frame"` -> `arm_tool_frame`; `ATTACHED_IGNORE` -> through `LINK_MAP` (a stale list is skipped silently and would make every held plan self-collide -- only the GPU `test_attached.py` catches it); the held YAML gets the same `lock_joints` / `urdf_path` / `mesh_link_names` keys; `test_attached.py` passes `urdf_path`.
- **#31 standoff override.** `from_config(name, overrides)` -> call as `overrides=` (keyword); the node can then apply `world` through `overrides` and drop this branch's `planner_config` helper.
