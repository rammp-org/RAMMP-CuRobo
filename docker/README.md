# The planning service as a container

One image = the whole pinned GPU stack (torch jp6/cu126, cuRobo v0.7.8,
warp 1.5.1) + ROS 2 Humble + this repo's planner node, planning-only.
Another codebase talks to it over two ROS actions and never has to touch
the cuRobo install again:

```
in:  /rammp_curobo/plan_to_pose   (geometry_msgs/Pose + start_joints)
     /rammp_curobo/plan_to_joints (7 joint angles    + start_joints)
out: trajectory_msgs/JointTrajectory — full cuRobo time-parameterization,
     joint_names arm_joint_1..7, header.frame_id "arm_mount"
needs: /arm/robot_description (std_msgs/String, latched) from arm_state_publisher
```

Poses in and out are `arm_tool_frame` in `arm_mount`, the root of the arm's
URDF. The container ships no robot model: it plans on the one the robot
publishes, read once at startup.

The container holds **no arm driver** and nothing inside it can move an
arm: executing the returned trajectory is the caller's job, on whatever
controller stack owns the robot.

## Where to build

On the machine that will RUN the container — any JetPack 6 (L4T r36.x)
Jetson with **~25 GB free during the build** (~15 GB image afterwards).
The lab's planning Jetson (57 GB eMMC, mostly full) deliberately does NOT
host this image; it was validated there only up to the pre-Docker line:
planner service + client contract proven live, base-image tag and the
cuRobo v0.7.8 tag→commit pin verified against the registries 2026-08-14.

**Built and verified on `abra` 2026-08-24** — L4T R36.5.2, Docker 27.5.1,
`nvidia` runtime registered, 796 GB free. The image builds, and the verify step
at the bottom of this file passes **23/23**. Checked inside the running
container:

```
torch 2.10.0, cuda_built 12.6      # --no-deps kept CUDA 12.9 out
sys.flags.utf8_mode == 1
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
libopenblas.so.0 and libcudss_*.so.0 both resolve via ldconfig
```

What that does *not* prove: layers 2–5 (apt/ROS, torch, the torch runtime deps,
cuRobo) were Docker **cache hits** from an earlier build on the same machine.
Layers 6–10 — including the Cyclone DDS layer, cache-busted by the source
`COPY` — rebuilt fresh. So a cold-cache build has not been timed here, and the
~1 h figure below is still an estimate rather than a measurement from this run.

## Prerequisites (once, on the target machine)

Engine + NVIDIA runtime, then let the runtime mount the GPU:

```bash
sudo apt-get update && sudo apt-get install -y docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker $USER   # then re-login
```

## Build (~1 h — cuRobo compiles its CUDA kernels)

```bash
git clone <this-repo> && cd RAMMP-CuRobo
docker build -f docker/Dockerfile -t rammp-curobo:jp6 .
```

## Run

```bash
docker run --rm -it --runtime nvidia --network host --ipc host \
    -e ROS_LOCALHOST_ONLY=1 \
    -v rammp-curobo-cache:/root/.cache \
    rammp-curobo:jp6
```

- `--network host --ipc host`: DDS discovery + shared-memory transport with
  nodes on the host. Keep `ROS_LOCALHOST_ONLY=1` matching the host shells
  (this lab's standard).
- The cache volume keeps compiled CUDA kernels — without it every cold
  container pays the minutes-long first-plan warmup again.
- **The model.** The node waits up to `robot_description_timeout_s`
  (default 30 s) for `arm_state_publisher` on `/arm/robot_description`
  (`ros2 launch kinova_gen3_description description.launch.py mount_file:=<robot>/mount.yaml`),
  in the same ROS domain. A later, different model is never used, and the
  node logs an ERROR each time one arrives: **restart the container whenever
  the robot's mount changes.**
- **Exit codes.** The image's CMD execs the planner node itself (PID 1), not
  `ros2 launch`, so the container's exit code is the node's: no model in
  time is a FATAL naming `arm_state_publisher` and a **non-zero exit (1)**,
  which a restart policy or supervisor sees; `docker stop` exits 0 promptly.
- Different world or parameters: replace the CMD with the node and your own
  `-p` args, e.g. `... rammp-curobo:jp6 /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node --ros-args -p config:=gen3_real.yaml -p world:=world_empty.yaml`
  (`world_empty.yaml`: the arm alone, for a robot with nothing in reach to
  model). `ros2 launch rammp_curobo_ros planner.launch.py ...` still works
  but always exits 0. Under sheppy, see below.
- The arm on the MEBot wheelchair (a release after 1.1.0). The chair's planner
  config, its world and its three layer files are NOT part of the installed
  package, so no bare name resolves to them: they describe one chair's
  geometry and ship with that robot, in rammp-deployments
  `config/robots/<robot>/curobo/` (issue #19). The repo's **fixture** copy is
  in the image, at `/opt/rammp_curobo_src/core/tests/data/mebot/`, because the
  build copies the whole tree and `make verify` needs it — never point
  `config:` at that path. It is whatever the image was built from, not what
  the robot is wearing. Mount the robot's directory and name the config by
  path:
  `... -v ./config/robots/current:/config/robot:ro rammp-curobo:jp6 /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node --ros-args -p config:=/config/robot/curobo/planner.yaml`.
  Set `world` to `""` if your manifest declares one: a non-empty `world`
  replaces whatever the planner config names.
  The chair's drive base and the person in it are then in every world,
  worlds pushed over `/rammp_curobo/set_world` included, and the node's log
  says so:
  `Collision world: 26 boxes: scene 0 | config layers: platform 16, person 9 | scene layers: floor 1`.
  See INTEGRATION.md, "Planning on the MEBot". An image without obstacle
  layers has no such config and refuses it.
- **Verify the bridge on first start** (the container runs as root; mixed
  root/non-root Fast DDS shared memory is a classic silent-delivery
  failure): from a host shell, `ros2 action list | grep rammp_curobo`
  must show the plan actions, and one round-trip plan must return. If
  discovery works but calls hang, run the container with
  `--user $(id -u):$(id -g) -e HOME=/tmp` (move the cache volume to
  `/tmp/.cache`) so DDS shared memory is same-UID as the host nodes.

### Under sheppy (rammp-deployments)

One manifest serves dojo and flatbot. To give flatbot `world_empty.yaml` and keep
dojo's bench:

```yaml
- id: curobo_planner
  kind: docker
  ros_node_name: rammp_curobo          # key for the params file sheppy writes
  container:
    image: ghcr.io/rammp-org/rammp-curobo:<release with #38>
    # ... (network, runtime, volumes as today)
    # sheppy appends --ros-args --params-file only after a command; without
    # one it replaces the image CMD and the container exits 2.
    command:
      - /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node
      - --ros-args
      - -p
      - config:=gen3_real.yaml
  params:
    world: world_real_bench.yaml       # dojo's bench; flatbot overrides
```

and in flatbot's profile `overrides: {planner: {world: world_empty.yaml}}` (sheppy
applies overrides only to params the alternative declares, so the `params:` entry is
required).

Why this and not a plain `-p world:=world_empty.yaml`: `params:` without a
`command:` makes sheppy replace the image CMD with `--ros-args --params-file ...`
alone, which `/entrypoint.sh` cannot exec, and the container exits 2; hard-coding the
world in the shared manifest's `command:` drops dojo's bench, so dojo plans through
real objects.

The planner exits 1 when no model arrives within 30 s
(`robot_description_timeout_s`; a slow first `arm_state_publisher` start does it), and
sheppy does not restart it: after fixing `arm_state_publisher`, restart the planner
(`sheppy restart planner`).

## Client side (your codebase)

Build `rammp_curobo_interfaces` in your workspace (copy the package or add
this repo to your `src/`; it is dependency-free rosidl by design), then:

```python
import rclpy
from rammp_curobo_interfaces.action import PlanToPose
from rclpy.action import ActionClient

rclpy.init()
node = rclpy.create_node("my_planner_client")
client = ActionClient(node, PlanToPose, "/rammp_curobo/plan_to_pose")
client.wait_for_server()

goal = PlanToPose.Goal()
goal.target.position.x, goal.target.position.y, goal.target.position.z = p
(goal.target.orientation.x, goal.target.orientation.y,
 goal.target.orientation.z, goal.target.orientation.w) = quat_xyzw
goal.start_joints = list(q_now)   # REQUIRED: the configuration to plan
                                  # from. You own the arm's state.

# action calls only complete while the node spins — spin between the two
# futures (send_goal() without a spinning executor would block forever)
send = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, send)
result_future = send.result().get_result_async()
rclpy.spin_until_future_complete(node, result_future)   # planning: ~0.2-2 s warm
res = result_future.result().result
if res.success:
    execute(res.trajectory)       # your controller, your gates
```

(Inside an application that already spins its executor, drop the
`spin_until_future_complete` calls and await/collect the same futures —
see `tour_demo.py` + `ros_util.spin_until_done` for the in-repo pattern.)

`start_joints` is what keeps the container stateless: it needs no
`/joint_states` from your side at all, and you can pre-plan chained
segments from planned endpoints (see `tour_demo.py` for the pattern,
including merging chained plans into one continuous trajectory).

## Verify the image (no arm needed)

```bash
docker run --rm --runtime nvidia -e RAMMP_CUROBO_GPU=1 rammp-curobo:jp6 \
    python3 -m pytest /opt/rammp_curobo_src/core/tests -q
```

`RAMMP_CUROBO_GPU=1` fails the run where the GPU tests would skip (no GPU
passed through, a torch or cuRobo that does not import), instead of
passing it on the offline tests alone.
