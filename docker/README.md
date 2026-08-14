# The planning service as a container

One image = the whole pinned GPU stack (torch jp6/cu126, cuRobo v0.7.8,
warp 1.5.1) + ROS 2 Humble + this repo's planner node, planning-only.
Another codebase talks to it over two ROS actions and never has to touch
the cuRobo install again:

    in:  /rammp_curobo/plan_to_pose   (geometry_msgs/Pose [+ start_joints])
         /rammp_curobo/plan_to_joints (7 joint angles    [+ start_joints])
    out: trajectory_msgs/JointTrajectory — full cuRobo time-parameterization

The container holds **no arm driver** and starts with `execute:=false`:
executing the returned trajectory is the caller's job, on whatever
controller stack owns the arm.

## Prerequisites (once per machine)

Docker is NOT currently installed on this lab's Jetson. Install engine +
NVIDIA runtime, then let the runtime mount the GPU:

```bash
sudo apt-get update && sudo apt-get install -y docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker $USER   # then re-login
```

## Build (on the Jetson, ~1 h — cuRobo compiles its CUDA kernels)

```bash
cd ~/RAMMP-CuRobo
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
- Different world: append the launch invocation with your own args, e.g.
  `... rammp-curobo:jp6 ros2 launch rammp_curobo_ros planner.launch.py
  config:=gen3_real.yaml world:=/path/mounted/world.yaml`.

## Client side (your codebase)

Build `rammp_curobo_interfaces` in your workspace (copy the package or add
this repo to your `src/`; it is dependency-free rosidl by design), then:

```python
from rammp_curobo_interfaces.action import PlanToPose
from rclpy.action import ActionClient

client = ActionClient(node, PlanToPose, "/rammp_curobo/plan_to_pose")
goal = PlanToPose.Goal()
goal.target.position.x, goal.target.position.y, goal.target.position.z = p
(goal.target.orientation.x, goal.target.orientation.y,
 goal.target.orientation.z, goal.target.orientation.w) = quat_xyzw
goal.start_joints = list(q_now)   # RECOMMENDED: explicit start; empty = the
                                  # planner's /joint_states subscription
res = client.send_goal(goal).result
if res.success:
    execute(res.trajectory)       # your controller, your gates
```

Passing `start_joints` explicitly keeps the container fully stateless — it
then needs no `/joint_states` from your side at all, and you can pre-plan
chained segments from planned endpoints (see `tour_demo.py` for the
pattern, including merging chained plans into one continuous trajectory).

## Verify the image (no arm needed)

```bash
docker run --rm --runtime nvidia rammp-curobo:jp6 \
    python3 -m pytest /opt/rammp_curobo_src/core/tests -q
```
