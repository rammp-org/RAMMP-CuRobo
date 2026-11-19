# Planning benchmark (not shipped)

`sweep_bench.py` measures how far, and how repeatably, `plan_to_pose` swings
the arm on the chair: per plan, the swept volume of the arm's collision
spheres, how far any sphere strays from a straight line, the tool's path, and
the final joints. Plans of one request that end more than 0.3 rad apart (any
joint) are different "families"; "final spread" is the median distance
between the final joints of one request's reps.

The default problem set is 80 start -> goal-pose pairs from the MEBot sim's
states (`sim/mebot_sim/config/scenarios.yaml`) on this checkout's chair fixture
(`core/tests/data/mebot/planner.yaml`, whose layers were generated for the
30-degree mount: `core/tests/data/arm_tilt30.urdf`), 10 reps each, the start
jittered by 3 mrad per joint every rep.

## Running it (an x86 GPU host with Docker and the NVIDIA runtime)

```bash
docker build -f bench/Dockerfile.x86 -t rammp-curobo-bench:x86 bench  # cuRobo v0.7.8, sm_86
bench/run.sh bench/out base
bench/run.sh bench/out candidate --set num_ik_seeds=512 --set num_trajopt_seeds=1 \
    --set reseed_each_plan=true --set graph_fallback=true
python3 bench/sweep_bench.py report bench/out/*.jsonl --baseline base   # numpy + PyYAML
```

`--set key=JSON` overrides one `planner:` key; anything in `PLANNER_DEFAULTS`
works. `bench/Dockerfile.x86` targets Ampere (`TORCH_CUDA_ARCH_LIST=8.6`);
change it for another GPU.

## Real requests

Start the planner node with `plan_log:=/some/file.jsonl` and it appends every
`plan_to_pose` / `plan_to_joints` request with the collision world it was
planned in and the result. Replay that under a condition with the robot's own
planner config:

```bash
docker run --rm --gpus all -v "$PWD":/work -v <robots>/<robot>:/robot:ro rammp-curobo-bench:x86 \
    python3 bench/sweep_bench.py replay bench/out/plan_log.jsonl \
    --config /robot/curobo/planner.yaml --name base --out bench/out/replay/base.jsonl
```

The report then adds a `live` row: the plans as they were executed.
