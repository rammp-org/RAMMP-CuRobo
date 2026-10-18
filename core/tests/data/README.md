# Test fixtures

`arm_level.urdf` and `arm_tilt30.urdf` are the robot model as the arm publishes it on
`/arm/robot_description`, rendered by kinova-gen3-ros2's `kinova_gen3_render`. They are
committed as rendered, plus one trailing newline (the `end-of-file-fixer` hook).

- Image: `kinova-gen3-ros2:review-sim` on abra (`sha256:b8a5a209e2c8…`, built 2026-10-16)
  from kinova-gen3-ros2 `feat/tilted-arm-mount` @ `e140639` against kinova-gen3-driver
  `dev` @ `e439d30`. Its `camera` xacro arg defaults to false, so the bracelet is
  `bracelet_no_vision_link`.
- Rendered 2026-10-18.

Mount files:

```yaml
# level.yaml
robot: fixture_level
rpy_deg: [0, 0, 0]
```

```yaml
# tilt30.yaml
robot: fixture_tilt30
rpy_deg: [0, 30, 0]
```

Commands (from the directory holding the mount files):

```bash
docker run --rm --network none -u $(id -u):$(id -g) -v $PWD:/w -w /w kinova-gen3-ros2:review-sim \
  bash -c "source /module_ws/install/setup.bash && python3 -m kinova_gen3_render --mount /w/level.yaml --out /w/arm_level.urdf"
docker run --rm --network none -u $(id -u):$(id -g) -v $PWD:/w -w /w kinova-gen3-ros2:review-sim \
  bash -c "source /module_ws/install/setup.bash && python3 -m kinova_gen3_render --mount /w/tilt30.yaml --out /w/arm_tilt30.urdf"
```

The two files differ only in `arm_base_joint`'s origin: `rpy="0.0 0.5235987755982988 0.0"`
in the tilt fixture.
