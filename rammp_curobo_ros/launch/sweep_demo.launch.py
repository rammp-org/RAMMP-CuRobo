"""Sweep-and-avoid demo: planner + environment camera + the sweep loop.

    ros2 launch rammp_curobo_ros sweep_demo.launch.py                # dry run
    ros2 launch rammp_curobo_ros sweep_demo.launch.py execute:=true  # it moves

Brings up everything this repo owns. It does NOT start the arm bringup or
the Orbbec driver — start those first, in their own terminals:

    ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 \
        dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false
    ros2 launch orbbec_camera gemini_330_series.launch.py

Perception defaults here are the REACTIVE ones, not the cameras node's:
5 Hz with occupied_at 2 confirms an obstacle in ~0.4 s instead of ~1.5 s.
self_radius stays at the node's 0.08 m margin — startup registration
(auto_register) absorbs the camera-extrinsic error.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("execute", "false", "allow motion (default: dry-run, nothing moves)"),
    ("config", "gen3_real.yaml", "planner YAML"),
    ("world", "world_real_bench.yaml", "baseline collision world — MEASURE IT"),
    ("speed_scale", "0.25", "execution speed; start low, raise once trusted"),
    ("activation_distance", "0.0", "collision standoff (m); 0 = config's 0.03"),
    ("world_padding", "0.0", "hard box inflation (m); 0 = config's 0.02"),
    ("max_left_deg", "75.0", "base yaw limit to the left (+y); 0 = none"),
    ("max_right_deg", "90.0", "base yaw limit to the right (-y); 0 = none"),
    ("camera", "camera_orbbec_bench.yaml", "environment camera config"),
    ("rate_hz", "5.0", "perception tick rate"),
    ("occupied_at", "2", "ticks before a voxel counts as an obstacle"),
    # 0.03 is the PROVEN value. Lowering it to table_top+2.5cm (-0.002)
    # flooded the map with 11-13 stable boxes: registration corrects the
    # mount's TRANSLATION only, so a small residual tilt lifts the bench's
    # far corners a couple of cm in base_link and the surface itself leaks
    # into the band. Revisit only after a rotation calibration.
    ("min_z", "0.03", "ignore depth below this height (m, base_link)"),
    ("self_radius", "0.08", "self-filter margin over the arm's collision spheres (m)"),
    ("auto_register", "true", "solve the camera's translation error off the arm at startup"),
    ("sweep_deg", "30", "base yaw each side of home (joint-space sweep)"),
    ("pose_a", "", "optional: tool xyz for pose-mode sweep (needs pose_b too)"),
    ("pose_b", "", "optional: tool xyz for pose-mode sweep"),
    ("watchdog_hz", "5.0", "how often the in-flight path is re-checked"),
    ("clearance_margin", "0.0", "extra standoff over cuRobo's measured 0.02 m"),
]


def _nodes(context, *_args, **_kwargs):
    """Resolve every argument to a real Python value before building the
    nodes.

    Substitutions are strings, and a parameter's type is inferred from
    what it is given: `{"cameras": [LaunchConfiguration("camera")]}` does
    NOT produce a STRING_ARRAY, it produces a STRING, and the cameras
    node dies at startup with InvalidParameterTypeException. Resolving
    here instead makes every parameter's type explicit.
    """
    def val(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return val(name).strip().lower() in ("1", "true", "yes", "on")

    def xyz(name):
        raw = val(name).replace(" ", "")
        if not raw:
            return [0.0, 0.0, 0.0]                # unset: yaw-sweep mode
        parts = [float(v) for v in raw.split(",")]
        if len(parts) != 3:
            raise RuntimeError("%s must be 'x,y,z', got %r" % (name, val(name)))
        return parts

    execute = flag("execute")
    world = val("world")
    speed = float(val("speed_scale"))
    planner = Node(
        package="rammp_curobo_ros",
        executable="planner_node",
        name="rammp_curobo",
        output="screen",
        emulate_tty=True,
        # an in-flight cuRobo solve runs ~5 s and cannot be interrupted;
        # shutdown waits for it, so give launch more than its 5 s default
        # before it escalates SIGINT to SIGTERM
        sigterm_timeout="12",
        parameters=[{
            "config": val("config"),
            "world": world,
            "execute": execute,
            "speed_scale": speed,
            "activation_distance": float(val("activation_distance")),
            "world_padding": float(val("world_padding")),
            "max_left_deg": float(val("max_left_deg")),
            "max_right_deg": float(val("max_right_deg")),
        }],
    )
    cameras = Node(
        package="rammp_curobo_ros",
        executable="cameras",
        name="cameras",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "cameras": [val("camera")],          # a real list of str
            "baseline": world,
            "rate_hz": float(val("rate_hz")),
            "occupied_at": int(val("occupied_at")),
            "min_z": float(val("min_z")),
            "self_radius": float(val("self_radius")),
            "auto_register": flag("auto_register"),
        }],
    )
    demo = Node(
        package="rammp_curobo_ros",
        executable="sweep_demo",
        name="sweep_demo",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "execute": execute,
            "speed_scale": speed,
            "sweep_deg": float(val("sweep_deg")),
            "pose_a": xyz("pose_a"),
            "pose_b": xyz("pose_b"),
            "watchdog_hz": float(val("watchdog_hz")),
            "clearance_margin": float(val("clearance_margin")),
        }],
    )
    return [planner, cameras, demo]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [OpaqueFunction(function=_nodes)]
    )
