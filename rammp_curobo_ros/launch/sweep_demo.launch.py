"""Sweep-and-avoid demo: planner + environment camera + the sweep loop.

    ros2 launch rammp_curobo_ros sweep_demo.launch.py                # dry run
    ros2 launch rammp_curobo_ros sweep_demo.launch.py execute:=true  # it moves

Brings up everything this repo owns. It does NOT start the arm bringup or
the Orbbec driver — start those first, in their own terminals:

    ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 \
        dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false
    ros2 launch orbbec_camera gemini_330_series.launch.py

Perception defaults here are the REACTIVE ones, not the cameras node's:
5 Hz with occupied_at 2 confirms an obstacle in ~0.4 s instead of ~1.5 s,
and self_radius is opened up to cover the camera-extrinsic residual plus
TF/depth skew while the arm is sweeping.
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
    ("camera", "camera_orbbec_bench.yaml", "environment camera config"),
    ("rate_hz", "5.0", "perception tick rate"),
    ("occupied_at", "2", "ticks before a voxel counts as an obstacle"),
    ("self_radius", "0.16", "arm self-filter radius (m)"),
    ("pose_a", "0.55,-0.30,0.35", "sweep waypoint A, tool xyz in base_link"),
    ("pose_b", "0.55,0.30,0.35", "sweep waypoint B"),
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
        parts = [float(v) for v in val(name).replace(" ", "").split(",")]
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
        parameters=[{
            "config": val("config"),
            "world": world,
            "execute": execute,
            "speed_scale": speed,
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
            "self_radius": float(val("self_radius")),
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
