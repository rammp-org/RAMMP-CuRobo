"""What every process of the MEBot sim stack shares.

The stack runs this checkout's planner node twice beside /mebot_sim, a
MuJoCo arm that serves the real arm driver's names: /joint_states and
/execute_joint_trajectory. On the bench's ROS domain, 0, those names belong
to the real arm, so no process of the stack ever joins it. isolate() moves
a process whose ROS_DOMAIN_ID is anything but a plain domain from 1 to 232
(unset, 0, or one of the other values ROS reads as 0) to 88 before it
starts ROS, and sim.launch.py does the same for itself and everything it
launches.

The sim and the planners must also run this checkout's rammp_curobo: this
host has an editable install of another checkout, which is what a bare
`import rammp_curobo` finds.
"""

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import yaml

SIM_DOMAIN = "88"
JOINTS = ["joint_%d" % i for i in range(1, 8)]

# The two planner nodes the launch runs. The chair planner composes the
# platform and person layers into every world it plans in; the control
# planner composes none, so the world pushed to it decides which layers it
# sees: that is the same request planned without a layer.
CHAIR = "/rammp_curobo"
CONTROL = "/rammp_curobo_control"
CHAIR_CONFIG = "gen3_mebot.yaml"
CONTROL_CONFIG = "gen3.yaml"
# The chair planner's start-up world (the floor, under its config layers);
# pushing it again puts the planner back in it.
CHAIR_WORLD = "world_mebot.yaml"


def on_bench_domain(environ):
    """Whether `environ` could put ROS on the bench's domain, 0: whether its
    ROS_DOMAIN_ID is anything but a domain from 1 to 232 written plainly.

    rcl reads an unset or empty ROS_DOMAIN_ID as 0, and "00", "0x0" and " 0"
    too; it reads "-1" and 2**64 - 1 as its default domain, which Cyclone
    DDS makes 0. Rather than follow every such reading, only a plain number
    is believed (Cyclone has no domain above 232)."""
    value = environ.get("ROS_DOMAIN_ID", "")
    return not (re.fullmatch(r"[1-9][0-9]{0,2}", value) and int(value) <= 232)


def isolate():
    """Move this process off the bench's ROS domain, saying so. Call it before
    rclpy.init(): rcl reads ROS_DOMAIN_ID when the context starts."""
    if on_bench_domain(os.environ):
        value = os.environ.get("ROS_DOMAIN_ID")
        os.environ["ROS_DOMAIN_ID"] = SIM_DOMAIN
        print(
            "[mebot_sim] ROS_DOMAIN_ID is %s, not a domain from 1 to 232, and "
            "ROS can read that as 0, the bench's domain, where the real arm "
            "driver serves the names this stack serves: running on "
            "ROS_DOMAIN_ID=%s instead"
            % ("unset" if value is None else repr(value), SIM_DOMAIN),
            file=sys.stderr,
            flush=True,
        )
    # the bench's middleware; the stack's processes must all agree on one
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")


def checkout():
    """The RAMMP-CuRobo checkout this package was built from.

    `make sim-build` installs it into the checkout's .sim_ws/, as links to
    these sources or as copies of them, so the checkout is the nearest
    directory above this file that holds core/ and scripts/.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "core" / "rammp_curobo").is_dir() and (
            parent / "scripts" / "generate_layers.py"
        ).is_file():
            return parent
    raise RuntimeError(
        "%s is not inside a RAMMP-CuRobo checkout; build mebot_sim with "
        "`make sim-build` in the checkout it should run" % __file__
    )


def core():
    """This checkout's rammp_curobo, imported from its core/ whatever else is
    on the path. Refused if another rammp_curobo was imported first."""
    path = checkout() / "core"
    if sys.path[:1] != [str(path)]:
        sys.path.insert(0, str(path))
    import rammp_curobo

    found = Path(rammp_curobo.__file__).resolve().parents[1]
    if found != path:
        raise RuntimeError(
            "rammp_curobo is imported from %s, not this checkout's core (%s)"
            % (found, path)
        )
    return rammp_curobo


def generator():
    """This checkout's scripts/generate_layers.py, as a module. The sim reads
    the generator's data file and fetches the URDF with its fetch_pinned, so
    the MEBot it simulates is the one the layers were generated from."""
    core()  # the generator imports rammp_curobo
    spec = importlib.util.spec_from_file_location(
        "generate_layers", checkout() / "scripts" / "generate_layers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planner_environment(environ):
    """What the launch adds to the planner nodes' environment: this checkout's
    core/ first on PYTHONPATH."""
    paths = [str(checkout() / "core")]
    if environ.get("PYTHONPATH"):
        paths.append(environ["PYTHONPATH"])
    return {"PYTHONPATH": os.pathsep.join(paths)}


def check_planner_imports(env):
    """Refuse an environment in which the planner node would run another
    checkout's code: rammp_curobo from anywhere but this checkout's core/,
    or rammp_curobo_ros from outside this checkout (its build in .sim_ws/
    is inside). `env` is added to this process's environment, as the launch
    adds it to the planners'."""
    probe = (
        "import importlib.util as u\n"
        "for name in ('rammp_curobo', 'rammp_curobo_ros'):\n"
        "    spec = u.find_spec(name)\n"
        "    print(spec.origin if spec else '')\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", probe],
        env=dict(os.environ, **env),
        capture_output=True,
        text=True,
        check=True,
    )
    core_origin, node_origin = run.stdout.splitlines()
    here = checkout()
    for name, origin, ours in (
        ("rammp_curobo", core_origin, lambda p: p.parents[1] == here / "core"),
        ("rammp_curobo_ros", node_origin, lambda p: here in p.parents),
    ):
        if not origin or not ours(Path(origin).resolve()):
            raise RuntimeError(
                "the planner nodes would import %s from %s, not from this "
                "checkout (%s); PYTHONPATH is %s"
                % (name, origin or "nowhere", here, env.get("PYTHONPATH"))
            )


def config(name):
    """The path of one of this package's config files, in the checkout: the
    package runs from it in any case."""
    return checkout() / "sim" / "mebot_sim" / "config" / name


def scenarios():
    """config/scenarios.yaml: the stand-in's postures and the suite's
    thresholds, and what find_pairs found: the sim's start state and the
    requests the suite flies."""
    with open(config("scenarios.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def chair_layers():
    """The chair planner's layers, as rammp_curobo loads them: (the ones its
    config lists, which every world it plans in holds, and its start-up
    world, whose own layer is the floor)."""
    core()
    from rammp_curobo.config import load_planner_config, resolve_config
    from rammp_curobo.scene import load_layers, load_scene

    cfg, where = load_planner_config(CHAIR_CONFIG)
    world = load_scene(resolve_config(cfg["world"], relative_to=where))
    layers = load_layers(cfg["layers"], relative_to=where, listed_in=CHAIR_CONFIG)
    return layers, world


def layer_files():
    """The layer files the generator writes, in this checkout: {layer: path}."""
    outputs = generator().load_data()["outputs"]
    return {name: checkout() / path for name, path in outputs.items()}


def generated_layers():
    """The layer files the generator writes, as the planner loads them:
    {name: Layer}. What the suite holds the planners to, whatever a planner
    config lists: a chair config that drops a layer is a regression for the
    suite to catch."""
    core()
    from rammp_curobo.scene import load_layer

    loaded = [load_layer(path) for path in layer_files().values()]
    return {layer.name: layer for layer in loaded}


def layer_digests():
    """sha256 of each layer file the generator writes, by file name: what the
    suite's requests are found with."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in layer_files().values()
    }


def worlds():
    """What a request can be planned in: {key: (planner, world to push,
    what the planner then sees)}."""
    return {
        "all": (CHAIR, CHAIR_WORLD, "floor, platform, person"),
        "no_person": (
            CONTROL,
            str(config("control_floor_platform.yaml")),
            "floor, platform",
        ),
        "floor_only": (CONTROL, str(config("control_floor.yaml")), "floor"),
    }


def case(name, where, outcome, gates, **details):
    """One case of the sim suite: what was run, where (the planner and the
    world it planned in, or what was checked against what), what came of it,
    and its verdict on each gate it bears on, {gate: passed}. It passes when
    every verdict does; `details` are its numbers, for the result file."""
    return dict(
        case=name,
        where=where,
        outcome=outcome,
        gates=gates,
        passed=all(gates.values()),
        **details,
    )


def case_line(case):
    """A case as a line of the suite's report."""
    return "%-26s %-40s %-64s %s" % (
        case["case"],
        case["where"],
        case["outcome"],
        "PASS" if case["passed"] else "FAIL",
    )


def trajectory_arrays(traj):
    """(times, rows) of a trajectory_msgs/JointTrajectory: seconds from its
    start, and each point's positions.

    The arm driver never reads joint_names: it takes each point's positions
    as joint_1..joint_7. Read by name, a trajectory naming its joints in
    another order would fly here as the arm would not fly it, so one that
    names anything but joint_1..joint_7 in that order is refused."""
    if list(traj.joint_names) != JOINTS:
        raise ValueError(
            "the trajectory names %s, not joint_1..joint_7 in order; the arm "
            "driver takes each point's positions in that order whatever the "
            "names say" % (", ".join(traj.joint_names) or "no joints")
        )
    times = [
        p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points
    ]
    rows = [[float(v) for v in p.positions] for p in traj.points]
    return times, rows


def wait(future, timeout, what):
    """The result of an rclpy future that an executor in another thread
    completes."""
    done = threading.Event()
    future.add_done_callback(lambda _future: done.set())
    if not done.wait(timeout):
        raise TimeoutError("%s: no answer in %.0f s" % (what, timeout))
    return future.result()


class Planner:
    """A client of one planner node (CHAIR or CONTROL): its set_world service
    and its two plan actions. Its node must be spun in another thread."""

    def __init__(self, node, name):
        from rclpy.action import ActionClient
        from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
        from rammp_curobo_interfaces.srv import SetWorld

        self.name = name
        self._types = PlanToPose, PlanToJoints, SetWorld
        self._pose = ActionClient(node, PlanToPose, name + "/plan_to_pose")
        self._joints = ActionClient(node, PlanToJoints, name + "/plan_to_joints")
        self._world = node.create_client(SetWorld, name + "/set_world")

    def ready(self):
        return (
            self._pose.server_is_ready()
            and self._joints.server_is_ready()
            and self._world.service_is_ready()
        )

    def set_world(self, world):
        """(success, message): the planner's reply to a push of `world`, a
        path or the name of a packaged world."""
        request = self._types[2].Request(world=str(world))
        reply = wait(self._world.call_async(request), 60.0, "%s/set_world" % self.name)
        return reply.success, reply.message

    def plan(self, start, pos=None, rpy_deg=None, joints=None):
        """The planner's result for a tool pose (`pos`, metres, and `rpy_deg`,
        the ROS fixed-axis roll, pitch and yaw) or a joint goal (`joints`),
        planned from `start`; None if it rejected the goal."""
        PlanToPose, PlanToJoints, _ = self._types
        if joints is None:
            core()
            from rammp_curobo.geometry import euler_deg_to_quat_xyzw

            goal, client = PlanToPose.Goal(), self._pose
            p, q = goal.target.position, goal.target.orientation
            p.x, p.y, p.z = (float(v) for v in pos)
            q.x, q.y, q.z, q.w = euler_deg_to_quat_xyzw(rpy_deg)
        else:
            goal, client = PlanToJoints.Goal(), self._joints
            goal.target_joints = [float(v) for v in joints]
        goal.start_joints = [float(v) for v in start]
        what = "%s planning" % self.name
        handle = wait(client.send_goal_async(goal), 30.0, what)
        if not handle.accepted:
            return None
        return wait(handle.get_result_async(), 300.0, what).result
