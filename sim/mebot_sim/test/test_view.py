"""The live view (mebot_sim/view.py): its stream, its state and its commands,
served on a free port with stand-in planner clients: no ROS graph. It
renders on EGL, and needs what test_scene.py needs; without them it skips.
"""

import json
import os
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
msgs = pytest.importorskip("trajectory_msgs.msg")

from mebot_sim import scene, sim_arm, stack, view  # noqa: E402

try:
    ASSETS = scene.Assets()
except (FileNotFoundError, OSError) as exc:  # pragma: no cover
    pytest.skip("scene inputs unavailable: %s" % exc, allow_module_level=True)


class Planner:
    """What the view asks of a stack.Planner, answered with `result`."""

    def __init__(self, name, result):
        self.name, self.result, self.calls = name, result, []

    def ready(self):
        return True

    def set_world(self, world):
        self.calls.append(("set_world", world))
        return True, "world set to %s" % world

    def plan(self, start, pos=None, rpy_deg=None, joints=None):
        self.calls.append(("plan", list(start), pos, rpy_deg))
        return self.result


def _plan(rows, dt=0.02):
    traj = msgs.JointTrajectory(joint_names=list(stack.JOINTS))
    for k, row in enumerate(rows):
        point = msgs.JointTrajectoryPoint(positions=list(row))
        point.time_from_start.nanosec = int(round((k + 1) * dt * 1e9))
        traj.points.append(point)
    return types.SimpleNamespace(
        success=True, message="planned", planning_time=0.2, trajectory=traj
    )


@pytest.fixture
def live():
    sim = sim_arm.Sim(ASSETS, "upright_lap", stack.scenarios()["start"], view.SIZE)
    start = sim.q
    planners = {
        stack.CHAIR: Planner(stack.CHAIR, _plan([start + 0.01 * k for k in (1, 2, 3)])),
        stack.CONTROL: Planner(stack.CONTROL, None),
    }
    page = view.View(sim, planners)
    try:
        page.render(sim)
    except Exception as exc:  # pragma: no cover
        pytest.skip("no EGL rendering here: %r" % exc)
    (url,) = page.serve(["127.0.0.1"], 0)
    yield sim, page, url, planners
    page.close()


def _post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(url + "/cmd", data=data, timeout=5) as reply:
            return reply.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_the_sim_renders_however_it_is_started():
    # as `ros2 run` and the launch start it: no MUJOCO_GL, and the sim's
    # modules imported before anything renders
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    code = (
        "from mebot_sim import sim_arm\n"
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_string(\n"
        "    '<mujoco><worldbody><geom size=\"0.1\"/></worldbody></mujoco>')\n"
        "r = mujoco.Renderer(m, 32, 32)\n"
        "assert r.render().shape == (32, 32, 3)\n"
        "r.close()\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr


def test_the_stream_serves_jpeg_frames(live):
    _sim, page, url, _ = live
    assert page.frame[:2] == b"\xff\xd8"  # a JPEG
    with urllib.request.urlopen(url + "/stream", timeout=5) as stream:
        assert "multipart/x-mixed-replace" in stream.headers["Content-Type"]
        head = stream.read(100)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
    assert b"\r\n\r\n\xff\xd8" in head
    with urllib.request.urlopen(url + "/", timeout=5) as html:
        assert b"/stream" in html.read()


def test_the_page_reads_the_state_and_moves_the_joints(live):
    sim, _page, url, _ = live
    with urllib.request.urlopen(url + "/state", timeout=5) as reply:
        state = json.load(reply)
    assert len(state["q"]) == 7 and len(state["limits"]) == 7
    assert state["planners"] == "ready" and state["posture"] == "upright_lap"

    target = state["ref"][1] + 0.1
    assert _post(url, {"cmd": "jog", "joint": 1, "value": target}) == 200
    sim.poll()
    assert sim.ref[1] == pytest.approx(target)

    for bad in (
        {"cmd": "jog", "joint": 9, "value": 0},
        {"cmd": "jog", "joint": 1, "value": "nan"},
        {"cmd": "fly"},
        {"cmd": "speed", "value": 3},
        {"cmd": "posture", "name": "asleep"},
        {"cmd": "plan", "world": "kitchen", "pos": [0, 0, 0], "rpy_deg": [0, 0, 0]},
        b"not json",
    ):
        assert _post(url, bad) == 400, bad


def test_a_goal_is_planned_and_flown_at_the_pages_speed(live):
    sim, page, url, planners = live
    assert _post(url, {"cmd": "speed", "value": 0.25}) == 200
    goal = {
        "cmd": "plan",
        "world": "all",
        "pos": [0.5, -0.4, 0.3],
        "rpy_deg": [90, 0, 90],
    }
    assert _post(url, goal) == 200
    deadline = time.monotonic() + 5
    while page.planning and time.monotonic() < deadline:
        time.sleep(0.01)
    chair = planners[stack.CHAIR]
    assert chair.calls[0] == ("set_world", stack.CHAIR_WORLD)
    assert chair.calls[1][2:] == ([0.5, -0.4, 0.3], [90.0, 0.0, 90.0])
    assert chair.calls[1][1] == pytest.approx(sim.q.tolist())
    sim.poll()
    # the plan's three points at a quarter of their speed, from where the arm is
    assert sim.flight.times == pytest.approx([0.0, 0.08, 0.16, 0.24])
    assert page.goal == [0.5, -0.4, 0.3]
    assert "planned" in page.log[0]
