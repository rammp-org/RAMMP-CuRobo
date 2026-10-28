"""The sim renders on EGL, as /mebot_sim does on a headless Jetson. MuJoCo
picks its GL backend when it is first imported, so this is set first.

`make sim-test` sets MEBOT_SIM_STRICT=1, under which a test or a module that
would be skipped fails instead, giving the skip's reason. Without an input
from outside this checkout (MuJoCo, EGL, the menagerie, the humanoid, the
URDF, ROS or the overlay's messages) the tests that need it skip, and the
gate would pass on the rest.
"""

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
STRICT = os.environ.get("MEBOT_SIM_STRICT") == "1"


def _no_skip(outcome):
    report = outcome.get_result()
    if STRICT and report.skipped:
        reason = report.longrepr
        if isinstance(reason, tuple):  # (path, line, reason)
            reason = reason[2]
        report.outcome = "failed"
        report.longrepr = "MEBOT_SIM_STRICT=1 skips nothing: %s" % reason


# old-style wrappers: Ubuntu 22.04's pytest 6 has no wrapper=True
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    _no_skip((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    _no_skip((yield))


@pytest.fixture(scope="session")
def assets():
    """scene.Assets() as the sim builds it by default. A test that needs it
    skips only when an input from outside this checkout is missing: the
    menagerie Gen3 or 2F-85, the humanoid, or every source of the pinned
    MEBot URDF. Whatever this checkout's own files raise fails it."""
    from mebot_sim import scene, stack

    menagerie = os.path.expanduser(scene.MENAGERIE)
    for path in (
        os.path.join(menagerie, "kinova_gen3", "gen3.xml"),
        os.path.join(menagerie, "robotiq_2f85", "2f85.xml"),
        scene.HUMANOID,
    ):
        if not os.path.isfile(path):
            pytest.skip("%s not found" % path)
    generator = stack.generator()
    try:
        generator.fetch_pinned(generator.load_data()["source"])
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return scene.Assets()
