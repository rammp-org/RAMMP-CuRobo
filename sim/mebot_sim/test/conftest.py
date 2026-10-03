"""The sim renders on EGL, as /mebot_sim does on a headless Jetson. MuJoCo
picks its GL backend when it is first imported, so this is set first."""

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


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
