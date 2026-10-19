"""Shared fixtures: the arm's model as the planner receives it.

The planner takes the URDF from /arm/robot_description as a file; these write
the committed renders (data/README.md) the way the node will — sanitised by
prepare_urdf, then write_urdf — and return the paths. No cuRobo here, so the
offline suite can share this file.
"""

from pathlib import Path

import pytest

from rammp_curobo.urdf_model import prepare_urdf, write_urdf

DATA = Path(__file__).parent / "data"


def _written(tmp_path_factory, name):
    xml = prepare_urdf((DATA / name).read_text())
    return write_urdf(xml, tmp_path_factory.mktemp(Path(name).stem))


@pytest.fixture(scope="session")
def urdf_level(tmp_path_factory):
    return _written(tmp_path_factory, "arm_level.urdf")


@pytest.fixture(scope="session")
def urdf_tilt30(tmp_path_factory):
    return _written(tmp_path_factory, "arm_tilt30.urdf")
