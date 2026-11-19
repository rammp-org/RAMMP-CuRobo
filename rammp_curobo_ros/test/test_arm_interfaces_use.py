"""The planner never reaches the arm (issue #6), checked by machine.

rammp_curobo_ros depends on rammp_arm_interfaces only to SERVE AddObstacles
(with CollisionBox). Every arm action and service (ExecuteJointTrajectory,
GoTo*) is importable from that same dependency, so this pins that the
package's code names nothing else from it. No ROS needed: it reads source.
A tripwire for static imports, not a sandbox: importlib or a type looked up
by string (rosidl_runtime_py's get_action) gets past it.
"""

import re
from pathlib import Path

ALLOWED = {"AddObstacles", "CollisionBox"}
PACKAGE = Path(__file__).resolve().parents[1] / "rammp_curobo_ros"


def _arm_names(text):
    names = set()
    for m in re.finditer(
        r"from\s+rammp_arm_interfaces(?:\.\w+)*\s+import\s+(\([^)]*\)|[^\n]+)", text
    ):
        body = re.sub(r"#[^\n]*", "", m.group(1))  # trailing comments
        listed = body.strip().strip("()").split(",")
        names |= {n.split(" as ")[0].strip() for n in listed} - {""}
    # any other spelling (import rammp_arm_interfaces..., a dotted use) is
    # refused outright: it could reach any type
    other = re.findall(
        r"import\s+rammp_arm_interfaces|rammp_arm_interfaces\.\w+\.\w+", text
    )
    return names, other


def test_the_node_uses_only_the_service_it_serves():
    used, other = set(), []
    for path in PACKAGE.rglob("*.py"):
        names, rest = _arm_names(path.read_text())
        used |= names
        other += ["%s: %s" % (path.name, r) for r in rest]
    assert used and used <= ALLOWED, "rammp_arm_interfaces names: %s" % sorted(used)
    assert not other, other


def test_the_check_sees_a_forbidden_import():
    names, _ = _arm_names(
        "from rammp_arm_interfaces.action import (\n    GoToEEPose,\n)\n"
    )
    assert names == {"GoToEEPose"}
    allowed = "from rammp_arm_interfaces.srv import AddObstacles  # noqa: E402\n"
    assert _arm_names(allowed)[0] == {"AddObstacles"}
    assert _arm_names("import rammp_arm_interfaces.action\n")[1]
