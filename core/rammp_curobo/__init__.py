"""rammp_curobo: pure-Python cuRobo planning core for the RAMMP Gen3.

No ROS imports anywhere in this package. cuRobo/torch are imported lazily
inside CuRoboPlanner so the scene/geometry/retime/validate modules work on
machines without the GPU stack.
"""

from rammp_curobo.planner import CuRoboPlanner
from rammp_curobo.retime import scale_trajectory
from rammp_curobo.scene import Scene, load_scene
from rammp_curobo.types import PlanResult, Trajectory
from rammp_curobo.validate import start_state_matches, validate_trajectory

__all__ = [
    "CuRoboPlanner",
    "PlanResult",
    "Trajectory",
    "Scene",
    "load_scene",
    "scale_trajectory",
    "validate_trajectory",
    "start_state_matches",
]

__version__ = "1.0.0"
