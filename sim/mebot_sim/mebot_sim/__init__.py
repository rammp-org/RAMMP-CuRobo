"""The MEBot sim stack: a MuJoCo arm beside this checkout's planner nodes."""

import os

# A headless Jetson renders on EGL, and MuJoCo picks its GL backend when it
# is first imported: this runs before any module here imports it.
os.environ.setdefault("MUJOCO_GL", "egl")
