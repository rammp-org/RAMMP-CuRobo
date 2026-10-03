"""The sim renders on EGL, as /mebot_sim does on a headless Jetson. MuJoCo
picks its GL backend when it is first imported, so this is set first."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
