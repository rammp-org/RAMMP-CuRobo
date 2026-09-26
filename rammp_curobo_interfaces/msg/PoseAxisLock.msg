# Tool-pose components to hold fixed for the WHOLE trajectory.
#
# A locked component is held AT THE GOAL'S VALUE, so the start must already
# match the goal on that component (0.05 rad / 0.005 m). "Locked" means
# unchanged, not level: a tool at 45 degrees planning to a 45-degree goal
# stays at 45 the whole way. Level is simply the case where the goal is level.
#
# All fields false (the default) means unconstrained.

# Which frame the locked axes are measured in.
uint8 FRAME_BASE=0    # the robot base frame — gravity-relative, what "level" means
uint8 FRAME_GOAL=1    # projected into the goal frame (cuRobo's own default)
uint8 reference_frame 0

bool lock_roll
bool lock_pitch
bool lock_yaw
bool lock_x
bool lock_y
bool lock_z
