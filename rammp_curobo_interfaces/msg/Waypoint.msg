# One waypoint of PlanThrough: a tool pose or a joint configuration.
#
# kind POSE: `pose`, the tool_frame in the base frame (arm_mount), reached by
# a leg PlanToPose would plan, under `hold` (PlanToPose's HOLD_* values).
# kind JOINTS: `joints` (rad, joint_1..7), reached by a leg PlanToJoints would
# plan or, `straight`, by a straight line in joint space: no planner, so no
# detour round an obstacle, and a straight leg that collides fails the goal.
#
# blend_radius (m): how far the tool may pass from the planned path at this
# waypoint, still moving, instead of stopping on it. 0, the default, stops
# here, and the last waypoint always stops. It is a bound, not a target: the
# corner is cut only as far as the joint limits and the collision world also
# allow, which may be not at all (the result's corner_m says how far).

uint8 POSE=0
uint8 JOINTS=1

uint8 kind
geometry_msgs/Pose pose
uint8 hold 0
float64[] joints
bool straight
float64 blend_radius
