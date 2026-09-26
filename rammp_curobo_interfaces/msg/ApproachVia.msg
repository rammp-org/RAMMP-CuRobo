# One blended via point for the final approach to the goal.
#
# It is a COST, not a waypoint: the path passes NEAR the offset without
# stopping there and without hitting it exactly. If an exact intermediate
# pose is required, send two goals and accept the stop between them.
#
# offset of 0.0 (the default) means no via point.

uint8 AXIS_X=0
uint8 AXIS_Y=1
uint8 AXIS_Z=2        # the tool approach axis — the usual choice

float64 offset        # metres back along `axis` from the goal
uint8 axis 2
float64 at_fraction 0.8   # fraction of the motion at which it engages, in (0, 1)
