# One box of a collision world, as a world or layer file writes it (cuRobo v0.7.8 plans against boxes only). The
# orientation is roll-pitch-yaw, not a quaternion: the planner builds every box's quaternion from rpy_deg itself.
string name                   # within its layer; no '/'
float64[3] position           # its centre, m, in the planner's base frame (base_link)
float64[3] rpy_deg            # fixed-axis roll, pitch, yaw, degrees
float64[3] dims               # full extents along its own axes, m, each > 0
