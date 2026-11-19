# One box obstacle for AddObstacles, in the request's frame.
string name                 # unique in the collision world
geometry_msgs/Pose pose     # the box's centre and orientation
# Full extents along the box's own x, y, z (m), each > 0: the object's own
# size. The planner pads every side by its world_padding (2 cm by default),
# as it pads every scene box, so do not add a margin of your own.
geometry_msgs/Vector3 dims
