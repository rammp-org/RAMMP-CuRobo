# One box obstacle for AddObstacles, in the request's frame.
string name                 # unique in the collision world
geometry_msgs/Pose pose     # the box's centre and orientation
geometry_msgs/Vector3 dims  # full extents along the box's own x, y, z (m)
