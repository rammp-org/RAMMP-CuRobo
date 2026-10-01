# How much of the tool's ORIENTATION to keep for the whole trajectory.
#
# Two modes, and no frame field — see RAMMP-CuRobo#16. Per-axis holds expressed
# a mechanism ("hold rx and ry in the base frame") rather than a request, in a
# frame whose axes mean nothing to a caller. Position holds are gone entirely:
# they could only express "travel along one base axis", and a straight line in
# an arbitrary direction is not expressible through cuRobo's DIAGONAL
# hold_vec_weight at all. A Cartesian linear move is a different operation.

uint8 HOLD_NONE=0     # orientation free
uint8 HOLD_LEVEL=1    # keep the tool's tilt; spin about vertical stays free
uint8 HOLD_FIXED=2    # keep the orientation entirely
uint8 hold 0

# HOLD_LEVEL PRESERVES TILT, IT DOES NOT CREATE LEVEL. Roll and pitch are
# pinned at the GOAL's values, so a gripper that starts 20 degrees off stays
# 20 degrees off for the whole motion — faithfully, just not level. "Level"
# means "as level as you already are".
#
# Consequently a goal whose orientation disagrees with the start on the held
# components is REFUSED, not quietly re-aimed: it is asking for two different
# orientations at once. Pass the start's orientation as the goal's, or get
# there with an unconstrained move first.
#
# "LEVEL WITH THE WORLD" MEANS "LEVEL WITH BASE Z". True while the arm is
# mounted level, and silently false the moment it is not: HOLD_LEVEL frees
# rotation about base Z, so on a tilted mount that freedom is about a tilted
# axis and tips the very object the hold was asked to protect. The fix is
# declaring the mount transform so base_link IS gravity-aligned (#20), not a
# field here — an undeclared tilt breaks gravity compensation in the driver
# too.
#
# WHICH AXIS IS FREE IS AN ASSUMPTION ABOUT THE GRASP. HOLD_LEVEL frees the
# spin about vertical, which is right for something rotationally symmetric
# about its upright axis — a cup, a bottle, a plate. A spoon is not symmetric:
# the bowl has to face a particular way, so no spin is free and it needs
# HOLD_FIXED. When the planner can see the held object (#23) the free axis
# should become a property of that object rather than this assumption.
#
# IT IS A COST, NOT A CLAMP. cuRobo penalises deviation without forbidding it,
# so the planner VERIFIES the returned trajectory and refuses a plan whose
# worst deviation exceeds constraint_tolerance_deg rather than handing over
# something that quietly tipped.
