# One leg of a path for path_follower_node.  The vehicle is Ackermann and
# cannot rotate in place, so a TURN is driven as an arc: the node commands a
# nonzero forward speed together with a yaw rate, and the segment completes
# once heading has rotated by turn_angle, not after any fixed radius or time.

uint8 STRAIGHT=0
uint8 TURN=1

uint8 type

# STRAIGHT: target distance, metres.  Negative reverses.  Ignored for TURN.
float32 distance

# TURN: target heading change, radians.  Positive is left (counter-clockwise,
# REP-103), matching DriveCommand's steering sign.  Must satisfy
# |turn_angle| <= pi; split larger turns into multiple segments.  Ignored for
# STRAIGHT.
float32 turn_angle
