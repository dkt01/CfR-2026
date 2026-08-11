# Normalized actuator command produced by the autonomy stack and consumed by
# arduino_bridge_node, which converts it to the on-wire [0,255] byte commands.

std_msgs/Header header

# Autonomy stack is alive and willing to drive.  The Arduino will not leave
# AUTO_ARMED until this is true and both axes are within the neutral deadband.
bool auto_ready

# [-1.0, 1.0].  +1.0 is full left (right-hand rule about +Z, REP-103).
float32 steering

# [-1.0, 1.0].  +1.0 is full forward.
float32 throttle
