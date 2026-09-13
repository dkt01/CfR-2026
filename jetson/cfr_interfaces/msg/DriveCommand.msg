# Actuator command produced by the autonomy stack and consumed by
# arduino_bridge_node, which converts it to the on-wire steering byte and the
# target spur RPM tracked by the Arduino's closed loop speed controller.

std_msgs/Header header

# Autonomy stack is alive and willing to drive.  The Arduino will not leave
# AUTO_ARMED until this is true, steering is within the neutral deadband, and
# the speed target is zero.
bool auto_ready

# [-1.0, 1.0].  +1.0 is full left (right-hand rule about +Z, REP-103).
float32 steering

# Target ground speed in m/s.  Positive forward, negative reverse, zero stops.
# arduino_bridge_node clamps it to its max_speed parameter and rate limits it
# with speed_slew_rate.  The single spur magnet cannot resolve much below
# 0.3 m/s, so slower targets run mostly on feedforward.
float32 velocity
