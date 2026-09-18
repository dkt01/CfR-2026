# Decoded Arduino -> Jetson status frame.

uint8 MODE_ESTOP=0
uint8 MODE_RC_ARMED=1
uint8 MODE_RC_ACTIVE=2
uint8 MODE_AUTO_ARMED=3
uint8 MODE_AUTO_ACTIVE=4

std_msgs/Header header

# True while a valid status frame has been received within the link timeout.
# When false every other field is stale and must not be trusted.
bool link_ok

bool estop
bool auto_arm
bool manual_start

# One of the MODE_* constants above.
uint8 mode

# 0 is empty, 255 is full.
uint8 battery_level

# Spur gear revolutions per minute, from the hall sensor and its single trigger
# magnet.  This is spur revolutions, not motor or wheel revolutions.  The sensor
# cannot see direction, so the sign is the Arduino speed controller's estimate:
# the direction it last drove, positive when unknown (for example while
# coasting after leaving AUTO_ACTIVE).  Zero means stopped, no sensor, or a dead
# link - use link_ok to tell the last case apart.
int16 rpm

# rpm divided by the arduino_bridge spur_to_wheel_ratio parameter.  Signed.
float32 wheel_rpm

# Ground speed in metres per second from wheel_rpm and the tire_diameter
# parameter, assuming no wheel slip.  Signed like rpm.
float32 speed

# Speed in m/s the Arduino's controller is tracking right now.  Follows the
# commanded velocity after slew limiting, except that it holds zero while the
# car stops ahead of a reversal.
float32 target_speed

# Throttle pulse the controller is requesting, in microseconds, before the
# low speed dithering.  1500 is neutral.
uint16 throttle_us

# True once the Arduino reports running the speed gains currently set on
# arduino_bridge.  False after a parameter change until the new set is applied.
bool gains_applied
