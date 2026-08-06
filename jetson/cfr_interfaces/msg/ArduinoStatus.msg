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
