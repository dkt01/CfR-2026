# Current world-frame position of every hoop on the Obstacle Course.
#
# Published by obstacle_randomizer_node, transient local, whenever it moves
# the hoops: at startup (the nominal layout the world spawns with), and
# after every ~/randomize or ~/reset call. hoop_monitor_node is the reader --
# it needs to know where the gates currently stand without duplicating the
# randomizer's own draw logic.
#
# Orientation is not carried here: each hoop's yaw is fixed by the course
# layout (obstacle_course_layout.yaml's hoops.<name>.yaw), only its position
# along the drawing's dashed line varies between draws.

string[] names
geometry_msgs/Point[] positions
