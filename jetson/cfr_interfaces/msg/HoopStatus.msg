# Whether the car has threaded every hoop on the Obstacle Course this run.
#
# Missing even one hoop fails the run outright per the rules, so `missed`
# latches per hoop and `any_missed` latches for the whole run -- nothing
# clears either short of hoop_monitor_node's ~/reset service.

std_msgs/Header header

string[] names
bool[] passed
bool[] missed
bool any_missed
bool all_passed
