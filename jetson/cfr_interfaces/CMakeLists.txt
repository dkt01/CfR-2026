cmake_minimum_required(VERSION 3.16)
project(cfr_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DriveCommand.msg"
  "msg/ArduinoStatus.msg"
  "msg/PathSegment.msg"
  "msg/StartSignal.msg"
  "msg/LapCount.msg"
  "msg/HoopLayout.msg"
  "msg/HoopStatus.msg"
  "action/DrivePath.action"
  DEPENDENCIES std_msgs action_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
