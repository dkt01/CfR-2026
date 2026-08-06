cmake_minimum_required(VERSION 3.16)
project(cfr_arduino_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Collect built executables in one place rather than scattering them through
# the build tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(ament_cmake REQUIRED)
find_package(cfr_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

# protocol and serial handling, kept free of ROS so it can be unit tested.
add_library(cfr_arduino_protocol
  src/protocol.cpp
  src/serial_port.cpp
)
target_include_directories(cfr_arduino_protocol PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(arduino_bridge_node src/arduino_bridge_node.cpp)
target_link_libraries(arduino_bridge_node cfr_arduino_protocol)
ament_target_dependencies(arduino_bridge_node rclcpp cfr_interfaces)

add_executable(cmd_vel_to_drive_node src/cmd_vel_to_drive_node.cpp)
ament_target_dependencies(cmd_vel_to_drive_node rclcpp cfr_interfaces geometry_msgs)

install(TARGETS cfr_arduino_protocol
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# `ros2 run` and launch_ros resolve executables from lib/<package>, so that
# copy has to stay.  bin/ is installed alongside it so the nodes can also be
# invoked directly off the install prefix.
install(TARGETS arduino_bridge_node cmd_vel_to_drive_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS arduino_bridge_node cmd_vel_to_drive_node
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_link_libraries(test_protocol cfr_arduino_protocol)

  # Loopback test over a pseudo terminal; openpty() lives in libutil.
  ament_add_gtest(test_serial_port test/test_serial_port.cpp)
  target_link_libraries(test_serial_port cfr_arduino_protocol util)
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp cfr_interfaces geometry_msgs)

ament_package()
