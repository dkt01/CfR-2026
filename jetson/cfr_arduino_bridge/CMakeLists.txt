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
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ament_index_cpp REQUIRED)
find_package(cfr_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tinyxml2_vendor REQUIRED)
find_package(TinyXML2 REQUIRED)

# protocol and serial handling, kept free of ROS so it can be unit tested.
add_library(cfr_arduino_protocol
  src/protocol.cpp
  src/serial_port.cpp
)
target_include_directories(cfr_arduino_protocol PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# path_follower_node's control law, kept free of ROS so it can be unit tested.
add_library(cfr_path_geometry
  src/path_geometry.cpp
)
target_include_directories(cfr_path_geometry PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# The point cloud segmenter, free of ROS.  Shared, because besides
# cloud_segmentation_node it is loaded by src/cloud_segmentation.py through its
# C interface -- which is how the fixture tests, and any training code, run
# the same compiled segmenter the car does.
add_library(cfr_cloud_segmentation SHARED
  src/cloud_segmentation.cpp
  src/cloud_segmentation_c.cpp
)
target_include_directories(cfr_cloud_segmentation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Always optimized: a Debug build of the workspace would otherwise leave the
# fixture tests and the simulator's segmented cloud several times slower.
target_compile_options(cfr_cloud_segmentation PRIVATE -O2)

# The left-wall follower's control laws and the simulated camera's stereo
# noise, free of ROS so they can be unit tested.
add_library(cfr_perception STATIC
  src/left_wall.cpp
  src/stereo_noise.cpp
)
target_include_directories(cfr_perception PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cfr_perception tinyxml2::tinyxml2)

add_executable(arduino_bridge_node src/arduino_bridge_node.cpp)
target_link_libraries(arduino_bridge_node cfr_arduino_protocol)
ament_target_dependencies(arduino_bridge_node rclcpp cfr_interfaces)

add_executable(cmd_vel_to_drive_node src/cmd_vel_to_drive_node.cpp)
ament_target_dependencies(cmd_vel_to_drive_node rclcpp cfr_interfaces geometry_msgs)

add_executable(sim_vehicle_node src/sim_vehicle_node.cpp)
ament_target_dependencies(sim_vehicle_node rclcpp cfr_interfaces geometry_msgs)

add_executable(path_follower_node src/path_follower_node.cpp)
target_link_libraries(path_follower_node cfr_path_geometry)
ament_target_dependencies(path_follower_node rclcpp rclcpp_action cfr_interfaces geometry_msgs nav_msgs std_srvs)

# The sensor-to-actuator path, and the simulator's camera and visualization
# path: C++ so that what runs per frame costs the same in Gazebo, in training
# and on the Orin.
add_executable(left_wall_follower_node src/left_wall_follower_node.cpp)
target_link_libraries(left_wall_follower_node cfr_perception)
ament_target_dependencies(left_wall_follower_node
  rclcpp ament_index_cpp cfr_interfaces geometry_msgs sensor_msgs std_msgs std_srvs)

add_executable(zed_cloud_noise_node src/zed_cloud_noise_node.cpp)
target_link_libraries(zed_cloud_noise_node cfr_perception)
ament_target_dependencies(zed_cloud_noise_node rclcpp sensor_msgs)

add_executable(cloud_segmentation_node src/cloud_segmentation_node.cpp)
target_link_libraries(cloud_segmentation_node cfr_cloud_segmentation)
ament_target_dependencies(cloud_segmentation_node rclcpp geometry_msgs sensor_msgs)

install(TARGETS cfr_arduino_protocol cfr_path_geometry cfr_cloud_segmentation
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# `ros2 run` and launch_ros resolve executables from lib/<package>, so that
# copy has to stay.  bin/ is installed alongside it so the nodes can also be
# invoked directly off the install prefix.
set(CFR_NODES arduino_bridge_node cmd_vel_to_drive_node sim_vehicle_node path_follower_node
  left_wall_follower_node zed_cloud_noise_node cloud_segmentation_node)

install(TARGETS ${CFR_NODES}
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${CFR_NODES}
  RUNTIME DESTINATION bin
)

install(PROGRAMS src/teleport_api.py src/maneuver_runner_node.py src/obstacle_randomizer_node.py
  src/start_signal_detector_node.py src/lap_counter_node.py src/hoop_monitor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# These are imported from beside their nodes, so they are installed into the
# same directory -- but as modules rather than programs, since they have no
# shebang and are not meant to be run.  cloud_segmentation.py finds
# libcfr_cloud_segmentation.so one directory up, in lib/.
install(FILES src/start_signal_detector.py src/lap_counter.py src/hoop_monitor.py src/obstacle_layout_draw.py src/cloud_segmentation.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# meshes/ is referenced from the worlds as model://cfr_arduino_bridge/meshes,
# which Gazebo resolves against GZ_SIM_RESOURCE_PATH -- simulation.launch.py
# points that at the share root.
install(DIRECTORY config launch meshes worlds
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_link_libraries(test_protocol cfr_arduino_protocol)

  # Loopback test over a pseudo terminal; openpty() lives in libutil.
  ament_add_gtest(test_serial_port test/test_serial_port.cpp)
  target_link_libraries(test_serial_port cfr_arduino_protocol util)

  ament_add_gtest(test_path_geometry test/test_path_geometry.cpp)
  target_link_libraries(test_path_geometry cfr_path_geometry)

  # Header only; linking the protocol library just brings its include path.
  ament_add_gtest(test_drivetrain test/test_drivetrain.cpp)
  target_link_libraries(test_drivetrain cfr_arduino_protocol)

  # left_wall_follower_node's control laws and zed_cloud_noise_node's noise
  # model.  Reads the speed course's world and path out of the source tree.
  ament_add_gtest(test_left_wall test/test_left_wall.cpp)
  target_link_libraries(test_left_wall cfr_perception)
  target_compile_definitions(test_left_wall PRIVATE CFR_PACKAGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  # Offline characterization analysis: trace parsers, fits, the run loader and
  # the vehicle.yaml patcher.  Plain unittest, so it also runs as a bare script
  # on a laptop with no test runner installed - which is where runs get analysed.
  ament_add_pytest_test(test_characterize test/test_characterize.py)

  # The Gazebo vehicle model is generated from config/vehicle.yaml into EVERY
  # world in WORLD_SPAWN_POSES (speed_course.sdf and obstacle_course.sdf).
  # This fails the build if someone edited one without regenerating the
  # other, rather than letting a simulator quietly describe a different car.
  find_package(ament_cmake_test REQUIRED)
  # --check reports through its exit code and prints a human readable reason;
  # it does not write a JUnit file, which ament_add_test otherwise insists on
  # and fails the test for missing however the check itself came out.  No
  # --world: omitting it checks every world the generator knows about.
  ament_add_test(test_vehicle_model_up_to_date
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_vehicle_model.py"
            --check
            --vehicle "${CMAKE_CURRENT_SOURCE_DIR}/config/vehicle.yaml"
    OUTPUT_FILE "${CMAKE_BINARY_DIR}/test_results/${PROJECT_NAME}/test_vehicle_model_up_to_date.txt"
  )

  # start_signal_detector_node's color decision, which the test imports
  # straight out of src/ -- it is free of ROS, so it needs no workspace.
  ament_add_pytest_test(test_start_signal_detector test/test_start_signal_detector.py)

  # lap_counter_node's crossing logic, imported straight out of src/ the same
  # way -- also free of ROS, so it also needs no workspace.
  ament_add_pytest_test(test_lap_counter test/test_lap_counter.py)

  # cloud_segmentation, scored against Gazebo-rendered views of both courses
  # with ground-truth labels (test/fixtures/segmentation, rendered by
  # scripts/capture_segmentation_fixtures.py).  ROS-free like the two above,
  # but it scores the compiled segmenter -- the one cloud_segmentation_node
  # runs -- through src/cloud_segmentation.py, so it is pointed at this
  # build's library.
  ament_add_pytest_test(test_cloud_segmentation test/test_cloud_segmentation.py
    ENV CFR_CLOUD_SEGMENTATION_LIB=$<TARGET_FILE:cfr_cloud_segmentation>
    TIMEOUT 300)

  # hoop_monitor_node's per-hoop crossing logic, same pattern again.
  ament_add_pytest_test(test_hoop_monitor test/test_hoop_monitor.py)

  # obstacle_randomizer_node's layout draw, which rl/obstacleRacer's exported
  # training layouts must keep agreeing with.
  ament_add_pytest_test(test_obstacle_layout_draw test/test_obstacle_layout_draw.py)

  # maneuver_runner_node's arming/abort state machine -- unlike the two tests
  # above, this one has no ROS-free module to import: it constructs a real
  # rclpy Node and drives it with cfr_interfaces/nav_msgs messages on a fake
  # clock, so it needs the built workspace rather than running standalone.
  ament_add_pytest_test(test_maneuver_runner test/test_maneuver_runner.py)
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rclcpp_action cfr_interfaces geometry_msgs nav_msgs sensor_msgs std_srvs)

ament_package()
