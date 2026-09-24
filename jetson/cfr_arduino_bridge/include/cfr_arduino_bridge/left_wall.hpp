// The left-wall follower's control laws, kept free of ROS so they can be
// unit tested.  left_wall_follower_node is the wiring.
//
// Two sources, two laws:
//
//   * cloud: the physical car.  The ZED's registered cloud is reduced to the
//     nearest returns in three bearing sectors -- two on the left to fit the
//     wall's tangent, one ahead for the ends of the oval -- and those steer.
//   * simulation: Gazebo with ground truth.  The wall-derived centerline is
//     pure-pursued, with a small correction from one ray cast against the
//     bales read out of the world file.  It needs pose and world, so it only
//     runs in the simulator.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cfr_arduino_bridge/point_cloud.hpp"

namespace cfr_arduino_bridge::left_wall {

  constexpr double kWheelbase = 0.335;
  constexpr double kMaxSteer = 0.45;

  struct Command {
    double speed = 0.0;  // m/s
    double steer = 0.0;  // rad, positive left
  };

  // ------------------------------------------------------------------ cloud

  // The cloud reduced to what cloud_command reads: how many returns passed
  // the height band at all, and the ranges of those inside each sector.
  struct SectorScan {
    size_t points = 0;
    std::vector<float> near;   // 50 deg left, +/-5 deg
    std::vector<float> far;    // 30 deg left, +/-5 deg
    std::vector<float> front;  // dead ahead, +/-15 deg
  };

  // Bearing and range of every return in the ZED's body-frame registered
  // cloud (+x forward, +y left, +z up, heights relative to the camera),
  // discarding the floor below the bale faces, reduced to the three sectors.
  // `scan` is cleared and reused.  Throws std::invalid_argument on a layout
  // the buffer cannot hold.
  void ScanCloud(const uint8_t* data, size_t size, const CloudLayout& layout, SectorScan* scan);

  // 10th percentile of a sector's ranges, or 3 m when it has fewer than five.
  // Reorders `ranges`.
  double SectorRange(std::vector<float>& ranges);

  // Slow forward speed and steering angle from two left rays and front
  // space.  Stops when the cloud has too few returns to trust.
  Command CloudCommand(SectorScan& scan);

  // ------------------------------------------------------------- simulation

  // A bale's footprint: center, yaw, and half extents along its own axes.
  struct Bale {
    double x, y, yaw, half_x, half_y;
  };

  // The bales of the course_bales model in a world SDF.  Throws
  // std::runtime_error if the file or the model is missing.
  std::vector<Bale> LoadBales(const std::string& sdf_path);

  // Nearest intersection of a horizontal ray and an oriented bale box, or
  // max_range if it hits nothing nearer.
  double RangeToBales(
      const std::vector<Bale>& bales, double x, double y, double yaw, double bearing, double max_range = 5.0);

  // Follow the wall-derived centerline with a small left-range correction.
  //
  // The centerline resolves gaps between individual bales and prevents a
  // reactive range controller from circling a single bale near the start.
  class CourseFollower {
   public:
    // The `x` and `y` arrays of config/speed_course_path.json, reversed: the
    // stored polyline runs opposite the speed course's starting heading.
    // Throws std::runtime_error if the file cannot be read.
    explicit CourseFollower(const std::string& path_file);
    explicit CourseFollower(std::vector<std::pair<double, double>> path);

    Command Step(const std::vector<Bale>& bales, double x, double y, double yaw);

   private:
    std::vector<std::pair<double, double>> path_;
    long index_ = -1;
  };

}  // namespace cfr_arduino_bridge::left_wall
