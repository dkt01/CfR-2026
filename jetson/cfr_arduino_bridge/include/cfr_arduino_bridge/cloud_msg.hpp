// CloudLayout from a sensor_msgs/PointCloud2, shared by the nodes that read
// clouds.  The layout itself (point_cloud.hpp) stays free of ROS.

#pragma once

#include <stdexcept>

#include "cfr_arduino_bridge/point_cloud.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace cfr_arduino_bridge {

  // How a cloud consumer subscribes: the newest frame only, reliably unless
  // told otherwise.  Best effort drops most of a camera-sized cloud between
  // processes -- each is thousands of fragments, and losing any one loses the
  // frame -- which measured 35% to 80% of frames never arriving.  But a
  // reliable subscriber hears nothing at all from a best-effort publisher, so
  // it stays a parameter for a camera driver configured that way.
  inline rclcpp::QoS CloudQoS(bool reliable) {
    return reliable ? rclcpp::QoS(1).reliable() : rclcpp::QoS(1).best_effort();
  }

  // The x/y/z layout of a cloud, or std::invalid_argument if it has none.
  inline CloudLayout LayoutOf(const sensor_msgs::msg::PointCloud2& msg) {
    CloudLayout layout;
    layout.width = msg.width;
    layout.height = msg.height;
    layout.point_step = msg.point_step;
    layout.row_step = msg.row_step;
    layout.big_endian = msg.is_bigendian;
    int found = 0;
    for (const auto& field : msg.fields) {
      if (field.datatype != sensor_msgs::msg::PointField::FLOAT32) {
        continue;
      }
      if (field.name == "x") {
        layout.x = field.offset, found |= 1;
      } else if (field.name == "y") {
        layout.y = field.offset, found |= 2;
      } else if (field.name == "z") {
        layout.z = field.offset, found |= 4;
      }
    }
    if (found != 7) {
      throw std::invalid_argument("point cloud has no FLOAT32 x/y/z fields");
    }
    layout.Check(msg.data.size());
    return layout;
  }

}  // namespace cfr_arduino_bridge
