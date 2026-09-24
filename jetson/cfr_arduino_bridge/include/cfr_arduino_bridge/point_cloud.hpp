// Reading x/y/z out of a packed point cloud without copying it.
//
// sensor_msgs/PointCloud2 is a byte buffer plus a field table, and Gazebo's
// rgbd_camera and the ZED wrapper do not agree on what else rides along in
// each point.  So the layout is read from the message, not assumed.  Kept free
// of ROS: a node fills CloudLayout from the message, tests fill it by hand.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cfr_arduino_bridge {

  struct CloudLayout {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t point_step = 0;
    uint32_t row_step = 0;
    // Byte offsets of the three FLOAT32 coordinates within a point.
    uint32_t x = 0;
    uint32_t y = 4;
    uint32_t z = 8;
    bool big_endian = false;

    size_t Count() const { return static_cast<size_t>(width) * height; }

    // Throws std::invalid_argument when `size` bytes cannot hold the cloud
    // this describes, or its coordinates overrun a point.
    void Check(size_t size) const {
      if (point_step < 12 || x + 4 > point_step || y + 4 > point_step || z + 4 > point_step) {
        throw std::invalid_argument("point cloud x/y/z do not fit in its point_step");
      }
      if (height > 0 &&
          (row_step < static_cast<uint64_t>(width) * point_step || static_cast<uint64_t>(row_step) * height > size)) {
        throw std::invalid_argument("point cloud is shorter than its width, height and row_step");
      }
    }

    // Byte offset of point `i` in row-major order.
    size_t Offset(size_t i) const { return (i / width) * row_step + (i % width) * point_step; }
  };

  inline float ReadFloat(const uint8_t* bytes, bool big_endian) {
    uint8_t raw[4];
    std::memcpy(raw, bytes, 4);
    if (big_endian) {
      std::swap(raw[0], raw[3]);
      std::swap(raw[1], raw[2]);
    }
    float value;
    std::memcpy(&value, raw, 4);
    return value;
  }

  inline void WriteFloat(uint8_t* bytes, float value, bool big_endian) {
    uint8_t raw[4];
    std::memcpy(raw, &value, 4);
    if (big_endian) {
      std::swap(raw[0], raw[3]);
      std::swap(raw[1], raw[2]);
    }
    std::memcpy(bytes, raw, 4);
  }

  // Calls fn(index, x, y, z) for every point, in row-major order.
  template <typename Fn>
  void ForEachPoint(const uint8_t* data, const CloudLayout& layout, Fn&& fn) {
    for (uint32_t row = 0; row < layout.height; ++row) {
      const uint8_t* point = data + static_cast<size_t>(row) * layout.row_step;
      for (uint32_t col = 0; col < layout.width; ++col, point += layout.point_step) {
        fn(static_cast<size_t>(row) * layout.width + col,
           ReadFloat(point + layout.x, layout.big_endian),
           ReadFloat(point + layout.y, layout.big_endian),
           ReadFloat(point + layout.z, layout.big_endian));
      }
    }
  }

}  // namespace cfr_arduino_bridge
