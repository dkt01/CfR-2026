// Run cloud_segmentation on the live ZED cloud and republish it by class.
//
// Subscribes to the ZED's registered cloud and its pose, classifies every
// point, and publishes the cloud again with each point's color set by its
// class, so RViz and the browser viewer can show what the segmenter decided
// rather than what the camera saw.  The pose is only read for its pitch and
// roll -- on the car the ZED's map frame is gravity-aligned, in the sim the
// bridged ground-truth pose stands in for it -- which is what the segmenter
// needs to level the cloud.
//
// Colors: ground gray-green, obstacle red, hoop magenta, car wash cyan,
// overhead blue; unknown points are left out.
//
// | Interface | Type | Direction |
// | --------- | ---- | --------- |
// | `cloud` | `sensor_msgs/PointCloud2` | subscribed |
// | `pose` | `geometry_msgs/PoseStamped` | subscribed |
// | `~/cloud` | `sensor_msgs/PointCloud2` | published, x/y/z/rgb |
//
// Parameters: `stride` thins what is published (not what is segmented) to
// every stride-th row and column, because the browser draws it and a quarter
// of 230k points is plenty to see.  Every cloud_segmentation::Params field is
// a parameter too, under its own name, read at startup.  `cloud_reliable`
// (default true) subscribes reliably; see CloudQoS.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "cfr_arduino_bridge/cloud_msg.hpp"
#include "cfr_arduino_bridge/cloud_segmentation.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace cfr_arduino_bridge {

  namespace seg = segmentation;

  // uint32 0x00RRGGBB per class, the PCL/ROS packed-rgb convention.
  inline uint32_t ClassColor(uint8_t label) {
    const auto rgb = [](uint32_t r, uint32_t g, uint32_t b) { return (r << 16) | (g << 8) | b; };
    switch (label) {
      case seg::kGround:
        return rgb(96, 128, 96);
      case seg::kObstacle:
        return rgb(230, 50, 40);
      case seg::kHoop:
        return rgb(240, 60, 240);
      case seg::kCarWash:
        return rgb(40, 220, 230);
      case seg::kOverhead:
        return rgb(70, 110, 255);
      default:
        return rgb(60, 60, 60);
    }
  }

  class CloudSegmentationNode : public rclcpp::Node {
   public:
    CloudSegmentationNode() : rclcpp::Node("cloud_segmentation") {
      stride_ = std::max<int64_t>(1, declare_parameter<int64_t>("stride", 2));
      const bool reliable = declare_parameter<bool>("cloud_reliable", true);
#define CFR_DECLARE(field) \
  params_.field = static_cast<decltype(params_.field)>(declare_parameter(#field, params_.field));
      CFR_SEGMENTATION_PARAMS(CFR_DECLARE)
#undef CFR_DECLARE

      publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud", 1);
      pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          "pose", 10, [this](const geometry_msgs::msg::PoseStamped& msg) { OnPose(msg); });
      // Depth 1: a frame that waited behind a slow one is stale by the time
      // it is read, so only the newest is kept.
      cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          "cloud", CloudQoS(reliable), [this](const sensor_msgs::msg::PointCloud2& msg) { OnCloud(msg); });
    }

   private:
    // Nose-down pitch and left-up roll from a REP-103 orientation.
    void OnPose(const geometry_msgs::msg::PoseStamped& msg) {
      const auto& q = msg.pose.orientation;
      pitch_ = std::asin(std::clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0));
      roll_ = std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
    }

    void OnCloud(const sensor_msgs::msg::PointCloud2& msg) {
      CloudLayout layout;
      try {
        layout = LayoutOf(msg);
      } catch (const std::invalid_argument& error) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cannot segment this cloud: %s", error.what());
        return;
      }
      const size_t n = layout.Count();
      xyz_.resize(3 * n);
      ForEachPoint(msg.data.data(), layout, [this](size_t i, float x, float y, float z) {
        xyz_[3 * i] = x;
        xyz_[3 * i + 1] = y;
        xyz_[3 * i + 2] = z;
      });
      // Both the ZED and Gazebo publish registered color, but their extra
      // point fields are not laid out identically. Accept packed rgb or rgba
      // in either FLOAT32 or UINT32 form, and keep geometry-only behavior if
      // the cloud does not provide color.
      const sensor_msgs::msg::PointField* color_field = nullptr;
      for (const auto& field : msg.fields) {
        if ((field.name == "rgb" || field.name == "rgba") &&
            (field.datatype == sensor_msgs::msg::PointField::FLOAT32 ||
             field.datatype == sensor_msgs::msg::PointField::UINT32) &&
            field.offset + 4 <= layout.point_step) {
          color_field = &field;
          break;
        }
      }
      if (color_field) {
        rgb_.resize(n);
        for (size_t i = 0; i < n; ++i) {
          const uint8_t* bytes = msg.data.data() + layout.Offset(i) + color_field->offset;
          rgb_[i] = msg.is_bigendian ? (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
                                           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]) :
                                       (uint32_t(bytes[3]) << 24) | (uint32_t(bytes[2]) << 16) |
                                           (uint32_t(bytes[1]) << 8) | uint32_t(bytes[0]);
        }
      }
      seg::Segment(xyz_.data(), n, pitch_, roll_, params_, &result_, color_field ? rgb_.data() : nullptr);

      // Everything is segmented; only every stride-th row and column of an
      // organized cloud is published.
      const size_t stride = layout.height > 1 ? static_cast<size_t>(stride_) : 1;
      auto out = std::make_unique<sensor_msgs::msg::PointCloud2>();
      out->header = msg.header;
      out->height = 1;
      out->is_bigendian = false;
      out->point_step = 16;
      out->is_dense = true;
      const char* names[] = {"x", "y", "z", "rgb"};
      for (uint32_t k = 0; k < 4; ++k) {
        sensor_msgs::msg::PointField field;
        field.name = names[k];
        field.offset = 4 * k;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        out->fields.push_back(field);
      }
      out->data.resize(16 * ((layout.height + stride - 1) / stride) * ((layout.width + stride - 1) / stride));
      uint8_t* write = out->data.data();
      for (size_t row = 0; row < layout.height; row += stride) {
        for (size_t col = 0; col < layout.width; col += stride) {
          const size_t i = row * layout.width + col;
          const uint8_t label = result_.labels[i];
          if (label == seg::kUnknown) {
            continue;
          }
          const float point[3] = {static_cast<float>(xyz_[3 * i]),
                                  static_cast<float>(xyz_[3 * i + 1]),
                                  static_cast<float>(xyz_[3 * i + 2])};
          const uint32_t color = ClassColor(label);
          std::memcpy(write, point, 12);
          std::memcpy(write + 12, &color, 4);
          write += 16;
        }
      }
      const size_t count = static_cast<size_t>(write - out->data.data()) / 16;
      out->data.resize(16 * count);
      out->width = static_cast<uint32_t>(count);
      out->row_step = static_cast<uint32_t>(16 * count);
      publisher_->publish(std::move(out));
    }

    int64_t stride_ = 2;
    seg::Params params_;
    double pitch_ = 0.0;
    double roll_ = 0.0;
    std::vector<double> xyz_;
    std::vector<uint32_t> rgb_;
    seg::Segmentation result_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::CloudSegmentationNode>());
  rclcpp::shutdown();
  return 0;
}
