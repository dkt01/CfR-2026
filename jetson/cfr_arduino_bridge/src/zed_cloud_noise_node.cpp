// Add stereo-like error to Gazebo's perfect cloud before downstream nodes
// see it. Modify the message in place to avoid another full cloud copy.
// Noise sigma = noise_a + noise_b * range^2; dropout removes returns.

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "cfr_arduino_bridge/cloud_msg.hpp"
#include "cfr_arduino_bridge/stereo_noise.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace cfr_arduino_bridge {

  class ZedCloudNoiseNode : public rclcpp::Node {
   public:
    ZedCloudNoiseNode() : rclcpp::Node("zed_cloud_noise") {
      model_.noise_a = declare_parameter<double>("noise_a", model_.noise_a);
      model_.noise_b = declare_parameter<double>("noise_b", model_.noise_b);
      model_.dropout = declare_parameter<double>("dropout", model_.dropout);
      const int64_t seed = declare_parameter<int64_t>("seed", 0);
      state_ = seed != 0 ? static_cast<uint64_t>(seed) :
                           (static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}();

      // Reliable, because rl/bale_follower's environment subscribes reliably
      // and a reliable subscriber hears nothing from a best-effort publisher.
      // Depth 1: a consumer that falls behind wants the newest frame.
      publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/zed/zed_node/point_cloud/cloud_registered",
                                                                   rclcpp::QoS(1).reliable());
      subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          "/zed/gz/rgbd/points", rclcpp::QoS(1).reliable(), [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
            OnCloud(std::move(msg));
          });
    }

   private:
    void OnCloud(sensor_msgs::msg::PointCloud2::UniquePtr msg) {
      try {
        const CloudLayout layout = LayoutOf(*msg);
        ApplyStereoNoise(msg->data.data(), msg->data.size(), layout, model_, state_);
      } catch (const std::invalid_argument& error) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cannot corrupt this cloud: %s", error.what());
        return;
      }
      msg->is_dense = false;
      publisher_->publish(std::move(msg));
    }

    StereoNoise model_;
    uint64_t state_ = 0;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::ZedCloudNoiseNode>());
  rclcpp::shutdown();
  return 0;
}
