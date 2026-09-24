// Apply a simple stereo-like depth error to Gazebo's perfect RGB-D cloud.
//
// Sits between the bridged Gazebo cloud and the ZED's topic, so everything
// downstream reads a cloud with the real camera's kind of error.  That puts it
// on the simulator's sensor path -- every frame the follower, the segmenter or
// a training environment sees has come through here -- and the real car has
// no such hop at all.  So it does as little as it can: the message it is
// handed is corrupted in place and published on, no copy, keeping the latest
// frame only, and the latency it adds is the sim's alone to carry.
//
// | Interface | Type | Direction |
// | --------- | ---- | --------- |
// | `/zed/gz/rgbd/points` | `sensor_msgs/PointCloud2` | subscribed |
// | `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | published |
//
// Parameters: noise_a, noise_b (sigma = a + b x^2, meters), dropout (fraction
// of returns lost), seed (0 draws one from the OS).

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
