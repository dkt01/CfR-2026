// Converts geometry_msgs/Twist velocity commands from the autonomy stack into
// the normalized DriveCommand that arduino_bridge_node puts on the wire.
//
// The Slash is an Ackermann platform, so the yaw rate in a Twist is turned into
// a steering angle with the bicycle model:
//
//     delta = atan(wheelbase * yaw_rate / speed)
//
// Republished at a fixed rate so the bridge always has a fresh command, and
// zeroed when the upstream planner goes quiet.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "cfr_interfaces/msg/drive_command.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cfr_arduino_bridge {

class CmdVelToDriveNode : public rclcpp::Node {
public:
  CmdVelToDriveNode() : rclcpp::Node("cmd_vel_to_drive") {
    wheelbase_ = declare_parameter<double>("wheelbase", 0.324);
    max_speed_ = declare_parameter<double>("max_speed", 4.0);
    max_steering_angle_ = declare_parameter<double>("max_steering_angle", 0.40);
    min_speed_for_steering_ =
        declare_parameter<double>("min_speed_for_steering", 0.3);
    cmd_timeout_ = declare_parameter<double>("cmd_timeout", 0.3);
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);

    if (max_speed_ <= 0.0 || max_steering_angle_ <= 0.0 || wheelbase_ <= 0.0) {
      RCLCPP_FATAL(
          get_logger(),
          "wheelbase, max_speed and max_steering_angle must all be positive");
      throw std::invalid_argument("invalid vehicle parameters");
    }

    publisher_ = create_publisher<cfr_interfaces::msg::DriveCommand>(
        "drive_cmd", rclcpp::SensorDataQoS());
    subscription_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", rclcpp::SensorDataQoS(),
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
          last_twist_ = *msg;
          last_twist_time_ = now();
        });

    const auto period =
        std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 1.0));
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this] { OnTimer(); });

    RCLCPP_INFO(get_logger(),
                "cmd_vel_to_drive: wheelbase=%.3f m max_speed=%.2f m/s "
                "max_steer=%.2f rad",
                wheelbase_, max_speed_, max_steering_angle_);
  }

private:
  void OnTimer() {
    const rclcpp::Time stamp = now();
    const bool fresh = last_twist_time_.nanoseconds() != 0 &&
                       (stamp - last_twist_time_).seconds() < cmd_timeout_;

    cfr_interfaces::msg::DriveCommand msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";
    msg.auto_ready = fresh;
    msg.steering = 0.0F;
    msg.throttle = 0.0F;

    if (fresh) {
      const double speed = last_twist_.linear.x;
      const double yaw_rate = last_twist_.angular.z;

      const double steering_speed =
          std::max(std::abs(speed), min_speed_for_steering_);
      const double steering_angle =
          std::atan2(wheelbase_ * yaw_rate, steering_speed);

      msg.steering = static_cast<float>(
          std::clamp(steering_angle / max_steering_angle_, -1.0, 1.0));
      msg.throttle =
          static_cast<float>(std::clamp(speed / max_speed_, -1.0, 1.0));
    } else if (last_twist_time_.nanoseconds() != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "cmd_vel stale (> %.2f s), commanding neutral",
                           cmd_timeout_);
    }

    publisher_->publish(msg);
  }

  double wheelbase_ = 0.324;
  double max_speed_ = 4.0;
  double max_steering_angle_ = 0.40;
  double min_speed_for_steering_ = 0.3;
  double cmd_timeout_ = 0.3;
  double publish_rate_hz_ = 50.0;

  geometry_msgs::msg::Twist last_twist_;
  rclcpp::Time last_twist_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<cfr_interfaces::msg::DriveCommand>::SharedPtr publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace cfr_arduino_bridge

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::CmdVelToDriveNode>());
  rclcpp::shutdown();
  return 0;
}
