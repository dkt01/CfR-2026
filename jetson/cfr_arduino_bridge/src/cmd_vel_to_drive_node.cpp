// Converts geometry_msgs/Twist velocity commands from the autonomy stack into
// the DriveCommand that arduino_bridge_node puts on the wire.  Speed passes
// straight through as a velocity target for the Arduino's speed controller;
// only steering needs a model.
//
// The Slash is an Ackermann platform, so the yaw rate in a Twist is turned into
// a steering angle with the bicycle model:
//
//     delta = atan(wheelbase * yaw_rate / speed)
//
// That angle then has to become the normalized steering command on the wire,
// and THAT is where this node used to be wrong.  It divided by a single
// symmetric max_steering_angle, but the measured car is asymmetric -- 0.512
// rad left against 0.382 rad right (vehicle.yaml steering.effective_angle_
// table, and steering_angle_points below).  Dividing 0.20 rad by 0.40 gives a
// command of 0.50, which the table renders as 0.256 rad: 28% MORE steering
// than was asked for.  The same arithmetic under-steers right by 4.5%.  The
// error is a constant fraction, so it does not wash out -- every left turn is
// a third sharper than the planner intended, and the car walks left.
//
// So the table is INVERTED here instead: given a desired angle, interpolate
// the command that produces it.  With no table configured the old symmetric
// scaling is kept, so a stack that has not set the points behaves as before.
//
// Republished at a fixed rate so the bridge always has a fresh command, and
// zeroed when the upstream planner goes quiet.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

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
      min_speed_for_steering_ = declare_parameter<double>("min_speed_for_steering", 0.3);
      cmd_timeout_ = declare_parameter<double>("cmd_timeout", 0.3);
      publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);

      // Measured command -> angle table, the same one sim_vehicle_node uses.
      // Empty keeps the old symmetric behaviour.
      steering_commands_ =
          declare_parameter<std::vector<double>>("steering_command_points", std::vector<double>{});
      steering_angles_ =
          declare_parameter<std::vector<double>>("steering_angle_points", std::vector<double>{});
      if (steering_commands_.size() != steering_angles_.size()) {
        RCLCPP_FATAL(get_logger(), "steering_command_points and steering_angle_points must match");
        throw std::invalid_argument("steering table mismatch");
      }
      for (size_t i = 1; i < steering_angles_.size(); ++i) {
        if (steering_angles_[i] <= steering_angles_[i - 1]) {
          RCLCPP_FATAL(get_logger(), "steering_angle_points must be strictly increasing to invert");
          throw std::invalid_argument("steering table not invertible");
        }
      }

      if (max_speed_ <= 0.0 || max_steering_angle_ <= 0.0 || wheelbase_ <= 0.0) {
        RCLCPP_FATAL(get_logger(), "wheelbase, max_speed and max_steering_angle must all be positive");
        throw std::invalid_argument("invalid vehicle parameters");
      }

      publisher_ = create_publisher<cfr_interfaces::msg::DriveCommand>("drive_cmd", rclcpp::SensorDataQoS());
      subscription_ = create_subscription<geometry_msgs::msg::Twist>(
          "cmd_vel", rclcpp::SensorDataQoS(), [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
            last_twist_ = *msg;
            last_twist_time_ = now();
          });

      const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 1.0));
      timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { OnTimer(); });

      RCLCPP_INFO(get_logger(),
                  "cmd_vel_to_drive: wheelbase=%.3f m max_speed=%.2f m/s "
                  "max_steer=%.2f rad",
                  wheelbase_,
                  max_speed_,
                  max_steering_angle_);
    }

   private:
    void OnTimer() {
      const rclcpp::Time stamp = now();
      const bool fresh = last_twist_time_.nanoseconds() != 0 && (stamp - last_twist_time_).seconds() < cmd_timeout_;

      cfr_interfaces::msg::DriveCommand msg;
      msg.header.stamp = stamp;
      msg.header.frame_id = "base_link";
      msg.auto_ready = fresh;
      msg.steering = 0.0F;
      msg.velocity = 0.0F;

      if (fresh) {
        const double speed = last_twist_.linear.x;
        const double yaw_rate = last_twist_.angular.z;

        const double steering_speed = std::max(std::abs(speed), min_speed_for_steering_);
        const double steering_angle = std::atan2(wheelbase_ * yaw_rate, steering_speed);

        msg.steering = static_cast<float>(CommandForAngle(steering_angle));
        msg.velocity = static_cast<float>(std::clamp(speed, -max_speed_, max_speed_));
      } else if (last_twist_time_.nanoseconds() != 0) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000, "cmd_vel stale (> %.2f s), commanding neutral", cmd_timeout_);
      }

      publisher_->publish(msg);
    }

    // Desired steering angle -> normalized command, by inverting the measured
    // table.  Falls back to the symmetric scaling when no table is set.
    double CommandForAngle(double angle) const {
      if (steering_angles_.size() < 2) {
        return std::clamp(angle / max_steering_angle_, -1.0, 1.0);
      }
      if (angle <= steering_angles_.front()) {
        return steering_commands_.front();
      }
      if (angle >= steering_angles_.back()) {
        return steering_commands_.back();
      }
      for (size_t i = 1; i < steering_angles_.size(); ++i) {
        if (angle <= steering_angles_[i]) {
          const double span = steering_angles_[i] - steering_angles_[i - 1];
          const double fraction = (angle - steering_angles_[i - 1]) / span;
          return steering_commands_[i - 1] +
                 fraction * (steering_commands_[i] - steering_commands_[i - 1]);
        }
      }
      return steering_commands_.back();
    }

    double wheelbase_ = 0.324;
    double max_speed_ = 4.0;
    double max_steering_angle_ = 0.40;
    double min_speed_for_steering_ = 0.3;
    double cmd_timeout_ = 0.3;
    double publish_rate_hz_ = 50.0;

    std::vector<double> steering_commands_;
    std::vector<double> steering_angles_;
    geometry_msgs::msg::Twist last_twist_;
    rclcpp::Time last_twist_time_{0, 0, RCL_ROS_TIME};

    rclcpp::Publisher<cfr_interfaces::msg::DriveCommand>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::CmdVelToDriveNode>());
  rclcpp::shutdown();
  return 0;
}
