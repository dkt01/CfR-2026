// Replaces the Arduino bridge for Gazebo runs. It accepts the same normalized
// command and emits a Twist for Gazebo plus the status heartbeat expected by
// monitoring tools. Gazebo remains source of odometry and collision physics.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "cfr_interfaces/msg/arduino_status.hpp"
#include "cfr_interfaces/msg/drive_command.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cfr_arduino_bridge {

  class SimVehicleNode : public rclcpp::Node {
   public:
    SimVehicleNode() : rclcpp::Node("sim_vehicle") {
      wheelbase_ = declare_parameter<double>("wheelbase", 0.324);
      max_speed_ = declare_parameter<double>("max_speed", 8.0);
      max_steering_angle_ = declare_parameter<double>("max_steering_angle", 0.40);
      command_timeout_ = declare_parameter<double>("command_timeout", 0.3);
      publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);
      neutral_throttle_deadband_ = declare_parameter<double>("neutral_throttle_deadband", 0.03);
      neutral_coast_deceleration_ = declare_parameter<double>("neutral_coast_deceleration", 1.0);
      forward_neutral_coast_deceleration_ =
          declare_parameter<double>("forward_neutral_coast_deceleration", neutral_coast_deceleration_);
      reverse_neutral_coast_deceleration_ =
          declare_parameter<double>("reverse_neutral_coast_deceleration", neutral_coast_deceleration_);

      if (wheelbase_ <= 0.0 || max_speed_ <= 0.0 || max_steering_angle_ <= 0.0 || neutral_throttle_deadband_ < 0.0 ||
          neutral_coast_deceleration_ < 0.0 || forward_neutral_coast_deceleration_ < 0.0 ||
          reverse_neutral_coast_deceleration_ < 0.0) {
        throw std::invalid_argument("vehicle dimensions and limits must be non-negative, with positive dimensions");
      }

      twist_publisher_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", rclcpp::QoS(10));
      status_publisher_ = create_publisher<cfr_interfaces::msg::ArduinoStatus>("~/status", rclcpp::QoS(10));
      command_subscription_ = create_subscription<cfr_interfaces::msg::DriveCommand>(
          "~/drive_cmd", rclcpp::SensorDataQoS(), [this](const cfr_interfaces::msg::DriveCommand::SharedPtr msg) {
            command_ = *msg;
            command_time_ = now();
          });

      const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 1.0));
      timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { OnTimer(); });
    }

   private:
    void OnTimer() {
      const rclcpp::Time stamp = now();
      const bool fresh = command_time_.nanoseconds() != 0 && (stamp - command_time_).seconds() < command_timeout_;
      const bool active = fresh && command_.auto_ready;
      const double elapsed =
          last_publish_time_.nanoseconds() == 0 ? 0.0 : std::max(0.0, (stamp - last_publish_time_).seconds());
      last_publish_time_ = stamp;

      geometry_msgs::msg::Twist twist;
      if (active) {
        const double throttle = std::clamp(static_cast<double>(command_.throttle), -1.0, 1.0);
        if (std::abs(throttle) <= neutral_throttle_deadband_) {
          CoastToStop(elapsed);
        } else {
          simulated_speed_ = throttle * max_speed_;
        }
        const double steering = std::clamp(static_cast<double>(command_.steering), -1.0, 1.0) * max_steering_angle_;
        twist.linear.x = simulated_speed_;
        twist.angular.z = simulated_speed_ * std::tan(steering) / wheelbase_;
      } else {
        CoastToStop(elapsed);
        twist.linear.x = simulated_speed_;
      }
      twist_publisher_->publish(twist);

      cfr_interfaces::msg::ArduinoStatus status;
      status.header.stamp = stamp;
      status.header.frame_id = "base_link";
      status.link_ok = true;
      status.estop = false;
      status.auto_arm = true;
      status.manual_start = true;
      status.mode = active ? cfr_interfaces::msg::ArduinoStatus::MODE_AUTO_ACTIVE :
                             cfr_interfaces::msg::ArduinoStatus::MODE_AUTO_ARMED;
      status.battery_level = 255;
      status.rpm = 0;
      status_publisher_->publish(status);
    }

    void CoastToStop(double elapsed) {
      const double deceleration =
          simulated_speed_ >= 0.0 ? forward_neutral_coast_deceleration_ : reverse_neutral_coast_deceleration_;
      const double speed_step = deceleration * elapsed;
      simulated_speed_ -= std::copysign(std::min(std::abs(simulated_speed_), speed_step), simulated_speed_);
    }

    double wheelbase_ = 0.324;
    double max_speed_ = 8.0;
    double max_steering_angle_ = 0.40;
    double command_timeout_ = 0.3;
    double publish_rate_hz_ = 50.0;
    double neutral_throttle_deadband_ = 0.03;
    double neutral_coast_deceleration_ = 1.0;
    double forward_neutral_coast_deceleration_ = 1.0;
    double reverse_neutral_coast_deceleration_ = 1.0;
    double simulated_speed_ = 0.0;
    cfr_interfaces::msg::DriveCommand command_;
    rclcpp::Time command_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_publish_time_{0, 0, RCL_ROS_TIME};

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_publisher_;
    rclcpp::Publisher<cfr_interfaces::msg::ArduinoStatus>::SharedPtr status_publisher_;
    rclcpp::Subscription<cfr_interfaces::msg::DriveCommand>::SharedPtr command_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::SimVehicleNode>());
  rclcpp::shutdown();
  return 0;
}
