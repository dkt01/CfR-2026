// Drives a sequence of straight/turn segments (e.g. an L-shape) using odometry
// for closed-loop feedback and publishes cmd_vel, same as a human operator
// would with `ros2 topic pub /cmd_vel`.  Sits upstream of cmd_vel_to_drive_node
// and arduino_bridge_node and knows nothing about either -- it only reasons in
// Twist and odometry.
//
// Exposed as an action (not a topic or service) so a client gets goal
// acceptance, live progress feedback, and cancellation for free, which matters
// here: this is code that drives a moving car unattended between feedback
// ticks, so a clean way to abort mid-path is not optional.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "cfr_arduino_bridge/path_geometry.hpp"
#include "cfr_interfaces/action/drive_path.hpp"
#include "cfr_interfaces/msg/path_segment.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace cfr_arduino_bridge {

  using DrivePath = cfr_interfaces::action::DrivePath;
  using GoalHandleDrivePath = rclcpp_action::ServerGoalHandle<DrivePath>;
  using ResetOdometry = std_srvs::srv::Trigger;

  class PathFollowerNode : public rclcpp::Node {
   public:
    PathFollowerNode() : rclcpp::Node("path_follower") {
      params_.cruise_speed = declare_parameter("cruise_speed", params_.cruise_speed);
      params_.turn_speed = declare_parameter("turn_speed", params_.turn_speed);
      params_.turn_rate = declare_parameter("turn_rate", params_.turn_rate);
      params_.heading_kp = declare_parameter("heading_kp", params_.heading_kp);
      params_.heading_deadband = declare_parameter("heading_deadband", params_.heading_deadband);
      params_.max_angular = declare_parameter("max_angular", params_.max_angular);
      params_.decel_distance = declare_parameter("decel_distance", params_.decel_distance);
      params_.decel_angle = declare_parameter("decel_angle", params_.decel_angle);
      params_.min_creep_speed = declare_parameter("min_creep_speed", params_.min_creep_speed);
      params_.min_turn_rate = declare_parameter("min_turn_rate", params_.min_turn_rate);
      params_.distance_tolerance = declare_parameter("distance_tolerance", params_.distance_tolerance);
      params_.angle_tolerance = declare_parameter("angle_tolerance", params_.angle_tolerance);

      control_rate_hz_ = declare_parameter("control_rate_hz", 20.0);
      odom_timeout_ = declare_parameter("odom_timeout", 0.5);
      max_segment_duration_ = declare_parameter("max_segment_duration", 20.0);
      keep_auto_active_when_idle_ = declare_parameter("keep_auto_active_when_idle", false);

      cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", rclcpp::SensorDataQoS());
      odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
          "~/odom", rclcpp::SensorDataQoS(), [this](const nav_msgs::msg::Odometry::SharedPtr msg) { OnOdom(*msg); });

      action_server_ = rclcpp_action::create_server<DrivePath>(
          this,
          "~/drive_path",
          [this](const rclcpp_action::GoalUUID&, std::shared_ptr<const DrivePath::Goal> goal) {
            return HandleGoal(goal);
          },
          [this](const std::shared_ptr<GoalHandleDrivePath>& goal_handle) { return HandleCancel(goal_handle); },
          [this](const std::shared_ptr<GoalHandleDrivePath>& goal_handle) { HandleAccepted(goal_handle); });

      reset_odometry_service_ = create_service<ResetOdometry>(
          "~/reset_odometry",
          [this](const std::shared_ptr<ResetOdometry::Request>&, std::shared_ptr<ResetOdometry::Response> response) {
            HandleResetOdometry(response);
          });

      const auto period = std::chrono::duration<double>(1.0 / std::max(control_rate_hz_, 1.0));
      timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { OnTimer(); });

      RCLCPP_INFO(get_logger(),
                  "path_follower ready, action ~/drive_path, "
                  "waiting for odometry on ~/odom");
    }

   private:
    void OnOdom(const nav_msgs::msg::Odometry& msg) {
      Pose2D raw_pose;
      raw_pose.x = msg.pose.pose.position.x;
      raw_pose.y = msg.pose.pose.position.y;
      const auto& q = msg.pose.pose.orientation;
      raw_pose.yaw = YawFromQuaternion(q.w, q.x, q.y, q.z);
      latest_raw_pose_ = raw_pose;
      have_raw_pose_ = true;
      if (!have_odom_reference_) {
        odom_reference_pose_ = raw_pose;
        have_odom_reference_ = true;
        RCLCPP_INFO(
            get_logger(), "initialized local odometry at x=%.3f y=%.3f yaw=%.3f", raw_pose.x, raw_pose.y, raw_pose.yaw);
      }
      current_pose_ = PoseRelativeTo(raw_pose, odom_reference_pose_);
      have_odom_ = true;
      last_odom_time_ = now();
    }

    bool OdomStale() const { return !have_odom_ || (now() - last_odom_time_).seconds() > odom_timeout_; }

    void HandleResetOdometry(const std::shared_ptr<ResetOdometry::Response>& response) {
      if (goal_handle_ != nullptr) {
        response->success = false;
        response->message = "cannot reset odometry while a path is executing";
        return;
      }
      if (!have_raw_pose_) {
        response->success = false;
        response->message = "no odometry received yet";
        return;
      }

      odom_reference_pose_ = latest_raw_pose_;
      have_odom_reference_ = true;
      current_pose_ = Pose2D{};
      response->success = true;
      response->message = "local odometry reset to current robot pose";
      RCLCPP_INFO(get_logger(),
                  "reset local odometry at x=%.3f y=%.3f yaw=%.3f",
                  odom_reference_pose_.x,
                  odom_reference_pose_.y,
                  odom_reference_pose_.yaw);
    }

    rclcpp_action::GoalResponse HandleGoal(const std::shared_ptr<const DrivePath::Goal>& goal) {
      // Set synchronously so a second goal arriving before handle_accepted()
      // runs for this one is rejected too, regardless of executor timing.
      if (goal_pending_or_active_) {
        RCLCPP_WARN(get_logger(), "rejecting goal, one is already pending or active");
        return rclcpp_action::GoalResponse::REJECT;
      }
      if (goal->segments.empty()) {
        RCLCPP_WARN(get_logger(), "rejecting goal, empty segment list");
        return rclcpp_action::GoalResponse::REJECT;
      }
      for (const auto& segment : goal->segments) {
        if (segment.type == cfr_interfaces::msg::PathSegment::TURN && std::abs(segment.turn_angle) > kPi) {
          RCLCPP_WARN(get_logger(),
                      "rejecting goal, a turn segment exceeds 180 "
                      "degrees; split it into multiple");
          return rclcpp_action::GoalResponse::REJECT;
        }
      }
      goal_pending_or_active_ = true;
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse HandleCancel(const std::shared_ptr<GoalHandleDrivePath>& /*goal_handle*/) {
      return rclcpp_action::CancelResponse::ACCEPT;
    }

    void HandleAccepted(const std::shared_ptr<GoalHandleDrivePath>& goal_handle) {
      if (OdomStale()) {
        auto result = std::make_shared<DrivePath::Result>();
        result->success = false;
        result->message = "no recent odometry, refusing to start";
        goal_handle->abort(result);
        goal_pending_or_active_ = false;
        return;
      }

      goal_handle_ = goal_handle;
      segments_ = goal_handle->get_goal()->segments;
      segment_index_ = 0;
      segment_start_pose_ = current_pose_;
      segment_start_time_ = now();
      RCLCPP_INFO(get_logger(), "starting path with %zu segment(s)", segments_.size());
    }

    void OnTimer() {
      if (goal_handle_ == nullptr) {
        if (keep_auto_active_when_idle_) {
          PublishTwist(0.0, 0.0);
        }
        return;
      }

      const auto stamp = now();

      if (OdomStale()) {
        AbortGoal("odometry lost mid-path");
        return;
      }

      if (goal_handle_->is_canceling()) {
        PublishTwist(0.0, 0.0);
        auto result = std::make_shared<DrivePath::Result>();
        result->success = false;
        result->message = "canceled";
        goal_handle_->canceled(result);
        FinishGoal();
        return;
      }

      if ((stamp - segment_start_time_).seconds() > max_segment_duration_) {
        AbortGoal("segment did not complete within max_segment_duration, check "
                  "for wheel slip or lost odometry");
        return;
      }

      const auto& ros_segment = segments_[segment_index_];
      Segment segment;
      segment.type =
          ros_segment.type == cfr_interfaces::msg::PathSegment::TURN ? SegmentType::kTurn : SegmentType::kStraight;
      segment.distance = ros_segment.distance;
      segment.turn_angle = ros_segment.turn_angle;

      const SegmentCommand command = ComputeSegmentCommand(segment, segment_start_pose_, current_pose_, params_);
      PublishTwist(command.linear_x, command.angular_z);
      const DesiredPose desired = ComputeDesiredPose(segment, segment_start_pose_);

      auto feedback = std::make_shared<DrivePath::Feedback>();
      feedback->current_segment = static_cast<uint32_t>(segment_index_);
      feedback->total_segments = static_cast<uint32_t>(segments_.size());
      feedback->segment_progress = static_cast<float>(command.progress);
      feedback->current_x = current_pose_.x;
      feedback->current_y = current_pose_.y;
      feedback->current_yaw = current_pose_.yaw;
      feedback->desired_x = desired.pose.x;
      feedback->desired_y = desired.pose.y;
      feedback->desired_yaw = desired.pose.yaw;
      feedback->desired_position_valid = desired.position_valid;
      feedback->error_x = desired.pose.x - current_pose_.x;
      feedback->error_y = desired.pose.y - current_pose_.y;
      feedback->error_yaw = WrapToPi(desired.pose.yaw - current_pose_.yaw);
      feedback->commanded_linear_x = command.linear_x;
      feedback->commanded_angular_z = command.angular_z;
      goal_handle_->publish_feedback(feedback);

      if (!command.complete) {
        return;
      }

      ++segment_index_;
      if (segment_index_ >= segments_.size()) {
        PublishTwist(0.0, 0.0);
        auto result = std::make_shared<DrivePath::Result>();
        result->success = true;
        result->message = "path complete";
        goal_handle_->succeed(result);
        RCLCPP_INFO(get_logger(), "path complete");
        FinishGoal();
        return;
      }

      segment_start_pose_ = current_pose_;
      segment_start_time_ = stamp;
    }

    void AbortGoal(const std::string& reason) {
      PublishTwist(0.0, 0.0);
      RCLCPP_ERROR(get_logger(), "aborting path: %s", reason.c_str());
      auto result = std::make_shared<DrivePath::Result>();
      result->success = false;
      result->message = reason;
      goal_handle_->abort(result);
      FinishGoal();
    }

    void FinishGoal() {
      goal_handle_.reset();
      goal_pending_or_active_ = false;
    }

    void PublishTwist(double linear_x, double angular_z) {
      geometry_msgs::msg::Twist twist;
      twist.linear.x = linear_x;
      twist.angular.z = angular_z;
      cmd_vel_publisher_->publish(twist);
    }

    ControllerParams params_;
    double control_rate_hz_ = 20.0;
    double odom_timeout_ = 0.5;
    double max_segment_duration_ = 20.0;
    bool keep_auto_active_when_idle_ = false;

    Pose2D current_pose_;
    Pose2D odom_reference_pose_;
    Pose2D latest_raw_pose_;
    bool have_odom_ = false;
    bool have_odom_reference_ = false;
    bool have_raw_pose_ = false;
    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};

    std::vector<cfr_interfaces::msg::PathSegment> segments_;
    size_t segment_index_ = 0;
    Pose2D segment_start_pose_;
    rclcpp::Time segment_start_time_{0, 0, RCL_ROS_TIME};

    bool goal_pending_or_active_ = false;
    std::shared_ptr<GoalHandleDrivePath> goal_handle_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp_action::Server<DrivePath>::SharedPtr action_server_;
    rclcpp::Service<ResetOdometry>::SharedPtr reset_odometry_service_;
    rclcpp::TimerBase::SharedPtr timer_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::PathFollowerNode>());
  rclcpp::shutdown();
  return 0;
}
