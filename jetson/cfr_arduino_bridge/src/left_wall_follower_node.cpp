// Drive the speed course after either visual or manual start; stop after
// three laps.
//
//     ros2 run cfr_arduino_bridge left_wall_follower_node --ros-args -p source:=cloud
//
// Two sources (left_wall.hpp has the control laws):
//
//   * `cloud`, the physical car: steers off the ZED's registered cloud.  This
//     is the car's sensor-to-actuator path, so a command goes out the moment
//     a cloud has been read -- not at the next tick of a timer, which is up to
//     a whole period of latency the car would be driving blind through.
//   * `simulation`: follows the centerline against Gazebo ground truth pose
//     and the bales in the world file, on the timer.
//
// Either way the timer keeps /cmd_vel alive at 10 Hz: the latest command
// while running and fresh, and a stop whenever not started, stopped, done, or
// when the source has gone quiet for half a second.
//
// | Interface | Type | Direction |
// | --------- | ---- | --------- |
// | `/start_signal_detector/go` | `std_msgs/Bool` | subscribed, latched |
// | `/lap_counter/done` | `std_msgs/Bool` | subscribed, latched |
// | `cloud_topic` (cloud) | `sensor_msgs/PointCloud2` | subscribed |
// | `/zed/zed_node/pose` (simulation) | `geometry_msgs/PoseStamped` | subscribed |
// | `/obstacle_randomizer/start_signal_green` (simulation) | `std_msgs/Bool` | subscribed, latched |
// | `/start_signal_detector/state` (simulation) | `cfr_interfaces/StartSignal` | subscribed |
// | `/cmd_vel` | `geometry_msgs/Twist` | published |
// | `/left_wall_follower/manual_go` | `std_msgs/Bool` | published, latched |
// | `~/manual_start` | `std_srvs/SetBool` | service |
//
// Parameters: `source` (above), `cloud_topic`, `cloud_reliable` (default
// true; see CloudQoS in cloud_msg.hpp), `auto_start_signal` (simulation: turn
// the simulated signal once the detector is armed).

#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "cfr_arduino_bridge/cloud_msg.hpp"
#include "cfr_arduino_bridge/left_wall.hpp"
#include "cfr_interfaces/msg/start_signal.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace cfr_arduino_bridge {

  namespace lw = left_wall;
  using Bool = std_msgs::msg::Bool;
  using Twist = geometry_msgs::msg::Twist;

  class LeftWallFollowerNode : public rclcpp::Node {
   public:
    LeftWallFollowerNode() : rclcpp::Node("left_wall_follower") {
      source_ = declare_parameter<std::string>("source", "simulation");
      const auto cloud_topic =
          declare_parameter<std::string>("cloud_topic", "/zed/zed_node/point_cloud/cloud_registered");
      declare_parameter<bool>("auto_start_signal", true);
      const bool reliable = declare_parameter<bool>("cloud_reliable", true);
      if (source_ != "simulation" && source_ != "cloud") {
        throw std::invalid_argument("source must be 'simulation' or 'cloud'");
      }
      simulation_ = source_ == "simulation";
      if (simulation_) {
        const std::string share = ament_index_cpp::get_package_share_directory("cfr_arduino_bridge");
        bales_ = lw::LoadBales(share + "/worlds/speed_course.sdf");
        follower_.emplace(share + "/config/speed_course_path.json");
      }

      const auto latched = rclcpp::QoS(1).transient_local();
      go_sub_ =
          create_subscription<Bool>("/start_signal_detector/go", latched, [this](const Bool& msg) { go_ = msg.data; });
      done_sub_ = create_subscription<Bool>("/lap_counter/done", latched, [this](const Bool& msg) { OnDone(msg); });
      if (simulation_) {
        signal_client_ = create_client<std_srvs::srv::SetBool>("/obstacle_randomizer/start_signal");
        green_sub_ =
            create_subscription<Bool>("/obstacle_randomizer/start_signal_green", latched, [this](const Bool& msg) {
              if (msg.data && !signal_green_ && go_) {
                manual_stop_ = false;
              }
              signal_green_ = msg.data;
            });
        state_sub_ = create_subscription<cfr_interfaces::msg::StartSignal>(
            "/start_signal_detector/state", 10, [this](const cfr_interfaces::msg::StartSignal& msg) {
              OnSignalState(msg);
            });
        pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10, [this](const geometry_msgs::msg::PoseStamped& msg) { OnPose(msg); });
      } else {
        // Only the newest cloud matters to a controller.
        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, CloudQoS(reliable), [this](const sensor_msgs::msg::PointCloud2& msg) { OnCloud(msg); });
      }
      manual_publisher_ = create_publisher<Bool>("/left_wall_follower/manual_go", latched);
      manual_service_ = create_service<std_srvs::srv::SetBool>(
          "~/manual_start",
          [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                 std::shared_ptr<std_srvs::srv::SetBool::Response> response) { OnManualStart(*request, *response); });
      publisher_ = create_publisher<Twist>("/cmd_vel", 10);
      // On the node's clock, so it runs on sim time in Gazebo.
      timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(100), [this] { Tick(); });

      // A stop on the way out, while the context can still publish it.
      shutdown_stop_ =
          get_node_base_interface()->get_context()->add_pre_shutdown_callback([this] { publisher_->publish(Twist()); });
      RCLCPP_INFO(get_logger(), "%s wall follower waiting for start", source_.c_str());
    }

    ~LeftWallFollowerNode() override {
      get_node_base_interface()->get_context()->remove_pre_shutdown_callback(shutdown_stop_);
    }

   private:
    bool Driving() const {
      const bool visual_start = go_ && (simulation_ ? signal_green_ : true);
      return (manual_go_ || visual_start) && !manual_stop_ && !done_;
    }

    // The input's age in seconds is within [0, 0.5).
    bool Fresh(const std::optional<rclcpp::Time>& stamp) const {
      if (!stamp) {
        return false;
      }
      const double age = (now() - *stamp).seconds();
      return age >= 0.0 && age < 0.5;
    }

    static Twist ToTwist(const lw::Command& command) {
      Twist twist;
      twist.linear.x = command.speed;
      twist.angular.z = command.speed * std::tan(command.steer) / lw::kWheelbase;
      return twist;
    }

    void OnDone(const Bool& msg) {
      done_ = msg.data;
      if (done_) {
        publisher_->publish(Twist());
        RCLCPP_INFO(get_logger(), "lap counter complete; stopped");
      }
    }

    void OnSignalState(const cfr_interfaces::msg::StartSignal& msg) {
      // Wait for a confirmed red frame before turning the simulated arm.
      if (get_parameter("auto_start_signal").as_bool() && msg.armed && !signal_requested_ &&
          signal_client_->service_is_ready()) {
        signal_requested_ = true;
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = true;
        // With a callback, so the client does not keep the request pending.
        signal_client_->async_send_request(request, [](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture) {});
      }
    }

    void OnManualStart(const std_srvs::srv::SetBool::Request& request, std_srvs::srv::SetBool::Response& response) {
      manual_go_ = request.data;
      manual_stop_ = !request.data;
      Bool msg;
      msg.data = request.data;
      manual_publisher_->publish(msg);
      if (!request.data) {
        publisher_->publish(Twist());
      }
      response.success = true;
      response.message = request.data ? "manual start" : "manual stop";
    }

    void OnPose(const geometry_msgs::msg::PoseStamped& msg) {
      const auto& q = msg.pose.orientation;
      const double yaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
      pose_ = {msg.pose.position.x, msg.pose.position.y, yaw};
      pose_time_ = now();
    }

    void OnCloud(const sensor_msgs::msg::PointCloud2& msg) {
      try {
        lw::ScanCloud(msg.data.data(), msg.data.size(), LayoutOf(msg), &scan_);
      } catch (const std::invalid_argument& error) {
        RCLCPP_WARN(get_logger(), "unusable ZED cloud: %s", error.what());
        return;
      }
      cloud_command_ = lw::CloudCommand(scan_);
      cloud_time_ = now();
      if (Driving()) {
        publisher_->publish(ToTwist(*cloud_command_));
      }
    }

    void Tick() {
      Twist twist;
      if (Driving()) {
        if (simulation_ && pose_ && Fresh(pose_time_)) {
          twist = ToTwist(follower_->Step(bales_, pose_->x, pose_->y, pose_->yaw));
        } else if (!simulation_ && cloud_command_ && Fresh(cloud_time_)) {
          twist = ToTwist(*cloud_command_);
        } else if (!simulation_) {
          RCLCPP_WARN_THROTTLE(get_logger(),
                               *get_clock(),
                               2000,
                               "started, but no fresh cloud on %s; stopped.  If its publisher is best effort, "
                               "set cloud_reliable:=false",
                               cloud_sub_->get_topic_name());
        }
      }
      publisher_->publish(twist);
    }

    struct Pose {
      double x, y, yaw;
    };

    std::string source_;
    bool simulation_ = true;
    std::vector<lw::Bale> bales_;
    std::optional<lw::CourseFollower> follower_;
    lw::SectorScan scan_;

    bool go_ = false;
    bool manual_go_ = false;
    bool manual_stop_ = false;
    bool signal_green_ = false;
    bool done_ = false;
    bool signal_requested_ = false;
    std::optional<Pose> pose_;
    std::optional<rclcpp::Time> pose_time_;
    std::optional<lw::Command> cloud_command_;
    std::optional<rclcpp::Time> cloud_time_;

    rclcpp::Subscription<Bool>::SharedPtr go_sub_, done_sub_, green_sub_;
    rclcpp::Subscription<cfr_interfaces::msg::StartSignal>::SharedPtr state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr signal_client_;
    rclcpp::Publisher<Bool>::SharedPtr manual_publisher_;
    rclcpp::Publisher<Twist>::SharedPtr publisher_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr manual_service_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::PreShutdownCallbackHandle shutdown_stop_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  // Held past shutdown, which runs the node's stop-on-exit callback.
  const auto node = std::make_shared<cfr_arduino_bridge::LeftWallFollowerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
