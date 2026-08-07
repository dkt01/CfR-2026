// Owns the USB serial link to the Arduino.
//
// Responsibilities:
//   * transmit a command frame every cycle, so the Arduino's 200 ms comms
//     watchdog never expires while we are alive,
//   * hold the axes neutral until the Arduino reports AUTO_ACTIVE, which is
//     what lets it complete the AUTO_ARMED -> AUTO_ACTIVE handshake,
//   * fall back to neutral whenever commands go stale, the link drops, or
//     E-Stop is asserted,
//   * decode and republish the Arduino status frame.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "cfr_arduino_bridge/protocol.hpp"
#include "cfr_arduino_bridge/serial_port.hpp"
#include "cfr_interfaces/msg/arduino_status.hpp"
#include "cfr_interfaces/msg/drive_command.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cfr_arduino_bridge {

using namespace std::chrono_literals; // NOLINT(build/namespaces)

class ArduinoBridgeNode : public rclcpp::Node {
public:
  ArduinoBridgeNode() : rclcpp::Node("arduino_bridge") {
    device_ = declare_parameter<std::string>("device", "/dev/ttyACM0");
    baud_ = static_cast<unsigned>(declare_parameter<int>("baud", 115200));
    tx_rate_hz_ = declare_parameter<double>("tx_rate_hz", 50.0);
    command_timeout_ = declare_parameter<double>("command_timeout", 0.2);
    link_timeout_ = declare_parameter<double>("link_timeout", 0.5);
    reconnect_period_ = declare_parameter<double>("reconnect_period", 1.0);
    boot_delay_ = declare_parameter<double>("boot_delay", 2.0);
    max_throttle_ = declare_parameter<double>("max_throttle", 0.25);
    max_steering_ = declare_parameter<double>("max_steering", 1.0);
    throttle_slew_per_s_ =
        declare_parameter<double>("throttle_slew_per_s", 2.0);
    invert_steering_ = declare_parameter<bool>("invert_steering", false);
    invert_throttle_ = declare_parameter<bool>("invert_throttle", false);
    require_auto_active_ = declare_parameter<bool>("require_auto_active", true);

    if (tx_rate_hz_ < 10.0) {
      RCLCPP_WARN(get_logger(),
                  "tx_rate_hz %.1f is below the Arduino's 200 ms watchdog "
                  "margin, clamping to 10 Hz",
                  tx_rate_hz_);
      tx_rate_hz_ = 10.0;
    }
    max_throttle_ = std::clamp(max_throttle_, 0.0, 1.0);
    max_steering_ = std::clamp(max_steering_, 0.0, 1.0);

    status_publisher_ = create_publisher<cfr_interfaces::msg::ArduinoStatus>(
        "~/status", rclcpp::QoS(10));
    command_subscription_ =
        create_subscription<cfr_interfaces::msg::DriveCommand>(
            "~/drive_cmd", rclcpp::SensorDataQoS(),
            [this](const cfr_interfaces::msg::DriveCommand::SharedPtr msg) {
              OnDriveCommand(*msg);
            });

    const auto period = std::chrono::duration<double>(1.0 / tx_rate_hz_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this] { OnTimer(); });

    RCLCPP_INFO(get_logger(),
                "arduino_bridge starting: device=%s baud=%u tx_rate=%.1f Hz "
                "max_throttle=%.2f",
                device_.c_str(), baud_, tx_rate_hz_, max_throttle_);
  }

  ~ArduinoBridgeNode() override {
    // Best effort: leave the car neutral and unarmed on the way out.
    if (port_.IsOpen()) {
      port_.Write(Serialize(JetsonToArduino{}));
    }
  }

private:
  void OnDriveCommand(const cfr_interfaces::msg::DriveCommand &msg) {
    last_command_ = msg;
    last_command_time_ = now();
  }

  void OnTimer() {
    const rclcpp::Time stamp = now();

    if (!EnsureConnected(stamp)) {
      PublishStatus(stamp, false);
      return;
    }

    ServiceReceive(stamp);

    const bool link_ok = link_established_ &&
                         (stamp - last_status_time_).seconds() < link_timeout_;
    if (link_established_ && !link_ok && !link_lost_logged_) {
      RCLCPP_WARN(get_logger(), "no Arduino status frame for %.2f s",
                  link_timeout_);
      link_lost_logged_ = true;
    }

    SendCommand(stamp, link_ok);
    PublishStatus(stamp, link_ok);
  }

  bool EnsureConnected(const rclcpp::Time &stamp) {
    if (port_.IsOpen()) {
      return true;
    }
    if (have_reconnect_time_ &&
        (stamp - last_reconnect_attempt_).seconds() < reconnect_period_) {
      return false;
    }
    last_reconnect_attempt_ = stamp;
    have_reconnect_time_ = true;

    if (!port_.Open(device_, baud_)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s",
                           port_.LastError().c_str());
      return false;
    }

    // Opening the port asserts DTR, which resets an Arduino Uno into its
    // bootloader.  Anything written during that window is lost.
    port_open_time_ = stamp;
    link_established_ = false;
    link_lost_logged_ = false;
    RCLCPP_INFO(get_logger(),
                "opened %s, waiting %.1f s for the Arduino to boot",
                device_.c_str(), boot_delay_);
    return true;
  }

  void HandlePortFailure(const std::string &context) {
    RCLCPP_ERROR(get_logger(), "%s: %s, reconnecting", context.c_str(),
                 port_.LastError().c_str());
    port_.Close();
    link_established_ = false;
  }

  void ServiceReceive(const rclcpp::Time &stamp) {
    lines_.clear();
    if (!port_.ReadLines(lines_)) {
      HandlePortFailure("serial read");
      return;
    }

    for (const std::string &line : lines_) {
      ArduinoToJetson decoded;
      if (!Deserialize(line, decoded)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "dropping malformed status frame '%s'",
                             line.c_str());
        continue;
      }
      if (!link_established_ || decoded.mode != status_.mode) {
        RCLCPP_INFO(get_logger(), "Arduino mode %s -> %s",
                    link_established_ ? ModeName(status_.mode) : "(none)",
                    ModeName(decoded.mode));
      }
      status_ = decoded;
      last_status_time_ = stamp;
      link_established_ = true;
      link_lost_logged_ = false;
    }
  }

  void SendCommand(const rclcpp::Time &stamp, bool link_ok) {
    JetsonToArduino frame; // defaults to neutral and not ready

    if ((stamp - port_open_time_).seconds() < boot_delay_) {
      return; // Arduino is still in its bootloader
    }

    const bool command_fresh =
        HaveCommand() &&
        (stamp - last_command_time_).seconds() < command_timeout_;
    const bool estopped =
        link_ok && (status_.estop || status_.mode == Mode::kEstop);
    const bool auto_active = link_ok && status_.mode == Mode::kAutoActive;

    double steering = 0.0;
    double throttle = 0.0;
    bool passthrough = false;

    if (command_fresh && !estopped) {
      frame.auto_ready = last_command_.auto_ready;
      // Hold neutral until the Arduino has actually entered AUTO_ACTIVE.  It
      // will not make that transition unless both axes sit in the neutral
      // deadband while AUTO_ARMED, so passing commands through early would
      // deadlock the handshake.
      if (auto_active || !require_auto_active_) {
        passthrough = true;
        steering =
            std::clamp(static_cast<double>(last_command_.steering), -1.0, 1.0) *
            max_steering_;
        throttle =
            std::clamp(static_cast<double>(last_command_.throttle), -1.0, 1.0) *
            max_throttle_;
      }
    } else if (command_fresh && estopped) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "E-Stop asserted, holding neutral");
    } else if (HaveCommand()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "drive command stale (> %.2f s), holding neutral",
                           command_timeout_);
    }

    throttle = ApplyThrottleSlew(stamp, throttle, passthrough);

    if (invert_steering_) {
      steering = -steering;
    }
    if (invert_throttle_) {
      throttle = -throttle;
    }

    // DriveCommand uses +1 = left (REP-103); the wire uses 255 = right.
    frame.steering = NormalizedToCommand(-steering);
    frame.throttle = NormalizedToCommand(throttle);

    if (!port_.Write(Serialize(frame))) {
      HandlePortFailure("serial write");
    }
  }

  /// Rate limit the throttle so the ESC is not asked for a step change.  Any
  /// fall back to neutral (stale command, lost link, E-Stop) snaps to zero
  /// immediately rather than ramping.
  double ApplyThrottleSlew(const rclcpp::Time &stamp, double target,
                           bool passthrough) {
    // A missed cycle or a clock jump must not turn into a huge step, so fall
    // back to the nominal period whenever dt looks implausible.
    double dt = have_slew_time_ ? (stamp - last_slew_time_).seconds() : 0.0;
    if (dt <= 0.0 || dt > 1.0) {
      dt = 1.0 / tx_rate_hz_;
    }
    last_slew_time_ = stamp;
    have_slew_time_ = true;

    if (!passthrough) {
      commanded_throttle_ = 0.0;
    } else if (throttle_slew_per_s_ <= 0.0) {
      commanded_throttle_ = target;
    } else {
      const double max_step = throttle_slew_per_s_ * dt;
      commanded_throttle_ +=
          std::clamp(target - commanded_throttle_, -max_step, max_step);
    }
    return commanded_throttle_;
  }

  void PublishStatus(const rclcpp::Time &stamp, bool link_ok) {
    cfr_interfaces::msg::ArduinoStatus msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";
    msg.link_ok = link_ok;
    msg.estop = link_ok ? status_.estop : false;
    msg.auto_arm = link_ok ? status_.auto_arm : false;
    msg.manual_start = link_ok ? status_.manual_start : false;
    msg.mode = link_ok ? static_cast<uint8_t>(status_.mode)
                       : static_cast<uint8_t>(Mode::kEstop);
    msg.battery_level = link_ok ? status_.battery_level : 0;
    status_publisher_->publish(msg);
  }

  bool HaveCommand() const { return last_command_time_.nanoseconds() != 0; }

  // Parameters
  std::string device_;
  unsigned baud_ = 115200;
  double tx_rate_hz_ = 50.0;
  double command_timeout_ = 0.2;
  double link_timeout_ = 0.5;
  double reconnect_period_ = 1.0;
  double boot_delay_ = 2.0;
  double max_throttle_ = 0.25;
  double max_steering_ = 1.0;
  double throttle_slew_per_s_ = 2.0;
  bool invert_steering_ = false;
  bool invert_throttle_ = false;
  bool require_auto_active_ = true;

  // State
  SerialPort port_;
  std::vector<std::string> lines_;
  ArduinoToJetson status_;
  cfr_interfaces::msg::DriveCommand last_command_;
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_reconnect_attempt_{0, 0, RCL_ROS_TIME};
  rclcpp::Time port_open_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_slew_time_{0, 0, RCL_ROS_TIME};
  bool have_reconnect_time_ = false;
  bool have_slew_time_ = false;
  bool link_established_ = false;
  bool link_lost_logged_ = false;
  double commanded_throttle_ = 0.0;

  rclcpp::Publisher<cfr_interfaces::msg::ArduinoStatus>::SharedPtr
      status_publisher_;
  rclcpp::Subscription<cfr_interfaces::msg::DriveCommand>::SharedPtr
      command_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace cfr_arduino_bridge

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::ArduinoBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
