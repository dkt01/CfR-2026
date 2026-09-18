// Owns the USB serial link to the Arduino.
//
// Responsibilities:
//   * transmit a command frame every cycle, so the Arduino's 200 ms comms
//     watchdog never expires while we are alive,
//   * hold steering centered and the speed target at zero until the Arduino
//     reports AUTO_ACTIVE, which is what lets it complete the AUTO_ARMED ->
//     AUTO_ACTIVE handshake,
//   * fall back to a zero speed target whenever commands go stale, the link
//     drops, or E-Stop is asserted,
//   * convert DriveCommand's m/s into the spur RPM target the Arduino's speed
//     controller tracks, and keep that controller's gains in step with this
//     node's parameters,
//   * decode and republish the Arduino status frame.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfr_arduino_bridge/drivetrain.hpp"
#include "cfr_arduino_bridge/protocol.hpp"
#include "cfr_arduino_bridge/serial_port.hpp"
#include "cfr_interfaces/msg/arduino_status.hpp"
#include "cfr_interfaces/msg/drive_command.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cfr_arduino_bridge {

  using namespace std::chrono_literals;  // NOLINT(build/namespaces)

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
      tx_trace_path_ = declare_parameter<std::string>("tx_trace_path", "");
      rx_trace_path_ = declare_parameter<std::string>("rx_trace_path", "");
      trace_timestamps_ = declare_parameter<bool>("trace_timestamps", false);
      max_speed_ = declare_parameter<double>("max_speed", 2.0);
      max_steering_ = declare_parameter<double>("max_steering", 1.0);
      speed_slew_rate_ = declare_parameter<double>("speed_slew_rate", 2.0);
      invert_steering_ = declare_parameter<bool>("invert_steering", false);
      invert_speed_ = declare_parameter<bool>("invert_speed", false);
      require_auto_active_ = declare_parameter<bool>("require_auto_active", true);
      spur_to_wheel_ratio_ = declare_parameter<double>("spur_to_wheel_ratio", kSpurToWheelRatio);
      tire_diameter_ = declare_parameter<double>("tire_diameter", kTireDiameterM);
      gains_resend_period_ = declare_parameter<double>("gains_resend_period", 1.0);

      // Speed controller gains, per m/s of ground speed.  These are the only
      // parameters that take effect when changed at runtime, so the loop can be
      // retuned on the ground with `ros2 param set`.  Defaults are the bench
      // tune, the firmware's compiled-in SpeedGains converted to per m/s.
      gains_.ks = declare_parameter<double>("speed_ks", 28.0);
      gains_.kv = declare_parameter<double>("speed_kv", 4.52);
      gains_.kp = declare_parameter<double>("speed_kp", 7.62);
      gains_.ki = declare_parameter<double>("speed_ki", 4.76);
      gains_.kd = declare_parameter<double>("speed_kd", 0.0);
      gains_.i_limit = declare_parameter<double>("speed_i_limit", 60.0);
      gains_.output_limit = declare_parameter<double>("speed_output_limit", 128.0);
      gains_.brake_limit = declare_parameter<double>("speed_brake_limit", 0.0);
      gains_.trace = declare_parameter<bool>("speed_trace", false);

      if (tx_rate_hz_ < 10.0) {
        RCLCPP_WARN(get_logger(),
                    "tx_rate_hz %.1f is below the Arduino's 200 ms watchdog "
                    "margin, clamping to 10 Hz",
                    tx_rate_hz_);
        tx_rate_hz_ = 10.0;
      }
      max_speed_ = std::max(max_speed_, 0.0);
      max_steering_ = std::clamp(max_steering_, 0.0, 1.0);
      if (!(spur_to_wheel_ratio_ > 0.0)) {
        RCLCPP_WARN(get_logger(),
                    "spur_to_wheel_ratio %.3f must be positive, using %.2f",
                    spur_to_wheel_ratio_,
                    kSpurToWheelRatio);
        spur_to_wheel_ratio_ = kSpurToWheelRatio;
      }
      if (!(tire_diameter_ > 0.0)) {
        RCLCPP_WARN(get_logger(), "tire_diameter %.4f must be positive, using %.4f m", tire_diameter_, kTireDiameterM);
        tire_diameter_ = kTireDiameterM;
      }
      if (ToTargetRpm(SpeedToSpurRpm(max_speed_, spur_to_wheel_ratio_, tire_diameter_)) == kMaxTargetRpm) {
        RCLCPP_WARN(get_logger(),
                    "max_speed %.2f m/s exceeds the %d spur RPM the Arduino accepts; targets will saturate",
                    max_speed_,
                    kMaxTargetRpm);
      }
      if (Serialize(ToFirmwareGains(gains_)).empty()) {
        RCLCPP_FATAL(get_logger(), "speed_* parameters are out of range for the Arduino, see arduino_bridge.yaml");
        throw std::invalid_argument("invalid speed controller gains");
      }
      if (!tx_trace_path_.empty()) {
        tx_trace_.open(tx_trace_path_, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!tx_trace_) {
          RCLCPP_ERROR(get_logger(), "unable to open tx trace %s", tx_trace_path_.c_str());
        } else {
          RCLCPP_INFO(get_logger(), "tracing exact Arduino TX frames to %s", tx_trace_path_.c_str());
        }
      }
      if (!rx_trace_path_.empty()) {
        rx_trace_.open(rx_trace_path_, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!rx_trace_) {
          RCLCPP_ERROR(get_logger(), "unable to open rx trace %s", rx_trace_path_.c_str());
        } else {
          RCLCPP_INFO(get_logger(), "tracing Arduino RX frames to %s", rx_trace_path_.c_str());
        }
      }

      parameter_callback_ = add_on_set_parameters_callback(
          [this](const std::vector<rclcpp::Parameter>& parameters) { return OnSetParameters(parameters); });

      status_publisher_ = create_publisher<cfr_interfaces::msg::ArduinoStatus>("~/status", rclcpp::QoS(10));
      command_subscription_ = create_subscription<cfr_interfaces::msg::DriveCommand>(
          "~/drive_cmd", rclcpp::SensorDataQoS(), [this](const cfr_interfaces::msg::DriveCommand::SharedPtr msg) {
            OnDriveCommand(*msg);
          });

      const auto period = std::chrono::duration<double>(1.0 / tx_rate_hz_);
      timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { OnTimer(); });

      RCLCPP_INFO(get_logger(),
                  "arduino_bridge starting: device=%s baud=%u tx_rate=%.1f Hz "
                  "max_speed=%.2f m/s",
                  device_.c_str(),
                  baud_,
                  tx_rate_hz_,
                  max_speed_);
      LogGains("speed gains");
    }

    ~ArduinoBridgeNode() override {
      // Best effort: leave the car stopped and unarmed on the way out.
      if (port_.IsOpen()) {
        WriteWire(Serialize(JetsonToArduino{}));
      }
    }

   private:
    void OnDriveCommand(const cfr_interfaces::msg::DriveCommand& msg) {
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

      const bool link_ok = link_established_ && (stamp - last_status_time_).seconds() < link_timeout_;
      if (link_established_ && !link_ok && !link_lost_logged_) {
        RCLCPP_WARN(get_logger(), "no Arduino status frame for %.2f s", link_timeout_);
        link_lost_logged_ = true;
      }

      SendCommand(stamp, link_ok);
      ServiceGains(stamp, link_ok);
      PublishStatus(stamp, link_ok);
    }

    bool EnsureConnected(const rclcpp::Time& stamp) {
      if (port_.IsOpen()) {
        return true;
      }
      if (have_reconnect_time_ && (stamp - last_reconnect_attempt_).seconds() < reconnect_period_) {
        return false;
      }
      last_reconnect_attempt_ = stamp;
      have_reconnect_time_ = true;

      if (!port_.Open(device_, baud_)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", port_.LastError().c_str());
        return false;
      }

      // Opening the port asserts DTR, which resets an Arduino Uno into its
      // bootloader.  Anything written during that window is lost, and the
      // reset also drops any gains sent before it.
      port_open_time_ = stamp;
      link_established_ = false;
      link_lost_logged_ = false;
      have_gains_tx_ = false;
      RCLCPP_INFO(get_logger(), "opened %s, waiting %.1f s for the Arduino to boot", device_.c_str(), boot_delay_);
      return true;
    }

    void HandlePortFailure(const std::string& context) {
      RCLCPP_ERROR(get_logger(), "%s: %s, reconnecting", context.c_str(), port_.LastError().c_str());
      port_.Close();
      link_established_ = false;
    }

    void ServiceReceive(const rclcpp::Time& stamp) {
      lines_.clear();
      if (!port_.ReadLines(lines_)) {
        HandlePortFailure("serial read");
        return;
      }

      for (const std::string& line : lines_) {
        TraceReceivedLine(line);
        // Debug ("D,") and speed controller trace ("T,") lines are for the rx
        // trace only.
        if (line.rfind("D,", 0) == 0 || line.rfind("T,", 0) == 0) {
          continue;
        }
        ArduinoToJetson decoded;
        if (!Deserialize(line, decoded)) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "dropping malformed status frame '%s'", line.c_str());
          continue;
        }
        if (!link_established_ || decoded.mode != status_.mode) {
          RCLCPP_INFO(get_logger(),
                      "Arduino mode %s -> %s",
                      link_established_ ? ModeName(status_.mode) : "(none)",
                      ModeName(decoded.mode));
        }
        status_ = decoded;
        last_status_time_ = stamp;
        link_established_ = true;
        link_lost_logged_ = false;
      }
    }

    void SendCommand(const rclcpp::Time& stamp, bool link_ok) {
      JetsonToArduino frame;  // defaults to centered, zero speed, not ready

      if ((stamp - port_open_time_).seconds() < boot_delay_) {
        return;  // Arduino is still in its bootloader
      }

      const bool command_fresh = HaveCommand() && (stamp - last_command_time_).seconds() < command_timeout_;
      const bool estopped = link_ok && (status_.estop || status_.mode == Mode::kEstop);
      const bool auto_active = link_ok && status_.mode == Mode::kAutoActive;

      double steering = 0.0;
      double speed = 0.0;
      bool passthrough = false;

      if (command_fresh && !estopped) {
        frame.auto_ready = last_command_.auto_ready;
        // Hold center and zero until the Arduino has actually entered
        // AUTO_ACTIVE.  It will not make that transition unless steering sits
        // in the neutral deadband with a zero speed target while AUTO_ARMED,
        // so passing commands through early would deadlock the handshake.
        if (auto_active || !require_auto_active_) {
          passthrough = true;
          steering = std::clamp(static_cast<double>(last_command_.steering), -1.0, 1.0) * max_steering_;
          speed = static_cast<double>(last_command_.velocity);
          speed = std::isfinite(speed) ? std::clamp(speed, -max_speed_, max_speed_) : 0.0;
        }
      } else if (command_fresh && estopped) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "E-Stop asserted, holding zero speed");
      } else if (HaveCommand()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000, "drive command stale (> %.2f s), holding zero speed", command_timeout_);
      }
      if (!std::isfinite(steering)) {
        steering = 0.0;
      }

      speed = ApplySpeedSlew(stamp, speed, passthrough);

      if (invert_steering_) {
        steering = -steering;
      }
      if (invert_speed_) {
        speed = -speed;
      }

      // DriveCommand uses +1 = left (REP-103); the wire uses 255 = right.
      frame.steering = NormalizedToCommand(-steering);
      frame.target_rpm = ToTargetRpm(SpeedToSpurRpm(speed, spur_to_wheel_ratio_, tire_diameter_));

      WriteWire(Serialize(frame));
    }

    /// Send the gains until the Arduino echoes their sequence number.  An
    /// Arduino reset reverts it to its compiled-in defaults (sequence 0), which
    /// shows up here as a mismatch and gets the set resent.
    void ServiceGains(const rclcpp::Time& stamp, bool link_ok) {
      if (!port_.IsOpen() || (stamp - port_open_time_).seconds() < boot_delay_) {
        return;
      }
      if (link_ok && status_.gains_seq == gains_seq_) {
        if (!gains_applied_logged_) {
          RCLCPP_INFO(get_logger(), "Arduino applied speed gains (seq %u)", static_cast<unsigned>(gains_seq_));
          gains_applied_logged_ = true;
        }
        return;
      }
      gains_applied_logged_ = false;
      if (have_gains_tx_ && (stamp - last_gains_tx_).seconds() < gains_resend_period_) {
        return;
      }
      if (WriteWire(Serialize(ToFirmwareGains(gains_)))) {
        last_gains_tx_ = stamp;
        have_gains_tx_ = true;
      }
    }

    /// Runtime-settable parameters.  The speed gains are here so the loop can be
    /// retuned on the ground without a restart; the drivetrain constants and the
    /// slew rate are here because a characterization run sweeps them between
    /// steps, and a restart between steps would cost the Arduino handshake and
    /// the run directory.  Everything else is launch-time only on purpose: the
    /// device, the safety clamps and the arming policy should not move under a
    /// car that is already armed.
    rcl_interfaces::msg::SetParametersResult OnSetParameters(const std::vector<rclcpp::Parameter>& parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;

      SpeedGains candidate = gains_;
      double candidate_ratio = spur_to_wheel_ratio_;
      double candidate_diameter = tire_diameter_;
      double candidate_slew = speed_slew_rate_;
      bool gains_touched = false;
      bool drivetrain_touched = false;
      bool slew_touched = false;

      for (const rclcpp::Parameter& parameter : parameters) {
        const std::string& name = parameter.get_name();
        if (name == "speed_trace") {
          candidate.trace = parameter.as_bool();
          gains_touched = true;
        } else if (name == "spur_to_wheel_ratio") {
          candidate_ratio = parameter.as_double();
          drivetrain_touched = true;
        } else if (name == "tire_diameter") {
          candidate_diameter = parameter.as_double();
          drivetrain_touched = true;
        } else if (name == "speed_slew_rate") {
          candidate_slew = parameter.as_double();
          slew_touched = true;
        } else if (double* field = GainField(candidate, name)) {
          *field = parameter.as_double();
          gains_touched = true;
        }
      }
      if (!gains_touched && !drivetrain_touched && !slew_touched) {
        return result;
      }

      if (drivetrain_touched && !(candidate_ratio > 0.0 && candidate_diameter > 0.0)) {
        result.successful = false;
        result.reason = "spur_to_wheel_ratio and tire_diameter must both be positive";
        return result;
      }
      if (slew_touched && !(candidate_slew >= 0.0 && std::isfinite(candidate_slew))) {
        result.successful = false;
        result.reason = "speed_slew_rate must be finite and non-negative (0 disables rate limiting)";
        return result;
      }
      // The firmware works per 1000 spur RPM, so the drivetrain constants are
      // part of the gain conversion: changing either has to be validated
      // against the gains it will be sent with.
      if (Serialize(ToFirmwareGains(candidate, candidate_ratio, candidate_diameter)).empty()) {
        result.successful = false;
        result.reason = "out of range for the Arduino: limits must be non-negative, speed_output_limit at most 500, "
                        "speed_brake_limit at most 440, and every converted value within +/-9999.999";
        return result;
      }

      if (slew_touched) {
        speed_slew_rate_ = candidate_slew;
        RCLCPP_INFO(get_logger(), "speed_slew_rate updated to %.2f m/s per second", speed_slew_rate_);
      }
      if (drivetrain_touched) {
        spur_to_wheel_ratio_ = candidate_ratio;
        tire_diameter_ = candidate_diameter;
        RCLCPP_INFO(get_logger(),
                    "drivetrain updated: spur_to_wheel_ratio=%.4f tire_diameter=%.4f m (1 m/s = %.0f spur RPM)",
                    spur_to_wheel_ratio_,
                    tire_diameter_,
                    SpeedToSpurRpm(1.0, spur_to_wheel_ratio_, tire_diameter_));
      }
      if (gains_touched || drivetrain_touched) {
        // A drivetrain change rescales every rate gain on the wire even when the
        // per-m/s values did not move, so it needs a resend just as much.
        gains_ = candidate;
        gains_seq_ = static_cast<uint8_t>((gains_seq_ % 255) + 1);  // 0 is reserved for firmware defaults
        have_gains_tx_ = false;                                     // send on the next cycle
        LogGains("speed gains updated");
      }
      return result;
    }

    static double* GainField(SpeedGains& gains, const std::string& name) {
      if (name == "speed_ks") {
        return &gains.ks;
      }
      if (name == "speed_kv") {
        return &gains.kv;
      }
      if (name == "speed_kp") {
        return &gains.kp;
      }
      if (name == "speed_ki") {
        return &gains.ki;
      }
      if (name == "speed_kd") {
        return &gains.kd;
      }
      if (name == "speed_i_limit") {
        return &gains.i_limit;
      }
      if (name == "speed_output_limit") {
        return &gains.output_limit;
      }
      if (name == "speed_brake_limit") {
        return &gains.brake_limit;
      }
      return nullptr;
    }

    /// Parameters are per m/s; the firmware works per 1000 spur RPM.
    SpeedGains ToFirmwareGains(const SpeedGains& per_mps, double spur_to_wheel_ratio, double tire_diameter) const {
      SpeedGains firmware = GainsPerMpsToPerKrpm(per_mps, SpeedToSpurRpm(1.0, spur_to_wheel_ratio, tire_diameter));
      firmware.seq = gains_seq_;
      return firmware;
    }

    SpeedGains ToFirmwareGains(const SpeedGains& per_mps) const {
      return ToFirmwareGains(per_mps, spur_to_wheel_ratio_, tire_diameter_);
    }

    void LogGains(const char* prefix) const {
      RCLCPP_INFO(get_logger(),
                  "%s (seq %u): ks=%.2f kv=%.3f kp=%.3f ki=%.3f kd=%.3f i_limit=%.1f output_limit=%.1f "
                  "brake_limit=%.1f trace=%s",
                  prefix,
                  static_cast<unsigned>(gains_seq_),
                  gains_.ks,
                  gains_.kv,
                  gains_.kp,
                  gains_.ki,
                  gains_.kd,
                  gains_.i_limit,
                  gains_.output_limit,
                  gains_.brake_limit,
                  gains_.trace ? "on" : "off");
    }

    /// Host timestamp for a trace line, or empty when trace_timestamps is off.
    ///
    /// The Arduino stamps its own lines with millis() since ITS boot, which
    /// cannot be aligned to a rosbag or to telemetry.csv.  This prefix is what
    /// makes the serial traces joinable with everything else recorded in a run.
    std::string TracePrefix() const {
      if (!trace_timestamps_) {
        return std::string();
      }
      char buffer[32];
      const int length = std::snprintf(buffer, sizeof(buffer), "%.6f ", now().seconds());
      return (length > 0 && length < static_cast<int>(sizeof(buffer))) ? std::string(buffer, length) : std::string();
    }

    bool WriteWire(const std::string& wire) {
      if (wire.empty()) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "refusing to send an unencodable frame");
        return false;
      }
      if (!port_.Write(wire)) {
        HandlePortFailure("serial write");
        return false;
      }
      if (tx_trace_.is_open()) {
        // Without trace_timestamps the trace is byte-for-byte identical to the
        // successfully completed USB write above.  With it, each line gains a
        // host timestamp and one space in front; everything after that space is
        // still the exact frame.
        const std::string prefix = TracePrefix();
        tx_trace_.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        tx_trace_.write(wire.data(), static_cast<std::streamsize>(wire.size()));
        tx_trace_.flush();
        if (!tx_trace_) {
          RCLCPP_ERROR(get_logger(), "failed writing tx trace %s; disabling it", tx_trace_path_.c_str());
          tx_trace_.close();
        }
      }
      return true;
    }

    void TraceReceivedLine(const std::string& line) {
      if (!rx_trace_.is_open()) {
        return;
      }
      const std::string prefix = TracePrefix();
      rx_trace_.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
      rx_trace_.write(line.data(), static_cast<std::streamsize>(line.size()));
      rx_trace_.put('\n');
      rx_trace_.flush();
      if (!rx_trace_) {
        RCLCPP_ERROR(get_logger(), "failed writing rx trace %s; disabling it", rx_trace_path_.c_str());
        rx_trace_.close();
      }
    }

    /// Rate limit the speed target in m/s per second, in both directions.  Any
    /// fall back to zero (stale command, lost link, E-Stop) snaps immediately
    /// rather than ramping; the Arduino's controller then coasts or brakes.
    double ApplySpeedSlew(const rclcpp::Time& stamp, double target, bool passthrough) {
      // A missed cycle or a clock jump must not turn into a huge step, so fall
      // back to the nominal period whenever dt looks implausible.
      double dt = have_slew_time_ ? (stamp - last_slew_time_).seconds() : 0.0;
      if (dt <= 0.0 || dt > 1.0) {
        dt = 1.0 / tx_rate_hz_;
      }
      last_slew_time_ = stamp;
      have_slew_time_ = true;

      if (!passthrough) {
        commanded_speed_ = 0.0;
      } else if (speed_slew_rate_ <= 0.0) {
        commanded_speed_ = target;
      } else {
        const double max_step = speed_slew_rate_ * dt;
        commanded_speed_ += std::clamp(target - commanded_speed_, -max_step, max_step);
      }
      return commanded_speed_;
    }

    void PublishStatus(const rclcpp::Time& stamp, bool link_ok) {
      cfr_interfaces::msg::ArduinoStatus msg;
      msg.header.stamp = stamp;
      msg.header.frame_id = "base_link";
      msg.link_ok = link_ok;
      msg.estop = link_ok ? status_.estop : false;
      msg.auto_arm = link_ok ? status_.auto_arm : false;
      msg.manual_start = link_ok ? status_.manual_start : false;
      msg.mode = link_ok ? static_cast<uint8_t>(status_.mode) : static_cast<uint8_t>(Mode::kEstop);
      msg.battery_level = link_ok ? status_.battery_level : 0;
      // Zero on a dead link rather than the last known value: a stale speed is
      // worse than no speed for anything closing a loop on it.  The firmware
      // caps both RPM fields at kMaxTargetRpm, so negating cannot overflow.
      const int sign = invert_speed_ ? -1 : 1;
      msg.rpm = link_ok ? static_cast<int16_t>(sign * status_.rpm) : 0;
      const double wheel_rpm = SpurRpmToWheelRpm(msg.rpm, spur_to_wheel_ratio_);
      msg.wheel_rpm = static_cast<float>(wheel_rpm);
      msg.speed = static_cast<float>(WheelRpmToSpeed(wheel_rpm, tire_diameter_));
      msg.target_speed =
          link_ok ?
              static_cast<float>(sign * SpurRpmToSpeed(status_.target_rpm, spur_to_wheel_ratio_, tire_diameter_)) :
              0.0F;
      msg.throttle_us = link_ok ? status_.throttle_us : 1500;
      msg.gains_applied = link_ok && status_.gains_seq == gains_seq_;
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
    std::string tx_trace_path_;
    std::string rx_trace_path_;
    bool trace_timestamps_ = false;
    double max_speed_ = 2.0;
    double max_steering_ = 1.0;
    double speed_slew_rate_ = 2.0;
    bool invert_steering_ = false;
    bool invert_speed_ = false;
    bool require_auto_active_ = true;
    double spur_to_wheel_ratio_ = kSpurToWheelRatio;
    double tire_diameter_ = kTireDiameterM;
    double gains_resend_period_ = 1.0;
    SpeedGains gains_;  // per m/s, as set on the parameters

    // State
    SerialPort port_;
    std::ofstream tx_trace_;
    std::ofstream rx_trace_;
    std::vector<std::string> lines_;
    ArduinoToJetson status_;
    cfr_interfaces::msg::DriveCommand last_command_;
    rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_reconnect_attempt_{0, 0, RCL_ROS_TIME};
    rclcpp::Time port_open_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_slew_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_gains_tx_{0, 0, RCL_ROS_TIME};
    bool have_reconnect_time_ = false;
    bool have_slew_time_ = false;
    bool have_gains_tx_ = false;
    bool gains_applied_logged_ = false;
    bool link_established_ = false;
    bool link_lost_logged_ = false;
    double commanded_speed_ = 0.0;
    uint8_t gains_seq_ = 1;

    rclcpp::Publisher<cfr_interfaces::msg::ArduinoStatus>::SharedPtr status_publisher_;
    rclcpp::Subscription<cfr_interfaces::msg::DriveCommand>::SharedPtr command_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  };

}  // namespace cfr_arduino_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cfr_arduino_bridge::ArduinoBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
