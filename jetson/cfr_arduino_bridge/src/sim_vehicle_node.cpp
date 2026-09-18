// Replaces the Arduino bridge for Gazebo runs. It accepts the same normalized
// command and emits a Twist for Gazebo plus the status heartbeat expected by
// monitoring tools. Gazebo remains source of odometry and collision physics.
//
// This node stands in for the WHOLE actuation chain -- the bridge's speed
// controller, the Arduino, the ESC and the plant -- so what it has to reproduce
// is the car's closed-loop response to a drive_cmd, not an idealisation of it.
// Until the Session B/C runs it reproduced none of it: a nonzero target was
// applied to the simulated speed instantly, in one tick.
//
// Four measured facts now shape the model (docs/characterization-results.md):
//
//   1. THE CAR HAS NO BRAKES.  Every non-zero brake limit tested (40/80/120 us)
//      clamped to the same ~1436 us pulse, and every one stopped the car SLOWER
//      than simply coasting -- the ESC reads that reverse-side pulse as reverse
//      drive.  So deceleration toward a lower target is limited by coast drag,
//      exactly as acceleration is limited by thrust.  On the car a 3.2 -> 0.8
//      m/s command takes ~2.6 s; the old model did it in one tick.
//   2. COAST DRAG IS 7-10x WHAT WAS ASSUMED.  decel = 0.606 + 0.130 v m/s^2,
//      against the 0.1 forward / 0.3 reverse pair this node used to carry.  That
//      pair was a command-side fudge for a Gazebo contact asymmetry; the real
//      car is symmetric to 3.3%, so the fudge is gone and one curve replaces it.
//   3. THE CAR UNDERSTEERS AND ITS STEERING IS ASYMMETRIC.  Left and right
//      differ by ~34% at matched command, and the achieved radius grows ~14%
//      between 0.7 and 2.9 m/s.  A single symmetric max_steering_angle and a
//      pure kinematic yaw rate cannot express either.
//   4. THE TACHOMETER IS COARSE AND GOES BLIND AT LOW SPEED.  It used to report
//      a hard-coded rpm = 0 and throttle_us = 1500 no matter what the car was
//      doing, so every sim run logged an empty speed channel while the car
//      logged a thousand distinct values.  TachModel below is a transcription
//      of the firmware's estimator, not an approximation of it.
//
// Everything here is a parameter, defaulted from config/vehicle.yaml through
// config/arduino_bridge.yaml, so a re-run that moves a number moves the twin.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <stdexcept>
#include <vector>

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
      neutral_speed_deadband_ = declare_parameter<double>("neutral_speed_deadband", 0.05);

      // Coastdown fit, m dv/dt = -(f0 + f1 v + f2 v^2).  f2 is zero by
      // measurement, not by omission: aero is ~0.08 N at 3 m/s on a 3.6 kg car.
      vehicle_mass_ = declare_parameter<double>("vehicle_mass", 3.599);
      coast_f0_ = declare_parameter<double>("coast_f0", 2.18);
      coast_f1_ = declare_parameter<double>("coast_f1", 0.47);
      coast_f2_ = declare_parameter<double>("coast_f2", 0.0);

      // p90 of achieved acceleration below 2.5 m/s across the whole campaign.
      max_acceleration_ = declare_parameter<double>("max_acceleration", 3.0);

      // Zero means "coast only", which is what the car does today.  It is a
      // parameter rather than a constant so that a firmware fix to the brake
      // path can be modelled here before it is trusted on the car.
      brake_deceleration_ = declare_parameter<double>("brake_deceleration", 0.0);

      // Jetson command to first movement.  Measured cleanly off the yaw step
      // response, where the onset is unambiguous.
      command_dead_time_ = declare_parameter<double>("command_dead_time", 0.19);

      understeer_gradient_ = declare_parameter<double>("understeer_gradient", 0.007);

      // Drivetrain, for the simulated tachometer.  These mirror the real
      // bridge's parameters of the same name so the sim reports spur RPM on
      // exactly the scale the controller and the analysis expect.
      tire_diameter_ = declare_parameter<double>("tire_diameter", 0.1132);
      spur_to_wheel_ratio_ = declare_parameter<double>("spur_to_wheel_ratio", 2.85);
      tach_window_ = declare_parameter<double>("tach_window", 0.1);
      tach_stall_timeout_ = declare_parameter<double>("tach_stall_timeout", 0.4);
      tach_pulses_per_rev_ = declare_parameter<int>("tach_pulses_per_rev", 1);

      // Open-loop pulse the throttle trace reports.  The real firmware arrives
      // here through a PID loop; in steady state that loop lands on the
      // feedforward, which is what was measured, so the sim reports the
      // feedforward directly rather than pretending to a closed loop it
      // does not run.
      throttle_neutral_us_ = declare_parameter<double>("throttle_neutral_us", 1504.0);
      throttle_ks_us_ = declare_parameter<double>("throttle_ks_us", 28.8);
      throttle_kv_us_ = declare_parameter<double>("throttle_kv_us", 11.13);

      // Effective steering angle against normalized command, interpolated.
      // Empty falls back to the symmetric max_steering_angle, which is what
      // this node did before the skidpad runs.
      steering_commands_ = declare_parameter<std::vector<double>>("steering_command_points", std::vector<double>{});
      steering_angles_ = declare_parameter<std::vector<double>>("steering_angle_points", std::vector<double>{});

      // Declared so characterization profiles built for the real bridge's
      // speed_* gains (and speed_trace) can set them here without the
      // parameter service rejecting an undeclared name and aborting the run.
      // This model still has no PID loop to tune -- the gains are ignored, on
      // purpose -- but it is no longer an ideal plant: the response envelope
      // above is the car's, so a dry run now exercises the profile's timing
      // against something with roughly the right dynamics.
      declare_parameter<bool>("speed_trace", false);
      for (const char* name : {"speed_ks",
                               "speed_kv",
                               "speed_kp",
                               "speed_ki",
                               "speed_kd",
                               "speed_i_limit",
                               "speed_output_limit",
                               "speed_brake_limit"}) {
        declare_parameter<double>(name, 0.0);
      }

      if (wheelbase_ <= 0.0 || max_speed_ <= 0.0 || max_steering_angle_ <= 0.0 || neutral_speed_deadband_ < 0.0 ||
          vehicle_mass_ <= 0.0 || max_acceleration_ <= 0.0 || brake_deceleration_ < 0.0 || command_dead_time_ < 0.0 ||
          understeer_gradient_ < 0.0 || coast_f0_ < 0.0 || coast_f1_ < 0.0 || coast_f2_ < 0.0) {
        throw std::invalid_argument("vehicle dimensions and limits must be non-negative, with positive dimensions");
      }
      if (tire_diameter_ <= 0.0 || spur_to_wheel_ratio_ <= 0.0 || tach_window_ <= 0.0 || tach_stall_timeout_ <= 0.0 ||
          tach_pulses_per_rev_ <= 0) {
        throw std::invalid_argument("drivetrain and tachometer parameters must be positive");
      }
      if (steering_commands_.size() != steering_angles_.size()) {
        throw std::invalid_argument("steering_command_points and steering_angle_points must be the same length");
      }
      for (size_t i = 1; i < steering_commands_.size(); ++i) {
        if (steering_commands_[i] <= steering_commands_[i - 1]) {
          throw std::invalid_argument("steering_command_points must be strictly increasing");
        }
      }

      twist_publisher_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", rclcpp::QoS(10));
      status_publisher_ = create_publisher<cfr_interfaces::msg::ArduinoStatus>("~/status", rclcpp::QoS(10));
      command_subscription_ = create_subscription<cfr_interfaces::msg::DriveCommand>(
          "~/drive_cmd", rclcpp::SensorDataQoS(), [this](const cfr_interfaces::msg::DriveCommand::SharedPtr msg) {
            command_ = *msg;
            command_time_ = now();
            pending_.push_back({command_time_, *msg});
          });

      const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 1.0));
      timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this] { OnTimer(); });
    }

   private:
    struct Stamped {
      rclcpp::Time time;
      cfr_interfaces::msg::DriveCommand command;
    };

    void OnTimer() {
      const rclcpp::Time stamp = now();
      const bool fresh = command_time_.nanoseconds() != 0 && (stamp - command_time_).seconds() < command_timeout_;
      const bool active = fresh && command_.auto_ready;
      const double elapsed =
          last_publish_time_.nanoseconds() == 0 ? 0.0 : std::max(0.0, (stamp - last_publish_time_).seconds());
      last_publish_time_ = stamp;

      // The command the actuators are acting on now is the one issued
      // command_dead_time_ ago, not the one that just arrived.
      const cfr_interfaces::msg::DriveCommand effective = EffectiveCommand(stamp);

      geometry_msgs::msg::Twist twist;
      if (active) {
        double target = static_cast<double>(effective.velocity);
        if (!std::isfinite(target) || std::abs(target) <= neutral_speed_deadband_) {
          target = 0.0;
        }
        ApproachTarget(std::clamp(target, -max_speed_, max_speed_), elapsed);
        AdvanceTachometer(elapsed);

        const double steering = EffectiveSteeringAngle(static_cast<double>(effective.steering));
        twist.linear.x = simulated_speed_;
        // Bicycle model with an understeer gradient: R = (L + K v^2) / delta,
        // so the achieved radius grows with speed instead of staying kinematic.
        // K = 0 collapses this back to the textbook form.
        const double effective_wheelbase = wheelbase_ + understeer_gradient_ * simulated_speed_ * simulated_speed_;
        twist.angular.z = simulated_speed_ * std::tan(steering) / effective_wheelbase;
      } else {
        ApproachTarget(0.0, elapsed);
        AdvanceTachometer(elapsed);
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
      // Direction-blind magnitude from the tach, signed by the commanded
      // direction, exactly as the firmware does it -- one magnet in the spur
      // gear cannot tell forward from reverse.
      const double magnitude = tach_.Rpm();
      const double signed_rpm = std::copysign(magnitude, simulated_speed_);
      status.rpm = static_cast<int16_t>(std::lround(signed_rpm));
      // Derived exactly as arduino_bridge_node derives them, so a sim run's
      // speed column means the same thing as the car's -- including inheriting
      // the tachometer's blind spots rather than leaking the true speed.
      status.wheel_rpm = static_cast<float>(signed_rpm / spur_to_wheel_ratio_);
      status.speed = static_cast<float>(status.wheel_rpm / 60.0 * M_PI * tire_diameter_);
      status.target_speed = static_cast<float>(target_speed_);
      status.throttle_us = static_cast<uint16_t>(std::lround(ThrottleMicroseconds()));
      status.gains_applied = true;
      status_publisher_->publish(status);
    }

    // Transcription of TachSensor in src/arduino_rcm/arduino_rcm.ino.  It times
    // whole revolutions instead of counting edges in a gate, and the two
    // consequences that matter for the twin are both emergent, not coded in:
    //
    //   * it needs TWO timestamps before it can report anything, so the first
    //     revolution after any stop reads zero, and
    //   * it forgets its history after tach_stall_timeout, which puts a hard
    //     floor at 60 / 0.4 = 150 spur RPM, about 0.3 m/s.
    //
    // That floor is why the steer_authority run is unusable: at its crawl the
    // tach sat right on the boundary and read exactly zero in ~50% of samples.
    // A simulator that reports a clean speed at 0.2 m/s does not reproduce the
    // single most confusing property of the real telemetry.
    class TachModel {
     public:
      void Configure(double window, double stall, int pulses_per_rev) {
        window_ = window;
        stall_ = stall;
        pulses_per_rev_ = pulses_per_rev;
      }

      void Reset() {
        held_ = 0;
        newest_ = 0;
        rpm_ = 0.0;
      }

      void Record(double when) { RecordStamp(when); }

      void Tick(double now) { Update(now); }

      double Rpm() const { return rpm_; }

     private:
      static constexpr int kHistory = 8;

      void RecordStamp(double when) {
        newest_ = (newest_ + 1) % kHistory;
        stamps_[newest_] = when;
        if (held_ < kHistory) {
          ++held_;
        }
      }

      void Update(double now) {
        const double since = now - stamps_[newest_];
        if (held_ > 0 && since >= stall_) {
          held_ = 0;
        }
        if (held_ < 2) {
          rpm_ = 0.0;
          return;
        }
        int spanned = 1;
        while (spanned + 1 < held_ &&
               (stamps_[newest_] - stamps_[((newest_ - spanned - 1) % kHistory + kHistory) % kHistory]) <= window_) {
          ++spanned;
        }
        double span = stamps_[newest_] - stamps_[((newest_ - spanned) % kHistory + kHistory) % kHistory];
        // A revolution still in progress that has already outlasted the
        // measured period bounds the speed from above, so an abrupt stop does
        // not hold its last reading until the stall timeout.
        if (span > 0.0 && since > span / spanned) {
          span = since;
          spanned = 1;
        }
        rpm_ = span > 0.0 ? (60.0 * spanned) / (span * pulses_per_rev_) : 0.0;
      }

      double stamps_[kHistory]{};
      int held_ = 0;
      int newest_ = 0;
      int pulses_per_rev_ = 1;
      double window_ = 0.1;
      double stall_ = 0.4;
      double rpm_ = 0.0;
    };

    // Turn the simulated ground speed into magnet passes at the spur gear.
    //
    // Each whole revolution is timestamped at the instant the spur actually
    // crossed that boundary, interpolated within the tick.  Timing the
    // crossings rather than the tick is what makes the quantisation come out
    // right: the firmware measures periods between magnet passes, so a
    // timestamp placed at the wrong point in the tick biases every RPM read
    // that spans it.
    void AdvanceTachometer(double elapsed) {
      if (elapsed <= 0.0) {
        return;
      }
      tach_.Configure(tach_window_, tach_stall_timeout_, tach_pulses_per_rev_);
      const double seconds = now().seconds();
      const double circumference = M_PI * tire_diameter_;
      const double travelled = std::abs(simulated_speed_) * elapsed / circumference * spur_to_wheel_ratio_;
      if (travelled > 0.0) {
        const double reached = spur_fraction_ + travelled;
        for (double boundary = 1.0; boundary <= reached; boundary += 1.0) {
          const double fraction = (boundary - spur_fraction_) / travelled;
          tach_.Record(seconds - elapsed + fraction * elapsed);
        }
        spur_fraction_ = reached - std::floor(reached);
      }
      tach_.Tick(seconds);
    }

    // Steady-state feedforward: neutral, plus the measured deadband once the
    // car is asked to move, plus kV per m/s.
    double ThrottleMicroseconds() const {
      if (std::abs(simulated_speed_) < 1e-3) {
        return throttle_neutral_us_;
      }
      const double magnitude = throttle_ks_us_ + throttle_kv_us_ * std::abs(simulated_speed_);
      return throttle_neutral_us_ + std::copysign(magnitude, simulated_speed_);
    }

    // Newest command at least command_dead_time_ old, holding the last one it
    // returned once the queue runs dry so a stall does not look like a zero.
    cfr_interfaces::msg::DriveCommand EffectiveCommand(const rclcpp::Time& stamp) {
      while (!pending_.empty() && (stamp - pending_.front().time).seconds() >= command_dead_time_) {
        delayed_ = pending_.front().command;
        have_delayed_ = true;
        pending_.pop_front();
      }
      if (!have_delayed_) {
        cfr_interfaces::msg::DriveCommand idle;
        idle.velocity = 0.0F;
        idle.steering = 0.0F;
        idle.auto_ready = command_.auto_ready;
        return idle;
      }
      return delayed_;
    }

    // Coast drag in m/s^2 at a given speed magnitude.
    double CoastDeceleration(double speed) const {
      return (coast_f0_ + coast_f1_ * speed + coast_f2_ * speed * speed) / vehicle_mass_;
    }

    // Move the simulated speed toward `target` under an asymmetric limit:
    // thrust when speeding up, coast drag (plus whatever braking exists) when
    // slowing down.  The asymmetry is the whole point -- a car with no brakes
    // reaches a lower speed far more slowly than it reaches a higher one.
    void ApproachTarget(double target, double elapsed) {
      if (elapsed <= 0.0) {
        return;
      }
      target_speed_ = target;
      const double delta = target - simulated_speed_;
      const bool speeding_up = std::abs(target) > std::abs(simulated_speed_) && target * simulated_speed_ >= 0.0;
      const double limit =
          speeding_up ? max_acceleration_ : CoastDeceleration(std::abs(simulated_speed_)) + brake_deceleration_;
      const double step = std::max(limit, 0.0) * elapsed;
      simulated_speed_ += std::clamp(delta, -step, step);
      if (std::abs(simulated_speed_) < 1e-4) {
        simulated_speed_ = 0.0;
      }
    }

    // Piecewise-linear interpolation of the measured command -> angle table,
    // which is asymmetric.  Falls back to the old symmetric scaling when no
    // table is configured.
    double EffectiveSteeringAngle(double command) const {
      const double clamped = std::clamp(std::isfinite(command) ? command : 0.0, -1.0, 1.0);
      if (steering_commands_.size() < 2) {
        return clamped * max_steering_angle_;
      }
      if (clamped <= steering_commands_.front()) {
        return steering_angles_.front();
      }
      if (clamped >= steering_commands_.back()) {
        return steering_angles_.back();
      }
      for (size_t i = 1; i < steering_commands_.size(); ++i) {
        if (clamped <= steering_commands_[i]) {
          const double span = steering_commands_[i] - steering_commands_[i - 1];
          const double fraction = (clamped - steering_commands_[i - 1]) / span;
          return steering_angles_[i - 1] + fraction * (steering_angles_[i] - steering_angles_[i - 1]);
        }
      }
      return steering_angles_.back();
    }

    double wheelbase_ = 0.324;
    double max_speed_ = 8.0;
    double max_steering_angle_ = 0.40;
    double command_timeout_ = 0.3;
    double publish_rate_hz_ = 50.0;
    double neutral_speed_deadband_ = 0.05;
    double vehicle_mass_ = 3.599;
    double coast_f0_ = 2.18;
    double coast_f1_ = 0.47;
    double coast_f2_ = 0.0;
    double max_acceleration_ = 3.0;
    double brake_deceleration_ = 0.0;
    double command_dead_time_ = 0.19;
    double understeer_gradient_ = 0.007;
    double tire_diameter_ = 0.1132;
    double spur_to_wheel_ratio_ = 2.85;
    double tach_window_ = 0.1;
    double tach_stall_timeout_ = 0.4;
    int tach_pulses_per_rev_ = 1;
    double throttle_neutral_us_ = 1504.0;
    double throttle_ks_us_ = 28.8;
    double throttle_kv_us_ = 11.13;
    double spur_fraction_ = 0.0;
    double target_speed_ = 0.0;
    TachModel tach_;
    std::vector<double> steering_commands_;
    std::vector<double> steering_angles_;
    double simulated_speed_ = 0.0;
    cfr_interfaces::msg::DriveCommand command_;
    cfr_interfaces::msg::DriveCommand delayed_;
    bool have_delayed_ = false;
    std::deque<Stamped> pending_;
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
