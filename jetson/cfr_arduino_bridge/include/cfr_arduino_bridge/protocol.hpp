// Wire protocol for the onboard Jetson <-> Arduino USB serial link.
//
// Both directions are ASCII, comma separated, newline terminated.  See the
// "Onboard Protocol" section of the repository README for the field tables.
#pragma once

#include <cstdint>
#include <string>

namespace cfr_arduino_bridge {

  /// Run mode reported by the Arduino.  Values match the on-wire enum.
  enum class Mode : uint8_t {
    kEstop = 0,
    kRcArmed = 1,
    kRcActive = 2,
    kAutoArmed = 3,
    kAutoActive = 4,
  };

  /// Neutral value for the steering command.
  constexpr uint8_t kNeutralCommand = 127;

  /// Deadband the Arduino uses to decide steering is centered (IsNearlyCenter).
  constexpr uint8_t kCenterTolerance = 5;

  /// Largest target spur RPM the firmware accepts, in either direction.
  constexpr int16_t kMaxTargetRpm = 20000;

  /// Largest drive and brake limits the firmware accepts, in microseconds.
  /// Both keep the throttle pulse inside [1000, 2000] us.
  constexpr double kMaxOutputLimitUs = 500.0;
  constexpr double kMaxBrakeLimitUs = 440.0;

  /// Jetson -> Arduino drive command frame.
  struct JetsonToArduino {
    bool auto_ready = false;
    uint8_t steering = kNeutralCommand;  ///< 0 full left, 255 full right
    /// Signed spur gear RPM for the Arduino's speed controller, positive
    /// forward.  Zero stops.
    int16_t target_rpm = 0;
  };

  /// Arduino speed controller tuning, in the firmware's units: the output is
  /// a throttle pulse offset from neutral in microseconds, speed is spur RPM,
  /// and the rate gains are per 1000 RPM.  See SpeedGains in arduino_rcm.ino.
  struct SpeedGains {
    double ks = 0.0;            ///< us, added whenever the target is nonzero
    double kv = 0.0;            ///< us per 1000 RPM of target
    double kp = 0.0;            ///< us per 1000 RPM of error
    double ki = 0.0;            ///< us per 1000 RPM of error, per second
    double kd = 0.0;            ///< us per 1000 RPM per second of measured speed change
    double i_limit = 0.0;       ///< us, integrator clamp
    double output_limit = 0.0;  ///< us, largest drive offset
    double brake_limit = 0.0;   ///< us of braking past the ESC threshold; 0 coasts
    /// Echoed back in ArduinoToJetson::gains_seq once applied.  Zero is
    /// reserved for the firmware's compiled-in defaults.
    uint8_t seq = 0;
    bool trace = false;  ///< ask the firmware for a per-tick "T," trace line
  };

  /// Arduino -> Jetson status frame.
  struct ArduinoToJetson {
    bool estop = false;
    bool auto_arm = false;
    bool manual_start = false;
    Mode mode = Mode::kEstop;
    uint8_t battery_level = 0;
    /// Spur gear revolutions per minute, signed by the firmware's direction
    /// estimate.  The hall sensor cannot see direction, so the sign is the
    /// direction the controller last drove and is positive when unknown.  Zero
    /// also means stopped or no sensor connected.
    int16_t rpm = 0;
    /// Signed spur RPM the controller is tracking.  Zero while it stops the
    /// car ahead of a reversal, so it can lag the commanded target.
    int16_t target_rpm = 0;
    uint16_t throttle_us = 1500;  ///< requested throttle pulse, before dithering
    uint8_t gains_seq = 0;        ///< seq of the gains in use, 0 for firmware defaults
  };

  /// Encode a drive command frame, including the trailing newline.  Always
  /// exactly "C,b,sss,+rrrrr\n" (15 bytes): the firmware requires the exact
  /// length, which is how a byte dropped on the link becomes a rejected frame
  /// rather than a different command.  target_rpm is clamped to
  /// +/-kMaxTargetRpm.
  std::string Serialize(const JetsonToArduino& command);

  /// Encode a gains frame, including the trailing newline.  Returns an empty
  /// string if any value is not finite, does not fit the wire's +/-9999.999
  /// range, or is a limit outside what the firmware accepts.
  std::string Serialize(const SpeedGains& gains);

  /// Decode one status frame.  @p line must not contain the trailing newline.
  /// Returns false and leaves @p status untouched if the frame is malformed.
  bool Deserialize(const std::string& line, ArduinoToJetson& status);

  /// Round a spur RPM to the wire's integer, clamped to +/-kMaxTargetRpm.  NaN
  /// becomes zero.
  int16_t ToTargetRpm(double spur_rpm);

  /// Convert gains whose rate terms are per m/s of ground speed into the
  /// firmware's per-1000-spur-RPM units.  ks and the limits are microseconds
  /// either way and pass through.  @p spur_rpm_per_mps must be positive.
  SpeedGains GainsPerMpsToPerKrpm(const SpeedGains& per_mps, double spur_rpm_per_mps);

  /// Map a normalized [-1, 1] axis to the on-wire [0, 255] range, where 0.0
  /// lands exactly on kNeutralCommand.  Out of range inputs and NaN are clamped.
  uint8_t NormalizedToCommand(double value);

  /// Mirror of the Arduino's IsNearlyCenter().
  bool IsNearlyCenter(uint8_t value, uint8_t center = kNeutralCommand, uint8_t tolerance = kCenterTolerance);

  /// Human readable mode name for logging.
  const char* ModeName(Mode mode);

}  // namespace cfr_arduino_bridge
