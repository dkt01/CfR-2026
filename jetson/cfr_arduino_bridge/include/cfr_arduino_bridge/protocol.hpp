// Wire protocol for the onboard Jetson <-> Arduino USB serial link.
//
// Both directions are ASCII, comma separated, newline terminated.  See the
// "Onboard Protocol" section of the repository README for the field tables.
#pragma once

#include <cstdint>
#include <string>

namespace cfr_arduino_bridge
{

  /// Run mode reported by the Arduino.  Values match the on-wire enum.
  enum class Mode : uint8_t
  {
    kEstop = 0,
    kRcArmed = 1,
    kRcActive = 2,
    kAutoArmed = 3,
    kAutoActive = 4,
  };

  /// Neutral value for both axis commands.
  constexpr uint8_t kNeutralCommand = 127;

  /// Deadband the Arduino uses to decide an axis is centered (IsNearlyCenter).
  constexpr uint8_t kCenterTolerance = 5;

  /// Jetson -> Arduino command frame.
  struct JetsonToArduino
  {
    bool auto_ready = false;
    uint8_t steering = kNeutralCommand; ///< 0 full left, 255 full right
    uint8_t throttle = kNeutralCommand; ///< 0 full reverse, 255 full forward
  };

  /// Arduino -> Jetson status frame.
  struct ArduinoToJetson
  {
    bool estop = false;
    bool auto_arm = false;
    bool manual_start = false;
    Mode mode = Mode::kEstop;
    uint8_t battery_level = 0;
  };

  /// Encode a command frame, including the trailing newline.
  ///
  /// Integer fields are always emitted zero padded to three digits, which makes
  /// every frame exactly "b,nnn,nnn\n" (10 bytes).  The Arduino rejects payloads
  /// shorter than 6 characters, so an unpadded frame such as "1,0,0" would be
  /// silently dropped; the fixed width sidesteps that entirely.
  std::string Serialize(const JetsonToArduino &command);

  /// Decode one status frame.  @p line must not contain the trailing newline.
  /// Returns false and leaves @p status untouched if the frame is malformed.
  bool Deserialize(const std::string &line, ArduinoToJetson &status);

  /// Map a normalized [-1, 1] axis to the on-wire [0, 255] range, where 0.0
  /// lands exactly on kNeutralCommand.  Out of range inputs and NaN are clamped.
  uint8_t NormalizedToCommand(double value);

  /// Mirror of the Arduino's IsNearlyCenter().
  bool IsNearlyCenter(uint8_t value, uint8_t center = kNeutralCommand, uint8_t tolerance = kCenterTolerance);

  /// Human readable mode name for logging.
  const char *ModeName(Mode mode);

} // namespace cfr_arduino_bridge
