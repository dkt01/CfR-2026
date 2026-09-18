#include "cfr_arduino_bridge/protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cfr_arduino_bridge {

  namespace {

    /// Largest magnitude a gains field can carry: seven digits of thousandths.
    constexpr long long kMaxGainThousandths = 9999999;

    /// split on commas.  A trailing comma (which the Arduino always emits) does
    /// not produce an extra field.
    std::vector<std::string> SplitFields(const std::string& line) {
      std::vector<std::string> fields;
      std::string current;
      for (const char c : line) {
        if (c == ',') {
          fields.push_back(current);
          current.clear();
        } else if (c != '\r' && c != '\n') {
          current.push_back(c);
        }
      }
      if (!current.empty()) {
        fields.push_back(current);
      }
      return fields;
    }

    bool ParseBool(const std::string& field, bool& value) {
      if (field == "0") {
        value = false;
        return true;
      }
      if (field == "1") {
        value = true;
        return true;
      }
      return false;
    }

    /// Unsigned decimal of at most @p max_digits digits, no larger than @p max.
    bool ParseUnsigned(const std::string& field, size_t max_digits, uint32_t max, uint32_t& value) {
      if (field.empty() || field.size() > max_digits) {
        return false;
      }
      uint32_t accumulator = 0;
      for (const char c : field) {
        if (c < '0' || c > '9') {
          return false;
        }
        accumulator = (accumulator * 10) + static_cast<uint32_t>(c - '0');
      }
      if (accumulator > max) {
        return false;
      }
      value = accumulator;
      return true;
    }

    bool ParseUint8(const std::string& field, uint8_t& value) {
      uint32_t parsed = 0;
      if (!ParseUnsigned(field, 3, 255, parsed)) {
        return false;
      }
      value = static_cast<uint8_t>(parsed);
      return true;
    }

    bool ParseUint16(const std::string& field, uint16_t& value) {
      uint32_t parsed = 0;
      if (!ParseUnsigned(field, 5, 65535, parsed)) {
        return false;
      }
      value = static_cast<uint16_t>(parsed);
      return true;
    }

    /// Optional leading '-', then at most five digits.  The firmware prints
    /// with %d, so there is never a '+'.
    bool ParseInt16(const std::string& field, int16_t& value) {
      const bool negative = !field.empty() && field[0] == '-';
      uint32_t magnitude = 0;
      if (!ParseUnsigned(negative ? field.substr(1) : field, 5, negative ? 32768U : 32767U, magnitude)) {
        return false;
      }
      value = static_cast<int16_t>(negative ? -static_cast<int32_t>(magnitude) : static_cast<int32_t>(magnitude));
      return true;
    }

    bool AppendThousandths(std::string& frame, double value) {
      if (!std::isfinite(value)) {
        return false;
      }
      const double scaled = std::round(value * 1000.0);
      if (scaled > static_cast<double>(kMaxGainThousandths) || scaled < -static_cast<double>(kMaxGainThousandths)) {
        return false;
      }
      std::array<char, 16> buffer{};
      const int written = std::snprintf(buffer.data(), buffer.size(), "%+08lld,", static_cast<long long>(scaled));
      if (written <= 0) {
        return false;
      }
      frame.append(buffer.data(), static_cast<size_t>(written));
      return true;
    }

  }  // namespace

  std::string Serialize(const JetsonToArduino& command) {
    const int target = std::clamp(
        static_cast<int>(command.target_rpm), -static_cast<int>(kMaxTargetRpm), static_cast<int>(kMaxTargetRpm));
    std::array<char, 24> buffer{};
    const int written = std::snprintf(buffer.data(),
                                      buffer.size(),
                                      "C,%c,%03u,%+06d\n",
                                      command.auto_ready ? '1' : '0',
                                      static_cast<unsigned>(command.steering),
                                      target);
    if (written <= 0) {
      return std::string();
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
  }

  std::string Serialize(const SpeedGains& gains) {
    // Mirror the firmware's own range checks, so a frame it would reject is
    // reported here instead of being resent forever.
    if (!(gains.i_limit >= 0.0) || !(gains.output_limit >= 0.0) || !(gains.output_limit <= kMaxOutputLimitUs) ||
        !(gains.brake_limit >= 0.0) || !(gains.brake_limit <= kMaxBrakeLimitUs)) {
      return std::string();
    }

    std::array<char, 16> header{};
    const int header_length = std::snprintf(
        header.data(), header.size(), "G,%03u,%c,", static_cast<unsigned>(gains.seq), gains.trace ? '1' : '0');
    if (header_length <= 0) {
      return std::string();
    }
    std::string frame(header.data(), static_cast<size_t>(header_length));

    for (const double value :
         {gains.ks, gains.kv, gains.kp, gains.ki, gains.kd, gains.i_limit, gains.output_limit, gains.brake_limit}) {
      if (!AppendThousandths(frame, value)) {
        return std::string();
      }
    }

    uint8_t sum = 0;
    for (const char c : frame) {
      sum = static_cast<uint8_t>(sum + static_cast<uint8_t>(c));
    }
    std::array<char, 4> checksum{};
    std::snprintf(checksum.data(), checksum.size(), "%02X", static_cast<unsigned>(sum));
    frame.append(checksum.data(), 2);
    frame.push_back('\n');
    return frame;
  }

  bool Deserialize(const std::string& line, ArduinoToJetson& status) {
    const auto fields = SplitFields(line);
    if (fields.size() != 9) {
      return false;
    }

    ArduinoToJetson parsed;
    uint8_t raw_mode = 0;
    if (!ParseBool(fields[0], parsed.estop) || !ParseBool(fields[1], parsed.auto_arm) ||
        !ParseBool(fields[2], parsed.manual_start) || !ParseUint8(fields[3], raw_mode) ||
        !ParseUint8(fields[4], parsed.battery_level) || !ParseInt16(fields[5], parsed.rpm) ||
        !ParseInt16(fields[6], parsed.target_rpm) || !ParseUint16(fields[7], parsed.throttle_us) ||
        !ParseUint8(fields[8], parsed.gains_seq)) {
      return false;
    }
    if (raw_mode > static_cast<uint8_t>(Mode::kAutoActive)) {
      return false;
    }
    parsed.mode = static_cast<Mode>(raw_mode);

    status = parsed;
    return true;
  }

  int16_t ToTargetRpm(double spur_rpm) {
    if (std::isnan(spur_rpm)) {
      return 0;
    }
    const double clamped =
        std::clamp(spur_rpm, -static_cast<double>(kMaxTargetRpm), static_cast<double>(kMaxTargetRpm));
    return static_cast<int16_t>(std::lround(clamped));
  }

  SpeedGains GainsPerMpsToPerKrpm(const SpeedGains& per_mps, double spur_rpm_per_mps) {
    SpeedGains per_krpm = per_mps;
    if (!(spur_rpm_per_mps > 0.0)) {
      return per_krpm;
    }
    // A gain of G us per m/s is G us per spur_rpm_per_mps RPM, which is
    // G * 1000 / spur_rpm_per_mps us per 1000 RPM.
    const double scale = 1000.0 / spur_rpm_per_mps;
    per_krpm.kv *= scale;
    per_krpm.kp *= scale;
    per_krpm.ki *= scale;
    per_krpm.kd *= scale;
    return per_krpm;
  }

  uint8_t NormalizedToCommand(double value) {
    if (std::isnan(value)) {
      return kNeutralCommand;
    }
    if (value > 1.0) {
      value = 1.0;
    } else if (value < -1.0) {
      value = -1.0;
    }
    // 127 is neutral, so the positive half has 128 counts and the negative 127.
    const double span = (value >= 0.0) ? 128.0 : 127.0;
    const double scaled = static_cast<double>(kNeutralCommand) + (value * span);
    const long rounded = std::lround(scaled);
    if (rounded < 0) {
      return 0;
    }
    if (rounded > 255) {
      return 255;
    }
    return static_cast<uint8_t>(rounded);
  }

  bool IsNearlyCenter(uint8_t value, uint8_t center, uint8_t tolerance) {
    const int low = static_cast<int>(center) - static_cast<int>(tolerance);
    const int high = static_cast<int>(center) + static_cast<int>(tolerance);
    const int sample = static_cast<int>(value);
    return sample >= low && sample <= high;
  }

  const char* ModeName(Mode mode) {
    switch (mode) {
      case Mode::kEstop:
        return "E-STOP";
      case Mode::kRcArmed:
        return "RC_ARMED";
      case Mode::kRcActive:
        return "RC_ACTIVE";
      case Mode::kAutoArmed:
        return "AUTO_ARMED";
      case Mode::kAutoActive:
        return "AUTO_ACTIVE";
    }
    return "UNKNOWN";
  }

}  // namespace cfr_arduino_bridge
