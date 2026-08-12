#include "cfr_arduino_bridge/protocol.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cfr_arduino_bridge {

  namespace {

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

    bool ParseUint8(const std::string& field, uint8_t& value) {
      if (field.empty() || field.size() > 3) {
        return false;
      }
      uint32_t accumulator = 0;
      for (const char c : field) {
        if (c < '0' || c > '9') {
          return false;
        }
        accumulator = (accumulator * 10) + static_cast<uint32_t>(c - '0');
      }
      if (accumulator > 255) {
        return false;
      }
      value = static_cast<uint8_t>(accumulator);
      return true;
    }

    bool ParseUint16(const std::string& field, uint16_t& value) {
      if (field.empty() || field.size() > 5) {
        return false;
      }
      uint32_t accumulator = 0;
      for (const char c : field) {
        if (c < '0' || c > '9') {
          return false;
        }
        accumulator = (accumulator * 10) + static_cast<uint32_t>(c - '0');
      }
      if (accumulator > 65535) {
        return false;
      }
      value = static_cast<uint16_t>(accumulator);
      return true;
    }

  }  // namespace

  std::string Serialize(const JetsonToArduino& command) {
    std::array<char, 16> buffer{};
    const int written = std::snprintf(buffer.data(),
                                      buffer.size(),
                                      "%c,%03u,%03u\n",
                                      command.auto_ready ? '1' : '0',
                                      static_cast<unsigned>(command.steering),
                                      static_cast<unsigned>(command.throttle));
    if (written <= 0) {
      return std::string();
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
  }

  bool Deserialize(const std::string& line, ArduinoToJetson& status) {
    const auto fields = SplitFields(line);
    if (fields.size() != 6) {
      return false;
    }

    ArduinoToJetson parsed;
    uint8_t raw_mode = 0;
    if (!ParseBool(fields[0], parsed.estop) || !ParseBool(fields[1], parsed.auto_arm) ||
        !ParseBool(fields[2], parsed.manual_start) || !ParseUint8(fields[3], raw_mode) ||
        !ParseUint8(fields[4], parsed.battery_level) || !ParseUint16(fields[5], parsed.rpm)) {
      return false;
    }
    if (raw_mode > static_cast<uint8_t>(Mode::kAutoActive)) {
      return false;
    }
    parsed.mode = static_cast<Mode>(raw_mode);

    status = parsed;
    return true;
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
