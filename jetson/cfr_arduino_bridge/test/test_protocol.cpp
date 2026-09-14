#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "cfr_arduino_bridge/protocol.hpp"

using cfr_arduino_bridge::ArduinoToJetson;
using cfr_arduino_bridge::Deserialize;
using cfr_arduino_bridge::GainsPerMpsToPerKrpm;
using cfr_arduino_bridge::IsNearlyCenter;
using cfr_arduino_bridge::JetsonToArduino;
using cfr_arduino_bridge::kMaxTargetRpm;
using cfr_arduino_bridge::kNeutralCommand;
using cfr_arduino_bridge::Mode;
using cfr_arduino_bridge::NormalizedToCommand;
using cfr_arduino_bridge::Serialize;
using cfr_arduino_bridge::SpeedGains;
using cfr_arduino_bridge::ToTargetRpm;

namespace {

  SpeedGains BenchGains() {
    SpeedGains gains;
    gains.ks = 28.0;
    gains.kv = 9.5;
    gains.kp = 8.0;
    gains.ki = 16.0;
    gains.kd = 0.0;
    gains.i_limit = 25.0;
    gains.output_limit = 128.0;
    gains.brake_limit = 0.0;
    gains.seq = 7;
    gains.trace = true;
    return gains;
  }

}  // namespace

TEST(SerializeCommand, NeutralFrame) {
  EXPECT_EQ(Serialize(JetsonToArduino{}), "C,0,127,+00000\n");
}

TEST(SerializeCommand, ReadyFrame) {
  JetsonToArduino command;
  command.auto_ready = true;
  command.steering = 200;
  command.target_rpm = -1234;
  EXPECT_EQ(Serialize(command), "C,1,200,-01234\n");
}

// FromJetson::deSerialize() accepts exactly 14 bytes before the newline.  A
// frame of any other length is how a byte lost on the link gets rejected, so
// every encodable command must hit it.
TEST(SerializeCommand, AlwaysExactlyFifteenBytes) {
  for (int steering = 0; steering <= 255; ++steering) {
    for (int target : {-32768, -20000, -9999, -1, 0, 1, 99, 12345, 20000, 32767}) {
      JetsonToArduino command;
      command.steering = static_cast<uint8_t>(steering);
      command.target_rpm = static_cast<int16_t>(target);
      const std::string frame = Serialize(command);
      ASSERT_EQ(frame.size(), 15U) << frame;
      EXPECT_EQ(frame.back(), '\n');
    }
  }
}

TEST(SerializeCommand, ClampsTargetToFirmwareLimit) {
  JetsonToArduino command;
  command.target_rpm = std::numeric_limits<int16_t>::max();
  EXPECT_EQ(Serialize(command), "C,0,127,+20000\n");
  command.target_rpm = std::numeric_limits<int16_t>::min();
  EXPECT_EQ(Serialize(command), "C,0,127,-20000\n");
}

// Byte for byte the frame the firmware accepted and echoed as seq 7 on the
// bench, checksum included.
TEST(SerializeGains, MatchesFrameAcceptedByFirmware) {
  EXPECT_EQ(Serialize(BenchGains()),
            "G,007,1,+0028000,+0009500,+0008000,+0016000,+0000000,+0025000,+0128000,+0000000,04\n");
}

TEST(SerializeGains, ChecksumIsSumOfPrecedingBytes) {
  SpeedGains gains = BenchGains();
  gains.kd = -1.2345;
  gains.seq = 255;
  gains.trace = false;
  const std::string frame = Serialize(gains);
  ASSERT_EQ(frame.size(), 83U) << frame;
  uint8_t sum = 0;
  for (size_t index = 0; index < frame.size() - 3; ++index) {
    sum = static_cast<uint8_t>(sum + static_cast<uint8_t>(frame[index]));
  }
  EXPECT_EQ(std::stoi(frame.substr(frame.size() - 3, 2), nullptr, 16), sum);
  EXPECT_NE(frame.find(",-0001235,"), std::string::npos) << "rounds half away from zero: " << frame;
  EXPECT_EQ(frame.rfind("G,255,0,", 0), 0U);
}

TEST(SerializeGains, RejectsValuesTheWireOrFirmwareCannotTake) {
  const auto rejects = [](void (*mutate)(SpeedGains&)) {
    SpeedGains gains = BenchGains();
    mutate(gains);
    return Serialize(gains).empty();
  };
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.kp = std::nan(""); }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.ki = std::numeric_limits<double>::infinity(); }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.kv = 10000.0; }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.kd = -10000.0; }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.i_limit = -1.0; }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.output_limit = 500.5; }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.brake_limit = 440.5; }));
  EXPECT_TRUE(rejects([](SpeedGains& g) { g.brake_limit = -0.5; }));
  EXPECT_FALSE(rejects([](SpeedGains& g) {
    g.output_limit = 500.0;
    g.brake_limit = 440.0;
    g.kv = 9999.999;
  }));
}

// A real frame from the bench, reversing at -800 target RPM.
TEST(Deserialize, StatusFrameFromArduino) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("0,1,0,4,170,-872,-800,1464,7,", status));
  EXPECT_FALSE(status.estop);
  EXPECT_TRUE(status.auto_arm);
  EXPECT_FALSE(status.manual_start);
  EXPECT_EQ(status.mode, Mode::kAutoActive);
  EXPECT_EQ(status.battery_level, 170);
  EXPECT_EQ(status.rpm, -872);
  EXPECT_EQ(status.target_rpm, -800);
  EXPECT_EQ(status.throttle_us, 1464);
  EXPECT_EQ(status.gains_seq, 7);
}

TEST(Deserialize, WithoutTrailingComma) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("1,0,1,3,255,0,0,1500,0", status));
  EXPECT_TRUE(status.estop);
  EXPECT_EQ(status.mode, Mode::kAutoArmed);
  EXPECT_EQ(status.rpm, 0);
}

TEST(Deserialize, SignedFieldsSpanSixteenBits) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("0,0,0,4,1,32767,-32768,2000,255,", status));
  EXPECT_EQ(status.rpm, 32767);
  EXPECT_EQ(status.target_rpm, -32768);
  EXPECT_EQ(status.gains_seq, 255);
  EXPECT_FALSE(Deserialize("0,0,0,4,1,32768,0,1500,0,", status));
  EXPECT_FALSE(Deserialize("0,0,0,4,1,0,-32769,1500,0,", status));
}

TEST(Deserialize, RejectsMalformedFrames) {
  ArduinoToJetson status;
  EXPECT_FALSE(Deserialize("", status));
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7,", status)) << "pre speed control six field frame";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7,0,1500", status)) << "too few fields";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7,0,1500,0,9", status)) << "too many fields";
  EXPECT_FALSE(Deserialize("0,1,0,9,255,7,0,1500,0", status)) << "mode out of range";
  EXPECT_FALSE(Deserialize("0,1,0,3,300,7,0,1500,0", status)) << "battery out of range";
  EXPECT_FALSE(Deserialize("2,1,0,3,255,7,0,1500,0", status)) << "non-boolean flag";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,+7,0,1500,0", status)) << "firmware never prints a plus";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,-,0,1500,0", status)) << "bare sign";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,--7,0,1500,0", status)) << "double sign";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,1e3,0,1500,0", status)) << "non-numeric rpm";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7,0,65536,0", status)) << "throttle out of range";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7,0,1500,256", status)) << "seq out of range";
}

// The firmware also sends "D," debug and "T," trace lines on the same link.
// The bridge skips them by prefix, but they must never decode as status.
TEST(Deserialize, RejectsDiagnosticLines) {
  ArduinoToJetson status;
  EXPECT_FALSE(Deserialize("D,99612,4138,628,1,127,0,4,1504,1500,0,11771,2,502C,0,7,3,0", status));
  EXPECT_FALSE(Deserialize("T,5000,800,812,356,-1,12,0,367,1537,0,170", status));
}

TEST(Deserialize, LeavesOutputUntouchedOnFailure) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("1,1,1,4,42,900,1000,1540,3", status));
  ASSERT_FALSE(Deserialize("garbage", status));
  EXPECT_EQ(status.mode, Mode::kAutoActive);
  EXPECT_EQ(status.battery_level, 42);
  EXPECT_EQ(status.rpm, 900);
  EXPECT_EQ(status.gains_seq, 3);
}

TEST(ToTargetRpm, RoundsClampsAndRejectsNan) {
  EXPECT_EQ(ToTargetRpm(0.0), 0);
  EXPECT_EQ(ToTargetRpm(476.2), 476);
  EXPECT_EQ(ToTargetRpm(-476.6), -477);
  EXPECT_EQ(ToTargetRpm(1.0e9), kMaxTargetRpm);
  EXPECT_EQ(ToTargetRpm(-1.0e9), -kMaxTargetRpm);
  EXPECT_EQ(ToTargetRpm(std::nan("")), 0);
}

TEST(GainsPerMpsToPerKrpm, ScalesOnlyTheRateTerms) {
  SpeedGains per_mps = BenchGains();
  // 500 spur RPM per m/s: a gain per m/s is a gain per 500 RPM, which is
  // twice that per 1000 RPM.
  const SpeedGains per_krpm = GainsPerMpsToPerKrpm(per_mps, 500.0);
  EXPECT_DOUBLE_EQ(per_krpm.kv, 19.0);
  EXPECT_DOUBLE_EQ(per_krpm.kp, 16.0);
  EXPECT_DOUBLE_EQ(per_krpm.ki, 32.0);
  EXPECT_DOUBLE_EQ(per_krpm.kd, 0.0);
  EXPECT_DOUBLE_EQ(per_krpm.ks, per_mps.ks);
  EXPECT_DOUBLE_EQ(per_krpm.i_limit, per_mps.i_limit);
  EXPECT_DOUBLE_EQ(per_krpm.output_limit, per_mps.output_limit);
  EXPECT_DOUBLE_EQ(per_krpm.brake_limit, per_mps.brake_limit);
  EXPECT_EQ(per_krpm.seq, per_mps.seq);
  EXPECT_EQ(per_krpm.trace, per_mps.trace);
}

TEST(GainsPerMpsToPerKrpm, NonPositiveScaleLeavesGainsAlone) {
  const SpeedGains per_mps = BenchGains();
  EXPECT_DOUBLE_EQ(GainsPerMpsToPerKrpm(per_mps, 0.0).kp, per_mps.kp);
  EXPECT_DOUBLE_EQ(GainsPerMpsToPerKrpm(per_mps, -1.0).kp, per_mps.kp);
  EXPECT_DOUBLE_EQ(GainsPerMpsToPerKrpm(per_mps, std::nan("")).kp, per_mps.kp);
}

TEST(NormalizedToCommand, EndpointsAndCenter) {
  EXPECT_EQ(NormalizedToCommand(0.0), kNeutralCommand);
  EXPECT_EQ(NormalizedToCommand(1.0), 255);
  EXPECT_EQ(NormalizedToCommand(-1.0), 0);
}

TEST(NormalizedToCommand, ClampsOutOfRangeAndNan) {
  EXPECT_EQ(NormalizedToCommand(5.0), 255);
  EXPECT_EQ(NormalizedToCommand(-5.0), 0);
  EXPECT_EQ(NormalizedToCommand(std::nan("")), kNeutralCommand);
}

// Small steering commands must not land inside the Arduino's arming deadband
// silently, and neutral must always be inside it.
TEST(NormalizedToCommand, NeutralIsInsideArmingDeadband) {
  EXPECT_TRUE(IsNearlyCenter(NormalizedToCommand(0.0)));
  EXPECT_FALSE(IsNearlyCenter(NormalizedToCommand(0.5)));
  EXPECT_FALSE(IsNearlyCenter(NormalizedToCommand(-0.5)));
}
