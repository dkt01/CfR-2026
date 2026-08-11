#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "cfr_arduino_bridge/protocol.hpp"

using cfr_arduino_bridge::ArduinoToJetson;
using cfr_arduino_bridge::Deserialize;
using cfr_arduino_bridge::IsNearlyCenter;
using cfr_arduino_bridge::JetsonToArduino;
using cfr_arduino_bridge::kNeutralCommand;
using cfr_arduino_bridge::Mode;
using cfr_arduino_bridge::NormalizedToCommand;
using cfr_arduino_bridge::Serialize;

TEST(Serialize, NeutralFrame) {
  EXPECT_EQ(Serialize(JetsonToArduino{}), "0,127,127\n");
}

TEST(Serialize, ReadyFrame) {
  JetsonToArduino command;
  command.auto_ready = true;
  command.steering = 200;
  command.throttle = 5;
  EXPECT_EQ(Serialize(command), "1,200,005\n");
}

// The Arduino's FromJetson::deSerialize() rejects payloads outside [6, 11]
// characters once readBytesUntil() has stripped the newline.  Zero padding
// keeps every frame at a legal, constant width.
TEST(Serialize, AlwaysWithinArduinoLengthLimits) {
  for (int steering = 0; steering <= 255; ++steering) {
    for (int throttle : {0, 1, 9, 10, 99, 100, 127, 255}) {
      JetsonToArduino command;
      command.steering = static_cast<uint8_t>(steering);
      command.throttle = static_cast<uint8_t>(throttle);
      const std::string frame = Serialize(command);
      ASSERT_EQ(frame.back(), '\n');
      const size_t payload_length = frame.size() - 1;
      EXPECT_GE(payload_length, 6U) << frame;
      EXPECT_LE(payload_length, 11U) << frame;
    }
  }
}

TEST(Deserialize, TrailingCommaFrameFromArduino) {
  // ToJetson::serialize() emits a comma after every field, including the last.
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("1,0,1,4,200,", status));
  EXPECT_TRUE(status.estop);
  EXPECT_FALSE(status.auto_arm);
  EXPECT_TRUE(status.manual_start);
  EXPECT_EQ(status.mode, Mode::kAutoActive);
  EXPECT_EQ(status.battery_level, 200);
}

TEST(Deserialize, WithoutTrailingComma) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("0,1,0,3,255", status));
  EXPECT_EQ(status.mode, Mode::kAutoArmed);
  EXPECT_EQ(status.battery_level, 255);
}

TEST(Deserialize, RejectsMalformedFrames) {
  ArduinoToJetson status;
  EXPECT_FALSE(Deserialize("", status));
  EXPECT_FALSE(Deserialize("0,1,0,3", status)) << "too few fields";
  EXPECT_FALSE(Deserialize("0,1,0,3,255,7", status)) << "too many fields";
  EXPECT_FALSE(Deserialize("0,1,0,9,255", status)) << "mode out of range";
  EXPECT_FALSE(Deserialize("0,1,0,3,300", status)) << "battery out of range";
  EXPECT_FALSE(Deserialize("2,1,0,3,255", status)) << "non-boolean flag";
  EXPECT_FALSE(Deserialize("0,1,0,3,2a5", status)) << "non-numeric battery";
}

TEST(Deserialize, LeavesOutputUntouchedOnFailure) {
  ArduinoToJetson status;
  ASSERT_TRUE(Deserialize("1,1,1,4,42", status));
  ASSERT_FALSE(Deserialize("garbage", status));
  EXPECT_EQ(status.mode, Mode::kAutoActive);
  EXPECT_EQ(status.battery_level, 42);
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

// Small commands must not land inside the Arduino's arming deadband silently,
// and neutral must always be inside it.
TEST(NormalizedToCommand, NeutralIsInsideArmingDeadband) {
  EXPECT_TRUE(IsNearlyCenter(NormalizedToCommand(0.0)));
  EXPECT_FALSE(IsNearlyCenter(NormalizedToCommand(0.5)));
  EXPECT_FALSE(IsNearlyCenter(NormalizedToCommand(-0.5)));
}
