#include <gtest/gtest.h>

#include <cmath>

#include "cfr_arduino_bridge/drivetrain.hpp"

using cfr_arduino_bridge::kPi;
using cfr_arduino_bridge::kSpurToWheelRatio;
using cfr_arduino_bridge::kTireDiameterM;
using cfr_arduino_bridge::SpeedToSpurRpm;
using cfr_arduino_bridge::SpurRpmToSpeed;
using cfr_arduino_bridge::SpurRpmToWheelRpm;
using cfr_arduino_bridge::WheelRpmToSpeed;

TEST(SpurRpmToWheelRpm, AppliesTransmissionRatio) {
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(2850.0), 1000.0);
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(0.0), 0.0);
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(300.0, 3.0), 100.0);
}

TEST(SpurRpmToWheelRpm, RejectsNonPositiveRatio) {
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(1000.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(1000.0, -2.85), 0.0);
  EXPECT_DOUBLE_EQ(SpurRpmToWheelRpm(1000.0, std::nan("")), 0.0);
}

TEST(WheelRpmToSpeed, OneRevPerSecondIsOneCircumference) {
  EXPECT_DOUBLE_EQ(WheelRpmToSpeed(60.0), kPi * kTireDiameterM);
  // A 1/pi meter tire has a 1 m circumference.
  EXPECT_NEAR(WheelRpmToSpeed(60.0, 1.0 / kPi), 1.0, 1e-12);
}

TEST(WheelRpmToSpeed, RejectsNonPositiveDiameter) {
  EXPECT_DOUBLE_EQ(WheelRpmToSpeed(1000.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(WheelRpmToSpeed(1000.0, -0.1), 0.0);
  EXPECT_DOUBLE_EQ(WheelRpmToSpeed(1000.0, std::nan("")), 0.0);
}

// 3502 spur RPM was measured on blocks at a 1600 us throttle pulse.
TEST(SpurRpmToSpeed, MatchesHandCalculation) {
  EXPECT_NEAR(SpurRpmToSpeed(3502.0), 3502.0 / 2.85 * kPi * 0.1143 / 60.0, 1e-9);
  EXPECT_NEAR(SpurRpmToSpeed(3502.0), 7.354, 0.001);
  EXPECT_DOUBLE_EQ(kSpurToWheelRatio, 2.85);
}

TEST(SpurRpmToSpeed, FullScaleRpmStaysFinite) {
  EXPECT_TRUE(std::isfinite(SpurRpmToSpeed(65535.0)));
}

TEST(SpurRpmToSpeed, SignFollowsRpm) {
  EXPECT_LT(SpurRpmToSpeed(-1000.0), 0.0);
  EXPECT_DOUBLE_EQ(SpurRpmToSpeed(-1000.0), -SpurRpmToSpeed(1000.0));
}

TEST(SpeedToSpurRpm, OneMetrePerSecond) {
  EXPECT_NEAR(SpeedToSpurRpm(1.0), 60.0 * 2.85 / (kPi * 0.1143), 1e-9);
  EXPECT_NEAR(SpeedToSpurRpm(1.0), 476.2, 0.05);
}

TEST(SpeedToSpurRpm, InvertsSpurRpmToSpeed) {
  for (const double rpm : {-5000.0, -1.0, 0.0, 1.0, 3502.0, 20000.0}) {
    EXPECT_NEAR(SpeedToSpurRpm(SpurRpmToSpeed(rpm)), rpm, 1e-9);
  }
  EXPECT_NEAR(SpeedToSpurRpm(SpurRpmToSpeed(1234.0, 3.0, 0.1), 3.0, 0.1), 1234.0, 1e-9);
}

TEST(SpeedToSpurRpm, RejectsNonPositiveGeometry) {
  EXPECT_DOUBLE_EQ(SpeedToSpurRpm(1.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(SpeedToSpurRpm(1.0, -2.85), 0.0);
  EXPECT_DOUBLE_EQ(SpeedToSpurRpm(1.0, std::nan("")), 0.0);
  EXPECT_DOUBLE_EQ(SpeedToSpurRpm(1.0, kSpurToWheelRatio, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(SpeedToSpurRpm(1.0, kSpurToWheelRatio, std::nan("")), 0.0);
}
