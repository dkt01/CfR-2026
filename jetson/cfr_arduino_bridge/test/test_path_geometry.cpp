#include <gtest/gtest.h>

#include <cmath>

#include "cfr_arduino_bridge/path_geometry.hpp"

using cfr_arduino_bridge::ComputeSegmentCommand;
using cfr_arduino_bridge::ControllerParams;
using cfr_arduino_bridge::kPi;
using cfr_arduino_bridge::Pose2D;
using cfr_arduino_bridge::Segment;
using cfr_arduino_bridge::SegmentType;
using cfr_arduino_bridge::WrapToPi;
using cfr_arduino_bridge::YawFromQuaternion;

namespace {
constexpr double kTolerance = 1e-6;

// Quaternion for a pure yaw rotation.
void YawQuaternion(double yaw, double &w, double &x, double &y, double &z) {
  w = std::cos(yaw / 2.0);
  x = 0.0;
  y = 0.0;
  z = std::sin(yaw / 2.0);
}
} // namespace

TEST(YawFromQuaternionTest, Identity) {
  EXPECT_NEAR(YawFromQuaternion(1.0, 0.0, 0.0, 0.0), 0.0, kTolerance);
}

TEST(YawFromQuaternionTest, NinetyDegreesLeft) {
  double w, x, y, z;
  YawQuaternion(kPi / 2.0, w, x, y, z);
  EXPECT_NEAR(YawFromQuaternion(w, x, y, z), kPi / 2.0, kTolerance);
}

TEST(YawFromQuaternionTest, NinetyDegreesRight) {
  double w, x, y, z;
  YawQuaternion(-kPi / 2.0, w, x, y, z);
  EXPECT_NEAR(YawFromQuaternion(w, x, y, z), -kPi / 2.0, kTolerance);
}

TEST(WrapToPiTest, WithinRangeUnchanged) {
  EXPECT_NEAR(WrapToPi(0.5), 0.5, kTolerance);
  EXPECT_NEAR(WrapToPi(-0.5), -0.5, kTolerance);
}

TEST(WrapToPiTest, WrapsPastPi) {
  EXPECT_NEAR(WrapToPi(3.0 * kPi / 2.0), -kPi / 2.0, kTolerance);
  EXPECT_NEAR(WrapToPi(-3.0 * kPi / 2.0), kPi / 2.0, kTolerance);
}

TEST(ComputeSegmentCommandTest, StraightMidway) {
  Segment segment;
  segment.type = SegmentType::kStraight;
  segment.distance = 1.0;
  Pose2D start;
  Pose2D current;
  current.x = 0.5;
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_NEAR(cmd.progress, 0.5, 1e-3);
  EXPECT_GT(cmd.linear_x, 0.0);
  EXPECT_NEAR(cmd.angular_z, 0.0, kTolerance);
}

TEST(ComputeSegmentCommandTest, StraightReachesTolerance) {
  Segment segment;
  segment.type = SegmentType::kStraight;
  segment.distance = 1.0;
  Pose2D start;
  Pose2D current;
  current.x = 0.99;
  ControllerParams params;
  params.distance_tolerance = 0.03;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_TRUE(cmd.complete);
}

TEST(ComputeSegmentCommandTest, StraightCorrectsHeadingDrift) {
  Segment segment;
  segment.type = SegmentType::kStraight;
  segment.distance = 1.0;
  Pose2D start;
  Pose2D current;
  current.x = 0.5;
  current.yaw = 0.1; // drifted left of the start heading
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  // Drifted left -> steer right (negative) to correct back.
  EXPECT_LT(cmd.angular_z, 0.0);
}

TEST(ComputeSegmentCommandTest, StraightDeceleratesNearTarget) {
  Segment segment;
  segment.type = SegmentType::kStraight;
  segment.distance = 1.0;
  Pose2D start;
  Pose2D current;
  current.x = 0.85; // 0.15 m remaining, inside the default 0.3 m decel window
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_LT(cmd.linear_x, params.cruise_speed);
  EXPECT_GE(cmd.linear_x, params.min_creep_speed);
}

TEST(ComputeSegmentCommandTest, StraightReverse) {
  Segment segment;
  segment.type = SegmentType::kStraight;
  segment.distance = -1.0;
  Pose2D start;
  Pose2D current;
  current.x = -0.5;
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_LT(cmd.linear_x, 0.0);
}

TEST(ComputeSegmentCommandTest, TurnLeftKeepsMovingUntilComplete) {
  Segment segment;
  segment.type = SegmentType::kTurn;
  segment.turn_angle = kPi / 2.0; // 90 degrees left
  Pose2D start;
  Pose2D current;
  current.yaw = 0.2; // partway through the turn
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_GT(cmd.angular_z, 0.0);
  // The car cannot rotate in place: it must keep driving forward mid-turn.
  EXPECT_GT(cmd.linear_x, 0.0);
}

TEST(ComputeSegmentCommandTest, TurnRightNegativeRate) {
  Segment segment;
  segment.type = SegmentType::kTurn;
  segment.turn_angle = -kPi / 2.0; // 90 degrees right
  Pose2D start;
  Pose2D current;
  current.yaw = -0.2;
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_LT(cmd.angular_z, 0.0);
  EXPECT_GT(cmd.linear_x, 0.0);
}

TEST(ComputeSegmentCommandTest, TurnCompletesAtTarget) {
  Segment segment;
  segment.type = SegmentType::kTurn;
  segment.turn_angle = kPi / 2.0;
  Pose2D start;
  Pose2D current;
  current.yaw = kPi / 2.0;
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_TRUE(cmd.complete);
}

TEST(ComputeSegmentCommandTest, TurnCorrectsBackOnOvershoot) {
  Segment segment;
  segment.type = SegmentType::kTurn;
  segment.turn_angle = kPi / 2.0;
  Pose2D start;
  Pose2D current;
  current.yaw = kPi / 2.0 + 0.1; // overshot the target by more than tolerance
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_LT(cmd.angular_z, 0.0); // steer back the other way
  EXPECT_LE(cmd.progress, 1.0);
}

TEST(ComputeSegmentCommandTest, TurnDeceleratesNearTarget) {
  Segment segment;
  segment.type = SegmentType::kTurn;
  segment.turn_angle = kPi / 2.0;
  Pose2D start;
  Pose2D current;
  current.yaw =
      kPi / 2.0 -
      0.1; // 0.1 rad remaining, inside the default 0.3 rad decel window
  ControllerParams params;

  const auto cmd = ComputeSegmentCommand(segment, start, current, params);
  EXPECT_FALSE(cmd.complete);
  EXPECT_LT(cmd.angular_z, params.turn_rate);
  EXPECT_GE(cmd.angular_z, params.min_turn_rate);
}
