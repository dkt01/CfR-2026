// Pure geometry and control-law helpers for path_follower_node, kept free of
// ROS so they can be unit tested directly.
//
// The vehicle is Ackermann (see cmd_vel_to_drive_node), so it cannot rotate in
// place: a TURN segment commands a nonzero forward speed and a yaw rate
// simultaneously, i.e. it drives an arc, and completes once heading has
// rotated by the requested angle -- not after any particular radius or
// duration.
#pragma once

#include <cstdint>

namespace cfr_arduino_bridge
{

  inline constexpr double kPi = 3.14159265358979323846;

  enum class SegmentType : uint8_t
  {
    kStraight = 0,
    kTurn = 1,
  };

  /// One leg of a path.  Units match ROS conventions: metres, radians,
  /// positive turn_angle is left (counter-clockwise, REP-103), matching the
  /// sign of DriveCommand::steering.
  struct Segment
  {
    SegmentType type = SegmentType::kStraight;
    double distance = 0.0;   ///< STRAIGHT: target distance, metres.  Negative reverses.
    double turn_angle = 0.0; ///< TURN: target heading change, radians.  Should satisfy |turn_angle| <= kPi.
  };

  /// Planar pose extracted from odometry.
  struct Pose2D
  {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
  };

  struct ControllerParams
  {
    double cruise_speed = 0.3;        ///< m/s commanded on a STRAIGHT segment
    double turn_speed = 0.2;          ///< m/s commanded on a TURN segment
    double turn_rate = 0.5;           ///< rad/s commanded on a TURN segment
    double heading_kp = 1.5;          ///< P gain holding heading on a STRAIGHT segment
    double max_angular = 1.0;         ///< rad/s clamp on commanded angular.z
    double decel_distance = 0.3;      ///< m, begin slowing within this of the STRAIGHT target
    double decel_angle = 0.3;         ///< rad, begin slowing within this of the TURN target
    double min_creep_speed = 0.08;    ///< m/s floor while decelerating a STRAIGHT segment
    double min_turn_rate = 0.15;      ///< rad/s floor while decelerating a TURN segment
    double distance_tolerance = 0.03; ///< m, STRAIGHT segment is done within this
    double angle_tolerance = 0.02;    ///< rad (~1.1 deg), TURN segment is done within this
  };

  /// Twist plus bookkeeping for the current tick of a segment.
  struct SegmentCommand
  {
    double linear_x = 0.0;
    double angular_z = 0.0;
    double progress = 0.0; ///< 0..1
    bool complete = false;
  };

  /// Extract yaw from a quaternion.  Assumes roll/pitch are small, which holds
  /// for a ground vehicle on a reasonably flat course.
  double YawFromQuaternion(double w, double x, double y, double z);

  /// Wrap an angle (in particular, a raw yaw difference) to (-pi, pi].
  double WrapToPi(double angle);

  /// Compute the Twist for one control tick of @p segment, given the pose at
  /// the start of the segment and the current pose.
  SegmentCommand ComputeSegmentCommand(const Segment &segment, const Pose2D &start, const Pose2D &current,
                                       const ControllerParams &params);

} // namespace cfr_arduino_bridge
