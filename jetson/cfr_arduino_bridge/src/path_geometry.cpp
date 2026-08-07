#include "cfr_arduino_bridge/path_geometry.hpp"

#include <algorithm>
#include <cmath>

namespace cfr_arduino_bridge {

double YawFromQuaternion(double w, double x, double y, double z) {
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double WrapToPi(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

namespace {

SegmentCommand ComputeStraight(const Segment &segment, const Pose2D &start,
                               const Pose2D &current,
                               const ControllerParams &p) {
  SegmentCommand cmd;
  const double dx = current.x - start.x;
  const double dy = current.y - start.y;
  const double traveled = std::hypot(dx, dy);
  const double target = std::abs(segment.distance);
  const double remaining = target - traveled;

  cmd.progress = target > 0.0 ? std::clamp(traveled / target, 0.0, 1.0) : 1.0;
  cmd.complete = remaining <= p.distance_tolerance;
  if (cmd.complete) {
    return cmd;
  }

  const double heading_error = WrapToPi(start.yaw - current.yaw);
  const double direction = segment.distance >= 0.0 ? 1.0 : -1.0;

  double speed = p.cruise_speed;
  if (p.decel_distance > 0.0 && remaining < p.decel_distance) {
    speed = std::max(p.min_creep_speed,
                     p.cruise_speed * (remaining / p.decel_distance));
  }

  cmd.linear_x = direction * speed;
  cmd.angular_z =
      std::clamp(p.heading_kp * heading_error, -p.max_angular, p.max_angular);
  return cmd;
}

SegmentCommand ComputeTurn(const Segment &segment, const Pose2D &start,
                           const Pose2D &current, const ControllerParams &p) {
  SegmentCommand cmd;
  const double delta = WrapToPi(current.yaw - start.yaw);
  const double target = segment.turn_angle;
  const double remaining = target - delta;

  cmd.progress = target != 0.0 ? std::clamp(delta / target, 0.0, 1.0) : 1.0;
  cmd.complete = std::abs(remaining) <= p.angle_tolerance;
  if (cmd.complete) {
    return cmd;
  }

  // Sign comes from the remaining error, not the original target: if the
  // heading overshoots past the target by more than tolerance, remaining
  // flips sign and the car needs to correct back the other way.
  const double sign = remaining >= 0.0 ? 1.0 : -1.0;
  double rate = p.turn_rate;
  if (p.decel_angle > 0.0 && std::abs(remaining) < p.decel_angle) {
    rate = std::max(p.min_turn_rate,
                    p.turn_rate * (std::abs(remaining) / p.decel_angle));
  }

  // Ackermann steering cannot rotate in place: a turn is only realized by
  // driving forward while steering, so linear_x must stay nonzero for the
  // whole segment.
  cmd.angular_z = sign * rate;
  cmd.linear_x = p.turn_speed;
  return cmd;
}

} // namespace

SegmentCommand ComputeSegmentCommand(const Segment &segment,
                                     const Pose2D &start, const Pose2D &current,
                                     const ControllerParams &params) {
  if (segment.type == SegmentType::kStraight) {
    return ComputeStraight(segment, start, current, params);
  }
  return ComputeTurn(segment, start, current, params);
}

} // namespace cfr_arduino_bridge
