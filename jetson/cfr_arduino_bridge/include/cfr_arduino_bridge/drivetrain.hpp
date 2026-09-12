// Drivetrain geometry for turning the Arduino's spur RPM into wheel RPM and
// ground speed.  Header only and free of ROS so it can be unit tested.
#pragma once

namespace cfr_arduino_bridge {

  /// Spur gear to wheel reduction of the Slash 4X4 transmission and
  /// differentials.  The owner's manual gives the final ratio as
  /// (spur teeth / pinion teeth) x 2.85, so the pinion only sets motor to spur
  /// and a pinion swap (9T fitted) does not change this.
  constexpr double kSpurToWheelRatio = 2.85;

  /// Nominal outer diameter of the Traxxas 6764 Gravix 2.8" tire, 4.5".  Foam
  /// filled tires grow with speed and squash under load, so calibrate against a
  /// measured roll-out if speed accuracy matters.
  constexpr double kTireDiameterM = 0.1143;

  constexpr double kPi = 3.14159265358979323846;

  /// Wheel revolutions per minute from spur revolutions per minute.  Returns
  /// zero for a non-positive (or NaN) ratio rather than dividing by it.
  inline double SpurRpmToWheelRpm(double spur_rpm, double spur_to_wheel_ratio = kSpurToWheelRatio) {
    return spur_to_wheel_ratio > 0.0 ? spur_rpm / spur_to_wheel_ratio : 0.0;
  }

  /// Ground speed in m/s from wheel RPM, assuming no wheel slip.  The hall
  /// sensor cannot see direction, so this is a magnitude.  Returns zero for a
  /// non-positive (or NaN) diameter.
  inline double WheelRpmToSpeed(double wheel_rpm, double tire_diameter_m = kTireDiameterM) {
    return tire_diameter_m > 0.0 ? wheel_rpm * kPi * tire_diameter_m / 60.0 : 0.0;
  }

  /// Ground speed in m/s straight from spur RPM.
  inline double SpurRpmToSpeed(double spur_rpm,
                               double spur_to_wheel_ratio = kSpurToWheelRatio,
                               double tire_diameter_m = kTireDiameterM) {
    return WheelRpmToSpeed(SpurRpmToWheelRpm(spur_rpm, spur_to_wheel_ratio), tire_diameter_m);
  }

}  // namespace cfr_arduino_bridge
