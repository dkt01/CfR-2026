// Per-point classes for the ZED cloud: ground, obstacle, hoop, car wash,
// overhead.
//
// rl/bale_follower/cloud_scan.py decides "obstacle or not" with one fixed
// height band, which is enough on the Speed Course's flat floor and nowhere
// else.  The Obstacle Course breaks it four ways:
//
//   * The ground is not flat.  The overpass ramp climbs 0.64 m in 3.3 m, so
//     2 m up it the surface is higher than a straw bale is tall.  The helix
//     descends, the bank rolls the car 9 degrees, the pothole board and gravel
//     tray stand proud of the floor and are dished or littered on top.
//   * Hoops are to be driven through, and seen from the car they are two
//     posts and a bar at bale height.
//   * The car wash is to be driven through although its ribbons render as a
//     curtain from 0.12 m up to 0.54 m -- a wall, to a height band.
//   * Some structure is overhead: the tunnel roof, the deck above it.
//
// So instead of one band, this
//
//   1. levels the cloud with the IMU's pitch and roll;
//   2. grows the drivable surface outward from under the car's own wheels on
//      a plan grid, accepting a cell when its lowest return continues the
//      surface it borders (within a grade and a step -- ramps and dishes
//      continue it, a bale face does not);
//   3. measures every point's height above the surface under it: close to it
//      is GROUND, below it is ground too (a lower level seen over an edge);
//   4. calls a column OVERHEAD when nothing in it comes lower than the car's
//      roof, and OBSTACLE otherwise;
//   5. finds *gates* -- thin structures standing on two feet with the span
//      between them open underneath -- and relabels them HOOP when the span
//      is open to the car's height and CARWASH when it is hung with a curtain.
//
// Every step reads only the cloud, the camera's mounting, and the IMU's
// gravity vector, so this runs unchanged on the robot.  Nothing reads sim pose
// or the world file.
//
// Frames: points arrive in REP-103 body convention relative to the camera
// (+x forward, +y left, +z up), which is what Gazebo's rgbd_camera and the ZED
// wrapper's point_cloud/cloud_registered both publish.  Pitch is nose-down
// positive and roll left-side-up positive, as cloud_scan takes them.
//
// This is the ONE implementation.  cloud_segmentation_node runs it on the car
// and in Gazebo, and src/cloud_segmentation.py wraps the same shared library
// through its C interface for test/test_cloud_segmentation.py and for any
// training code that wants it -- so what is scored against the rendered
// fixtures is exactly what drives.  It was ported from a numpy original step
// for step, including the order its relaxations visit cells in, so the
// fixture scores carried across unchanged.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cfr_arduino_bridge::segmentation {

  constexpr uint8_t kGround = 0;
  constexpr uint8_t kObstacle = 1;
  constexpr uint8_t kHoop = 2;
  constexpr uint8_t kCarWash = 3;
  constexpr uint8_t kOverhead = 4;
  // Beyond max_range, non-finite, or behind the camera.
  constexpr uint8_t kUnknown = 255;

  // Tunables, in meters unless named otherwise.
  //
  // The defaults come from the course and the car, not from fitting the
  // fixtures: each says which measurement it is.
  //
  // CFR_SEGMENTATION_PARAMS lists every field once, so the C interface can
  // hand Python the names and defaults instead of Python keeping a second
  // copy of them.  Add a field there, not here.
  struct Params {
    double max_range = 6.0;
    // The ZED 2i's minimum depth.  Anything nearer is a stereo artifact.
    double min_depth = 0.30;
    // Non-ground returns a 5 cm column needs to be believed: at least this
    // many, and at least speckle_fill of what a face at that range gives.
    int min_column_points = 3;
    double speckle_fill = 0.05;
    // Angle between neighboring pixels: 1.92 rad over 640 columns.
    double pixel_angle = 0.003;
    // Where the camera sits relative to the chassis origin, which is on the
    // ground plane between the wheels (sensors_world.SENSORS_CAMERA).
    double camera_forward = 0.315;
    double camera_height = 0.20;
    // Wheel contacts relative to the chassis origin (generate_vehicle_model).
    double half_wheelbase = 0.162;
    double half_track = 0.145;

    // Plan grid the drivable surface is grown on.
    double cell = 0.10;
    // A neighboring cell continues the surface if its lowest return is within
    // step + grade * distance of it.  The steepest drivable surface is the
    // overpass ramp at 19%; the tallest drivable step is the 2 in gravel tray
    // rail.  A bale is 0.356 m, a bucket 0.38 m, a guard rail 0.152 m.
    double max_grade = 0.25;
    double max_step = 0.06;
    // How far growth may carry the surface across cells with no returns --
    // shadows behind bumps and ribbons, and the rows of floor that thin out
    // with range.  Grows with range because the camera's row spacing on the
    // ground does: about 0.015 * r^2 m for a 0.2 m high camera.
    double gap_base = 0.25;
    double gap_per_range_sq = 0.02;
    // Which low percentile of a cell's heights stands for its surface.
    double surface_percentile = 0.10;
    double surface_band = 0.04;
    // Seeds: cells within this plan distance of the camera that lie on the
    // plane of the car's own wheels.
    double seed_radius = 1.2;
    double seed_tolerance = 0.06;

    // A point this high or less above the surface under it is ground.  Above
    // the 19 mm pothole bumps, the 13 mm gate base plates and the pebbles;
    // below the car's 40 mm belly, so anything it would scrape is not ground.
    double ground_tolerance = 0.05;
    // A column whose lowest non-ground return is above this is overhead: the
    // car, camera housing included, stands 0.23 m; the tunnel roof is 0.62.
    double clearance = 0.32;

    // Gates.  The plan grid is finer than the surface grid because a hoop's
    // tube is 34 mm across.
    double gate_cell = 0.05;
    // Hoop top 0.537 m, car wash arch top 0.537 m.  The start signal board is
    // 1.22 m and the deck's guard rails top out at 0.79 m; neither may qualify.
    double gate_min_top = 0.40;
    double gate_max_top = 0.70;
    // A foot touches the ground; the span between the feet does not.  A post's
    // lowest non-ground return sits just above ground_tolerance; the car wash
    // ribbons stop 0.115 m up, which over its 13 mm base plate is 0.10 m above
    // the surface under them.
    double foot_height = 0.08;
    // The camera's horizontal field of view (sensors_world.SENSORS_CAMERA),
    // and how near its edge a gate's end counts as running out of frame.
    double hfov_deg = 110.0;
    double frame_margin_deg = 3.0;
    // The span must fit the car (0.30 m) with room either side.
    double gate_min_span = 0.40;
    double gate_max_span = 1.40;
    // The wash arches span 1.151 m; the course hoops span 0.584 m. Thin
    // streamers may vanish from a stereo frame, but the rigid arch remains.
    double carwash_min_span = 0.85;
    // Thin in the direction of travel: a hoop's base is 0.4 m deep but it is
    // ground; above it the frame is one 34 mm tube.
    double gate_max_thickness = 0.15;
    // Bearing resolution of the see-through test for open space under a
    // column: about 3 cm at 3 m.
    double see_through_deg = 0.5;
    // How many gate cells from a hanging run's end its support may be: the
    // bale walls either side of the car wash stand 0.12 m off its arches.
    int support_reach = 3;
    // A grounded run longer than this is a wall, not a gate's foot; one no
    // bigger than post_size may be a post (the uprights are 34 mm tubes).
    double wall_length = 0.30;
    double post_size = 0.15;
    // The car wash is five arches at 0.457 m pitch: 1.83 m first to last.
    double carwash_depth = 1.95;
    // Of the span's length, how much hangs below car height for it to be a
    // curtain (car wash) rather than open (hoop).
    double curtain_fraction = 0.25;
    // Depth noise to allow for in the thickness test, as the stereo model
    // zed_cloud_noise_node applies: sigma = a + b r^2.
    double noise_a = 0.01;
    double noise_b = 0.008;
  };

// X(name) for every Params field, in declaration order.
#define CFR_SEGMENTATION_PARAMS(X) \
  X(max_range)                     \
  X(min_depth)                     \
  X(min_column_points)             \
  X(speckle_fill)                  \
  X(pixel_angle)                   \
  X(camera_forward)                \
  X(camera_height)                 \
  X(half_wheelbase)                \
  X(half_track)                    \
  X(cell)                          \
  X(max_grade)                     \
  X(max_step)                      \
  X(gap_base)                      \
  X(gap_per_range_sq)              \
  X(surface_percentile)            \
  X(surface_band)                  \
  X(seed_radius)                   \
  X(seed_tolerance)                \
  X(ground_tolerance)              \
  X(clearance)                     \
  X(gate_cell)                     \
  X(gate_min_top)                  \
  X(gate_max_top)                  \
  X(foot_height)                   \
  X(hfov_deg)                      \
  X(frame_margin_deg)              \
  X(gate_min_span)                 \
  X(gate_max_span)                 \
  X(carwash_min_span)              \
  X(gate_max_thickness)            \
  X(see_through_deg)               \
  X(support_reach)                 \
  X(wall_length)                   \
  X(post_size)                     \
  X(carwash_depth)                 \
  X(curtain_fraction)              \
  X(noise_a)                       \
  X(noise_b)

  struct Gate {
    uint8_t kind = kHoop;
    // Center of the span, and its two feet, in the leveled camera frame.
    double center[2] = {0.0, 0.0};
    double feet[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double span = 0.0;
    double top = 0.0;
    // Unit vector along the span, feet[0] to feet[1].
    double axis[2] = {0.0, 1.0};
  };

  struct Segmentation {
    std::vector<uint8_t> labels;
    // Height above the drivable surface under each point (nan if unknown).
    std::vector<double> height;
    // Points the car cannot pass through: obstacles, and the feet of gates.
    std::vector<uint8_t> blocking;
    // The cloud leveled against gravity, camera at the origin, N x 3.
    std::vector<double> level;
    std::vector<Gate> gates;
  };

  // Rotates body-frame points so +z is up: p_level = Ry(pitch) Rx(roll) p.
  class Leveler {
   public:
    Leveler(double pitch, double roll);
    void operator()(const double* in, double* out) const;

   private:
    double m_[9];
  };

  // Classify an N x 3 row-major body-frame cloud.  `out`'s buffers are
  // resized to N and reused, so a caller that segments every frame keeps
  // one Segmentation and allocates nothing after the first.
  // rgb may be null when the input cloud has no registered color. Packed as
  // 0x00RRGGBB; color only supports the geometric car wash classification.
  void Segment(const double* xyz,
               size_t n,
               double pitch,
               double roll,
               const Params& params,
               Segmentation* out,
               const uint32_t* rgb = nullptr);

}  // namespace cfr_arduino_bridge::segmentation
