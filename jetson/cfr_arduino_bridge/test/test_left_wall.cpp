// left_wall's control laws, stereo_noise, and the CloudLayout they read
// clouds through.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "cfr_arduino_bridge/left_wall.hpp"
#include "cfr_arduino_bridge/point_cloud.hpp"
#include "cfr_arduino_bridge/stereo_noise.hpp"

namespace lw = cfr_arduino_bridge::left_wall;
using cfr_arduino_bridge::CloudLayout;

namespace {

  constexpr double kPi = 3.14159265358979323846;

  // An unorganized x/y/z cloud with `pad` bytes of something else per point.
  struct Cloud {
    std::vector<uint8_t> data;
    CloudLayout layout;

    explicit Cloud(const std::vector<std::array<float, 3>>& points, uint32_t pad = 0) {
      layout.width = static_cast<uint32_t>(points.size());
      layout.height = 1;
      layout.point_step = 12 + pad;
      layout.row_step = layout.width * layout.point_step;
      data.assign(layout.row_step, 0xAB);
      for (size_t i = 0; i < points.size(); ++i) {
        std::memcpy(&data[i * layout.point_step], points[i].data(), 12);
      }
    }

    std::array<float, 3> At(size_t i) const {
      std::array<float, 3> p;
      std::memcpy(p.data(), &data[i * layout.point_step], 12);
      return p;
    }
  };

  // A straight wall `y` meters to the left, as the camera would see its face.
  lw::SectorScan Wall(double y) {
    std::vector<std::array<float, 3>> points;
    for (int i = 0; i < 1000; ++i) {
      const float x = static_cast<float>(0.18 + (3.0 - 0.18) * i / 999.0);
      points.push_back({x, static_cast<float>(y), 0.0F});
    }
    Cloud cloud(points);
    lw::SectorScan scan;
    lw::ScanCloud(cloud.data.data(), cloud.data.size(), cloud.layout, &scan);
    return scan;
  }

  std::string Share(const std::string& relative) {
    return std::string(CFR_PACKAGE_SOURCE_DIR) + "/" + relative;
  }

}  // namespace

TEST(LeftWallCloud, SteersTowardADistantLeftWallAndAwayFromACloseOne) {
  auto close = Wall(0.25);
  auto far = Wall(0.75);
  EXPECT_LT(lw::CloudCommand(close).steer, 0.0);
  EXPECT_GT(lw::CloudCommand(far).steer, 0.0);
}

TEST(LeftWallCloud, EmptyCloudCommandsStop) {
  lw::SectorScan scan;
  const auto command = lw::CloudCommand(scan);
  EXPECT_EQ(command.speed, 0.0);
  EXPECT_EQ(command.steer, 0.0);
}

TEST(LeftWallCloud, RegisteredCloudUsesBodyAxesAndRejectsFloor) {
  // Left and ahead at 50 degrees, the same to the right, and one on the floor
  // below the band.  Padded, since the ZED's points carry color too.
  const float x = std::cos(50 * kPi / 180), y = std::sin(50 * kPi / 180);
  std::vector<std::array<float, 3>> points;
  for (int i = 0; i < 5; ++i) {
    points.push_back({x, y, 0.0F});
    points.push_back({x, -y, 0.0F});
    points.push_back({x, y, -0.4F});
  }
  Cloud cloud(points, 4);
  lw::SectorScan scan;
  lw::ScanCloud(cloud.data.data(), cloud.data.size(), cloud.layout, &scan);
  EXPECT_EQ(scan.points, 10U);
  ASSERT_EQ(scan.near.size(), 5U);  // only the left ones
  EXPECT_NEAR(scan.near[0], 1.0, 1e-6);
  EXPECT_TRUE(scan.far.empty());
  EXPECT_TRUE(scan.front.empty());
}

TEST(LeftWallCloud, ReacquiresWhenNoWallIsInReach) {
  std::vector<std::array<float, 3>> points(30, {2.0F, -2.9F, 0.0F});
  Cloud cloud(points);
  lw::SectorScan scan;
  lw::ScanCloud(cloud.data.data(), cloud.data.size(), cloud.layout, &scan);
  const auto command = lw::CloudCommand(scan);
  EXPECT_DOUBLE_EQ(command.speed, 0.35);
  EXPECT_DOUBLE_EQ(command.steer, 0.25);
}

TEST(LeftWallCloud, SectorRangeIsTheLinearTenthPercentile) {
  std::vector<float> values = {5, 1, 4, 2, 3, 9, 8, 7, 6, 0};
  // numpy.percentile(range(10), 10) == 0.9
  EXPECT_NEAR(lw::SectorRange(values), 0.9, 1e-6);
  std::vector<float> few = {0.5F, 0.5F};
  EXPECT_DOUBLE_EQ(lw::SectorRange(few), 3.0);
}

TEST(LeftWallCloud, RejectsACloudShorterThanItsLayout) {
  Cloud cloud({{1, 0, 0}, {1, 0, 0}});
  lw::SectorScan scan;
  EXPECT_THROW(lw::ScanCloud(cloud.data.data(), cloud.data.size() - 1, cloud.layout, &scan), std::invalid_argument);
}

TEST(LeftWallSimulation, RayHitsTheNearFaceOfABale) {
  const std::vector<lw::Bale> bales = {{2.0, 0.0, 0.0, 0.5, 0.25}};
  EXPECT_NEAR(lw::RangeToBales(bales, 0, 0, 0, 0), 1.5, 1e-9);
  // Rotated a quarter turn, the short side faces the ray.
  const std::vector<lw::Bale> turned = {{2.0, 0.0, kPi / 2, 0.5, 0.25}};
  EXPECT_NEAR(lw::RangeToBales(turned, 0, 0, 0, 0), 1.75, 1e-9);
  // Looking away, or past it, reads max range.
  EXPECT_DOUBLE_EQ(lw::RangeToBales(bales, 0, 0, kPi, 0), 5.0);
  EXPECT_DOUBLE_EQ(lw::RangeToBales(bales, 0, 1.0, 0, 0), 5.0);
  // The bearing is relative to the heading.
  EXPECT_NEAR(lw::RangeToBales(bales, 0, 0, -kPi / 2, kPi / 2), 1.5, 1e-9);
}

TEST(LeftWallSimulation, LoadsTheSpeedCoursesBales) {
  const auto bales = lw::LoadBales(Share("worlds/speed_course.sdf"));
  ASSERT_EQ(bales.size(), 202U);
  // bale_0 in the world file.
  EXPECT_NEAR(bales[0].x, 26.6032, 1e-9);
  EXPECT_NEAR(bales[0].y, 4.0673, 1e-9);
  EXPECT_NEAR(bales[0].half_x, 0.4572, 1e-9);
  EXPECT_NEAR(bales[0].half_y, 0.2286, 1e-9);
  EXPECT_THROW(lw::LoadBales(Share("worlds/no_such_world.sdf")), std::runtime_error);
}

TEST(LeftWallSimulation, FollowsTheCenterlineFromTheStart) {
  lw::CourseFollower follower(Share("config/speed_course_path.json"));
  const auto bales = lw::LoadBales(Share("worlds/speed_course.sdf"));
  // At the origin facing +x, the start of the course: nearly straight on.
  const auto command = follower.Step(bales, 0.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(command.speed, 0.8);
  EXPECT_LT(std::abs(command.steer), 0.2);
  // Turned 30 degrees left of the line, it steers back right; bounded.
  const auto turned = follower.Step(bales, 0.0, 0.0, 30 * kPi / 180);
  EXPECT_LT(turned.steer, command.steer);
  EXPECT_GE(turned.steer, -lw::kMaxSteer);
}

TEST(LeftWallSimulation, FollowerSearchesOnlyAheadAfterTheFirstStep) {
  // A counterclockwise unit circle, the car first placed on point 0.
  std::vector<std::pair<double, double>> path;
  for (int i = 0; i < 100; ++i) {
    path.emplace_back(std::cos(2 * kPi * i / 100), std::sin(2 * kPi * i / 100));
  }
  lw::CourseFollower follower(path);
  const std::vector<lw::Bale> none;
  follower.Step(none, 1.0, 0.0, kPi / 2);
  // Then across the circle, on point 50, facing +x.  Searching only 16 points
  // ahead, the index stops at 15 and the target (27) is ahead and to the
  // left.  A search of the whole path would take point 50 itself, aim at 62,
  // behind and to the right, and steer the other way.
  const auto command = follower.Step(none, -1.0, 0.0, 0.0);
  EXPECT_GT(command.steer, 0.2);
}

TEST(StereoNoise, GrowsWithRangeDropsAboutTheStatedFractionAndKeepsColor) {
  std::vector<std::array<float, 3>> points;
  for (int i = 0; i < 20000; ++i) {
    points.push_back({1.0F, 0.1F, 0.0F});
    points.push_back({5.0F, 0.5F, 0.0F});
  }
  points.push_back({std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F});
  points.push_back({-1.0F, 0.0F, 0.0F});  // behind the camera: untouched
  Cloud cloud(points, 4);
  uint64_t state = 7;
  cfr_arduino_bridge::StereoNoise model;
  cfr_arduino_bridge::ApplyStereoNoise(cloud.data.data(), cloud.data.size(), cloud.layout, model, state);

  double sum[2] = {0, 0}, sum_sq[2] = {0, 0};
  int kept[2] = {0, 0}, dropped = 0;
  for (size_t i = 0; i < 40000; ++i) {
    const auto p = cloud.At(i);
    if (std::isnan(p[0])) {
      ++dropped;
      continue;
    }
    const int near_far = static_cast<int>(i % 2);
    const double truth = near_far == 0 ? 1.0 : 5.0;
    // Moved along its own ray: bearing kept.
    EXPECT_NEAR(p[1] / p[0], 0.1, 1e-5);
    sum[near_far] += p[0] - truth;
    sum_sq[near_far] += (p[0] - truth) * (p[0] - truth);
    ++kept[near_far];
  }
  const double sigma_near = std::sqrt(sum_sq[0] / kept[0]);
  const double sigma_far = std::sqrt(sum_sq[1] / kept[1]);
  EXPECT_NEAR(sigma_near, model.noise_a + model.noise_b * 1.0, 0.002);
  EXPECT_NEAR(sigma_far, model.noise_a + model.noise_b * 25.0, 0.01);
  EXPECT_NEAR(dropped / 40000.0, model.dropout, 0.005);
  EXPECT_NEAR(sum[1] / kept[1], 0.0, 0.01);  // unbiased
  EXPECT_EQ(cloud.At(40001)[0], -1.0F);
  // The four bytes after each point, color on the ZED, are untouched.
  for (size_t i = 0; i < 40002; ++i) {
    ASSERT_EQ(cloud.data[i * 16 + 12], 0xAB);
  }
}

TEST(StereoNoise, RefusesABigEndianCloud) {
  Cloud cloud({{1, 0, 0}});
  cloud.layout.big_endian = true;
  uint64_t state = 1;
  EXPECT_THROW(cfr_arduino_bridge::ApplyStereoNoise(
                   cloud.data.data(), cloud.data.size(), cloud.layout, cfr_arduino_bridge::StereoNoise(), state),
               std::invalid_argument);
}
