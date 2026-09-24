// A simple stereo-like depth error for Gazebo's perfect RGB-D cloud.
//
// Range along each camera ray is perturbed by sigma = a + b x^2, x the
// forward distance -- stereo error grows with the square of depth -- and a
// fraction of the returns drop out entirely, as stereo matching does on
// textureless or occluded patches.  Bearing and packed color are kept.
//
// cloud_segmentation's thickness test allows for exactly this sigma, so the
// two share their defaults (Params::noise_a / noise_b there).

#pragma once

#include <cstddef>
#include <cstdint>

#include "cfr_arduino_bridge/point_cloud.hpp"

namespace cfr_arduino_bridge {

  struct StereoNoise {
    double noise_a = 0.01;
    double noise_b = 0.008;
    double dropout = 0.03;
  };

  // Corrupt a little-endian x/y/z cloud in place, drawing from and advancing
  // the random `state` (any seed will do).  Returns the number of points
  // perturbed.  Throws std::invalid_argument on a big-endian cloud or a
  // layout the buffer cannot hold.
  size_t ApplyStereoNoise(
      uint8_t* data, size_t size, const CloudLayout& layout, const StereoNoise& model, uint64_t& state);

}  // namespace cfr_arduino_bridge
