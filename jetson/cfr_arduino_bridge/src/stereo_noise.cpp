#include "cfr_arduino_bridge/stereo_noise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace cfr_arduino_bridge {

  namespace {

    // Standard normal samples, drawn once.  A frame needs a couple of hundred
    // thousand of them and drawing each afresh cost more than everything else
    // this does; indexing a table with random bits costs almost nothing, and
    // 65536 of them still reach about four sigma into the tails.
    constexpr int kTableBits = 16;

    const std::array<float, 1U << kTableBits>& NormalTable() {
      static const auto table = [] {
        std::array<float, 1U << kTableBits> values{};
        std::mt19937_64 rng(0x5eed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (float& value : values) {
          value = normal(rng);
        }
        return values;
      }();
      return table;
    }

    // splitmix64: one 64-bit draw per point, split between the table index
    // and the dropout test.
    uint64_t Next(uint64_t& state) {
      uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      return z ^ (z >> 31);
    }

  }  // namespace

  size_t ApplyStereoNoise(
      uint8_t* data, size_t size, const CloudLayout& layout, const StereoNoise& model, uint64_t& state) {
    if (layout.big_endian) {
      throw std::invalid_argument("expected a little endian XYZ cloud");
    }
    layout.Check(size);
    const auto& table = NormalTable();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float a = static_cast<float>(model.noise_a);
    const float b = static_cast<float>(model.noise_b);
    // The top 48 bits as a uniform draw on [0, 1).
    const uint64_t dropout = static_cast<uint64_t>(std::clamp(model.dropout, 0.0, 1.0) * 0x1p48);
    size_t perturbed = 0;
    for (uint32_t row = 0; row < layout.height; ++row) {
      uint8_t* point = data + static_cast<size_t>(row) * layout.row_step;
      for (uint32_t col = 0; col < layout.width; ++col, point += layout.point_step) {
        const float x = ReadFloat(point + layout.x, false);
        const float y = ReadFloat(point + layout.y, false);
        const float z = ReadFloat(point + layout.z, false);
        if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x > 0.0F)) {
          continue;
        }
        const uint64_t draw = Next(state);
        if ((draw >> 16) < dropout) {
          WriteFloat(point + layout.x, nan, false);
          WriteFloat(point + layout.y, nan, false);
          WriteFloat(point + layout.z, nan, false);
          continue;
        }
        // Scaling all three coordinates moves the return along its own ray.
        const float delta = table[draw & ((1U << kTableBits) - 1)] * (a + b * x * x);
        const float scale = std::max(0.01F, x + delta) / x;
        WriteFloat(point + layout.x, x * scale, false);
        WriteFloat(point + layout.y, y * scale, false);
        WriteFloat(point + layout.z, z * scale, false);
        ++perturbed;
      }
    }
    return perturbed;
  }

}  // namespace cfr_arduino_bridge
