// C interface to cloud_segmentation, for src/cloud_segmentation.py.
//
// Python loads this library with ctypes, so the test suite that scores the
// segmenter against the rendered fixtures -- and any training code -- runs the
// same compiled code as cloud_segmentation_node.  Params travel as a flat
// array of doubles in CFR_SEGMENTATION_PARAMS order, and the names, integer
// flags and defaults are exported too, so Python never keeps its own copy of
// the tunables to drift out of step with these.

#include <cstdint>
#include <type_traits>

#include "cfr_arduino_bridge/cloud_segmentation.hpp"

namespace seg = cfr_arduino_bridge::segmentation;

namespace {

#define CFR_PARAM_NAME(field) #field,
  constexpr const char* kNames[] = {CFR_SEGMENTATION_PARAMS(CFR_PARAM_NAME)};
#undef CFR_PARAM_NAME

#define CFR_PARAM_IS_INT(field) std::is_integral_v<decltype(seg::Params::field)>,
  constexpr bool kIsInt[] = {CFR_SEGMENTATION_PARAMS(CFR_PARAM_IS_INT)};
#undef CFR_PARAM_IS_INT

  constexpr int kCount = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));

  seg::Params FromArray(const double* values) {
    seg::Params p;
    int i = 0;
#define CFR_PARAM_SET(field) p.field = static_cast<decltype(p.field)>(values[i++]);
    CFR_SEGMENTATION_PARAMS(CFR_PARAM_SET)
#undef CFR_PARAM_SET
    return p;
  }

  double Default(int index) {
    const seg::Params p;
    int i = 0;
    double value = 0.0;
#define CFR_PARAM_GET(field)              \
  if (i++ == index) {                     \
    value = static_cast<double>(p.field); \
  }
    CFR_SEGMENTATION_PARAMS(CFR_PARAM_GET)
#undef CFR_PARAM_GET
    return value;
  }

}  // namespace

// Doubles per gate row written by cfr_segment.
constexpr int kGateFields = 11;

extern "C" {

int cfr_segmentation_param_count() {
  return kCount;
}

const char* cfr_segmentation_param_name(int index) {
  return index >= 0 && index < kCount ? kNames[index] : nullptr;
}

int cfr_segmentation_param_is_int(int index) {
  return index >= 0 && index < kCount && kIsInt[index];
}

double cfr_segmentation_param_default(int index) {
  return Default(index);
}

int cfr_segmentation_gate_fields() {
  return kGateFields;
}

// Segment an n x 3 row-major cloud.  Outputs are caller-allocated: labels,
// blocking (n bytes each), height (n doubles), level (3n doubles), and gates
// (max_gates rows of kind, center x/y, foot a x/y, foot b x/y, span, top,
// axis x/y).  Returns the number of gates found, which may exceed max_gates
// (only the first max_gates are written), or -1 if param_count is wrong.
int64_t cfr_segment(const double* xyz,
                    int64_t n,
                    double pitch,
                    double roll,
                    const double* params,
                    int param_count,
                    uint8_t* labels,
                    double* height,
                    uint8_t* blocking,
                    double* level,
                    double* gates,
                    int64_t max_gates) {
  if (param_count != kCount) {
    return -1;
  }
  seg::Segmentation out;
  seg::Segment(xyz, static_cast<size_t>(n), pitch, roll, FromArray(params), &out);
  for (int64_t k = 0; k < n; ++k) {
    labels[k] = out.labels[static_cast<size_t>(k)];
    height[k] = out.height[static_cast<size_t>(k)];
    blocking[k] = out.blocking[static_cast<size_t>(k)];
  }
  for (int64_t k = 0; k < 3 * n; ++k) {
    level[k] = out.level[static_cast<size_t>(k)];
  }
  const int64_t found = static_cast<int64_t>(out.gates.size());
  for (int64_t g = 0; g < found && g < max_gates; ++g) {
    const seg::Gate& gate = out.gates[static_cast<size_t>(g)];
    double* row = gates + g * kGateFields;
    row[0] = gate.kind;
    row[1] = gate.center[0];
    row[2] = gate.center[1];
    row[3] = gate.feet[0][0];
    row[4] = gate.feet[0][1];
    row[5] = gate.feet[1][0];
    row[6] = gate.feet[1][1];
    row[7] = gate.span;
    row[8] = gate.top;
    row[9] = gate.axis[0];
    row[10] = gate.axis[1];
  }
  return found;
}

}  // extern "C"
