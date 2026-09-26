// See cloud_segmentation.hpp for what this does and why.
//
// The numpy original this was ported from built every step out of whole-grid
// shifts, and several of its loops are order dependent: a relaxation that
// updates in place sees, in the next direction it tries, what the previous
// direction changed.  Those orders are reproduced here deliberately, and each
// place that depends on one says so, because changing it changes labels.

#include "cfr_arduino_bridge/cloud_segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cfr_arduino_bridge::segmentation {

  namespace {

    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    constexpr double kPi = 3.14159265358979323846;

    // The eight neighbors, in the order every relaxation below visits them.
    constexpr int kNeighbors[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    // Per-point scratch, kept between frames.  Allocated fresh, these are
    // big enough that the allocator maps new pages for them every call, and
    // faulting those in cost several milliseconds a frame.
    struct Workspace {
      std::vector<int64_t> valid, ci, cj, fi, fj, candidates, bearing, band_start, band_fill;
      std::vector<double> plan_range, vx, vy, vz, h, column_low, band, steepest, keys;
      std::vector<uint8_t> is_ground, beside_ground, seen_under;
      std::vector<std::pair<double, double>> rays;
    };

    template <typename T>
    struct Grid {
      int64_t rows = 0;
      int64_t cols = 0;
      std::vector<T> cells;

      Grid() = default;
      Grid(int64_t r, int64_t c, T fill) : rows(r), cols(c), cells(static_cast<size_t>(r * c), fill) {}

      T& operator()(int64_t i, int64_t j) { return cells[static_cast<size_t>(i * cols + j)]; }
      const T& operator()(int64_t i, int64_t j) const { return cells[static_cast<size_t>(i * cols + j)]; }
      bool Inside(int64_t i, int64_t j) const { return i >= 0 && i < rows && j >= 0 && j < cols; }
      // The value at (i - dx, j - dy), or `fill` off the grid.
      T Neighbor(int64_t i, int64_t j, int dx, int dy, T fill) const {
        const int64_t si = i - dx;
        const int64_t sj = j - dy;
        return Inside(si, sj) ? (*this)(si, sj) : fill;
      }
    };

    using Mask = Grid<uint8_t>;

    // Cell or its 8 neighbors set.
    Mask Dilate(const Mask& mask) {
      Mask out = mask;
      for (int64_t i = 0; i < mask.rows; ++i) {
        for (int64_t j = 0; j < mask.cols; ++j) {
          if (out(i, j)) {
            continue;
          }
          for (const auto& d : kNeighbors) {
            if (mask.Neighbor(i, j, d[0], d[1], 0)) {
              out(i, j) = 1;
              break;
            }
          }
        }
      }
      return out;
    }

    // 8-connected components of `occupied`, each labeled by the smallest
    // row-major index it contains (-1 where unoccupied).  That is what a
    // min-label relaxation converges to, and the gate search visits
    // components in the order those labels sort in.
    Grid<int64_t> Components(const Mask& occupied) {
      Grid<int64_t> label(occupied.rows, occupied.cols, -1);
      std::vector<int64_t> stack;
      for (int64_t start = 0; start < static_cast<int64_t>(occupied.cells.size()); ++start) {
        if (!occupied.cells[static_cast<size_t>(start)] || label.cells[static_cast<size_t>(start)] >= 0) {
          continue;
        }
        label.cells[static_cast<size_t>(start)] = start;
        stack.assign(1, start);
        while (!stack.empty()) {
          const int64_t at = stack.back();
          stack.pop_back();
          const int64_t i = at / occupied.cols;
          const int64_t j = at % occupied.cols;
          for (const auto& d : kNeighbors) {
            const int64_t ni = i + d[0];
            const int64_t nj = j + d[1];
            if (occupied.Inside(ni, nj) && occupied(ni, nj) && label(ni, nj) < 0) {
              label(ni, nj) = start;
              stack.push_back(ni * occupied.cols + nj);
            }
          }
        }
      }
      return label;
    }

    // Drivable-surface height per cell, and which cells were reached.
    //
    // Relaxation on the grid: a cell joins the surface when a neighbor already
    // on it predicts its lowest return within step + grade * distance.  Empty
    // cells carry the neighbor's height forward for as long as the gap
    // allowance at that range permits, so the surface can cross shadows and
    // the thinning rows of far floor without inventing a surface where
    // something solid stands.
    //
    // Each direction reads its neighbors as they stood before that direction
    // began (the numpy original shifted whole arrays) but after the previous
    // direction's updates.  Visiting cells so that a neighbor at (i-dx, j-dy)
    // is always visited *after* (i, j) gives exactly that without copying.
    void GrowSurface(const Grid<double>& zmin,
                     const Mask& seeds,
                     const Grid<double>& allowance,
                     const Params& p,
                     Grid<double>* surface,
                     Mask* reached) {
      const int64_t rows = zmin.rows;
      const int64_t cols = zmin.cols;
      *surface = Grid<double>(rows, cols, kNan);
      *reached = seeds;
      Grid<double> gap(rows, cols, kInf);
      for (size_t k = 0; k < zmin.cells.size(); ++k) {
        if (seeds.cells[k]) {
          surface->cells[k] = zmin.cells[k];
          gap.cells[k] = 0.0;
        }
      }
      const int64_t iterations = 4 * std::max(rows, cols);
      for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        bool changed = false;
        for (const auto& d : kNeighbors) {
          const int dx = d[0];
          const int dy = d[1];
          const double step = p.cell * std::hypot(static_cast<double>(dx), static_cast<double>(dy));
          const double tolerance_base = p.max_step;
          for (int64_t a = 0; a < rows; ++a) {
            const int64_t i = dx > 0 ? rows - 1 - a : a;
            for (int64_t b = 0; b < cols; ++b) {
              const int64_t j = dy > 0 ? cols - 1 - b : b;
              const int64_t si = i - dx;
              const int64_t sj = j - dy;
              if (!zmin.Inside(si, sj) || !(*reached)(si, sj)) {
                continue;
              }
              const double n_surface = (*surface)(si, sj);
              const double n_gap = gap(si, sj);
              const double z = zmin(i, j);
              if (std::isfinite(z)) {
                // Occupied: accept when the return continues the surface.  The
                // tolerance grows with the gap crossed to get here, since the
                // surface may have kept climbing unseen.
                if ((*reached)(i, j)) {
                  continue;
                }
                const double reach = step + (std::isfinite(n_gap) ? n_gap : 0.0);
                if (std::abs(z - n_surface) <= tolerance_base + p.max_grade * reach) {
                  (*surface)(i, j) = z;
                  gap(i, j) = 0.0;
                  (*reached)(i, j) = 1;
                  changed = true;
                }
              } else {
                // Empty: carry the surface across, within the allowance.
                const double carried = n_gap + step;
                if (carried <= allowance(i, j) && carried < gap(i, j)) {
                  (*surface)(i, j) = n_surface;
                  gap(i, j) = carried;
                  (*reached)(i, j) = 1;
                  changed = true;
                }
              }
            }
          }
        }
        if (!changed) {
          break;
        }
      }
      for (size_t k = 0; k < zmin.cells.size(); ++k) {
        reached->cells[k] = reached->cells[k] && std::isfinite(zmin.cells[k]);
      }
    }

    // Spread known surface heights into unknown cells, `passes` cells deep.
    // Each pass reads the grid as it stood before the pass.
    void Fill(Grid<double>* surface, Mask* known, int passes) {
      const int64_t rows = surface->rows;
      const int64_t cols = surface->cols;
      Grid<double> total(rows, cols, 0.0);
      Grid<double> count(rows, cols, 0.0);
      for (int pass = 0; pass < passes; ++pass) {
        bool any = false;
        for (int64_t i = 0; i < rows; ++i) {
          for (int64_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            double n = 0.0;
            for (const auto& d : kNeighbors) {
              if (known->Neighbor(i, j, d[0], d[1], 0)) {
                sum += (*surface)(i - d[0], j - d[1]);
                n += 1.0;
              } else {
                sum += 0.0;
              }
            }
            total(i, j) = sum;
            count(i, j) = n;
            any = any || (!(*known)(i, j) && n > 0.0);
          }
        }
        if (!any) {
          break;
        }
        for (size_t k = 0; k < surface->cells.size(); ++k) {
          if (!known->cells[k] && count.cells[k] > 0.0) {
            surface->cells[k] = total.cells[k] / count.cells[k];
            known->cells[k] = 1;
          }
        }
      }
    }

    // atan2 to within 1.15e-5 rad (measured over 2e7 random points): a degree
    // 9 minimax polynomial for atan on [0, 1], folded out to the full circle.
    constexpr double kFastAtan2Error = 1.15e-5;
    double FastAtan2(double y, double x) {
      const double ax = std::abs(x);
      const double ay = std::abs(y);
      const double big = std::max(ax, ay);
      if (big == 0.0) {
        return std::atan2(y, x);
      }
      const double t = std::min(ax, ay) / big;
      const double t2 = t * t;
      double a = t * (0.9998660 + t2 * (-0.3302995 + t2 * (0.1801410 + t2 * (-0.0851330 + t2 * 0.0208351))));
      if (ay > ax) {
        a = kPi / 2 - a;
      }
      if (x < 0) {
        a = kPi - a;
      }
      return y < 0 ? -a : a;
    }

    // The see-through bearing bin of (x, y), exactly as the numpy original
    // computed it from atan2.  Wherever the fast arctangent lands more than
    // four times its worst error from a bin edge the bin cannot differ, and
    // at the default half-degree bins that is all but 1% of points; only the
    // rest pay for the exact call.
    int64_t BearingBin(double y, double x, int64_t bins) {
      const double scale = static_cast<double>(bins) / (2 * kPi);
      const double guard = 4 * kFastAtan2Error * scale;
      const double fast = (FastAtan2(y, x) + kPi) * scale;
      const double fraction = fast - std::floor(fast);
      if (fraction > guard && fraction < 1.0 - guard) {
        return static_cast<int64_t>(fast) % bins;
      }
      return static_cast<int64_t>((std::atan2(y, x) + kPi) / (2 * kPi) * static_cast<double>(bins)) % bins;
    }

    // Whether a ray to the ground further out passed below each point that
    // is not ground itself (false for ground points, where it is not used).
    //
    // Per bearing bin, the steepest ground ray beyond a point's range is the
    // one that passes lowest under it; the point is seen under if that ray is
    // below it at its range.  Exact bearing bins only -- a ray past the edge
    // of a board has not been under the board.
    //
    // `r` is each point's plan range.  The ground rays are ordered by (bin,
    // range) with a counting sort on the bin, which is what makes this cheap;
    // the running minimum and the search keys are the numpy original's,
    // arithmetic included.
    void SeenUnder(const std::vector<double>& x,
                   const std::vector<double>& y,
                   const std::vector<double>& z,
                   const std::vector<double>& r,
                   const std::vector<uint8_t>& is_ground,
                   const Params& p,
                   Workspace* ws,
                   std::vector<uint8_t>* out) {
      const size_t n = x.size();
      std::vector<uint8_t>& result = *out;
      result.assign(n, 0);
      const int64_t bins = static_cast<int64_t>(std::nearbyint(360.0 / p.see_through_deg));
      std::vector<int64_t>& bearing = ws->bearing;
      bearing.resize(n);
      std::vector<int64_t> start(static_cast<size_t>(bins) + 1, 0);
      for (size_t k = 0; k < n; ++k) {
        bearing[k] = BearingBin(y[k], x[k], bins);
        if (is_ground[k]) {
          ++start[static_cast<size_t>(bearing[k]) + 1];
        }
      }
      for (size_t b = 1; b < start.size(); ++b) {
        start[b] += start[b - 1];
      }
      const size_t m = static_cast<size_t>(start.back());
      if (m == 0) {
        return;
      }
      // Ground rays grouped by bin, then each bin by range.  How equal ranges
      // are ordered cannot matter: the first of them is where every search
      // lands, and its running minimum already covers all of them.
      std::vector<std::pair<double, double>>& rays = ws->rays;  // (range, slope)
      rays.resize(m);
      {
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (size_t k = 0; k < n; ++k) {
          if (is_ground[k]) {
            rays[static_cast<size_t>(fill[static_cast<size_t>(bearing[k])]++)] = {r[k], z[k] / std::max(r[k], 1e-6)};
          }
        }
        for (int64_t b = 0; b < bins; ++b) {
          std::sort(rays.begin() + start[static_cast<size_t>(b)],
                    rays.begin() + start[static_cast<size_t>(b) + 1],
                    [](const auto& a, const auto& c) { return a.first < c.first; });
        }
      }
      // Suffix minimum of slope within each bin: offsetting by bin keeps a
      // later bin's values from ever winning an earlier bin's minimum.
      const double span = p.max_range * 4 + 10.0;
      std::vector<double>& steepest = ws->steepest;
      std::vector<double>& keys = ws->keys;
      steepest.resize(m);
      keys.resize(m);
      double running = std::numeric_limits<double>::infinity();
      for (int64_t b = bins; b-- > 0;) {
        const double offset = 10.0 * static_cast<double>(b);
        for (int64_t k = start[static_cast<size_t>(b) + 1]; k-- > start[static_cast<size_t>(b)];) {
          const auto& ray = rays[static_cast<size_t>(k)];
          running = std::min(running, ray.second + offset);
          steepest[static_cast<size_t>(k)] = running - offset;
          keys[static_cast<size_t>(k)] = static_cast<double>(b) * span + ray.first;
        }
      }
      for (size_t k = 0; k < n; ++k) {
        // Only ever read for what is not ground itself.
        if (is_ground[k]) {
          continue;
        }
        // The first ground ray in this bin at least a cell further out.  Keys
        // of other bins can never match (they are a whole span away), so the
        // search stays inside the bin.
        const size_t b = static_cast<size_t>(bearing[k]);
        const double query = static_cast<double>(bearing[k]) * span + r[k] + p.cell;
        const auto first = keys.begin() + start[b];
        const auto last = keys.begin() + start[b + 1];
        const auto at = std::lower_bound(first, last, query);
        if (at == last) {
          continue;
        }
        result[k] = steepest[static_cast<size_t>(at - keys.begin())] * r[k] < z[k] - 0.02;
      }
    }

    // Unit eigenvectors of a symmetric 2x2 matrix [[a, b], [b, c]]: the major
    // one (largest eigenvalue) and the minor one.
    void Eigen2(double a, double b, double c, double major[2], double minor[2]) {
      if (b == 0.0) {
        if (a >= c) {
          major[0] = 1.0, major[1] = 0.0;
        } else {
          major[0] = 0.0, major[1] = 1.0;
        }
      } else {
        const double half = 0.5 * (a - c);
        const double root = std::hypot(half, b);
        const double lambda = 0.5 * (a + c) + root;
        // Of the two equivalent forms, the one without cancellation.
        double vx, vy;
        if (a >= c) {
          vx = lambda - c;
          vy = b;
        } else {
          vx = b;
          vy = lambda - a;
        }
        const double norm = std::hypot(vx, vy);
        major[0] = vx / norm;
        major[1] = vy / norm;
      }
      minor[0] = -major[1];
      minor[1] = major[0];
    }

    struct FoundGate {
      Gate gate;
      std::vector<int64_t> members;
      std::vector<int64_t> feet;
    };

    // Gates among the candidate points.
    //
    // A gate is found by what hangs, not by what stands.  On a plan grid, a
    // cell is *grounded* when its lowest return is within foot_height of the
    // surface, and *hanging* when its lowest return is above that and its
    // highest no taller than a gate.  A gate is a thin run of hanging cells --
    // a hoop's bar, a car wash arch with its ribbons -- held up at both ends
    // by grounded cells: its own posts, or a wall standing against them, as
    // the bale walls stand against the car wash's uprights.  An end may also
    // run out of the frame, which is how the car wash looks from inside it.
    std::vector<FoundGate> FindGates(const std::vector<double>& level,
                                     const std::vector<double>& height,
                                     const std::vector<int64_t>& index,
                                     const std::vector<double>& column_low,
                                     const Params& p) {
      std::vector<FoundGate> gates;
      if (index.empty()) {
        return gates;
      }
      const size_t count = index.size();
      const double g = p.gate_cell;
      std::vector<int64_t> ix(count), iy(count);
      int64_t ix_min = std::numeric_limits<int64_t>::max(), iy_min = ix_min;
      int64_t ix_max = std::numeric_limits<int64_t>::min(), iy_max = ix_max;
      for (size_t k = 0; k < count; ++k) {
        const size_t at = static_cast<size_t>(index[k]);
        ix[k] = static_cast<int64_t>(std::floor(level[3 * at] / g));
        iy[k] = static_cast<int64_t>(std::floor(level[3 * at + 1] / g));
        ix_min = std::min(ix_min, ix[k]);
        ix_max = std::max(ix_max, ix[k]);
        iy_min = std::min(iy_min, iy[k]);
        iy_max = std::max(iy_max, iy[k]);
      }
      const int64_t x0 = ix_min - 2;
      const int64_t y0 = iy_min - 2;
      const int64_t rows = ix_max - x0 + 3;
      const int64_t cols = iy_max - y0 + 3;
      std::vector<int64_t> cell_of(count);
      Grid<double> lowest(rows, cols, kInf);
      Grid<double> highest(rows, cols, -kInf);
      Mask bar_height(rows, cols, 0);
      Mask evidence(rows, cols, 0);
      for (size_t k = 0; k < count; ++k) {
        const int64_t cx = ix[k] - x0;
        const int64_t cy = iy[k] - y0;
        cell_of[k] = cx * cols + cy;
        const double h = height[static_cast<size_t>(index[k])];
        lowest(cx, cy) = std::min(lowest(cx, cy), h);
        highest(cx, cy) = std::max(highest(cx, cy), h);
        bar_height(cx, cy) |= h >= p.gate_min_top + 0.10;
        // Hanging needs the ground under it seen (column_low is -inf where it
        // was not); a far bucket whose foot hides behind a near one is not
        // hanging, it is merely half seen.  The last point in a cell decides.
        evidence(cx, cy) = column_low[static_cast<size_t>(index[k])] > p.foot_height;
      }
      Mask grounded(rows, cols, 0);
      Mask hanging(rows, cols, 0);
      for (size_t c = 0; c < lowest.cells.size(); ++c) {
        grounded.cells[c] = lowest.cells[c] <= p.foot_height;
        // Build the gate from its high bar, not its lower curtain. Streamers
        // can bow toward or away from the camera without thickening the bar.
        hanging.cells[c] =
            bar_height.cells[c] && !grounded.cells[c] && evidence.cells[c] && highest.cells[c] <= p.gate_max_top;
      }
      // Grounded runs no bigger than a post may be one.  (Longer ones are
      // walls: they may hold a gate up, but they are never part of it.)
      Mask post(rows, cols, 0);
      {
        const Grid<int64_t> runs = Components(grounded);
        struct Extent {
          int64_t i_lo, i_hi, j_lo, j_hi;
        };
        // Indexed by run label, which is the run's own first cell.
        std::vector<Extent> extents(runs.cells.size());
        for (int64_t i = 0; i < rows; ++i) {
          for (int64_t j = 0; j < cols; ++j) {
            const int64_t run = runs(i, j);
            if (run < 0) {
              continue;
            }
            Extent& e = extents[static_cast<size_t>(run)];
            if (run == i * cols + j) {
              e = {i, i, j, j};
            }
            e.i_lo = std::min(e.i_lo, i);
            e.i_hi = std::max(e.i_hi, i);
            e.j_lo = std::min(e.j_lo, j);
            e.j_hi = std::max(e.j_hi, j);
          }
        }
        std::vector<uint8_t> is_post(extents.size(), 0);
        for (size_t s = 0; s < extents.size(); ++s) {
          if (runs.cells[s] != static_cast<int64_t>(s)) {
            continue;
          }
          const Extent& e = extents[s];
          const double extent = std::max(static_cast<double>(e.i_hi) * g - static_cast<double>(e.i_lo) * g,
                                         static_cast<double>(e.j_hi) * g - static_cast<double>(e.j_lo) * g);
          is_post[s] = !(extent > p.wall_length) && extent <= p.post_size;
        }
        for (size_t c = 0; c < runs.cells.size(); ++c) {
          const int64_t run = runs.cells[c];
          if (run >= 0) {
            post.cells[c] = is_post[static_cast<size_t>(run)];
          }
        }
      }
      const Mask support = Dilate(grounded);
      Mask near_support = support;
      for (int k = 0; k < p.support_reach - 1; ++k) {
        near_support = Dilate(near_support);
      }

      const double edge = (p.hfov_deg / 2 - p.frame_margin_deg) * kPi / 180.0;
      // Close gaps of up to four cells between hanging cells before grouping
      // them -- but only across the line of sight.  Seen from inside the car
      // wash, the nearest curtain's bar is above the frame and what is left is
      // strips 51 mm wide with 76 mm between, and the arch beyond shows only
      // through those gaps, 0.15 m at a time.  Along the line of sight stereo
      // noise smears each arch by several centimetres, and bridging that way
      // welds five arches 0.457 m apart into one blob.
      Mask bridged = hanging;
      for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
          const double bearing = std::atan2(static_cast<double>(j + y0) + 0.5, static_cast<double>(i + x0) + 0.5);
          const bool across_is_y = std::abs(bearing) < kPi / 4;
          const int dx = across_is_y ? 0 : 1;
          const int dy = across_is_y ? 1 : 0;
          bool before = false;
          bool after = false;
          for (int k = 1; k <= 4; ++k) {
            before = before || hanging.Neighbor(i, j, k * dx, k * dy, 0);
            after = after || hanging.Neighbor(i, j, -k * dx, -k * dy, 0);
          }
          if (before && after) {
            bridged(i, j) = 1;
          }
        }
      }
      const Grid<int64_t> bridged_components = Components(bridged);
      // Hanging cells by component, components in label order.
      std::vector<int64_t> labels;
      for (size_t c = 0; c < hanging.cells.size(); ++c) {
        if (hanging.cells[c]) {
          labels.push_back(bridged_components.cells[c]);
        }
      }
      std::sort(labels.begin(), labels.end());
      labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
      std::vector<std::vector<int64_t>> components(labels.size());
      std::vector<int64_t> slot_of_cell(hanging.cells.size(), -1);
      for (size_t c = 0; c < hanging.cells.size(); ++c) {
        if (hanging.cells[c]) {
          const auto at = std::lower_bound(labels.begin(), labels.end(), bridged_components.cells[c]);
          const size_t slot = static_cast<size_t>(at - labels.begin());
          components[slot].push_back(static_cast<int64_t>(c));
          slot_of_cell[c] = static_cast<int64_t>(slot);
        }
      }
      // The candidate points in each component's hanging cells, and those
      // standing in post cells, gathered once rather than per component.
      std::vector<std::vector<size_t>> points_in(components.size());
      std::vector<size_t> post_points;
      {
        for (size_t k = 0; k < count; ++k) {
          const size_t c = static_cast<size_t>(cell_of[k]);
          if (slot_of_cell[c] >= 0) {
            points_in[static_cast<size_t>(slot_of_cell[c])].push_back(k);
          }
          if (post.cells[c]) {
            post_points.push_back(k);
          }
        }
      }

      std::vector<double> u, v;
      for (size_t slot = 0; slot < components.size(); ++slot) {
        const std::vector<int64_t>& cells = components[slot];
        const size_t m = cells.size();
        if (m < 3) {
          continue;
        }
        std::vector<double> xy(2 * m);
        double center[2] = {0.0, 0.0};
        for (size_t k = 0; k < m; ++k) {
          const int64_t i = cells[k] / cols;
          const int64_t j = cells[k] % cols;
          xy[2 * k] = (static_cast<double>(i + x0) + 0.5) * g;
          xy[2 * k + 1] = (static_cast<double>(j + y0) + 0.5) * g;
          center[0] += xy[2 * k];
          center[1] += xy[2 * k + 1];
        }
        center[0] /= static_cast<double>(m);
        center[1] /= static_cast<double>(m);
        double sxx = 0.0, sxy = 0.0, syy = 0.0;
        for (size_t k = 0; k < m; ++k) {
          const double ex = xy[2 * k] - center[0];
          const double ey = xy[2 * k + 1] - center[1];
          sxx += ex * ex;
          sxy += ex * ey;
          syy += ey * ey;
        }
        double axis[2], normal[2];
        Eigen2(sxx, sxy, syy, axis, normal);
        u.resize(m);
        v.resize(m);
        double u_lo = kInf, u_hi = -kInf, v_lo = kInf, v_hi = -kInf;
        for (size_t k = 0; k < m; ++k) {
          const double ex = xy[2 * k] - center[0];
          const double ey = xy[2 * k + 1] - center[1];
          u[k] = ex * axis[0] + ey * axis[1];
          v[k] = ex * normal[0] + ey * normal[1];
          u_lo = std::min(u_lo, u[k]);
          u_hi = std::max(u_hi, u[k]);
          v_lo = std::min(v_lo, v[k]);
          v_hi = std::max(v_hi, v[k]);
        }
        const double span = (u_hi - u_lo) + g;
        if (!(p.gate_min_span <= span && span <= p.gate_max_span)) {
          continue;
        }
        const double distance = std::hypot(center[0], center[1]);
        const double sigma = p.noise_a + p.noise_b * distance * distance;
        if ((v_hi - v_lo) + g > p.gate_max_thickness + 2 * sigma) {
          continue;
        }
        // The middle must hang over open floor.  A tapered bucket's rim
        // overhangs its body by a couple of centimeters and so hangs too, but
        // right beside something grounded all along its length.
        bool middle_supported = false;
        for (size_t k = 0; k < m && !middle_supported; ++k) {
          middle_supported = std::abs(u[k]) <= span / 4 && support.cells[static_cast<size_t>(cells[k])];
        }
        if (middle_supported) {
          continue;
        }
        // Each end must be held up: grounded cells beside its last cells, or
        // the frame's edge.
        enum End { kNone, kHeld, kCut };
        End ends[2];
        for (int side = 0; side < 2; ++side) {
          bool held = false;
          bool cut = false;
          for (size_t k = 0; k < m; ++k) {
            const bool at_end = side == 0 ? u[k] <= u_lo + g : u[k] >= u_hi - g;
            if (!at_end) {
              continue;
            }
            held = held || near_support.cells[static_cast<size_t>(cells[k])];
            cut = cut || std::abs(std::atan2(xy[2 * k + 1], xy[2 * k])) >= edge;
          }
          ends[side] = held ? kHeld : cut ? kCut : kNone;
        }
        if (ends[0] == kNone || ends[1] == kNone) {
          continue;
        }
        double top = -kInf;
        for (const size_t k : points_in[slot]) {
          top = std::max(top, height[static_cast<size_t>(index[k])]);
        }
        if (top < p.gate_min_top) {
          continue;
        }
        // Count low hanging returns across the span, allowing each flexible
        // streamer to move in plan independently of the rigid arch above it.
        // Count distinct across-span cells so a single folded strip cannot
        // masquerade as an entire curtain.
        const size_t bins = static_cast<size_t>(std::ceil(span / g));
        std::vector<uint8_t> curtain_bins(bins, 0);
        for (size_t k = 0; k < count; ++k) {
          const size_t at = static_cast<size_t>(index[k]);
          const double h = height[at];
          if (h <= p.foot_height || h >= p.clearance || column_low[at] <= p.foot_height) {
            continue;
          }
          const double rx = level[3 * at] - center[0];
          const double ry = level[3 * at + 1] - center[1];
          const double along = rx * axis[0] + ry * axis[1];
          const double across = rx * normal[0] + ry * normal[1];
          if (along < u_lo || along > u_hi || std::abs(across) > p.gate_max_thickness + 0.2) {
            continue;
          }
          const size_t b = std::min(static_cast<size_t>((along - u_lo) / g), bins - 1);
          curtain_bins[b] = 1;
        }
        const double curtain =
            static_cast<double>(std::count(curtain_bins.begin(), curtain_bins.end(), 1)) / static_cast<double>(bins);
        const uint8_t kind = span >= p.carwash_min_span || curtain >= p.curtain_fraction ? kCarWash : kHoop;
        // Unseen feet are only trusted for a curtain: two posts nobody saw do
        // not make a hoop.
        if ((ends[0] == kCut || ends[1] == kCut) && kind != kCarWash) {
          continue;
        }
        // The posts: grounded, no bigger than a post, near the hanging run.
        // The second growth ORs each direction into the running result, so
        // it reaches further along some diagonals than a plain 5x5 would.
        Mask grown(rows, cols, 0);
        for (const int64_t c : cells) {
          grown.cells[static_cast<size_t>(c)] = 1;
        }
        grown = Dilate(grown);
        for (const auto& d : kNeighbors) {
          const Mask before = grown;
          for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
              if (before.Neighbor(i, j, d[0], d[1], 0)) {
                grown(i, j) = 1;
              }
            }
          }
        }
        FoundGate found;
        std::vector<size_t> members = points_in[slot];
        for (const size_t k : post_points) {
          if (grown.cells[static_cast<size_t>(cell_of[k])]) {
            members.push_back(k);
            found.feet.push_back(index[k]);
          }
        }
        std::sort(members.begin(), members.end());
        members.erase(std::unique(members.begin(), members.end()), members.end());
        for (const size_t k : members) {
          found.members.push_back(index[k]);
        }
        Gate& gate = found.gate;
        gate.kind = kind;
        gate.center[0] = center[0];
        gate.center[1] = center[1];
        for (int a = 0; a < 2; ++a) {
          gate.feet[0][a] = center[a] + axis[a] * u_lo;
          gate.feet[1][a] = center[a] + axis[a] * u_hi;
          gate.axis[a] = axis[a];
        }
        gate.span = span;
        gate.top = top;
        gates.push_back(std::move(found));
      }
      return gates;
    }

    bool Claimable(uint8_t label) {
      return label == kObstacle || label == kOverhead;
    }

    // The real wash uses lemon-yellow party streamers. Their color survives
    // bends and tangles that make a point-cloud return look like a solid wall.
    // Keep the test conservative: Gazebo shades straw bales to brown pixels
    // with green/red near 0.82, while the yellow ribbons stay near 0.98.
    // Washed-out colors also need enough chroma.
    bool IsStreamerYellow(uint32_t rgb) {
      const int red = (rgb >> 16) & 0xff;
      const int green = (rgb >> 8) & 0xff;
      const int blue = rgb & 0xff;
      return red >= 50 && green >= 50 && 10 * green >= 9 * red && 5 * red >= 4 * green && green - blue >= 35 &&
             4 * (green - blue) >= green;
    }

    // The rest of a gate's bar, in its own thin footprint.
    //
    // The hanging run stops short of each post: the post hides the floor under
    // the bar beside it, so there is no evidence that stretch hangs.  Inside
    // the span everything above the ground belongs to the gate; past the
    // span's ends only what is higher than any bale does, because the walls a
    // gate may lean on stand there.  None of it blocks -- the posts were
    // claimed, and made blocking, with the gate itself.
    void ClaimFootprint(const Gate& gate, const std::vector<int64_t>& candidates, const Params& p, Segmentation* out) {
      const double distance = std::hypot(gate.center[0], gate.center[1]);
      const double sigma = p.noise_a + p.noise_b * distance * distance;
      const double half = gate.span / 2;
      const double beyond_limit = half + p.support_reach * p.gate_cell;
      const double thickness = p.gate_max_thickness / 2 + 2 * sigma;
      for (const int64_t at : candidates) {
        const size_t k = static_cast<size_t>(at);
        if (!Claimable(out->labels[k])) {
          continue;
        }
        const double rx = out->level[3 * k] - gate.center[0];
        const double ry = out->level[3 * k + 1] - gate.center[1];
        const double along = std::abs(rx * gate.axis[0] + ry * gate.axis[1]);
        const double across = std::abs(rx * -gate.axis[1] + ry * gate.axis[0]);
        const double h = out->height[k];
        const bool inside = along <= half;
        const bool beyond = !inside && along <= beyond_limit;
        if (across <= thickness && h <= p.gate_max_top && (inside || (beyond && h >= p.gate_min_top))) {
          out->labels[k] = gate.kind;
          out->blocking[k] = 0;
        }
      }
    }

    // Label the rest of a car wash once one of its arches is found.
    //
    // The arches behind the first are seen through its ribbons, in pieces,
    // and a piece has no feet to be recognized by.  But a car wash is a row
    // of arches carwash_depth deep, so what hangs inside that footprint --
    // above the ground, not standing on it -- is more of the same.  Anything
    // that does stand on the ground inside it keeps its own label.
    void ClaimCarWash(const Gate& gate,
                      const std::vector<int64_t>& valid,
                      const std::vector<double>& column_low,
                      const std::vector<uint8_t>& beside_ground,
                      const uint32_t* rgb,
                      const Params& p,
                      Segmentation* out) {
      const double half = gate.span / 2 + p.gate_cell;
      for (const int64_t at : valid) {
        const size_t k = static_cast<size_t>(at);
        const bool yellow = rgb && IsStreamerYellow(rgb[k]);
        // A bent ribbon may have been mistaken for a gate foot earlier in
        // this frame. Its yellow return must not keep that foot's blocking bit.
        if (!Claimable(out->labels[k]) && !(yellow && out->labels[k] == kCarWash)) {
          continue;
        }
        const double rx = out->level[3 * k] - gate.center[0];
        const double ry = out->level[3 * k + 1] - gate.center[1];
        const double along = rx * gate.axis[0] + ry * gate.axis[1];
        const double across = rx * -gate.axis[1] + ry * gate.axis[0];
        // Geometry alone keeps anything standing on the ground blocking.
        // Registered yellow is allowed farther sideways, because separately
        // attached, thin streamers can swing and tangle across the span.
        const bool hanging =
            std::abs(along) <= half + p.gate_cell && column_low[k] > p.foot_height && !beside_ground[k];
        const bool streamer_color =
            yellow && std::abs(along) <= gate.span / 2 + 0.425 && out->height[k] > p.ground_tolerance;
        if (std::abs(across) <= p.carwash_depth && out->height[k] <= p.gate_max_top && (hanging || streamer_color)) {
          out->labels[k] = kCarWash;
          out->blocking[k] = 0;
        }
      }
    }

  }  // namespace

  Leveler::Leveler(double pitch, double roll) {
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll), sr = std::sin(roll);
    m_[0] = cp, m_[1] = sp * sr, m_[2] = sp * cr;
    m_[3] = 0.0, m_[4] = cr, m_[5] = -sr;
    m_[6] = -sp, m_[7] = cp * sr, m_[8] = cp * cr;
  }

  void Leveler::operator()(const double* in, double* out) const {
    const double x = in[0], y = in[1], z = in[2];
    out[0] = m_[0] * x + m_[1] * y + m_[2] * z;
    out[1] = m_[3] * x + m_[4] * y + m_[5] * z;
    out[2] = m_[6] * x + m_[7] * y + m_[8] * z;
  }

  void Segment(
      const double* xyz, size_t n, double pitch, double roll, const Params& p, Segmentation* out, const uint32_t* rgb) {
    out->labels.assign(n, kUnknown);
    out->height.assign(n, kNan);
    out->blocking.assign(n, 0);
    out->level.resize(3 * n);
    out->gates.clear();

    thread_local Workspace ws;
    const Leveler leveler(pitch, roll);
    std::vector<int64_t>& valid = ws.valid;
    std::vector<double>& plan_range = ws.plan_range;
    valid.clear();
    plan_range.clear();
    for (size_t k = 0; k < n; ++k) {
      double* level = &out->level[3 * k];
      leveler(&xyz[3 * k], level);
      const bool finite = std::isfinite(level[0]) && std::isfinite(level[1]) && std::isfinite(level[2]);
      const double plan = std::sqrt(level[0] * level[0] + level[1] * level[1]);
      const double depth =
          std::sqrt(xyz[3 * k] * xyz[3 * k] + xyz[3 * k + 1] * xyz[3 * k + 1] + xyz[3 * k + 2] * xyz[3 * k + 2]);
      if (finite && plan <= p.max_range && level[0] > -0.2 && depth >= p.min_depth) {
        valid.push_back(static_cast<int64_t>(k));
        plan_range.push_back(plan);
      }
    }
    if (valid.empty()) {
      return;
    }
    const size_t nv = valid.size();

    // The plane the car's own wheels stand on, in the leveled frame: a least
    // squares fit of z = c0 + c1 x + c2 y through the four contacts.
    double plane[3];
    {
      double ata[3][3] = {{0}}, atb[3] = {0};
      for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
          const double wheel[3] = {sx * p.half_wheelbase - p.camera_forward, sy * p.half_track, -p.camera_height};
          double w[3];
          leveler(wheel, w);
          const double row[3] = {1.0, w[0], w[1]};
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              ata[a][b] += row[a] * row[b];
            }
            atb[a] += row[a] * w[2];
          }
        }
      }
      // Cramer's rule; the design is well conditioned (a rectangle of wheels).
      const auto det3 = [](const double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      };
      const double det = det3(ata);
      for (int col = 0; col < 3; ++col) {
        double m[3][3];
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            m[a][b] = b == col ? atb[a] : ata[a][b];
          }
        }
        plane[col] = det3(m) / det;
      }
    }

    const double cell = p.cell;
    const double x_lo = -0.5;
    const int64_t nx = static_cast<int64_t>(std::ceil((p.max_range - x_lo) / cell)) + 1;
    const int64_t ny = static_cast<int64_t>(std::ceil(2 * p.max_range / cell)) + 1;
    std::vector<double>&vx = ws.vx, &vy = ws.vy, &vz = ws.vz;
    std::vector<int64_t>&ci = ws.ci, &cj = ws.cj;
    for (auto* v : {&vx, &vy, &vz}) {
      v->resize(nv);
    }
    ci.resize(nv);
    cj.resize(nv);
    for (size_t k = 0; k < nv; ++k) {
      const double* level = &out->level[3 * static_cast<size_t>(valid[k])];
      vx[k] = level[0];
      vy[k] = level[1];
      vz[k] = level[2];
      ci[k] = std::clamp<int64_t>(static_cast<int64_t>((vx[k] - x_lo) / cell), 0, nx - 1);
      cj[k] = std::clamp<int64_t>(static_cast<int64_t>((vy[k] + p.max_range) / cell), 0, ny - 1);
    }

    // A low percentile of each cell's heights rather than its minimum: one
    // stereo outlier below the floor would otherwise set the cell's height,
    // and the next cell's genuine rise would then read as a step.  Taken over
    // the returns in the cell's bottom band only, so a cell that is mostly
    // bale face still reads the floor at the face's foot.
    Grid<double> zmin(nx, ny, kNan);
    {
      Grid<double> floor(nx, ny, kInf);
      for (size_t k = 0; k < nv; ++k) {
        floor(ci[k], cj[k]) = std::min(floor(ci[k], cj[k]), vz[k]);
      }
      // Bucket the band's heights by cell, then pick each cell's percentile.
      std::vector<int64_t>& start = ws.band_start;
      start.assign(static_cast<size_t>(nx * ny) + 1, 0);
      for (size_t k = 0; k < nv; ++k) {
        if (vz[k] <= floor(ci[k], cj[k]) + p.surface_band) {
          ++start[static_cast<size_t>(ci[k] * ny + cj[k]) + 1];
        }
      }
      for (size_t c = 1; c < start.size(); ++c) {
        start[c] += start[c - 1];
      }
      std::vector<double>& band = ws.band;
      band.resize(static_cast<size_t>(start.back()));
      std::vector<int64_t>& fill = ws.band_fill;
      fill.assign(start.begin(), start.end() - 1);
      for (size_t k = 0; k < nv; ++k) {
        if (vz[k] <= floor(ci[k], cj[k]) + p.surface_band) {
          band[static_cast<size_t>(fill[static_cast<size_t>(ci[k] * ny + cj[k])]++)] = vz[k];
        }
      }
      for (size_t c = 0; c + 1 < start.size(); ++c) {
        const int64_t count = start[c + 1] - start[c];
        if (count == 0) {
          continue;
        }
        const auto first = band.begin() + start[c];
        const int64_t pick = static_cast<int64_t>(std::floor(p.surface_percentile * static_cast<double>(count - 1)));
        std::nth_element(first, first + pick, first + count);
        zmin.cells[c] = *(first + pick);
      }
    }

    Grid<double> car_plane(nx, ny, 0.0);
    Grid<double> allowance(nx, ny, 0.0);
    Mask seeds(nx, ny, 0);
    for (int64_t i = 0; i < nx; ++i) {
      const double gx = x_lo + (static_cast<double>(i) + 0.5) * cell;
      // The car's own plane only holds under the car: pitched over the crest
      // of the ramp, it climbs 0.14 m per meter above the flat deck ahead.  So
      // the tolerance opens by the grade allowance with distance past the
      // front axle.
      const double ahead = std::max(gx - (p.half_wheelbase - p.camera_forward), 0.0);
      for (int64_t j = 0; j < ny; ++j) {
        const double gy = -p.max_range + (static_cast<double>(j) + 0.5) * cell;
        const double range = std::hypot(gx, gy);
        car_plane(i, j) = plane[0] + plane[1] * gx + plane[2] * gy;
        allowance(i, j) = p.gap_base + p.gap_per_range_sq * (range * range);
        const double z = zmin(i, j);
        seeds(i, j) = std::isfinite(z) && range <= p.seed_radius &&
                      std::abs(z - car_plane(i, j)) <= p.seed_tolerance + p.max_grade * ahead;
      }
    }
    Grid<double> surface;
    Mask known;
    GrowSurface(zmin, seeds, allowance, p, &surface, &known);
    Fill(&surface, &known, static_cast<int>(std::nearbyint(1.0 / cell)));
    for (size_t c = 0; c < surface.cells.size(); ++c) {
      if (!known.cells[c]) {
        surface.cells[c] = car_plane.cells[c];
      }
    }

    std::vector<double>& h = ws.h;
    std::vector<uint8_t>& is_ground = ws.is_ground;
    h.resize(nv);
    is_ground.resize(nv);
    for (size_t k = 0; k < nv; ++k) {
      h[k] = vz[k] - surface(ci[k], cj[k]);
      out->height[static_cast<size_t>(valid[k])] = h[k];
      is_ground[k] = h[k] <= p.ground_tolerance;
    }

    // Columns: the lowest non-ground return in each column of the finer gate
    // grid -- in a 0.1 m cell the tunnel's roof shares a column with the top
    // of its wall.  Overhead also needs evidence that the space under it is
    // open: ground seen in the column itself, or ground seen *beyond* it on
    // the same bearing -- a ray that reached the floor further out passed
    // under whatever is here.  Ground a fine cell away does not count: as
    // often as not it is the floor in front of whatever hides this column's
    // lower half (the start signal behind its bale wall, a bucket behind a
    // nearer one), or the one foot of it that does show.
    const double fine = p.gate_cell;
    std::vector<int64_t>&fi = ws.fi, &fj = ws.fj;
    fi.resize(nv);
    fj.resize(nv);
    int64_t fi_max = 0, fj_max = 0;
    for (size_t k = 0; k < nv; ++k) {
      fi[k] = static_cast<int64_t>(std::floor((vx[k] - x_lo) / fine));
      fj[k] = static_cast<int64_t>(std::floor((vy[k] + p.max_range) / fine));
      fi_max = std::max(fi_max, fi[k]);
      fj_max = std::max(fj_max, fj[k]);
    }
    Grid<double> fine_lowest(fi_max + 1, fj_max + 1, kInf);
    Grid<int64_t> fine_count(fi_max + 1, fj_max + 1, 0);
    Mask open_below(fi_max + 1, fj_max + 1, 0);
    for (size_t k = 0; k < nv; ++k) {
      if (is_ground[k]) {
        open_below(fi[k], fj[k]) = 1;
      } else {
        fine_lowest(fi[k], fj[k]) = std::min(fine_lowest(fi[k], fj[k]), h[k]);
        ++fine_count(fi[k], fj[k]);
      }
    }
    std::vector<uint8_t>& seen_under = ws.seen_under;
    SeenUnder(vx, vy, vz, plan_range, is_ground, p, &ws, &seen_under);
    for (size_t k = 0; k < nv; ++k) {
      if (!is_ground[k] && seen_under[k]) {
        open_below(fi[k], fj[k]) = 1;
      }
    }

    // Beside a grounded column: a tapered bucket's rim overhangs its body by
    // a couple of centimeters with floor showing under the lip, which is not
    // hanging in any sense a car wash cares about.
    Mask grounded_fine(fi_max + 1, fj_max + 1, 0);
    for (size_t c = 0; c < fine_lowest.cells.size(); ++c) {
      grounded_fine.cells[c] = fine_lowest.cells[c] <= p.foot_height;
    }
    const Mask beside = Dilate(grounded_fine);

    std::vector<int64_t>& candidates = ws.candidates;
    std::vector<double>& column_low = ws.column_low;
    std::vector<uint8_t>& beside_ground = ws.beside_ground;
    candidates.clear();
    column_low.assign(n, -kInf);
    beside_ground.assign(n, 0);
    for (size_t k = 0; k < nv; ++k) {
      const size_t at = static_cast<size_t>(valid[k]);
      const bool open = open_below(fi[k], fj[k]);
      const double low = fine_lowest(fi[k], fj[k]);
      const bool overhead = !is_ground[k] && low > p.clearance && open;
      // Speckle: a stray return, far off the surface it belongs to, standing
      // alone in its column.  Real structure puts several returns in a 5 cm
      // column even at the far end of the scan.  How many is several scales
      // with range: a face fills a 5 cm column with (0.05 / (range *
      // pixel_angle))^2 returns, some 200 at 1 m and 6 at 6 m.  Stereo error
      // scatters a far return anywhere along its ray, so a near column holding
      // a handful of returns is that, not an object.
      const double per_pixel = fine / (p.pixel_angle * std::max(plan_range[k], 0.1));
      const double needed =
          std::max(static_cast<double>(p.min_column_points), p.speckle_fill * (per_pixel * per_pixel));
      const bool speckle = !is_ground[k] && static_cast<double>(fine_count(fi[k], fj[k])) < needed;

      out->labels[at] = is_ground[k] ? kGround : speckle ? kUnknown : overhead ? kOverhead : kObstacle;
      out->blocking[at] = !is_ground[k] && !overhead && !speckle;
      if (!is_ground[k] && !speckle && h[k] <= 2.0) {
        candidates.push_back(valid[k]);
      }
      // What hangs, per point: the lowest return in its column is off the
      // ground, and the ground under it was actually seen.
      column_low[at] = open ? low : -kInf;
      beside_ground[at] = beside(fi[k], fj[k]);
    }

    for (FoundGate& found : FindGates(out->level, out->height, candidates, column_low, p)) {
      const Gate& gate = found.gate;
      // Right beside something grounded is part of that thing -- the top edge
      // of a bale wall the gate leans on -- unless it is the gate's own post.
      std::sort(found.feet.begin(), found.feet.end());
      for (const int64_t at : found.members) {
        const size_t k = static_cast<size_t>(at);
        if (!beside_ground[k] || std::binary_search(found.feet.begin(), found.feet.end(), at)) {
          out->labels[k] = gate.kind;
          out->blocking[k] = 0;
        }
      }
      for (const int64_t at : found.feet) {
        const size_t k = static_cast<size_t>(at);
        out->blocking[k] = out->height[k] <= p.clearance;
      }
      ClaimFootprint(gate, candidates, p, out);
      out->gates.push_back(gate);
      if (gate.kind == kCarWash) {
        ClaimCarWash(gate, valid, column_low, beside_ground, rgb, p, out);
      }
    }
  }

}  // namespace cfr_arduino_bridge::segmentation
