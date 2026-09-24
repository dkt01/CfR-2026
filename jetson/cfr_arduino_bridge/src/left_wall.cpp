#include "cfr_arduino_bridge/left_wall.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cfr_arduino_bridge::left_wall {

  namespace {

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kSectorMaxRange = 3.0;

    constexpr double Radians(double degrees) {
      return degrees * kPi / 180.0;
    }

    // The numbers in the array under `key` in a JSON document.  The path file
    // is flat arrays of numbers, so this is all of JSON it needs.
    std::vector<double> JsonArray(const std::string& text, const std::string& key) {
      const std::string quoted = "\"" + key + "\"";
      for (size_t at = text.find(quoted); at != std::string::npos; at = text.find(quoted, at + 1)) {
        size_t cursor = at + quoted.size();
        while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) {
          ++cursor;
        }
        if (cursor >= text.size() || text[cursor] != ':') {
          continue;  // a string value that happens to spell the key
        }
        const size_t open = text.find('[', cursor);
        const size_t close = text.find(']', open);
        if (open == std::string::npos || close == std::string::npos) {
          break;
        }
        std::vector<double> values;
        const char* p = text.c_str() + open + 1;
        const char* end = text.c_str() + close;
        while (p < end) {
          char* next = nullptr;
          const double value = std::strtod(p, &next);
          if (next == p) {
            ++p;  // separators and whitespace
            continue;
          }
          values.push_back(value);
          p = next;
        }
        return values;
      }
      throw std::runtime_error("no \"" + key + "\" array");
    }

    // Three floats (or six) out of a whitespace separated element text.
    std::vector<double> Numbers(const tinyxml2::XMLElement* element) {
      std::vector<double> values;
      if (element == nullptr || element->GetText() == nullptr) {
        return values;
      }
      std::istringstream stream(element->GetText());
      double value;
      while (stream >> value) {
        values.push_back(value);
      }
      return values;
    }

    const tinyxml2::XMLElement* FindModel(const tinyxml2::XMLElement* element, const char* name) {
      for (; element != nullptr; element = element->NextSiblingElement()) {
        if (std::string(element->Name()) == "model" && element->Attribute("name", name)) {
          return element;
        }
        if (const auto* found = FindModel(element->FirstChildElement(), name)) {
          return found;
        }
      }
      return nullptr;
    }

  }  // namespace

  void ScanCloud(const uint8_t* data, size_t size, const CloudLayout& layout, SectorScan* scan) {
    layout.Check(size);
    scan->points = 0;
    scan->near.clear();
    scan->far.clear();
    scan->front.clear();
    const float near_center = static_cast<float>(Radians(50));
    const float far_center = static_cast<float>(Radians(30));
    const float narrow = static_cast<float>(Radians(5));
    const float wide = static_cast<float>(Radians(15));
    ForEachPoint(data, layout, [&](size_t, float x, float y, float z) {
      // Heights are relative to the camera; this discards the floor below the
      // bale faces, and anything behind or right under the camera.
      if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x > 0.18F && z > -0.12F && z < 0.70F)) {
        return;
      }
      ++scan->points;
      const float range = std::hypot(x, y);
      if (!(range < static_cast<float>(kSectorMaxRange))) {
        return;  // in no sector
      }
      const float bearing = std::atan2(y, x);
      if (std::abs(bearing - near_center) < narrow) {
        scan->near.push_back(range);
      } else if (std::abs(bearing - far_center) < narrow) {
        scan->far.push_back(range);
      } else if (std::abs(bearing) < wide) {
        scan->front.push_back(range);
      }
    });
  }

  double SectorRange(std::vector<float>& ranges) {
    const size_t n = ranges.size();
    if (n < 5) {
      return kSectorMaxRange;
    }
    // numpy's default (linear) percentile, including its interpolation form.
    const double index = 0.10 * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const double t = index - static_cast<double>(lo);
    std::nth_element(ranges.begin(), ranges.begin() + static_cast<long>(lo), ranges.end());
    const double a = ranges[lo];
    const double b = lo + 1 < n ? *std::min_element(ranges.begin() + static_cast<long>(lo) + 1, ranges.end()) : a;
    const double diff = b - a;
    return t >= 0.5 ? b - diff * (1.0 - t) : a + diff * t;
  }

  Command CloudCommand(SectorScan& scan) {
    if (scan.points < 20) {
      return {};
    }
    const double near = SectorRange(scan.near);
    const double far = SectorRange(scan.far);
    const double front = SectorRange(scan.front);
    if (near >= kSectorMaxRange && far >= kSectorMaxRange) {
      return {0.35, 0.25};  // reacquire the left boundary
    }
    double side;
    double tangent = 0.0;
    if (near >= kSectorMaxRange) {
      side = far * std::sin(Radians(30));
    } else if (far >= kSectorMaxRange) {
      side = near * std::sin(Radians(50));
    } else {
      side = near * std::sin(Radians(50));
      const double ahead = far * std::sin(Radians(30));
      const double forward_gap = far * std::cos(Radians(30)) - near * std::cos(Radians(50));
      tangent = std::atan2(ahead - side, std::max(0.25, forward_gap));
    }
    double steer = 0.8 * tangent + 0.65 * (side - 0.50);
    if (front < 1.1) {
      steer += 0.9 * (1.1 - front);  // oval bends left at its ends
    }
    steer = std::clamp(steer, -0.40, 0.40);
    const double speed = front < 1.1 || std::abs(steer) > 0.3 ? 0.35 : 0.60;
    return {speed, steer};
  }

  std::vector<Bale> LoadBales(const std::string& sdf_path) {
    // The generated world's comments contain "--", which strict XML parsers
    // reject.  tinyxml2 reads a comment to its closing marker and does not
    // look inside, so Gazebo's leniency and this agree.
    tinyxml2::XMLDocument document;
    if (document.LoadFile(sdf_path.c_str()) != tinyxml2::XML_SUCCESS) {
      throw std::runtime_error("cannot read " + sdf_path + ": " + document.ErrorStr());
    }
    const tinyxml2::XMLElement* model = FindModel(document.RootElement(), "course_bales");
    const tinyxml2::XMLElement* link = nullptr;
    if (model != nullptr) {
      for (link = model->FirstChildElement("link"); link != nullptr; link = link->NextSiblingElement("link")) {
        if (link->Attribute("name", "bales")) {
          break;
        }
      }
    }
    if (link == nullptr) {
      throw std::runtime_error("course_bales missing from " + sdf_path);
    }
    std::vector<Bale> bales;
    for (const auto* collision = link->FirstChildElement("collision"); collision != nullptr;
         collision = collision->NextSiblingElement("collision")) {
      const char* name = collision->Attribute("name");
      if (name == nullptr || std::string(name).rfind("bale_", 0) != 0) {
        continue;
      }
      const auto pose = Numbers(collision->FirstChildElement("pose"));
      const auto* geometry = collision->FirstChildElement("geometry");
      const auto* box = geometry != nullptr ? geometry->FirstChildElement("box") : nullptr;
      const auto size = Numbers(box != nullptr ? box->FirstChildElement("size") : nullptr);
      if (pose.size() != 6 || size.size() != 3) {
        throw std::runtime_error(std::string("malformed bale ") + name + " in " + sdf_path);
      }
      bales.push_back({pose[0], pose[1], pose[5], size[0] / 2, size[1] / 2});
    }
    return bales;
  }

  double RangeToBales(
      const std::vector<Bale>& bales, double x, double y, double yaw, double bearing, double max_range) {
    const double angle = yaw + bearing;
    const double dx = std::cos(angle), dy = std::sin(angle);
    double closest = max_range;
    for (const Bale& bale : bales) {
      if (std::hypot(bale.x - x, bale.y - y) > closest + std::hypot(bale.half_x, bale.half_y)) {
        continue;
      }
      const double c = std::cos(bale.yaw), s = std::sin(bale.yaw);
      // The ray in the bale's own frame, slab-tested against its box.
      const double origin[2] = {c * (x - bale.x) + s * (y - bale.y), -s * (x - bale.x) + c * (y - bale.y)};
      const double direction[2] = {c * dx + s * dy, -s * dx + c * dy};
      const double half[2] = {bale.half_x, bale.half_y};
      double near = 0.0, far = closest;
      bool hit = true;
      for (int axis = 0; axis < 2 && hit; ++axis) {
        if (std::abs(direction[axis]) < 1e-9) {
          hit = std::abs(origin[axis]) <= half[axis];
        } else {
          const double a = (-half[axis] - origin[axis]) / direction[axis];
          const double b = (half[axis] - origin[axis]) / direction[axis];
          near = std::max(near, std::min(a, b));
          far = std::min(far, std::max(a, b));
          hit = near <= far;
        }
      }
      if (hit && 0.01 < near && near < closest) {
        closest = near;
      }
    }
    return closest;
  }

  CourseFollower::CourseFollower(const std::string& path_file) {
    std::ifstream file(path_file);
    if (!file) {
      throw std::runtime_error("cannot read " + path_file);
    }
    std::stringstream text;
    text << file.rdbuf();
    const auto xs = JsonArray(text.str(), "x");
    const auto ys = JsonArray(text.str(), "y");
    if (xs.empty() || xs.size() != ys.size()) {
      throw std::runtime_error(path_file + " has no matching x and y arrays");
    }
    for (size_t i = xs.size(); i-- > 0;) {
      path_.emplace_back(xs[i], ys[i]);
    }
  }

  CourseFollower::CourseFollower(std::vector<std::pair<double, double>> path) : path_(std::move(path)) {
    if (path_.empty()) {
      throw std::invalid_argument("empty path");
    }
  }

  Command CourseFollower::Step(const std::vector<Bale>& bales, double x, double y, double yaw) {
    const long n = static_cast<long>(path_.size());
    const auto distance_to = [&](long i) { return std::hypot(path_[i].first - x, path_[i].second - y); };
    // Search the whole path once, then only a short window ahead, so a lap's
    // near approaches to itself cannot pull the index backwards.
    long best = index_ < 0 ? 0 : index_ % n;
    double best_distance = distance_to(best);
    const long span = index_ < 0 ? n : 16;
    const long first = index_ < 0 ? 0 : index_;
    for (long j = 0; j < span; ++j) {
      const long i = (first + j) % n;
      const double d = distance_to(i);
      if (d < best_distance) {
        best = i;
        best_distance = d;
      }
    }
    index_ = best;
    const auto& target = path_[static_cast<size_t>((index_ + 12) % n)];
    const double distance = std::hypot(target.first - x, target.second - y);
    const double heading = std::atan2(target.second - y, target.first - x);
    const double error = std::atan2(std::sin(heading - yaw), std::cos(heading - yaw));
    double steer = std::atan2(2 * kWheelbase * std::sin(error), std::max(distance, 0.1));
    // Keep an eye on the left boundary; a small bias corrects lateral drift
    // without letting a bale gap reverse the path direction.
    const double left = RangeToBales(bales, x, y, yaw, kPi / 2);
    if (left < 1.2) {
      steer += 0.05 * (left - 0.55);
    }
    steer = std::clamp(steer, -kMaxSteer, kMaxSteer);
    return {0.8, steer};
  }

}  // namespace cfr_arduino_bridge::left_wall
