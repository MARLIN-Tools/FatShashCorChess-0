#pragma once

#include "position.h"

#include <array>

namespace makaira::lc0 {

constexpr int k_input_planes = 112;
constexpr int k_squares = 64;

using InputPlanes112 = std::array<float, k_input_planes * k_squares>;

InputPlanes112 extract_features_112(const Position& pos);

}  // namespace makaira::lc0
