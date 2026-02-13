#pragma once

#include <array>

namespace makaira::lc0 {

extern const std::array<int, 64 * 64 + 8 * 24> k_attn_policy_map;
extern const std::array<std::array<float, 64>, 64> k_pos_encoding;

}  // namespace makaira::lc0
