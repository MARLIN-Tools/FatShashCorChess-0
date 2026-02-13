#include "lc0_tables.h"

#include <array>

namespace makaira::lc0 {

namespace {
#include "lc0_tables_data.inc"
}

const std::array<int, 64 * 64 + 8 * 24> k_attn_policy_map = kAttnPolicyMap;
const std::array<std::array<float, 64>, 64> k_pos_encoding = kPosEncoding;

}  // namespace makaira::lc0
