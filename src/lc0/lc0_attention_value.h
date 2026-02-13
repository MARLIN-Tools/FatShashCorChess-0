#pragma once

#include "lc0_features112.h"
#include "lc0_weights.h"

namespace makaira::lc0 {

struct WdlOutput {
    float win = 0.0f;
    float draw = 0.0f;
    float loss = 0.0f;
};

WdlOutput forward_attention_value(const Weights& w, const InputPlanes112& input);

}  // namespace makaira::lc0
