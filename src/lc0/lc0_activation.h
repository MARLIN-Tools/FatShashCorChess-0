#pragma once

#include "lc0_weights.h"

#include <cstddef>
#include <vector>

namespace makaira::lc0 {

float activate_scalar(float x, Activation a);
void softmax_inplace(float* begin, std::size_t n);

void apply_activation(std::vector<float>& data, Activation a);

}  // namespace makaira::lc0
