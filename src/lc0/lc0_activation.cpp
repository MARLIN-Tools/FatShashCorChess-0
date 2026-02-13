#include "lc0_activation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace makaira::lc0 {

namespace {

float mish(float v) {
    const float e = std::exp(v);
    const float n = e * e + 2.0f * e;
    const float d = v / (n + 2.0f);
    if (v <= -0.125f) {
        return n * d;
    }
    return v - 2.0f * d;
}

float selu(float v) {
    constexpr float alpha = 1.67326324f;
    constexpr float scale = 1.05070098f;
    return v > 0.0f ? scale * v : scale * alpha * (std::exp(v) - 1.0f);
}

}  // namespace

float activate_scalar(float x, Activation a) {
    switch (a) {
        case Activation::RELU: return x > 0.0f ? x : 0.0f;
        case Activation::RELU2: return x > 0.0f ? x * x : 0.0f;
        case Activation::MISH: return mish(x);
        case Activation::TANH: return std::tanh(x);
        case Activation::SIGMOID: return 1.0f / (1.0f + std::exp(-x));
        case Activation::SELU: return selu(x);
        case Activation::SWISH: return x / (1.0f + std::exp(-x));
        case Activation::SOFTMAX:
        case Activation::NONE:
        case Activation::DEFAULT:
        default:
            return x;
    }
}

void softmax_inplace(float* begin, std::size_t n) {
    if (n == 0) {
        return;
    }
    float max_v = begin[0];
    for (std::size_t i = 1; i < n; ++i) {
        max_v = std::max(max_v, begin[i]);
    }

    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        begin[i] = std::exp(begin[i] - max_v);
        sum += begin[i];
    }
    if (sum <= std::numeric_limits<float>::min()) {
        const float uniform = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            begin[i] = uniform;
        }
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        begin[i] /= sum;
    }
}

void apply_activation(std::vector<float>& data, Activation a) {
    if (a == Activation::NONE || a == Activation::DEFAULT) {
        return;
    }
    if (a == Activation::SOFTMAX) {
        softmax_inplace(data.data(), data.size());
        return;
    }
    for (float& v : data) {
        v = activate_scalar(v, a);
    }
}

}  // namespace makaira::lc0
