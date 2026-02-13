#include "bitboard.h"
#include "lc0/lc0_attention_value.h"
#include "lc0/lc0_features112.h"
#include "lc0/lc0_weights.h"
#include "position.h"
#include "zobrist.h"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>

int main() {
    const std::string path = "t1-256x10-distilled-swa-2432500.pb.gz";
    if (!std::filesystem::exists(path)) {
        std::cout << "[SKIP] lc0 weights file not found: " << path << "\n";
        return 0;
    }

    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::Position pos;
    if (!pos.set_startpos()) {
        std::cerr << "[FAIL] could not set start position\n";
        return 1;
    }

    try {
        const auto w = makaira::lc0::load_from_pb_gz(path);
        makaira::lc0::validate_attention_value_shapes(w, true);

        const auto planes = makaira::lc0::extract_features_112(pos);
        const auto out1 = makaira::lc0::forward_attention_value(w, planes);
        const auto out2 = makaira::lc0::forward_attention_value(w, planes);

        const float sum = out1.win + out1.draw + out1.loss;
        if (std::abs(sum - 1.0f) > 1e-3f) {
            std::cerr << "[FAIL] WDL softmax sum invalid: " << sum << "\n";
            return 1;
        }

        const auto same = [](float a, float b) { return std::abs(a - b) < 1e-7f; };
        if (!same(out1.win, out2.win) || !same(out1.draw, out2.draw) || !same(out1.loss, out2.loss)) {
            std::cerr << "[FAIL] forward pass is non-deterministic\n";
            return 1;
        }

        std::cout << "[PASS] lc0 forward deterministic WDL\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] lc0 forward exception: " << e.what() << "\n";
        return 1;
    }
}
