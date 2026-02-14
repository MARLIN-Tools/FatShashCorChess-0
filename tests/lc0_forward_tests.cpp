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

namespace {

std::uint64_t plane_mask(const makaira::lc0::InputPlanes112& planes, int plane) {
    std::uint64_t mask = 0ULL;
    const int base = plane * 64;
    for (int sq = 0; sq < 64; ++sq) {
        if (planes[static_cast<std::size_t>(base + sq)] > 0.5f) {
            mask |= (1ULL << sq);
        }
    }
    return mask;
}

std::uint64_t rank_mask(int rank_zero_based) {
    return 0xFFULL << (rank_zero_based * 8);
}

}  // namespace

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

    // INPUT_CLASSICAL_112_PLANE should be oriented to side-to-move perspective.
    makaira::Position black_to_move;
    if (!black_to_move.set_from_fen("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1")) {
        std::cerr << "[FAIL] could not set black-to-move test position\n";
        return 1;
    }
    const auto oriented = makaira::lc0::extract_features_112(black_to_move);
    const std::uint64_t ours_pawns = plane_mask(oriented, 0);
    const std::uint64_t theirs_pawns = plane_mask(oriented, 6);
    if (ours_pawns != rank_mask(1) || theirs_pawns != rank_mask(6)) {
        std::cerr << "[FAIL] lc0 feature orientation mismatch for black-to-move position\n";
        return 1;
    }
    if (plane_mask(oriented, 108) != 0xFFFFFFFFFFFFFFFFULL) {
        std::cerr << "[FAIL] lc0 side-to-move aux plane mismatch\n";
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
