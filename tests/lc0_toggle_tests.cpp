#include "bitboard.h"
#include "hybrid_evaluator.h"
#include "position.h"
#include "zobrist.h"

#include <filesystem>
#include <iostream>

int main() {
    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::Position pos;
    if (!pos.set_startpos()) {
        std::cerr << "[FAIL] could not set start position\n";
        return 1;
    }

    makaira::HybridEvaluator eval;
    const int hce_score = eval.static_eval(pos);

    eval.set_use_lc0(false);
    if (eval.static_eval(pos) != hce_score) {
        std::cerr << "[FAIL] HCE path changed when lc0 disabled\n";
        return 1;
    }

    const std::string path = "t1-256x10-distilled-swa-2432500.pb.gz";
    if (!std::filesystem::exists(path)) {
        std::cout << "[SKIP] lc0 weights file not found: " << path << "\n";
        return 0;
    }

    if (!eval.load_lc0_weights(path, true)) {
        std::cerr << "[FAIL] could not load lc0 weights: " << eval.lc0_last_error() << "\n";
        return 1;
    }

    eval.set_lc0_cp_scale(220);
    eval.set_lc0_score_map(1);
    eval.set_lc0_eval_threads(2);
    eval.set_lc0_batch_max(8);
    eval.set_lc0_batch_wait_us(500);
    eval.set_backend(makaira::HybridEvaluator::Backend::LC0_FP32_ASYNC);

    const int nn_score = eval.static_eval(pos);
    if (nn_score < -30000 || nn_score > 30000) {
        std::cerr << "[FAIL] nn score out of bounds\n";
        return 1;
    }

    std::cout << "[PASS] lc0 toggle/hybrid evaluator\n";
    return 0;
}
