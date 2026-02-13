#include "hybrid_evaluator.h"

namespace makaira {

HybridEvaluator::HybridEvaluator() = default;

void HybridEvaluator::set_use_lc0(bool enabled) {
    use_lc0_ = enabled;
}

bool HybridEvaluator::load_lc0_weights(const std::string& path, bool strict_t1_shape) {
    return lc0_.load_weights(path, strict_t1_shape);
}

const IEvaluator& HybridEvaluator::active() const {
    if (use_lc0_ && lc0_.is_ready()) {
        return lc0_;
    }
    return hce_;
}

int HybridEvaluator::static_eval(const Position& pos) const {
    return active().static_eval(pos);
}

int HybridEvaluator::static_eval_trace(const Position& pos, EvalBreakdown* out) const {
    return active().static_eval_trace(pos, out);
}

EvalStats HybridEvaluator::stats() const {
    return active().stats();
}

void HybridEvaluator::clear_stats() {
    hce_.clear_stats();
    lc0_.clear_stats();
}

bool HybridEvaluator::requires_move_hooks() const {
    return active().requires_move_hooks();
}

void HybridEvaluator::on_make_move(const Position& pos, Move move) const {
    active().on_make_move(pos, move);
}

void HybridEvaluator::on_unmake_move(const Position& pos, Move move) const {
    active().on_unmake_move(pos, move);
}

}  // namespace makaira

