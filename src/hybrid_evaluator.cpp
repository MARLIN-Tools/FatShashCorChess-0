#include "hybrid_evaluator.h"

namespace makaira {

HybridEvaluator::HybridEvaluator() = default;

void HybridEvaluator::set_use_lc0(bool enabled) {
    use_lc0_ = enabled;
    if (!use_lc0_) {
        backend_ = Backend::HCE;
    } else if (backend_ == Backend::HCE) {
        backend_ = Backend::LC0_FP32;
    }
}

void HybridEvaluator::set_backend(Backend backend) {
    backend_ = backend;
    if (backend_ == Backend::HCE) {
        use_lc0_ = false;
        return;
    }

    use_lc0_ = true;
    if (backend_ == Backend::LC0_FP32) {
        lc0_.set_backend(Lc0Evaluator::Backend::FP32_SYNC);
    } else if (backend_ == Backend::LC0_FP32_ASYNC) {
        lc0_.set_backend(Lc0Evaluator::Backend::FP32_ASYNC);
    } else {
        lc0_.set_backend(Lc0Evaluator::Backend::INT8_PLACEHOLDER);
    }
}

void HybridEvaluator::set_backend_from_int(int backend) {
    if (backend <= 0) {
        set_backend(Backend::HCE);
    } else if (backend == 1) {
        set_backend(Backend::LC0_FP32);
    } else if (backend == 2) {
        set_backend(Backend::LC0_FP32_ASYNC);
    } else {
        set_backend(Backend::LC0_INT8);
    }
}

bool HybridEvaluator::load_lc0_weights(const std::string& path, bool strict_t1_shape) {
    return lc0_.load_weights(path, strict_t1_shape);
}

const IEvaluator& HybridEvaluator::active() const {
    if (backend_ != Backend::HCE && use_lc0_ && lc0_.is_ready()) {
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
