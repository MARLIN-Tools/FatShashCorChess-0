#include "lc0_evaluator.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

namespace makaira {

Lc0Evaluator::Lc0Evaluator() {
    eval_cache_.reserve(cache_limit_);
}

bool Lc0Evaluator::load_weights(const std::string& path, bool strict_t1_shape) {
    try {
        auto loaded = lc0::load_from_pb_gz(path);
        lc0::validate_attention_value_shapes(loaded, strict_t1_shape);

        weights_ = std::move(loaded);
        weights_path_ = path;
        last_error_.clear();
        ready_ = true;
        clear_cache();
        return true;
    } catch (const std::exception& e) {
        ready_ = false;
        last_error_ = e.what();
        return false;
    } catch (...) {
        ready_ = false;
        last_error_ = "unknown lc0 load error";
        return false;
    }
}

void Lc0Evaluator::set_cp_scale(int cp_scale) {
    cp_scale_ = std::clamp(cp_scale, 1, 2000);
}

void Lc0Evaluator::set_score_map(int score_map) {
    score_map_ = std::clamp(score_map, 0, 2);
}

int Lc0Evaluator::map_wdl_to_cp(const lc0::WdlOutput& wdl) const {
    const float p_w = std::clamp(wdl.win, 1e-6f, 1.0f - 1e-6f);
    const float p_d = std::clamp(wdl.draw, 1e-6f, 1.0f - 1e-6f);
    const float p_l = std::clamp(wdl.loss, 1e-6f, 1.0f - 1e-6f);
    const float expected = std::clamp(p_w - p_l, -0.999f, 0.999f);

    float cp = 0.0f;
    if (score_map_ == 0) {
        cp = static_cast<float>(cp_scale_) * expected;
    } else if (score_map_ == 1) {
        cp = static_cast<float>(cp_scale_) * static_cast<float>(std::atanh(expected));
    } else {
        // Logistic inverse using expected score in [0,1].
        const float score01 = std::clamp(p_w + 0.5f * p_d, 1e-5f, 1.0f - 1e-5f);
        cp = static_cast<float>(cp_scale_) * static_cast<float>(std::log(score01 / (1.0f - score01)));
    }

    if (!std::isfinite(cp)) {
        return 0;
    }

    constexpr int kMaxAbsEvalCp = 30000;
    return std::clamp(static_cast<int>(std::lround(cp)), -kMaxAbsEvalCp, kMaxAbsEvalCp);
}

int Lc0Evaluator::static_eval(const Position& pos) const {
    ++stats_.eval_calls;

    if (!ready_) {
        return 0;
    }

    const Key key = pos.key();
    if (auto it = eval_cache_.find(key); it != eval_cache_.end()) {
        return it->second;
    }

    const auto planes = lc0::extract_features_112(pos);
    const auto wdl = lc0::forward_attention_value(weights_, planes);
    const int cp = map_wdl_to_cp(wdl);

    if (eval_cache_.size() >= cache_limit_) {
        eval_cache_.clear();
    }
    eval_cache_.emplace(key, cp);
    return cp;
}

int Lc0Evaluator::static_eval_trace(const Position& pos, EvalBreakdown* out) const {
    const int score = static_eval(pos);
    if (out) {
        *out = EvalBreakdown{};
        out->total_white_pov = pos.side_to_move() == WHITE ? score : -score;
    }
    return score;
}

EvalStats Lc0Evaluator::stats() const {
    return stats_;
}

void Lc0Evaluator::clear_stats() {
    stats_ = EvalStats{};
}

void Lc0Evaluator::clear_cache() const {
    eval_cache_.clear();
}

void Lc0Evaluator::set_cache_limit(std::size_t entries) {
    cache_limit_ = std::max<std::size_t>(entries, 1024);
    eval_cache_.reserve(cache_limit_);
}

}  // namespace makaira
