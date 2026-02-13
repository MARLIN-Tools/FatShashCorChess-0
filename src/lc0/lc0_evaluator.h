#pragma once

#include "../evaluator.h"
#include "lc0_attention_value.h"
#include "lc0_features112.h"
#include "lc0_weights.h"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace makaira {

class Lc0Evaluator final : public IEvaluator {
   public:
    Lc0Evaluator();

    bool load_weights(const std::string& path, bool strict_t1_shape = true);
    bool is_ready() const { return ready_; }
    const std::string& last_error() const { return last_error_; }
    const std::string& weights_path() const { return weights_path_; }

    void set_cp_scale(int cp_scale);
    int cp_scale() const { return cp_scale_; }

    void set_score_map(int score_map);
    int score_map() const { return score_map_; }

    int static_eval(const Position& pos) const override;
    int static_eval_trace(const Position& pos, EvalBreakdown* out) const override;
    EvalStats stats() const override;
    void clear_stats() override;

    void clear_cache() const;
    void set_cache_limit(std::size_t entries);

   private:
    int map_wdl_to_cp(const lc0::WdlOutput& wdl) const;

    lc0::Weights weights_{};
    bool ready_ = false;
    std::string weights_path_{};
    std::string last_error_{};
    int cp_scale_ = 220;
    int score_map_ = 1;
    std::size_t cache_limit_ = 1u << 18;
    mutable std::unordered_map<Key, int> eval_cache_{};
    mutable EvalStats stats_{};
};

}  // namespace makaira

