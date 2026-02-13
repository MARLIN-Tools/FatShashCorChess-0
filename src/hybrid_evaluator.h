#pragma once

#include "evaluator.h"
#include "hce_evaluator.h"
#include "lc0/lc0_evaluator.h"

#include <string>

namespace makaira {

class HybridEvaluator final : public IEvaluator {
   public:
    HybridEvaluator();

    void set_use_lc0(bool enabled);
    bool use_lc0() const { return use_lc0_; }

    bool load_lc0_weights(const std::string& path, bool strict_t1_shape = true);
    bool lc0_ready() const { return lc0_.is_ready(); }
    const std::string& lc0_last_error() const { return lc0_.last_error(); }
    const std::string& lc0_weights_path() const { return lc0_.weights_path(); }

    void set_lc0_cp_scale(int cp_scale) { lc0_.set_cp_scale(cp_scale); }
    int lc0_cp_scale() const { return lc0_.cp_scale(); }

    void set_lc0_score_map(int score_map) { lc0_.set_score_map(score_map); }
    int lc0_score_map() const { return lc0_.score_map(); }

    int static_eval(const Position& pos) const override;
    int static_eval_trace(const Position& pos, EvalBreakdown* out) const override;
    EvalStats stats() const override;
    void clear_stats() override;
    bool requires_move_hooks() const override;
    void on_make_move(const Position& pos, Move move) const override;
    void on_unmake_move(const Position& pos, Move move) const override;

   private:
    const IEvaluator& active() const;

    HCEEvaluator hce_{};
    Lc0Evaluator lc0_{};
    bool use_lc0_ = false;
};

}  // namespace makaira

