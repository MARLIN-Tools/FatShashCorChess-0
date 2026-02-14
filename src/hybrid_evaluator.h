#pragma once

#include "evaluator.h"
#include "hce_evaluator.h"
#include "lc0/lc0_evaluator.h"

#include <cstddef>
#include <string>

namespace makaira {

class HybridEvaluator final : public IEvaluator {
   public:
    enum class Backend : int {
        HCE = 0,
        LC0_FP32 = 1,
        LC0_FP32_ASYNC = 2,
        LC0_INT8 = 3,
    };

    HybridEvaluator();

    void set_use_lc0(bool enabled);
    bool use_lc0() const { return use_lc0_; }

    void set_backend(Backend backend);
    Backend backend() const { return backend_; }
    void set_backend_from_int(int backend);
    int backend_as_int() const { return static_cast<int>(backend_); }

    bool load_lc0_weights(const std::string& path, bool strict_t1_shape = true);
    bool lc0_ready() const { return lc0_.is_ready(); }
    const std::string& lc0_last_error() const { return lc0_.last_error(); }
    const std::string& lc0_weights_path() const { return lc0_.weights_path(); }
    std::string lc0_backend_name() const { return lc0_.backend_name(); }

    void set_lc0_cp_scale(int cp_scale) { lc0_.set_cp_scale(cp_scale); }
    int lc0_cp_scale() const { return lc0_.cp_scale(); }

    void set_lc0_score_map(int score_map) { lc0_.set_score_map(score_map); }
    int lc0_score_map() const { return lc0_.score_map(); }

    void set_lc0_batch_max(int batch_max) { lc0_.set_batch_max(batch_max); }
    int lc0_batch_max() const { return lc0_.batch_max(); }
    void set_lc0_batch_wait_us(int wait_us) { lc0_.set_batch_wait_us(wait_us); }
    int lc0_batch_wait_us() const { return lc0_.batch_wait_us(); }
    void set_lc0_eval_threads(int threads) { lc0_.set_eval_threads(threads); }
    int lc0_eval_threads() const { return lc0_.eval_threads(); }
    void set_lc0_cache_entries(std::size_t entries) { lc0_.set_cache_limit(entries); }
    std::size_t lc0_cache_entries() const { return lc0_.cache_limit(); }
    void set_lc0_exec_backend(int backend) { lc0_.set_exec_backend(backend); }
    int lc0_exec_backend() const { return lc0_.exec_backend(); }
    std::string lc0_exec_backend_name() const { return lc0_.exec_backend_name(); }
    const std::string& lc0_exec_backend_error() const { return lc0_.exec_backend_error(); }
    bool lc0_eval_wdl(const Position& pos, float& w, float& d, float& l, int& cp) const {
        return lc0_.eval_wdl(pos, w, d, l, cp);
    }

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
    Backend backend_ = Backend::HCE;
    bool use_lc0_ = false;
};

}  // namespace makaira
