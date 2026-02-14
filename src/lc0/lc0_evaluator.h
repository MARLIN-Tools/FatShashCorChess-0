#pragma once

#include "../evaluator.h"
#include "lc0_attention_value.h"
#include "lc0_features112.h"
#include "lc0_linear_backend.h"
#include "lc0_weights.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace makaira {

class Lc0Evaluator final : public IEvaluator {
   public:
    enum class Backend : int {
        FP32_SYNC = 1,
        FP32_ASYNC = 2,
        INT8_PLACEHOLDER = 3,
    };

    Lc0Evaluator();
    ~Lc0Evaluator();

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
    bool eval_wdl(const Position& pos, float& w, float& d, float& l, int& cp) const;
    EvalStats stats() const override;
    void clear_stats() override;

    void clear_cache() const;
    void set_cache_limit(std::size_t entries);
    std::size_t cache_limit() const { return cache_limit_; }

    void set_backend(Backend backend);
    Backend backend() const { return backend_; }
    int backend_as_int() const { return static_cast<int>(backend_); }
    void set_backend_from_int(int backend);

    void set_batch_max(int batch_max);
    int batch_max() const { return batch_max_; }
    void set_batch_wait_us(int batch_wait_us);
    int batch_wait_us() const { return batch_wait_us_; }
    void set_eval_threads(int threads);
    int eval_threads() const { return eval_threads_; }
    void set_exec_backend(int backend);
    int exec_backend() const { return linear_backend_.type_as_int(); }
    std::string exec_backend_name() const { return linear_backend_.type_name(); }
    const std::string& exec_backend_error() const { return linear_backend_.last_error(); }

    std::string backend_name() const;

   private:
    struct CacheEntry {
        float w = 0.0f;
        float d = 0.0f;
        float l = 0.0f;
        int cp = 0;
    };

    struct EvalRequest {
        Key key = 0;
        lc0::InputPlanes112 planes{};
        std::promise<CacheEntry> promise{};
        std::chrono::steady_clock::time_point enqueued{};
    };

    int map_wdl_to_cp(const lc0::WdlOutput& wdl) const;
    CacheEntry run_forward_entry(const lc0::InputPlanes112& planes) const;
    CacheEntry evaluate_sync(Key key, const lc0::InputPlanes112& planes) const;
    CacheEntry evaluate_async(Key key, const lc0::InputPlanes112& planes) const;
    bool probe_cache(Key key, CacheEntry& out) const;
    void store_cache(Key key, const CacheEntry& entry) const;

    void restart_workers();
    void stop_workers();
    void worker_loop();

    lc0::Weights weights_{};
    bool ready_ = false;
    std::string weights_path_{};
    std::string last_error_{};
    int cp_scale_ = 220;
    int score_map_ = 1;
    std::size_t cache_limit_ = 1u << 18;
    mutable std::unordered_map<Key, CacheEntry> eval_cache_{};
    mutable std::mutex cache_mutex_{};
    mutable EvalStats stats_{};
    mutable lc0::LinearBackend linear_backend_{};
    mutable std::mutex stats_mutex_{};

    Backend backend_ = Backend::FP32_SYNC;
    int batch_max_ = 16;
    int batch_wait_us_ = 1000;
    int eval_threads_ = 1;
    mutable bool stop_workers_ = false;
    mutable std::mutex queue_mutex_{};
    mutable std::condition_variable queue_cv_{};
    mutable std::deque<std::unique_ptr<EvalRequest>> queue_{};
    mutable std::vector<std::thread> workers_{};
};

}  // namespace makaira
