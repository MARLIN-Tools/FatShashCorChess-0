#include "lc0_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>

namespace makaira {
namespace {

void record_latency_stats(EvalStats& stats, std::uint64_t latency_us) {
    ++stats.nn_eval_latency_samples;
    stats.nn_eval_latency_us += latency_us;
    stats.nn_eval_latency_max_us = std::max(stats.nn_eval_latency_max_us, latency_us);
    if (latency_us <= 250) {
        ++stats.nn_eval_latency_le_250us;
    } else if (latency_us <= 500) {
        ++stats.nn_eval_latency_le_500us;
    } else if (latency_us <= 1000) {
        ++stats.nn_eval_latency_le_1000us;
    } else if (latency_us <= 2000) {
        ++stats.nn_eval_latency_le_2000us;
    } else if (latency_us <= 5000) {
        ++stats.nn_eval_latency_le_5000us;
    } else {
        ++stats.nn_eval_latency_gt_5000us;
    }
}

}  // namespace

Lc0Evaluator::Lc0Evaluator() {
    eval_cache_.reserve(cache_limit_);
}

Lc0Evaluator::~Lc0Evaluator() {
    stop_workers();
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
        restart_workers();
        return true;
    } catch (const std::exception& e) {
        ready_ = false;
        last_error_ = e.what();
        stop_workers();
        return false;
    } catch (...) {
        ready_ = false;
        last_error_ = "unknown lc0 load error";
        stop_workers();
        return false;
    }
}

void Lc0Evaluator::set_cp_scale(int cp_scale) {
    cp_scale_ = std::clamp(cp_scale, 1, 2000);
}

void Lc0Evaluator::set_score_map(int score_map) {
    score_map_ = std::clamp(score_map, 0, 3);
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
    } else if (score_map_ == 2) {
        // Logistic inverse using expected score in [0,1].
        const float score01 = std::clamp(p_w + 0.5f * p_d, 1e-5f, 1.0f - 1e-5f);
        cp = static_cast<float>(cp_scale_) * static_cast<float>(std::log(score01 / (1.0f - score01)));
    } else {
        // lc0 "centipawn" style conversion from W-L value.
        // Reference: lc0 classic search score output path.
        cp = 90.0f * static_cast<float>(std::tan(1.5637541897f * expected));
    }

    if (!std::isfinite(cp)) {
        return 0;
    }

    constexpr int kMaxAbsEvalCp = 30000;
    return std::clamp(static_cast<int>(std::lround(cp)), -kMaxAbsEvalCp, kMaxAbsEvalCp);
}

Lc0Evaluator::CacheEntry Lc0Evaluator::run_forward_entry(const lc0::InputPlanes112& planes) const {
    const auto wdl = lc0::forward_attention_value(weights_, planes, &linear_backend_);

    CacheEntry entry{};
    entry.w = wdl.win;
    entry.d = wdl.draw;
    entry.l = wdl.loss;
    entry.cp = map_wdl_to_cp(wdl);
    return entry;
}

bool Lc0Evaluator::probe_cache(Key key, CacheEntry& out) const {
    std::scoped_lock lock(cache_mutex_);
    const auto it = eval_cache_.find(key);
    if (it == eval_cache_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void Lc0Evaluator::store_cache(Key key, const CacheEntry& entry) const {
    std::scoped_lock lock(cache_mutex_);
    if (eval_cache_.size() >= cache_limit_) {
        eval_cache_.clear();
    }
    eval_cache_[key] = entry;
}

Lc0Evaluator::CacheEntry Lc0Evaluator::evaluate_sync(Key key, const lc0::InputPlanes112& planes) const {
    CacheEntry entry{};
    if (probe_cache(key, entry)) {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.eval_cache_hits;
        return entry;
    }

    {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.eval_cache_misses;
    }

    const auto started = std::chrono::steady_clock::now();
    entry = run_forward_entry(planes);
    const auto infer_us = std::max<std::int64_t>(
      0, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
    {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.nn_batches;
        ++stats_.nn_batch_positions;
        stats_.nn_infer_us += static_cast<std::uint64_t>(infer_us);
        record_latency_stats(stats_, static_cast<std::uint64_t>(infer_us));
    }
    store_cache(key, entry);
    return entry;
}

Lc0Evaluator::CacheEntry Lc0Evaluator::evaluate_async(Key key, const lc0::InputPlanes112& planes) const {
    CacheEntry entry{};
    if (probe_cache(key, entry)) {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.eval_cache_hits;
        return entry;
    }

    {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.eval_cache_misses;
    }

    auto req = std::make_unique<EvalRequest>();
    req->key = key;
    req->planes = planes;
    req->enqueued = std::chrono::steady_clock::now();
    auto fut = req->promise.get_future();

    {
        std::scoped_lock lock(queue_mutex_);
        queue_.push_back(std::move(req));
    }
    queue_cv_.notify_one();
    return fut.get();
}

int Lc0Evaluator::static_eval(const Position& pos) const {
    {
        std::scoped_lock lock(stats_mutex_);
        ++stats_.eval_calls;
    }

    if (!ready_) {
        return 0;
    }

    const Key key = pos.key();
    const auto planes = lc0::extract_features_112(pos);
    CacheEntry entry{};
    if (backend_ == Backend::FP32_ASYNC && !workers_.empty()) {
        try {
            entry = evaluate_async(key, planes);
        } catch (...) {
            entry = evaluate_sync(key, planes);
        }
    } else {
        entry = evaluate_sync(key, planes);
    }

    return entry.cp;
}

int Lc0Evaluator::static_eval_trace(const Position& pos, EvalBreakdown* out) const {
    const int score = static_eval(pos);
    if (out) {
        *out = EvalBreakdown{};
        out->total_white_pov = pos.side_to_move() == WHITE ? score : -score;
    }
    return score;
}

bool Lc0Evaluator::eval_wdl(const Position& pos, float& w, float& d, float& l, int& cp) const {
    if (!ready_) {
        w = d = l = 0.0f;
        cp = 0;
        return false;
    }

    const Key key = pos.key();
    const auto planes = lc0::extract_features_112(pos);
    CacheEntry entry{};
    if (backend_ == Backend::FP32_ASYNC && !workers_.empty()) {
        try {
            entry = evaluate_async(key, planes);
        } catch (...) {
            entry = evaluate_sync(key, planes);
        }
    } else {
        entry = evaluate_sync(key, planes);
    }

    w = entry.w;
    d = entry.d;
    l = entry.l;
    cp = entry.cp;
    return true;
}

EvalStats Lc0Evaluator::stats() const {
    std::scoped_lock lock(stats_mutex_);
    return stats_;
}

void Lc0Evaluator::clear_stats() {
    std::scoped_lock lock(stats_mutex_);
    stats_ = EvalStats{};
}

void Lc0Evaluator::clear_cache() const {
    std::scoped_lock lock(cache_mutex_);
    eval_cache_.clear();
}

void Lc0Evaluator::set_cache_limit(std::size_t entries) {
    cache_limit_ = std::max<std::size_t>(entries, 1024);
    std::scoped_lock lock(cache_mutex_);
    eval_cache_.reserve(cache_limit_);
}

void Lc0Evaluator::set_backend(Backend backend) {
    backend_ = backend;
    restart_workers();
}

void Lc0Evaluator::set_backend_from_int(int backend) {
    if (backend <= 1) {
        set_backend(Backend::FP32_SYNC);
    } else if (backend == 2) {
        set_backend(Backend::FP32_ASYNC);
    } else {
        set_backend(Backend::INT8_PLACEHOLDER);
    }
}

void Lc0Evaluator::set_batch_max(int batch_max) {
    batch_max_ = std::clamp(batch_max, 1, 512);
}

void Lc0Evaluator::set_batch_wait_us(int batch_wait_us) {
    batch_wait_us_ = std::clamp(batch_wait_us, 0, 20000);
}

void Lc0Evaluator::set_eval_threads(int threads) {
    eval_threads_ = std::clamp(threads, 1, 64);
    restart_workers();
}

void Lc0Evaluator::set_exec_backend(int backend) {
    linear_backend_.set_type_from_int(backend);
}

std::string Lc0Evaluator::backend_name() const {
    switch (backend_) {
        case Backend::FP32_SYNC:
            return "fp32_sync";
        case Backend::FP32_ASYNC:
            return "fp32_async";
        case Backend::INT8_PLACEHOLDER:
            return "int8_placeholder";
        default:
            return "unknown";
    }
}

void Lc0Evaluator::restart_workers() {
    stop_workers();

    if (!ready_ || backend_ != Backend::FP32_ASYNC) {
        return;
    }

    stop_workers_ = false;
    workers_.reserve(static_cast<std::size_t>(eval_threads_));
    for (int i = 0; i < eval_threads_; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }
}

void Lc0Evaluator::stop_workers() {
    {
        std::scoped_lock lock(queue_mutex_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
    {
        std::scoped_lock lock(queue_mutex_);
        while (!queue_.empty()) {
            auto req = std::move(queue_.front());
            queue_.pop_front();
            try {
                req->promise.set_exception(std::make_exception_ptr(std::runtime_error("lc0 async worker stopped")));
            } catch (...) {
            }
        }
        stop_workers_ = false;
    }
}

void Lc0Evaluator::worker_loop() {
    while (true) {
        std::vector<std::unique_ptr<EvalRequest>> batch;
        batch.reserve(static_cast<std::size_t>(batch_max_));

        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() { return stop_workers_ || !queue_.empty(); });
            if (stop_workers_) {
                return;
            }

            auto first = std::move(queue_.front());
            queue_.pop_front();
            batch.push_back(std::move(first));

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(batch_wait_us_);
            const auto oldest_enqueued = batch.front()->enqueued;
            const auto soft_flush_threshold = std::chrono::microseconds(std::max(1, batch_wait_us_));
            while (batch.size() < static_cast<std::size_t>(batch_max_)) {
                if (std::chrono::steady_clock::now() - oldest_enqueued >= soft_flush_threshold) {
                    break;
                }
                if (queue_.empty()) {
                    if (batch_wait_us_ == 0) {
                        break;
                    }
                    if (queue_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        break;
                    }
                    if (stop_workers_) {
                        return;
                    }
                    if (queue_.empty()) {
                        continue;
                    }
                }

                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        const auto infer_started = std::chrono::steady_clock::now();
        for (auto& req : batch) {
            try {
                CacheEntry entry{};
                if (!probe_cache(req->key, entry)) {
                    entry = run_forward_entry(req->planes);
                    store_cache(req->key, entry);
                } else {
                    std::scoped_lock lock(stats_mutex_);
                    ++stats_.eval_cache_hits;
                }
                req->promise.set_value(entry);
            } catch (...) {
                req->promise.set_exception(std::current_exception());
            }
        }

        const auto infer_us = std::max<std::int64_t>(
          0, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - infer_started).count());
        const auto now = std::chrono::steady_clock::now();
        std::uint64_t queue_wait_us_sum = 0;
        std::vector<std::uint64_t> latencies_us{};
        latencies_us.reserve(batch.size());
        for (const auto& req : batch) {
            const std::uint64_t queue_wait_us = static_cast<std::uint64_t>(std::max<std::int64_t>(
              0, std::chrono::duration_cast<std::chrono::microseconds>(infer_started - req->enqueued).count()));
            const std::uint64_t latency_us = static_cast<std::uint64_t>(std::max<std::int64_t>(
              0, std::chrono::duration_cast<std::chrono::microseconds>(now - req->enqueued).count()));
            queue_wait_us_sum += queue_wait_us;
            latencies_us.push_back(latency_us);
        }

        {
            std::scoped_lock lock(stats_mutex_);
            ++stats_.nn_batches;
            stats_.nn_batch_positions += static_cast<std::uint64_t>(batch.size());
            stats_.nn_queue_wait_us += queue_wait_us_sum;
            stats_.nn_infer_us += static_cast<std::uint64_t>(infer_us);
            for (const std::uint64_t latency_us : latencies_us) {
                record_latency_stats(stats_, latency_us);
            }
        }
    }
}

}  // namespace makaira
