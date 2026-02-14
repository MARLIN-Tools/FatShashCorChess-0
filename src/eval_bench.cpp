#include "bitboard.h"
#include "hybrid_evaluator.h"
#include "movegen.h"
#include "position.h"
#include "zobrist.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    makaira::attacks::init();
    makaira::init_zobrist();

    const int positions_target = argc > 1 ? std::max(1, std::atoi(argv[1])) : 512;
    const int evals_per_position = argc > 2 ? std::max(1, std::atoi(argv[2])) : 64;
    std::string backend = "hce";
    std::string lc0_weights = "t1-256x10-distilled-swa-2432500.pb.gz";
    int lc0_batch_max = 16;
    int lc0_batch_wait_us = 1000;
    int lc0_batch_policy = 0;
    int lc0_root_priority = 0;
    int lc0_eval_threads = 1;
    int lc0_cache_entries = 1 << 18;
    int lc0_cache_policy = 1;
    int lc0_exec_backend = 0;
    int lc0_backend_strict = 0;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            lc0_weights = argv[++i];
        } else if (std::strcmp(argv[i], "--lc0-batch-max") == 0 && i + 1 < argc) {
            lc0_batch_max = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lc0-batch-wait-us") == 0 && i + 1 < argc) {
            lc0_batch_wait_us = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lc0-batch-policy") == 0 && i + 1 < argc) {
            lc0_batch_policy = std::clamp(std::atoi(argv[++i]), 0, 1);
        } else if (std::strcmp(argv[i], "--lc0-root-priority") == 0 && i + 1 < argc) {
            lc0_root_priority = std::clamp(std::atoi(argv[++i]), 0, 1);
        } else if (std::strcmp(argv[i], "--lc0-eval-threads") == 0 && i + 1 < argc) {
            lc0_eval_threads = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lc0-cache-entries") == 0 && i + 1 < argc) {
            lc0_cache_entries = std::max(1024, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lc0-cache-policy") == 0 && i + 1 < argc) {
            lc0_cache_policy = std::clamp(std::atoi(argv[++i]), 0, 1);
        } else if (std::strcmp(argv[i], "--lc0-exec-backend") == 0 && i + 1 < argc) {
            lc0_exec_backend = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lc0-backend-strict") == 0 && i + 1 < argc) {
            lc0_backend_strict = std::clamp(std::atoi(argv[++i]), 0, 1);
        }
    }

    makaira::HybridEvaluator eval;
    if (backend == "lc0_sync" || backend == "lc0_async" || backend == "lc0_int8") {
        eval.set_lc0_batch_max(lc0_batch_max);
        eval.set_lc0_batch_wait_us(lc0_batch_wait_us);
        eval.set_lc0_batch_policy_from_int(lc0_batch_policy);
        eval.set_lc0_root_priority(lc0_root_priority != 0);
        eval.set_lc0_eval_threads(lc0_eval_threads);
        eval.set_lc0_cache_entries(static_cast<std::size_t>(lc0_cache_entries));
        eval.set_lc0_cache_policy_from_int(lc0_cache_policy);
        eval.set_lc0_exec_backend(lc0_exec_backend);
        eval.set_lc0_backend_strict(lc0_backend_strict != 0);
        if (!eval.load_lc0_weights(lc0_weights, true)) {
            std::cerr << "failed to load lc0 weights: " << eval.lc0_last_error() << "\n";
            return 1;
        }

        if (backend == "lc0_sync") {
            eval.set_backend(makaira::HybridEvaluator::Backend::LC0_FP32);
        } else if (backend == "lc0_async") {
            eval.set_backend(makaira::HybridEvaluator::Backend::LC0_FP32_ASYNC);
        } else {
            eval.set_backend(makaira::HybridEvaluator::Backend::LC0_INT8);
        }
    } else {
        eval.set_backend(makaira::HybridEvaluator::Backend::HCE);
    }

    std::mt19937_64 rng(0xBADC0DEULL);

    std::vector<makaira::Position> positions;
    positions.reserve(positions_target);

    for (int i = 0; i < positions_target; ++i) {
        makaira::Position pos;
        pos.set_startpos();

        const int plies = 8 + static_cast<int>(rng() % 20ULL);
        for (int p = 0; p < plies; ++p) {
            makaira::MoveList moves;
            makaira::generate_legal(pos, moves);
            if (moves.count == 0) {
                break;
            }
            const int idx = static_cast<int>(rng() % static_cast<std::uint64_t>(moves.count));
            if (!pos.make_move(moves[idx])) {
                break;
            }
        }

        positions.push_back(pos);
    }

    volatile int sink = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& pos : positions) {
        for (int i = 0; i < evals_per_position; ++i) {
            sink += eval.static_eval(pos);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    const auto st = eval.stats();
    const std::uint64_t total_evals = static_cast<std::uint64_t>(positions.size()) * static_cast<std::uint64_t>(evals_per_position);
    const double secs = ms > 0 ? static_cast<double>(ms) / 1000.0 : 0.001;
    const std::uint64_t evals_per_sec = static_cast<std::uint64_t>(static_cast<double>(total_evals) / secs);

    std::cout << "evals " << total_evals << "\n";
    std::cout << "time_ms " << ms << "\n";
    std::cout << "evals_per_sec " << evals_per_sec << "\n";
    std::cout << "backend " << backend << "\n";
    std::cout << "lc0_exec_backend " << eval.lc0_exec_backend_name() << "\n";
    std::cout << "lc0_backend_strict " << (eval.lc0_backend_strict() ? 1 : 0) << "\n";
    std::cout << "lc0_batch_policy " << eval.lc0_batch_policy() << "\n";
    std::cout << "lc0_root_priority " << (eval.lc0_root_priority() ? 1 : 0) << "\n";
    std::cout << "lc0_cache_policy " << eval.lc0_cache_policy() << "\n";
    if (!eval.lc0_exec_backend_error().empty()) {
        std::cout << "lc0_exec_backend_error " << eval.lc0_exec_backend_error() << "\n";
    }
    std::cout << "eval_calls " << st.eval_calls << "\n";
    std::cout << "pawn_hash_hits " << st.pawn_hash_hits << "\n";
    std::cout << "pawn_hash_misses " << st.pawn_hash_misses << "\n";
    std::cout << "eval_cache_hits " << st.eval_cache_hits << "\n";
    std::cout << "eval_cache_misses " << st.eval_cache_misses << "\n";
    std::cout << "nn_batches " << st.nn_batches << "\n";
    std::cout << "nn_batch_positions " << st.nn_batch_positions << "\n";
    std::cout << "nn_queue_wait_us " << st.nn_queue_wait_us << "\n";
    std::cout << "nn_infer_us " << st.nn_infer_us << "\n";
    if (st.pawn_hash_hits + st.pawn_hash_misses > 0) {
        const double hit_rate = 100.0 * static_cast<double>(st.pawn_hash_hits)
                              / static_cast<double>(st.pawn_hash_hits + st.pawn_hash_misses);
        std::cout << "pawn_hash_hit_rate_pct " << hit_rate << "\n";
    }
    if (st.eval_cache_hits + st.eval_cache_misses > 0) {
        const double hit_rate = 100.0 * static_cast<double>(st.eval_cache_hits)
                              / static_cast<double>(st.eval_cache_hits + st.eval_cache_misses);
        std::cout << "eval_cache_hit_rate_pct " << hit_rate << "\n";
    }
    if (st.nn_batches > 0) {
        const double avg_batch = static_cast<double>(st.nn_batch_positions) / static_cast<double>(st.nn_batches);
        std::cout << "nn_avg_batch_size " << avg_batch << "\n";
    }
    if (st.eval_calls > 0) {
        const double avg_wait = static_cast<double>(st.nn_queue_wait_us) / static_cast<double>(st.eval_calls);
        const double avg_infer = static_cast<double>(st.nn_infer_us) / static_cast<double>(st.eval_calls);
        std::cout << "nn_avg_queue_wait_us " << avg_wait << "\n";
        std::cout << "nn_avg_infer_us " << avg_infer << "\n";
    }

    if (sink == 42) {
        std::cerr << "";
    }

    return 0;
}
