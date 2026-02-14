#pragma once

#include "move.h"
#include "position.h"

#include <cstdint>

namespace makaira {

struct Score {
    int mg = 0;
    int eg = 0;
};

constexpr Score make_score(int mg, int eg) {
    return Score{mg, eg};
}

constexpr Score operator+(Score a, Score b) {
    return Score{a.mg + b.mg, a.eg + b.eg};
}

constexpr Score operator-(Score a, Score b) {
    return Score{a.mg - b.mg, a.eg - b.eg};
}

constexpr Score& operator+=(Score& a, Score b) {
    a.mg += b.mg;
    a.eg += b.eg;
    return a;
}

constexpr Score operator*(Score a, int k) {
    return Score{a.mg * k, a.eg * k};
}

struct EvalBreakdown {
    Score material_psqt{};
    Score pawns{};
    Score pawns_passed{};
    Score pawns_isolated{};
    Score pawns_doubled{};
    Score pawns_backward{};
    Score pawns_candidate{};
    Score pawns_connected{};
    Score pawns_supported{};
    Score pawns_outside{};
    Score pawns_blocked{};
    Score mobility{};
    Score king_safety{};
    Score king_shelter{};
    Score king_storm{};
    Score king_danger{};
    Score piece_features{};
    Score piece_bishop_pair{};
    Score piece_rook_file{};
    Score piece_rook_seventh{};
    Score piece_knight_outpost{};
    Score piece_bad_bishop{};
    Score threats{};
    Score threat_hanging{};
    Score threat_pawn{};
    Score space{};
    Score endgame_terms{};
    Score endgame_king_activity{};
    int endgame_scale = 128;
    int tempo = 0;
    int phase = 0;
    int total_white_pov = 0;
};

struct EvalStats {
    std::uint64_t eval_calls = 0;
    std::uint64_t pawn_hash_hits = 0;
    std::uint64_t pawn_hash_misses = 0;
    std::uint64_t eval_cache_hits = 0;
    std::uint64_t eval_cache_misses = 0;
    std::uint64_t nn_batches = 0;
    std::uint64_t nn_batch_positions = 0;
    std::uint64_t nn_queue_wait_us = 0;
    std::uint64_t nn_infer_us = 0;
    std::uint64_t nn_eval_latency_samples = 0;
    std::uint64_t nn_eval_latency_us = 0;
    std::uint64_t nn_eval_latency_max_us = 0;
    std::uint64_t nn_eval_latency_le_250us = 0;
    std::uint64_t nn_eval_latency_le_500us = 0;
    std::uint64_t nn_eval_latency_le_1000us = 0;
    std::uint64_t nn_eval_latency_le_2000us = 0;
    std::uint64_t nn_eval_latency_le_5000us = 0;
    std::uint64_t nn_eval_latency_gt_5000us = 0;
};

class IEvaluator {
   public:
    virtual ~IEvaluator() = default;

    virtual int static_eval(const Position& pos) const = 0;
    virtual int static_eval_trace(const Position& pos, EvalBreakdown* out) const;

    // Evaluators that maintain incremental state (e.g. NNUE accumulators)
    // can request make/unmake callbacks in the hot search loop.
    virtual bool requires_move_hooks() const { return false; }

    virtual EvalStats stats() const;
    virtual void clear_stats();

    virtual void on_make_move(const Position&, Move) const {}
    virtual void on_unmake_move(const Position&, Move) const {}
};

class MaterialEvaluator final : public IEvaluator {
   public:
    int static_eval(const Position& pos) const override;
};

class HCEEvaluator;

}  // namespace makaira
