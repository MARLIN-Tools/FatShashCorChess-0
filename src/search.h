#pragma once

#include "evaluator.h"
#include "move.h"
#include "position.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace makaira {

constexpr int MAX_PLY = 128;
constexpr int VALUE_INFINITE = 32000;
constexpr int VALUE_MATE = 31000;

struct SearchLimits {
    int depth = 0;
    std::uint64_t nodes = 0;
    int movetime_ms = -1;
    int wtime_ms = -1;
    int btime_ms = -1;
    int winc_ms = 0;
    int binc_ms = 0;
    int movestogo = 0;
    int move_overhead_ms = 30;
    bool infinite = false;
    bool ponder = false;
    bool nodes_as_time = false;
};

struct SearchConfig {
    bool use_history = true;
    bool use_cont_history = true;
    bool use_nmp = true;
    bool use_lmr = true;

    int history_max = 20923;
    int history_bonus_scale = 1;
    int history_malus_divisor = 3;
    int cont_history_2ply_divisor = 4;

    int nmp_min_depth = 2;
    int nmp_base_reduction = 5;
    int nmp_depth_divisor = 6;
    int nmp_margin_base = 60;
    int nmp_margin_per_depth = 14;
    int nmp_non_pawn_min = 760;
    int nmp_verify_non_pawn_max = 2004;
    int nmp_verify_min_depth = 8;

    int lmr_min_depth = 2;
    int lmr_full_depth_moves = 2;
    int lmr_history_threshold = 3426;
};

struct SearchStats {
    std::uint64_t nodes = 0;
    std::uint64_t qnodes = 0;
    std::uint64_t tt_probes = 0;
    std::uint64_t tt_hits = 0;
    std::uint64_t beta_cutoffs = 0;
    std::uint64_t pvs_researches = 0;
    std::uint64_t movegen_calls = 0;
    std::uint64_t moves_generated = 0;
    std::uint64_t move_pick_iterations = 0;
    std::uint64_t cutoff_tt = 0;
    std::uint64_t cutoff_good_capture = 0;
    std::uint64_t cutoff_quiet = 0;
    std::uint64_t cutoff_bad_capture = 0;
    std::uint64_t history_updates = 0;
    std::uint64_t cont_history_updates = 0;
    std::uint64_t nmp_attempts = 0;
    std::uint64_t nmp_cutoffs = 0;
    std::uint64_t nmp_verifications = 0;
    std::uint64_t nmp_verification_fails = 0;
    std::uint64_t lmr_reduced = 0;
    std::uint64_t lmr_researches = 0;
    std::uint64_t lmr_fail_high_after_reduce = 0;
};

struct SearchResult {
    Move best_move{};
    int score = 0;
    int depth = 0;
    int seldepth = 0;
    std::vector<Move> pv{};
    SearchStats stats{};
    int time_ms = 0;
};

struct SearchIterationInfo {
    int depth = 0;
    int seldepth = 0;
    int score = 0;
    int score_delta = 0;
    int aspiration_fails = 0;
    int bestmove_changes = 0;
    int root_legal_moves = 0;
    int stability_score = 0;
    int complexity_x100 = 100;
    int optimum_time_ms = 0;
    int effective_optimum_ms = 0;
    int maximum_time_ms = 0;
    int time_ms = 0;
    std::uint64_t nodes = 0;
    std::uint64_t nodes_this_iter = 0;
    std::uint64_t nps = 0;
    std::vector<Move> pv{};
    SearchStats stats{};
};

using SearchInfoCallback = std::function<void(const SearchIterationInfo&)>;

class Searcher {
   public:
    explicit Searcher(const IEvaluator& evaluator);

    void set_hash_size_mb(std::size_t mb);
    void clear_hash();
    void clear_heuristics();

    void set_search_config(const SearchConfig& config) { config_ = config; }
    const SearchConfig& search_config() const { return config_; }

    SearchResult search(Position& pos, const SearchLimits& limits, SearchInfoCallback on_iteration = {});

   private:
    enum Bound : std::uint8_t {
        BOUND_NONE = 0,
        BOUND_UPPER = 1,
        BOUND_LOWER = 2,
        BOUND_EXACT = 3
    };

    struct TTEntry {
        Key key = 0;
        std::uint32_t move_raw = 0;
        std::int16_t score = 0;
        std::int16_t eval = 0;
        std::int8_t depth = 0;
        std::uint8_t bound = BOUND_NONE;
        std::uint8_t generation = 0;
        std::uint8_t padding = 0;
    };

    class TranspositionTable {
       public:
        void resize_mb(std::size_t mb);
        void clear();

        TTEntry* probe(Key key);
        void store(Key key,
                   Move move,
                   int score,
                   int eval,
                   int depth,
                   std::uint8_t bound,
                   std::uint8_t generation,
                   int ply);

       private:
        std::size_t index(Key key) const;

        std::vector<TTEntry> entries_{};
    };

    struct PVLine {
        std::array<Move, MAX_PLY> moves{};
        int length = 0;
    };

    struct IterationSummary {
        int depth = 0;
        int score = 0;
        int score_delta = 0;
        bool bestmove_changed = false;
        int bestmove_changes = 0;
        int aspiration_fails = 0;
        int root_legal_moves = 0;
        std::uint64_t nodes_this_iter = 0;
        std::uint64_t total_nodes = 0;
        std::uint64_t nps = 0;
    };

    class TimeManager {
       public:
        void init(const SearchLimits& limits, Color us, double session_nps_ema);
        bool should_stop_hard(std::uint64_t total_nodes, std::uint64_t explicit_node_limit, bool external_stop);
        bool should_stop_soft(const IterationSummary& iteration);
        void update_nps(std::uint64_t nps);

        int elapsed_ms() const;
        int optimum_ms() const { return optimum_time_ms_; }
        int effective_optimum_ms() const { return effective_optimum_ms_; }
        int maximum_ms() const { return maximum_time_ms_; }
        int stability_score() const { return last_stability_score_; }
        int complexity_x100() const { return last_complexity_x100_; }

       private:
        static int clamp_ms(int v);
        void refresh_check_period();

        SearchLimits limits_{};
        Color us_ = WHITE;
        std::chrono::steady_clock::time_point start_time_{};

        int time_left_ms_ = 0;
        int increment_ms_ = 0;
        int moves_to_go_ = 0;
        int available_ms_ = 0;
        int optimum_time_ms_ = 0;
        int effective_optimum_ms_ = 0;
        int maximum_time_ms_ = 0;

        bool fixed_movetime_ = false;
        bool emergency_mode_ = false;

        bool nodes_as_time_ = false;
        std::uint64_t soft_node_budget_ = 0;
        std::uint64_t hard_node_budget_ = 0;

        double nps_ema_ = 0.0;
        std::uint64_t next_check_node_ = 1024;
        std::uint64_t check_period_nodes_ = 1024;

        int last_stability_score_ = 0;
        int last_complexity_x100_ = 100;
    };

    int search_node(Position& pos, int depth, int alpha, int beta, int ply, bool is_pv, PVLine& pv);
    int qsearch(Position& pos, int alpha, int beta, int ply, PVLine& pv);

    static int score_to_tt(int score, int ply);
    static int score_from_tt(int score, int ply);
    static int move_index(Piece pc, Square to);

    bool should_stop_hard();

    int quiet_move_score(const Position& pos, Move move, int ply) const;
    void update_history_value(int& value, int bonus) const;
    void update_quiet_history(const Position& pos,
                              Color side,
                              Move best_move,
                              int ply,
                              int depth,
                              const std::array<Move, 256>& quiet_tried,
                              int quiet_count);

    int nmp_reduction(int depth) const;
    int lmr_reduction(int depth, int move_count, int quiet_score) const;

    static void update_pv(PVLine& dst, Move move, const PVLine& child);

    struct SearchStackEntry {
        int move_index = -1;
        bool did_null = false;
        int static_eval = 0;
    };

    const IEvaluator& evaluator_;
    TranspositionTable tt_{};
    TimeManager tm_{};

    SearchLimits limits_{};
    SearchStats stats_{};

    std::uint8_t generation_ = 0;
    bool stop_ = false;
    int root_depth_ = 0;
    int root_legal_moves_ = 0;
    int seldepth_ = 0;

    Move previous_root_best_move_{};
    int rolling_bestmove_changes_ = 0;
    double session_nps_ema_ = 0.0;
    bool use_eval_move_hooks_ = false;

    SearchConfig config_{};

    static constexpr int HISTORY_SIZE = static_cast<int>(COLOR_NB) * static_cast<int>(SQ_NB) * static_cast<int>(SQ_NB);
    static constexpr int MOVE_INDEX_NB = static_cast<int>(PIECE_NB) * static_cast<int>(SQ_NB);
    static constexpr int CONT_HISTORY_SIZE = MOVE_INDEX_NB * MOVE_INDEX_NB;
    std::vector<std::int16_t> history_{};
    std::vector<std::int16_t> cont_history_{};
    std::array<SearchStackEntry, MAX_PLY + 4> stack_{};
    std::vector<int> lmr_table_{};
};

}  // namespace makaira
