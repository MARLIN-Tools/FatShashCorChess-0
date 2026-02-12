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

struct SearchStats {
    std::uint64_t nodes = 0;
    std::uint64_t qnodes = 0;
    std::uint64_t tt_probes = 0;
    std::uint64_t tt_hits = 0;
    std::uint64_t beta_cutoffs = 0;
    std::uint64_t pvs_researches = 0;
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

    bool should_stop_hard();

    int score_move(const Position& pos, Move move, Move tt_move) const;
    static void update_pv(PVLine& dst, Move move, const PVLine& child);

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
};

}  // namespace makaira