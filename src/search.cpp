#include "search.h"

#include "movepicker.h"
#include "movegen.h"

#include <algorithm>
#include <limits>

namespace makaira {
namespace {

constexpr int ASPIRATION_INITIAL = 24;
constexpr int ASPIRATION_MAX = 1024;
constexpr int TIME_INF = std::numeric_limits<int>::max() / 4;

constexpr int MATE_SCORE_FOR_TT = VALUE_MATE - MAX_PLY;

}  // namespace

Searcher::Searcher(const IEvaluator& evaluator) :
    evaluator_(evaluator) {
    tt_.resize_mb(32);
}

void Searcher::set_hash_size_mb(std::size_t mb) {
    tt_.resize_mb(mb);
}

void Searcher::clear_hash() {
    tt_.clear();
}

void Searcher::TranspositionTable::resize_mb(std::size_t mb) {
    const std::size_t bytes = std::max<std::size_t>(mb, 1) * 1024ULL * 1024ULL;
    const std::size_t count = std::max<std::size_t>(bytes / sizeof(TTEntry), 1);
    entries_.assign(count, TTEntry{});
}

void Searcher::TranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), TTEntry{});
}

std::size_t Searcher::TranspositionTable::index(Key key) const {
    return static_cast<std::size_t>(key % entries_.size());
}

Searcher::TTEntry* Searcher::TranspositionTable::probe(Key key) {
    if (entries_.empty()) {
        return nullptr;
    }

    TTEntry& e = entries_[index(key)];
    return e.key == key ? &e : nullptr;
}

void Searcher::TranspositionTable::store(Key key,
                                         Move move,
                                         int score,
                                         int eval,
                                         int depth,
                                         std::uint8_t bound,
                                         std::uint8_t generation,
                                         int ply) {
    if (entries_.empty()) {
        return;
    }

    TTEntry& dst = entries_[index(key)];

    const bool replace = dst.key != key || bound == BOUND_EXACT || depth >= dst.depth || dst.generation != generation;
    if (!replace) {
        return;
    }

    dst.key = key;
    dst.move_raw = move.raw();
    dst.score = static_cast<std::int16_t>(Searcher::score_to_tt(score, ply));
    dst.eval = static_cast<std::int16_t>(std::clamp(eval, -VALUE_INFINITE, VALUE_INFINITE));
    dst.depth = static_cast<std::int8_t>(std::clamp(depth, -1, 127));
    dst.bound = bound;
    dst.generation = generation;
}

int Searcher::score_to_tt(int score, int ply) {
    if (score > MATE_SCORE_FOR_TT) {
        return score + ply;
    }
    if (score < -MATE_SCORE_FOR_TT) {
        return score - ply;
    }
    return score;
}

int Searcher::score_from_tt(int score, int ply) {
    if (score > MATE_SCORE_FOR_TT) {
        return score - ply;
    }
    if (score < -MATE_SCORE_FOR_TT) {
        return score + ply;
    }
    return score;
}

int Searcher::TimeManager::clamp_ms(int v) {
    return std::max(1, std::min(v, TIME_INF));
}

void Searcher::TimeManager::init(const SearchLimits& limits, Color us, double session_nps_ema) {
    limits_ = limits;
    us_ = us;
    start_time_ = std::chrono::steady_clock::now();

    time_left_ms_ = us_ == WHITE ? limits_.wtime_ms : limits_.btime_ms;
    increment_ms_ = us_ == WHITE ? limits_.winc_ms : limits_.binc_ms;
    moves_to_go_ = limits_.movestogo;

    fixed_movetime_ = limits_.movetime_ms > 0;
    nodes_as_time_ = limits_.nodes_as_time;
    emergency_mode_ = false;

    nps_ema_ = session_nps_ema > 1.0 ? session_nps_ema : 200000.0;
    check_period_nodes_ = std::clamp<std::uint64_t>(static_cast<std::uint64_t>(nps_ema_ / 50.0), 512, 32768);
    next_check_node_ = check_period_nodes_;

    soft_node_budget_ = 0;
    hard_node_budget_ = 0;

    if (limits_.infinite || limits_.ponder) {
        available_ms_ = TIME_INF;
        optimum_time_ms_ = TIME_INF;
        effective_optimum_ms_ = TIME_INF;
        maximum_time_ms_ = TIME_INF;
        return;
    }

    const int overhead = std::max(0, limits_.move_overhead_ms);

    if (fixed_movetime_) {
        available_ms_ = clamp_ms(limits_.movetime_ms - overhead);
        optimum_time_ms_ = clamp_ms((available_ms_ * 85) / 100);
        maximum_time_ms_ = available_ms_;
    } else if (time_left_ms_ > 0) {
        const int safety_reserve = moves_to_go_ > 0 ? std::max(20, time_left_ms_ / 50) : std::max(40, time_left_ms_ / 25);
        available_ms_ = clamp_ms(time_left_ms_ - overhead - safety_reserve);

        if (time_left_ms_ <= (overhead * 3 + 80)) {
            emergency_mode_ = true;
        }

        const int horizon = moves_to_go_ > 0 ? std::clamp(moves_to_go_, 1, 80) : std::clamp(20 + time_left_ms_ / 15000, 20, 40);
        const int base_time_per_move = available_ms_ / std::max(1, horizon);

        const int increment_spend = increment_ms_ / 2;
        optimum_time_ms_ = clamp_ms(base_time_per_move + increment_spend);

        if (moves_to_go_ > 0) {
            maximum_time_ms_ = std::min(available_ms_, std::max(optimum_time_ms_, optimum_time_ms_ * 3));
        } else {
            maximum_time_ms_ = std::min(available_ms_, std::max(optimum_time_ms_ * 4, base_time_per_move * 6));
        }

        if (emergency_mode_) {
            optimum_time_ms_ = std::max(1, std::min(optimum_time_ms_, available_ms_ / 4));
            maximum_time_ms_ = std::max(optimum_time_ms_, std::min(maximum_time_ms_, available_ms_ / 2));
        }

        optimum_time_ms_ = std::min(optimum_time_ms_, available_ms_);
        maximum_time_ms_ = std::max(optimum_time_ms_, std::min(maximum_time_ms_, available_ms_));
    } else {
        available_ms_ = TIME_INF;
        optimum_time_ms_ = TIME_INF;
        maximum_time_ms_ = TIME_INF;
    }

    effective_optimum_ms_ = optimum_time_ms_;

    if (nodes_as_time_ && maximum_time_ms_ < TIME_INF && nps_ema_ > 1.0) {
        soft_node_budget_ = static_cast<std::uint64_t>(std::max(1.0, (effective_optimum_ms_ * nps_ema_ * 0.90) / 1000.0));
        hard_node_budget_ = static_cast<std::uint64_t>(std::max(soft_node_budget_ + 1.0, (maximum_time_ms_ * nps_ema_ * 0.80) / 1000.0));
    }
}

bool Searcher::TimeManager::should_stop_hard(std::uint64_t total_nodes,
                                             std::uint64_t explicit_node_limit,
                                             bool external_stop) {
    if (external_stop) {
        return true;
    }

    if (explicit_node_limit > 0 && total_nodes >= explicit_node_limit) {
        return true;
    }

    if (nodes_as_time_ && hard_node_budget_ > 0 && total_nodes >= hard_node_budget_) {
        return true;
    }

    if (maximum_time_ms_ >= TIME_INF) {
        return false;
    }

    if (total_nodes < next_check_node_) {
        return false;
    }

    next_check_node_ = total_nodes + check_period_nodes_;
    return elapsed_ms() >= maximum_time_ms_;
}

bool Searcher::TimeManager::should_stop_soft(const IterationSummary& iteration) {
    const int elapsed = elapsed_ms();

    if (elapsed >= maximum_time_ms_) {
        return true;
    }

    if (effective_optimum_ms_ >= TIME_INF) {
        return false;
    }

    int complexity = 100;

    if (iteration.root_legal_moves <= 1) {
        complexity -= 45;
    } else if (iteration.root_legal_moves <= 3) {
        complexity -= 20;
    } else if (iteration.root_legal_moves >= 30) {
        complexity += 20;
    } else if (iteration.root_legal_moves >= 20) {
        complexity += 10;
    }

    if (iteration.bestmove_changed) {
        complexity += 18;
    }
    if (iteration.bestmove_changes >= 2) {
        complexity += 8;
    }

    if (iteration.score_delta >= 80) {
        complexity += 20;
    } else if (iteration.score_delta >= 35) {
        complexity += 10;
    }

    if (iteration.aspiration_fails >= 2) {
        complexity += 18;
    } else if (iteration.aspiration_fails == 1) {
        complexity += 10;
    }

    complexity = std::clamp(complexity, 55, 260);

    const int min_optimum = std::max(1, optimum_time_ms_ / 2);
    effective_optimum_ms_ = std::clamp((optimum_time_ms_ * complexity) / 100, min_optimum, maximum_time_ms_);

    int stability = 0;

    if (!iteration.bestmove_changed) {
        stability += 3;
    } else {
        stability -= 1;
    }

    if (iteration.bestmove_changes == 0) {
        stability += 1;
    }

    if (iteration.score_delta <= 10) {
        stability += 2;
    } else if (iteration.score_delta <= 25) {
        stability += 1;
    } else if (iteration.score_delta >= 80) {
        stability -= 2;
    }

    if (iteration.aspiration_fails == 0) {
        stability += 2;
    } else if (iteration.aspiration_fails >= 2) {
        stability -= 2;
    }

    if (iteration.root_legal_moves <= 1) {
        stability += 3;
    } else if (iteration.root_legal_moves <= 3) {
        stability += 1;
    }

    last_stability_score_ = stability;
    last_complexity_x100_ = complexity;

    if (nodes_as_time_ && soft_node_budget_ > 0 && iteration.total_nodes >= soft_node_budget_) {
        return stability >= 0;
    }

    return elapsed >= effective_optimum_ms_ && stability >= 3;
}

void Searcher::TimeManager::update_nps(std::uint64_t nps) {
    if (nps == 0) {
        return;
    }

    if (nps_ema_ <= 1.0) {
        nps_ema_ = static_cast<double>(nps);
    } else {
        nps_ema_ = 0.85 * nps_ema_ + 0.15 * static_cast<double>(nps);
    }

    check_period_nodes_ = std::clamp<std::uint64_t>(static_cast<std::uint64_t>(nps_ema_ / 50.0), 512, 32768);

    if (nodes_as_time_ && maximum_time_ms_ < TIME_INF) {
        soft_node_budget_ = static_cast<std::uint64_t>(std::max(1.0, (effective_optimum_ms_ * nps_ema_ * 0.90) / 1000.0));
        hard_node_budget_ = static_cast<std::uint64_t>(std::max(soft_node_budget_ + 1.0, (maximum_time_ms_ * nps_ema_ * 0.80) / 1000.0));
    }
}

int Searcher::TimeManager::elapsed_ms() const {
    const auto now = std::chrono::steady_clock::now();
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
}

bool Searcher::should_stop_hard() {
    stop_ = tm_.should_stop_hard(stats_.nodes, limits_.nodes, stop_);
    return stop_;
}

void Searcher::update_pv(PVLine& dst, Move move, const PVLine& child) {
    dst.moves[0] = move;
    dst.length = 1;
    for (int i = 0; i < child.length && dst.length < MAX_PLY; ++i) {
        dst.moves[dst.length++] = child.moves[i];
    }
}

SearchResult Searcher::search(Position& pos, const SearchLimits& limits, SearchInfoCallback on_iteration) {
    limits_ = limits;
    stats_ = {};
    stop_ = false;
    seldepth_ = 0;
    root_legal_moves_ = 0;
    previous_root_best_move_ = Move{};
    rolling_bestmove_changes_ = 0;

    ++generation_;

    tm_.init(limits_, pos.side_to_move(), session_nps_ema_);
    use_eval_move_hooks_ = evaluator_.requires_move_hooks();

    SearchResult result{};
    result.best_move = Move{};

    const int max_depth = limits_.depth > 0 ? limits_.depth : 64;

    bool have_prev_score = false;
    int prev_score = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        if (should_stop_hard()) {
            break;
        }

        const std::uint64_t nodes_before = stats_.nodes;

        root_depth_ = depth;
        int alpha = -VALUE_INFINITE;
        int beta = VALUE_INFINITE;

        int delta = ASPIRATION_INITIAL;
        if (depth >= 4 && have_prev_score) {
            alpha = std::max(-VALUE_INFINITE, prev_score - delta);
            beta = std::min(VALUE_INFINITE, prev_score + delta);
        }

        int aspiration_fails = 0;
        PVLine pv{};
        int score = 0;

        while (true) {
            pv = {};
            score = search_node(pos, depth, alpha, beta, 0, true, pv);
            if (stop_) {
                break;
            }

            if (score <= alpha) {
                ++aspiration_fails;
                beta = (alpha + beta) / 2;
                alpha = std::max(-VALUE_INFINITE, score - delta);
                delta = std::min(delta * 2, ASPIRATION_MAX);
                continue;
            }
            if (score >= beta) {
                ++aspiration_fails;
                beta = std::min(VALUE_INFINITE, score + delta);
                delta = std::min(delta * 2, ASPIRATION_MAX);
                continue;
            }
            break;
        }

        if (stop_) {
            break;
        }

        result.score = score;
        result.depth = depth;
        result.seldepth = seldepth_;

        if (pv.length > 0) {
            result.best_move = pv.moves[0];
            result.pv.assign(pv.moves.begin(), pv.moves.begin() + pv.length);
        } else {
            result.best_move = Move{};
            result.pv.clear();
        }

        const int elapsed = tm_.elapsed_ms();
        const int safe_elapsed = std::max(1, elapsed);
        const std::uint64_t nps = (stats_.nodes * 1000ULL) / static_cast<std::uint64_t>(safe_elapsed);
        tm_.update_nps(nps);

        const bool bestmove_changed = depth > 1 && !result.best_move.is_none() && previous_root_best_move_ != result.best_move;
        if (bestmove_changed) {
            rolling_bestmove_changes_ = std::min(rolling_bestmove_changes_ + 1, 8);
        } else if (depth > 1 && rolling_bestmove_changes_ > 0) {
            --rolling_bestmove_changes_;
        }

        const int score_delta = have_prev_score ? std::abs(score - prev_score) : 0;
        previous_root_best_move_ = result.best_move;
        prev_score = score;
        have_prev_score = true;

        const IterationSummary iteration = {
          depth,
          score,
          score_delta,
          bestmove_changed,
          rolling_bestmove_changes_,
          aspiration_fails,
          root_legal_moves_,
          stats_.nodes - nodes_before,
          stats_.nodes,
          nps,
        };

        const bool stop_soft = tm_.should_stop_soft(iteration);

        if (on_iteration) {
            on_iteration(SearchIterationInfo{
              depth,
              seldepth_,
              score,
              score_delta,
              aspiration_fails,
              rolling_bestmove_changes_,
              root_legal_moves_,
              tm_.stability_score(),
              tm_.complexity_x100(),
              tm_.optimum_ms(),
              tm_.effective_optimum_ms(),
              tm_.maximum_ms(),
              elapsed,
              stats_.nodes,
              stats_.nodes - nodes_before,
              nps,
              result.pv,
              stats_,
            });
        }

        if (stop_soft) {
            break;
        }
    }

    result.time_ms = tm_.elapsed_ms();
    result.stats = stats_;

    if (result.time_ms > 0 && stats_.nodes > 0) {
        const double nps = (static_cast<double>(stats_.nodes) * 1000.0) / static_cast<double>(result.time_ms);
        if (session_nps_ema_ <= 1.0) {
            session_nps_ema_ = nps;
        } else {
            session_nps_ema_ = 0.90 * session_nps_ema_ + 0.10 * nps;
        }
    }

    return result;
}

int Searcher::search_node(Position& pos, int depth, int alpha, int beta, int ply, bool is_pv, PVLine& pv) {
    pv.length = 0;
    seldepth_ = std::max(seldepth_, ply);
    ++stats_.nodes;

    if (should_stop_hard()) {
        return 0;
    }

    if (ply >= MAX_PLY - 1) {
        return evaluator_.static_eval(pos);
    }

    if (pos.is_draw()) {
        return 0;
    }

    if (depth <= 0) {
        return qsearch(pos, alpha, beta, ply, pv);
    }

    const int alpha_orig = alpha;
    const Key key = pos.key();

    Move tt_move{};
    int tt_eval = std::numeric_limits<int>::min();

    ++stats_.tt_probes;
    if (TTEntry* e = tt_.probe(key)) {
        ++stats_.tt_hits;
        tt_move = Move(static_cast<Square>(e->move_raw & 0x3F),
                       static_cast<Square>((e->move_raw >> 6) & 0x3F),
                       static_cast<std::uint8_t>((e->move_raw >> 16) & 0xFF),
                       static_cast<PieceType>((e->move_raw >> 12) & 0x0F));

        tt_eval = e->eval;
        if (!is_pv && e->depth >= depth) {
            const int tt_score = score_from_tt(e->score, ply);
            if (e->bound == BOUND_EXACT) {
                return tt_score;
            }
            if (e->bound == BOUND_LOWER && tt_score >= beta) {
                return tt_score;
            }
            if (e->bound == BOUND_UPPER && tt_score <= alpha) {
                return tt_score;
            }
        }
    }

    const bool in_check = pos.in_check(pos.side_to_move());

    ++stats_.movegen_calls;
    MovePicker picker(pos, tt_move, false);
    stats_.moves_generated += static_cast<std::uint64_t>(picker.generated_count());

    int legal_moves = 0;
    int best_score = -VALUE_INFINITE;
    Move best_move{};

    while (true) {
        MovePickPhase phase = MovePickPhase::END;
        const Move move = picker.next(&phase);
        if (move.is_none()) {
            break;
        }

        ++stats_.move_pick_iterations;
        if (!pos.make_move(move)) {
            continue;
        }

        if (use_eval_move_hooks_) {
            evaluator_.on_make_move(pos, move);
        }

        ++legal_moves;

        PVLine child_pv{};
        int score;

        if (legal_moves == 1) {
            score = -search_node(pos, depth - 1, -beta, -alpha, ply + 1, is_pv, child_pv);
        } else {
            score = -search_node(pos, depth - 1, -alpha - 1, -alpha, ply + 1, false, child_pv);
            if (score > alpha && score < beta) {
                ++stats_.pvs_researches;
                score = -search_node(pos, depth - 1, -beta, -alpha, ply + 1, is_pv, child_pv);
            }
        }

        pos.unmake_move();
        if (use_eval_move_hooks_) {
            evaluator_.on_unmake_move(pos, move);
        }

        if (stop_) {
            return 0;
        }

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }

        if (score > alpha) {
            alpha = score;
            update_pv(pv, move, child_pv);
        }

        if (alpha >= beta) {
            ++stats_.beta_cutoffs;
            switch (phase) {
                case MovePickPhase::TT: ++stats_.cutoff_tt; break;
                case MovePickPhase::GOOD_CAPTURE: ++stats_.cutoff_good_capture; break;
                case MovePickPhase::QUIET: ++stats_.cutoff_quiet; break;
                case MovePickPhase::BAD_CAPTURE: ++stats_.cutoff_bad_capture; break;
                case MovePickPhase::END: break;
            }
            break;
        }
    }

    if (ply == 0) {
        root_legal_moves_ = legal_moves;
    }

    if (legal_moves == 0) {
        if (in_check) {
            return -VALUE_MATE + ply;
        }
        return 0;
    }

    std::uint8_t bound = BOUND_UPPER;
    if (best_score >= beta) {
        bound = BOUND_LOWER;
    } else if (best_score > alpha_orig) {
        bound = BOUND_EXACT;
    }

    const int eval = tt_eval != std::numeric_limits<int>::min() ? tt_eval : evaluator_.static_eval(pos);
    tt_.store(key, best_move, best_score, eval, depth, bound, generation_, ply);

    return best_score;
}

int Searcher::qsearch(Position& pos, int alpha, int beta, int ply, PVLine& pv) {
    pv.length = 0;
    seldepth_ = std::max(seldepth_, ply);
    ++stats_.nodes;
    ++stats_.qnodes;

    if (should_stop_hard()) {
        return 0;
    }

    if (pos.is_draw()) {
        return 0;
    }

    int stand_pat = evaluator_.static_eval(pos);
    if (stand_pat >= beta) {
        return stand_pat;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    ++stats_.movegen_calls;
    MovePicker picker(pos, Move{}, true);
    stats_.moves_generated += static_cast<std::uint64_t>(picker.generated_count());

    while (true) {
        MovePickPhase phase = MovePickPhase::END;
        const Move move = picker.next(&phase);
        if (move.is_none()) {
            break;
        }

        ++stats_.move_pick_iterations;
        if (!pos.make_move(move)) {
            continue;
        }

        if (use_eval_move_hooks_) {
            evaluator_.on_make_move(pos, move);
        }

        PVLine child{};
        const int score = -qsearch(pos, -beta, -alpha, ply + 1, child);

        pos.unmake_move();
        if (use_eval_move_hooks_) {
            evaluator_.on_unmake_move(pos, move);
        }

        if (stop_) {
            return 0;
        }

        if (score >= beta) {
            return score;
        }

        if (score > alpha) {
            alpha = score;
            update_pv(pv, move, child);
        }
    }

    return alpha;
}

}  // namespace makaira
