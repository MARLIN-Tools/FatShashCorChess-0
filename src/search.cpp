#include "search.h"

#include "movepicker.h"
#include "movegen.h"

#include <algorithm>
#include <cmath>
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
    history_.assign(HISTORY_SIZE, 0);
    cont_history_.assign(CONT_HISTORY_SIZE, 0);
    lmr_table_.assign((MAX_PLY + 1) * 256, 0);
    clear_heuristics();

    for (int d = 0; d <= MAX_PLY; ++d) {
        for (int m = 0; m < 256; ++m) {
            if (d < 2 || m < 2) {
                lmr_table_[d * 256 + m] = 0;
                continue;
            }

            const double rd = std::log(static_cast<double>(d));
            const double rm = std::log(static_cast<double>(m));
            lmr_table_[d * 256 + m] = std::max(1, static_cast<int>(std::floor((rd * rm) / 2.0)));
        }
    }
}

void Searcher::set_hash_size_mb(std::size_t mb) {
    tt_.resize_mb(mb);
}

void Searcher::clear_hash() {
    tt_.clear();
}

void Searcher::clear_heuristics() {
    std::fill(history_.begin(), history_.end(), std::int16_t{0});
    std::fill(cont_history_.begin(), cont_history_.end(), std::int16_t{0});
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

int Searcher::move_index(Piece pc, Square to) {
    if (pc == NO_PIECE || !is_ok_square(to)) {
        return -1;
    }
    return static_cast<int>(pc) * SQ_NB + static_cast<int>(to);
}

int Searcher::quiet_move_score(const Position& pos, Move move, int ply) const {
    if (!config_.use_history && !config_.use_cont_history) {
        return 0;
    }

    int score = 0;
    if (config_.use_history) {
        const int idx =
          (static_cast<int>(pos.side_to_move()) * SQ_NB + static_cast<int>(move.from())) * SQ_NB + static_cast<int>(move.to());
        score += history_[idx];
    }

    if (config_.use_cont_history) {
        const int cur = move_index(pos.piece_on(move.from()), move.to());
        if (cur >= 0) {
            const int prev1 = stack_[ply].move_index;
            const int prev2 = ply > 0 ? stack_[ply - 1].move_index : -1;
            if (prev1 >= 0) {
                score += cont_history_[prev1 * MOVE_INDEX_NB + cur];
            }
            if (prev2 >= 0) {
                score += cont_history_[prev2 * MOVE_INDEX_NB + cur] / std::max(1, config_.cont_history_2ply_divisor);
            }
        }
    }

    return score;
}

void Searcher::update_history_value(int& value, int bonus) const {
    const int max_h = std::max(1, config_.history_max);
    int next = value + bonus - (value * std::abs(bonus)) / max_h;
    next = std::clamp(next, -max_h, max_h);
    value = next;
}

void Searcher::update_quiet_history(const Position& pos,
                                    Color side,
                                    Move best_move,
                                    int ply,
                                    int depth,
                                    const std::array<Move, 256>& quiet_tried,
                                    int quiet_count) {
    if (!config_.use_history && !config_.use_cont_history) {
        return;
    }

    const int bonus = std::max(1, depth * depth * std::max(1, config_.history_bonus_scale));
    const int malus = std::max(1, bonus / std::max(1, config_.history_malus_divisor));

    auto update_quiet = [&](Move m, int delta) {
        if (m.is_none() || m.is_capture() || m.is_promotion()) {
            return;
        }

        if (config_.use_history) {
            const int idx =
              (static_cast<int>(side) * SQ_NB + static_cast<int>(m.from())) * SQ_NB + static_cast<int>(m.to());
            int v = history_[idx];
            update_history_value(v, delta);
            history_[idx] = static_cast<std::int16_t>(v);
            ++stats_.history_updates;
        }

        if (config_.use_cont_history) {
            const int cur = move_index(pos.piece_on(m.from()), m.to());
            if (cur >= 0) {
                const int prev1 = stack_[ply].move_index;
                const int prev2 = ply > 0 ? stack_[ply - 1].move_index : -1;
                if (prev1 >= 0) {
                    const int idx = prev1 * MOVE_INDEX_NB + cur;
                    int v = cont_history_[idx];
                    update_history_value(v, delta);
                    cont_history_[idx] = static_cast<std::int16_t>(v);
                    ++stats_.cont_history_updates;
                }
                if (prev2 >= 0) {
                    const int idx = prev2 * MOVE_INDEX_NB + cur;
                    int v = cont_history_[idx];
                    update_history_value(v, delta / std::max(1, config_.cont_history_2ply_divisor));
                    cont_history_[idx] = static_cast<std::int16_t>(v);
                    ++stats_.cont_history_updates;
                }
            }
        }
    };

    update_quiet(best_move, bonus);
    for (int i = 0; i < quiet_count; ++i) {
        if (quiet_tried[i] == best_move) {
            continue;
        }
        update_quiet(quiet_tried[i], -malus);
    }
}

int Searcher::nmp_reduction(int depth) const {
    return std::clamp(config_.nmp_base_reduction + depth / std::max(1, config_.nmp_depth_divisor), 1, depth - 1);
}

int Searcher::lmr_reduction(int depth, int move_count, int quiet_score) const {
    if (depth <= 1) {
        return 0;
    }

    const int d = std::min(depth, MAX_PLY);
    const int m = std::min(move_count, 255);
    int r = lmr_table_[d * 256 + m];
    if (quiet_score >= config_.lmr_history_threshold) {
        --r;
    }
    return std::clamp(r, 0, depth - 1);
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
    for (auto& s : stack_) {
        s = SearchStackEntry{};
    }
    stack_[0].move_index = -1;
    stack_[0].did_null = false;

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
    const Color us = pos.side_to_move();

    int static_eval = tt_eval != std::numeric_limits<int>::min() ? tt_eval : evaluator_.static_eval(pos);
    stack_[ply].static_eval = static_eval;

    if (config_.use_nmp
        && depth >= config_.nmp_min_depth
        && !is_pv
        && !in_check
        && !stack_[ply].did_null
        && std::abs(beta) < MATE_SCORE_FOR_TT
        && pos.non_pawn_material(us) >= config_.nmp_non_pawn_min
        && static_eval >= beta - (config_.nmp_margin_base + config_.nmp_margin_per_depth * depth)) {
        ++stats_.nmp_attempts;

        const int r = nmp_reduction(depth);
        pos.make_null_move();
        stack_[ply + 1].move_index = -1;
        stack_[ply + 1].did_null = true;
        stack_[ply + 1].static_eval = 0;

        PVLine null_pv{};
        const int null_score = -search_node(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, false, null_pv);

        pos.unmake_null_move();
        stack_[ply + 1] = SearchStackEntry{};

        if (stop_) {
            return 0;
        }

        if (null_score >= beta) {
            const bool verify =
              depth >= config_.nmp_verify_min_depth || pos.non_pawn_material(us) <= config_.nmp_verify_non_pawn_max;
            if (verify) {
                ++stats_.nmp_verifications;
                PVLine verify_pv{};
                const int verify_score = search_node(pos, depth - 1 - r, beta - 1, beta, ply, false, verify_pv);
                if (verify_score >= beta) {
                    ++stats_.nmp_cutoffs;
                    return verify_score;
                }
                ++stats_.nmp_verification_fails;
            } else {
                ++stats_.nmp_cutoffs;
                return null_score;
            }
        }
    }

    QuietOrderContext quiet_ctx{};
    quiet_ctx.history = history_.data();
    quiet_ctx.cont_history = cont_history_.data();
    quiet_ctx.use_history = config_.use_history;
    quiet_ctx.use_cont_history = config_.use_cont_history;
    quiet_ctx.side = us;
    quiet_ctx.prev1_move_index = stack_[ply].move_index;
    quiet_ctx.prev2_move_index = ply > 0 ? stack_[ply - 1].move_index : -1;
    quiet_ctx.cont_history_2ply_divisor = config_.cont_history_2ply_divisor;

    ++stats_.movegen_calls;
    MovePicker picker(pos, tt_move, false, &quiet_ctx);
    stats_.moves_generated += static_cast<std::uint64_t>(picker.generated_count());

    int legal_moves = 0;
    int best_score = -VALUE_INFINITE;
    Move best_move{};
    std::array<Move, 256> quiet_tried{};
    int quiet_count = 0;

    while (true) {
        MovePickPhase phase = MovePickPhase::END;
        const Move move = picker.next(&phase);
        if (move.is_none()) {
            break;
        }

        ++stats_.move_pick_iterations;
        const bool is_quiet = !move.is_capture() && !move.is_promotion();
        const int quiet_score = is_quiet ? quiet_move_score(pos, move, ply) : 0;
        const int move_idx = move_index(pos.piece_on(move.from()), move.to());

        if (!pos.make_move(move)) {
            continue;
        }

        if (use_eval_move_hooks_) {
            evaluator_.on_make_move(pos, move);
        }

        ++legal_moves;
        if (is_quiet && quiet_count < static_cast<int>(quiet_tried.size())) {
            quiet_tried[quiet_count++] = move;
        }

        stack_[ply + 1].move_index = move_idx;
        stack_[ply + 1].did_null = false;
        stack_[ply + 1].static_eval = 0;

        PVLine child_pv{};
        int score = 0;

        const int next_depth = depth - 1;
        const bool gives_check = pos.in_check(pos.side_to_move());

        if (legal_moves == 1) {
            score = -search_node(pos, next_depth, -beta, -alpha, ply + 1, is_pv, child_pv);
        } else {
            bool reduced = false;
            if (config_.use_lmr
                && depth >= config_.lmr_min_depth
                && !is_pv
                && !in_check
                && is_quiet
                && move != tt_move
                && legal_moves > config_.lmr_full_depth_moves
                && !gives_check) {
                const int red = lmr_reduction(depth, legal_moves, quiet_score);
                if (red > 0) {
                    reduced = true;
                    ++stats_.lmr_reduced;
                    score = -search_node(pos, next_depth - red, -alpha - 1, -alpha, ply + 1, false, child_pv);

                    if (score > alpha) {
                        ++stats_.lmr_fail_high_after_reduce;
                        ++stats_.lmr_researches;
                        score = -search_node(pos, next_depth, -alpha - 1, -alpha, ply + 1, false, child_pv);
                    }
                }
            }

            if (!reduced) {
                score = -search_node(pos, next_depth, -alpha - 1, -alpha, ply + 1, false, child_pv);
            }

            if (score > alpha && score < beta) {
                ++stats_.pvs_researches;
                score = -search_node(pos, next_depth, -beta, -alpha, ply + 1, is_pv, child_pv);
            }
        }

        pos.unmake_move();
        stack_[ply + 1] = SearchStackEntry{};
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

            if (is_quiet) {
                update_quiet_history(pos, us, move, ply, depth, quiet_tried, quiet_count);
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

    tt_.store(key, best_move, best_score, static_eval, depth, bound, generation_, ply);

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
