#include "hce_evaluator.h"

#include "bitboard.h"
#include "eval_params.h"
#include "eval_params_tuned.h"
#include "eval_tables.h"
#include "zobrist.h"

#include <algorithm>
#include <cmath>

namespace makaira {
namespace {

inline int rel_rank(Color c, Square sq) {
    return c == WHITE ? static_cast<int>(rank_of(sq)) : 7 - static_cast<int>(rank_of(sq));
}

inline int clamp_index(int v) {
    return std::clamp(v, 0, 15);
}

inline Bitboard shift_up(Color c, Bitboard b) {
    return c == WHITE ? (b << 8) : (b >> 8);
}

inline int sign_for(Color c) {
    return c == WHITE ? 1 : -1;
}

inline Score apply_scale(Score s, int mg_scale, int eg_scale) {
    return make_score((s.mg * mg_scale) / 100, (s.eg * eg_scale) / 100);
}

inline int square_color(Square sq) {
    return (static_cast<int>(file_of(sq)) + static_cast<int>(rank_of(sq))) & 1;
}

inline int king_centralization(Square sq) {
    const int f = static_cast<int>(file_of(sq));
    const int r = static_cast<int>(rank_of(sq));
    const int df = std::abs(2 * f - 7);
    const int dr = std::abs(2 * r - 7);
    return 14 - (df + dr);
}

Bitboard center_mask() {
    Bitboard b = 0;
    for (int f = FILE_C; f <= FILE_F; ++f) {
        for (int r = RANK_3; r <= RANK_6; ++r) {
            b |= bb_from(make_square(static_cast<File>(f), static_cast<Rank>(r)));
        }
    }
    return b;
}

const Bitboard CENTER_MASK = center_mask();

}  // namespace

HCEEvaluator::HCEEvaluator() :
    pawn_hash_(1ULL << 16) {
    eval_tables::init_eval_tables();
}

EvalStats HCEEvaluator::stats() const {
    return stats_;
}

void HCEEvaluator::clear_stats() {
    stats_ = {};
    pawn_hash_.clear();
}

int HCEEvaluator::static_eval(const Position& pos) const {
    return evaluate(pos, true, nullptr);
}

int HCEEvaluator::static_eval_trace(const Position& pos, EvalBreakdown* out) const {
    return evaluate(pos, true, out);
}

int HCEEvaluator::static_eval_recompute(const Position& pos) const {
    return evaluate(pos, false, nullptr);
}

Score HCEEvaluator::evaluate_material_psqt(const Position& pos, bool use_incremental) const {
    if (use_incremental) {
        return make_score(pos.mg_psqt(WHITE) - pos.mg_psqt(BLACK), pos.eg_psqt(WHITE) - pos.eg_psqt(BLACK));
    }

    Score s{};
    for (int sq = SQ_A1; sq <= SQ_H8; ++sq) {
        const Piece pc = pos.piece_on(static_cast<Square>(sq));
        if (pc == NO_PIECE) {
            continue;
        }

        const Score ps = eval_tables::psqt(pc, static_cast<Square>(sq));
        const int sign = sign_for(color_of(pc));
        s.mg += sign * ps.mg;
        s.eg += sign * ps.eg;
    }
    return s;
}

PawnHashEntry HCEEvaluator::compute_pawn_entry(const Position& pos, Key pawn_key_with_kings) const {
    PawnHashEntry e{};
    e.key = pawn_key_with_kings;

    std::array<Bitboard, COLOR_NB> pawn_attacks{{0, 0}};
    for (Color c : {WHITE, BLACK}) {
        Bitboard pawns = pos.pieces(c, PAWN);
        while (pawns) {
            const Square sq = pop_lsb(pawns);
            pawn_attacks[c] |= attacks::pawn[c][sq];
        }
    }

    for (Color c : {WHITE, BLACK}) {
        const Color them = ~c;
        const Bitboard our = pos.pieces(c, PAWN);
        const Bitboard enemy = pos.pieces(them, PAWN);

        Bitboard pawns = our;
        while (pawns) {
            const Square sq = pop_lsb(pawns);
            const int rr = rel_rank(c, sq);
            const int file = static_cast<int>(file_of(sq));
            const Bitboard sq_bb = bb_from(sq);

            const bool isolated = (our & eval_tables::ADJACENT_FILE_MASK[file]) == 0;
            if (isolated) {
                const Score v = make_score(-eval_params::ISOLATED_PAWN_PENALTY_MG * sign_for(c),
                                           -eval_params::ISOLATED_PAWN_PENALTY_EG * sign_for(c));
                e.isolated_score += v;
                e.pawn_score += v;
            }

            if (popcount(our & eval_tables::FILE_MASK[file]) > 1) {
                const Score v = make_score(-eval_params::DOUBLED_PAWN_PENALTY_MG * sign_for(c),
                                           -eval_params::DOUBLED_PAWN_PENALTY_EG * sign_for(c));
                e.doubled_score += v;
                e.pawn_score += v;
            }

            const Bitboard passed_mask = eval_tables::PASSED_MASK[c][sq];
            const bool is_passed = (enemy & passed_mask) == 0;
            if (is_passed) {
                e.passed[c] |= sq_bb;
                const Score passed = make_score(eval_params::PASSED_PAWN_MG[rr] * sign_for(c),
                                                eval_params::PASSED_PAWN_EG[rr] * sign_for(c));
                e.passed_score += passed;
                e.pawn_score += passed;

                if (pawn_attacks[c] & sq_bb) {
                    const Score v = make_score(eval_params::SUPPORTED_PASSER_BONUS_MG * sign_for(c),
                                               eval_params::SUPPORTED_PASSER_BONUS_EG * sign_for(c));
                    e.supported_score += v;
                    e.pawn_score += v;
                }

                if ((our & eval_tables::ADJACENT_FILE_MASK[file]) != 0) {
                    const Score v = make_score(eval_params::CONNECTED_PASSER_BONUS_MG * sign_for(c),
                                               eval_params::CONNECTED_PASSER_BONUS_EG * sign_for(c));
                    e.connected_score += v;
                    e.pawn_score += v;
                }

                const bool outside = file <= FILE_B || file >= FILE_G;
                if (outside) {
                    const Score v = make_score(eval_params::OUTSIDE_PASSER_BONUS_MG * sign_for(c),
                                               eval_params::OUTSIDE_PASSER_BONUS_EG * sign_for(c));
                    e.outside_score += v;
                    e.pawn_score += v;
                }

                const Square stop = c == WHITE ? static_cast<Square>(sq + 8) : static_cast<Square>(sq - 8);
                if (is_ok_square(stop) && pos.piece_on(stop) != NO_PIECE) {
                    const Score v = make_score(-eval_params::BLOCKED_PASSER_PENALTY_MG * sign_for(c),
                                               -eval_params::BLOCKED_PASSER_PENALTY_EG * sign_for(c));
                    e.blocked_score += v;
                    e.pawn_score += v;
                }
            } else {
                const Bitboard forward = eval_tables::FORWARD_MASK[c][sq];
                if ((enemy & forward) == 0) {
                    const Score v = make_score(eval_params::CANDIDATE_PAWN_BONUS_MG * sign_for(c),
                                               eval_params::CANDIDATE_PAWN_BONUS_EG * sign_for(c));
                    e.candidate_score += v;
                    e.pawn_score += v;
                }
            }

            const Square stop = c == WHITE ? static_cast<Square>(sq + 8) : static_cast<Square>(sq - 8);
            if (is_ok_square(stop)) {
                const bool blocked = pos.piece_on(stop) != NO_PIECE;
                const bool no_support = (our & eval_tables::ADJACENT_FILE_MASK[file] & eval_tables::FORWARD_MASK[them][sq]) == 0;
                if (blocked && no_support && (pawn_attacks[them] & bb_from(stop))) {
                    const Score v = make_score(-eval_params::BACKWARD_PAWN_PENALTY_MG * sign_for(c),
                                               -eval_params::BACKWARD_PAWN_PENALTY_EG * sign_for(c));
                    e.backward_score += v;
                    e.pawn_score += v;
                }
            }
        }
    }

    for (Color c : {WHITE, BLACK}) {
        const Square ksq = pos.king_square(c);
        const int kf = static_cast<int>(file_of(ksq));
        const int kr = static_cast<int>(rank_of(ksq));
        int shelter = 0;

        for (int df = -1; df <= 1; ++df) {
            const int f = kf + df;
            if (f < FILE_A || f > FILE_H) {
                continue;
            }

            Bitboard file_pawns = pos.pieces(c, PAWN) & eval_tables::FILE_MASK[f];
            while (file_pawns) {
                const Square psq = pop_lsb(file_pawns);
                const int dist = c == WHITE ? static_cast<int>(rank_of(psq)) - kr : kr - static_cast<int>(rank_of(psq));
                if (dist >= 0 && dist <= 7) {
                    const int v = eval_params::SHELTER_PAWN_BONUS[dist];
                    shelter += v;
                    break;
                }
            }

            Bitboard storms = pos.pieces(~c, PAWN) & eval_tables::FILE_MASK[f];
            while (storms) {
                const Square esq = pop_lsb(storms);
                const int dist = c == WHITE ? kr - static_cast<int>(rank_of(esq)) : static_cast<int>(rank_of(esq)) - kr;
                if (dist >= 0 && dist <= 7) {
                    e.storm_penalty_mg[c] += eval_params::STORM_PAWN_PENALTY[dist];
                    break;
                }
            }
        }

        e.shelter_bonus_mg[c] = shelter;
    }

    return e;
}

HCEEvaluator::AttackInfo HCEEvaluator::build_attack_info(const Position& pos) const {
    AttackInfo ai{};
    const Bitboard occ = pos.occupancy();

    for (Color c : {WHITE, BLACK}) {
        Bitboard pawns = pos.pieces(c, PAWN);
        while (pawns) {
            const Square sq = pop_lsb(pawns);
            ai.pawn_attacks[c] |= attacks::pawn[c][sq];
            ai.all_attacks[c] |= attacks::pawn[c][sq];
        }
    }

    for (Color c : {WHITE, BLACK}) {
        const Color them = ~c;
        const Bitboard own_occ = pos.occupancy(c);
        const Bitboard enemy_pawn_attacks = ai.pawn_attacks[them];
        const Square enemy_king = pos.king_square(them);
        const Bitboard king_ring = attacks::king[enemy_king] | bb_from(enemy_king);

        for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN}) {
            Bitboard pieces = pos.pieces(c, pt);
            while (pieces) {
                const Square sq = pop_lsb(pieces);
                Bitboard atk = 0;
                if (pt == KNIGHT) {
                    atk = attacks::knight[sq];
                } else if (pt == BISHOP) {
                    atk = attacks::bishop_attacks(sq, occ);
                } else if (pt == ROOK) {
                    atk = attacks::rook_attacks(sq, occ);
                } else {
                    atk = attacks::bishop_attacks(sq, occ) | attacks::rook_attacks(sq, occ);
                }

                ai.all_attacks[c] |= atk;

                const Bitboard mobility_targets = atk & ~own_occ & ~enemy_pawn_attacks;
                const int mob = clamp_index(popcount(mobility_targets));
                ai.mobility.mg += sign_for(c) * eval_params::MOBILITY_BONUS_MG[pt][mob];
                ai.mobility.eg += sign_for(c) * eval_params::MOBILITY_BONUS_EG[pt][mob];

                const int ring_hits = popcount(atk & king_ring);
                if (ring_hits > 0) {
                    ai.king_attackers[c] += 1;
                    ai.king_attack_units[c] += ring_hits * eval_params::KING_ATTACK_UNIT[pt];
                }
            }
        }

        ai.all_attacks[c] |= attacks::king[pos.king_square(c)];
    }

    return ai;
}

Score HCEEvaluator::evaluate_piece_features(const Position& pos, const AttackInfo&, EvalBreakdown* out) const {
    Score s{};

    for (Color c : {WHITE, BLACK}) {
        const int sign = sign_for(c);

        if (popcount(pos.pieces(c, BISHOP)) >= 2) {
            const Score v = eval_params::BISHOP_PAIR_BONUS * sign;
            s += v;
            if (out) {
                out->piece_bishop_pair += v;
            }
        }

        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            const Square sq = pop_lsb(rooks);
            const int f = static_cast<int>(file_of(sq));
            const Bitboard file_mask = eval_tables::FILE_MASK[f];
            const bool own_pawn = (pos.pieces(c, PAWN) & file_mask) != 0;
            const bool enemy_pawn = (pos.pieces(~c, PAWN) & file_mask) != 0;

            if (!own_pawn && !enemy_pawn) {
                const Score v = eval_params::ROOK_OPEN_FILE_BONUS * sign;
                s += v;
                if (out) {
                    out->piece_rook_file += v;
                }
            } else if (!own_pawn && enemy_pawn) {
                const Score v = eval_params::ROOK_SEMIOPEN_FILE_BONUS * sign;
                s += v;
                if (out) {
                    out->piece_rook_file += v;
                }
            }

            const int rr = rel_rank(c, sq);
            if (rr == 6) {
                const Score v = eval_params::ROOK_ON_SEVENTH_BONUS * sign;
                s += v;
                if (out) {
                    out->piece_rook_seventh += v;
                }
            }
        }

        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            const Square sq = pop_lsb(knights);
            const int rr = rel_rank(c, sq);
            if (rr < 3 || rr > 5) {
                continue;
            }

            const Bitboard sq_bb = bb_from(sq);
            Bitboard support = 0;
            Bitboard pawns = pos.pieces(c, PAWN);
            while (pawns) {
                const Square psq = pop_lsb(pawns);
                support |= attacks::pawn[c][psq];
            }

            Bitboard enemy_attacks = 0;
            Bitboard epawns = pos.pieces(~c, PAWN);
            while (epawns) {
                const Square psq = pop_lsb(epawns);
                enemy_attacks |= attacks::pawn[~c][psq];
            }

            if ((support & sq_bb) && !(enemy_attacks & sq_bb)) {
                const Score v = eval_params::KNIGHT_OUTPOST_BONUS * sign;
                s += v;
                if (out) {
                    out->piece_knight_outpost += v;
                }
            }
        }

        Bitboard bishops = pos.pieces(c, BISHOP);
        int bad_bishop_pawns = 0;
        while (bishops) {
            const Square bsq = pop_lsb(bishops);
            const int bcolor = square_color(bsq);
            Bitboard pawns = pos.pieces(c, PAWN);
            while (pawns) {
                const Square psq = pop_lsb(pawns);
                if (square_color(psq) == bcolor) {
                    ++bad_bishop_pawns;
                }
            }
        }
        const Score v = eval_params::BAD_BISHOP_PENALTY * (-sign * bad_bishop_pawns / 2);
        s += v;
        if (out) {
            out->piece_bad_bishop += v;
        }
    }

    return s;
}

Score HCEEvaluator::evaluate_threats(const Position& pos, const AttackInfo& ai, EvalBreakdown* out) const {
    Score s{};

    for (Color c : {WHITE, BLACK}) {
        const Color them = ~c;
        const int sign = sign_for(c);

        Bitboard enemy_pieces = pos.occupancy(them) & ~pos.pieces(them, KING);
        Bitboard pawn_threats = ai.pawn_attacks[c] & enemy_pieces;
        while (pawn_threats) {
            const Square sq = pop_lsb(pawn_threats);
            const Bitboard bb = bb_from(sq);
            if (!(ai.all_attacks[them] & bb)) {
                const Score v = eval_params::THREAT_BY_PAWN_BONUS * sign;
                s += v;
                if (out) {
                    out->threat_pawn += v;
                }
            }
        }

        Bitboard hanging = ai.all_attacks[c] & enemy_pieces & ~ai.all_attacks[them];
        int n = popcount(hanging);
        const Score v = eval_params::HANGING_PIECE_BONUS * (sign * n);
        s += v;
        if (out) {
            out->threat_hanging += v;
        }
    }

    return s;
}

Score HCEEvaluator::evaluate_space(const Position& pos, const AttackInfo& ai) const {
    Score s{};

    for (Color c : {WHITE, BLACK}) {
        if (popcount(pos.pieces(c, PAWN)) < 4) {
            continue;
        }

        const Bitboard controlled = ai.all_attacks[c] & CENTER_MASK;
        const Bitboard free = controlled & ~pos.occupancy(c) & ~pos.pieces(c, PAWN);
        const int gain = popcount(free);
        s += eval_params::SPACE_BONUS * (sign_for(c) * gain);
    }

    return s;
}

Score HCEEvaluator::evaluate_endgame_terms(const Position& pos, EvalBreakdown* out) const {
    Score s{};

    const Square wk = pos.king_square(WHITE);
    const Square bk = pos.king_square(BLACK);

    const int w_center = king_centralization(wk);
    const int b_center = king_centralization(bk);
    s.eg += (w_center - b_center) * eval_params::KING_ACTIVITY_BONUS.eg / 8;
    if (out) {
        out->endgame_king_activity = s;
    }

    return s;
}

int HCEEvaluator::evaluate_endgame_scale(const Position& pos, int blended_white_pov) const {
    int scale = 128;

    const bool only_bishops =
      pos.pieces(WHITE, KNIGHT) == 0 && pos.pieces(BLACK, KNIGHT) == 0
      && pos.pieces(WHITE, ROOK) == 0 && pos.pieces(BLACK, ROOK) == 0
      && pos.pieces(WHITE, QUEEN) == 0 && pos.pieces(BLACK, QUEEN) == 0;

    if (only_bishops && popcount(pos.pieces(WHITE, BISHOP)) == 1 && popcount(pos.pieces(BLACK, BISHOP)) == 1) {
        scale = 96;
    }

    const int total_pawns = popcount(pos.pieces(WHITE, PAWN) | pos.pieces(BLACK, PAWN));
    if (total_pawns <= 2 && std::abs(blended_white_pov) < 120) {
        scale = std::min(scale, 88);
    }

    return scale;
}

int HCEEvaluator::evaluate(const Position& pos, bool use_incremental, EvalBreakdown* out) const {
    ++stats_.eval_calls;

    EvalBreakdown b{};

    b.material_psqt = evaluate_material_psqt(pos, use_incremental);

    Key pawn_key = pos.pawn_key()
                 ^ Zobrist.pawn_file_king[WHITE][file_of(pos.king_square(WHITE))]
                 ^ Zobrist.pawn_file_king[BLACK][file_of(pos.king_square(BLACK))];

    const PawnHashEntry* cached = pawn_hash_.probe(pawn_key);
    PawnHashEntry entry{};
    if (cached) {
        ++stats_.pawn_hash_hits;
        entry = *cached;
    } else {
        ++stats_.pawn_hash_misses;
        entry = compute_pawn_entry(pos, pawn_key);
        pawn_hash_.store(entry);
    }

    b.pawns_passed = entry.passed_score;
    b.pawns_isolated = entry.isolated_score;
    b.pawns_doubled = entry.doubled_score;
    b.pawns_backward = entry.backward_score;
    b.pawns_candidate = entry.candidate_score;
    b.pawns_connected = entry.connected_score;
    b.pawns_supported = entry.supported_score;
    b.pawns_outside = entry.outside_score;
    b.pawns_blocked = entry.blocked_score;

    b.pawns = b.pawns_passed + b.pawns_isolated + b.pawns_doubled + b.pawns_backward + b.pawns_candidate + b.pawns_connected
            + b.pawns_supported + b.pawns_outside + b.pawns_blocked;

    b.king_shelter.mg = entry.shelter_bonus_mg[WHITE] - entry.shelter_bonus_mg[BLACK];
    b.king_storm.mg = -(entry.storm_penalty_mg[WHITE] - entry.storm_penalty_mg[BLACK]);
    b.king_safety += b.king_shelter;
    b.king_safety += b.king_storm;

    const AttackInfo ai = build_attack_info(pos);
    b.mobility = ai.mobility;

    for (Color c : {WHITE, BLACK}) {
        const Color them = ~c;
        const int sign = sign_for(c);
        const int attackers = std::clamp(ai.king_attackers[c], 0, 7);
        const int base_units = ai.king_attack_units[c];
        const int np_scale = std::clamp(pos.non_pawn_material(c) / 8, 0, 128);
        int danger = (base_units * eval_params::KING_DANGER_SCALE[attackers] * np_scale) / 256;

        if (pos.non_pawn_material(them) < 1200) {
            danger -= danger / 3;
        }
        b.king_danger.mg += sign * danger;
        b.king_safety.mg += sign * danger;
    }

    b.piece_features = evaluate_piece_features(pos, ai, &b);
    b.threats = evaluate_threats(pos, ai, &b);
    b.space = evaluate_space(pos, ai);

    b.endgame_scale = evaluate_endgame_scale(pos, b.material_psqt.mg + b.pawns.mg);

    Score total{};
    total += apply_scale(
      b.material_psqt, eval_params_tuned::MATERIAL_PSQT_MG_SCALE, eval_params_tuned::MATERIAL_PSQT_EG_SCALE);
    {
        Score pawns_scaled{};
        pawns_scaled += apply_scale(
          b.pawns_passed, eval_params_tuned::PAWN_PASSED_MG_SCALE, eval_params_tuned::PAWN_PASSED_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_isolated, eval_params_tuned::PAWN_ISOLATED_MG_SCALE, eval_params_tuned::PAWN_ISOLATED_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_doubled, eval_params_tuned::PAWN_DOUBLED_MG_SCALE, eval_params_tuned::PAWN_DOUBLED_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_backward, eval_params_tuned::PAWN_BACKWARD_MG_SCALE, eval_params_tuned::PAWN_BACKWARD_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_candidate, eval_params_tuned::PAWN_CANDIDATE_MG_SCALE, eval_params_tuned::PAWN_CANDIDATE_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_connected, eval_params_tuned::PAWN_CONNECTED_MG_SCALE, eval_params_tuned::PAWN_CONNECTED_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_supported, eval_params_tuned::PAWN_SUPPORTED_MG_SCALE, eval_params_tuned::PAWN_SUPPORTED_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_outside, eval_params_tuned::PAWN_OUTSIDE_MG_SCALE, eval_params_tuned::PAWN_OUTSIDE_EG_SCALE);
        pawns_scaled += apply_scale(
          b.pawns_blocked, eval_params_tuned::PAWN_BLOCKED_MG_SCALE, eval_params_tuned::PAWN_BLOCKED_EG_SCALE);
        total += apply_scale(pawns_scaled, eval_params_tuned::PAWN_MG_SCALE, eval_params_tuned::PAWN_EG_SCALE);
    }
    total += apply_scale(b.mobility, eval_params_tuned::MOBILITY_MG_SCALE, eval_params_tuned::MOBILITY_EG_SCALE);
    {
        Score king_scaled{};
        king_scaled += apply_scale(
          b.king_shelter, eval_params_tuned::KING_SHELTER_MG_SCALE, eval_params_tuned::KING_SHELTER_EG_SCALE);
        king_scaled += apply_scale(
          b.king_storm, eval_params_tuned::KING_STORM_MG_SCALE, eval_params_tuned::KING_STORM_EG_SCALE);
        king_scaled += apply_scale(
          b.king_danger, eval_params_tuned::KING_DANGER_MG_SCALE, eval_params_tuned::KING_DANGER_EG_SCALE);
        total += apply_scale(king_scaled, eval_params_tuned::KING_MG_SCALE, eval_params_tuned::KING_EG_SCALE);
    }
    {
        Score piece_scaled{};
        piece_scaled += apply_scale(b.piece_bishop_pair,
                                    eval_params_tuned::PIECE_BISHOP_PAIR_MG_SCALE,
                                    eval_params_tuned::PIECE_BISHOP_PAIR_EG_SCALE);
        piece_scaled += apply_scale(
          b.piece_rook_file, eval_params_tuned::PIECE_ROOK_FILE_MG_SCALE, eval_params_tuned::PIECE_ROOK_FILE_EG_SCALE);
        piece_scaled += apply_scale(b.piece_rook_seventh,
                                    eval_params_tuned::PIECE_ROOK_SEVENTH_MG_SCALE,
                                    eval_params_tuned::PIECE_ROOK_SEVENTH_EG_SCALE);
        piece_scaled += apply_scale(b.piece_knight_outpost,
                                    eval_params_tuned::PIECE_KNIGHT_OUTPOST_MG_SCALE,
                                    eval_params_tuned::PIECE_KNIGHT_OUTPOST_EG_SCALE);
        piece_scaled += apply_scale(
          b.piece_bad_bishop, eval_params_tuned::PIECE_BAD_BISHOP_MG_SCALE, eval_params_tuned::PIECE_BAD_BISHOP_EG_SCALE);
        total += apply_scale(piece_scaled, eval_params_tuned::PIECE_MG_SCALE, eval_params_tuned::PIECE_EG_SCALE);
    }
    {
        Score threat_scaled{};
        threat_scaled += apply_scale(
          b.threat_hanging, eval_params_tuned::THREAT_HANGING_MG_SCALE, eval_params_tuned::THREAT_HANGING_EG_SCALE);
        threat_scaled += apply_scale(
          b.threat_pawn, eval_params_tuned::THREAT_PAWN_MG_SCALE, eval_params_tuned::THREAT_PAWN_EG_SCALE);
        total += apply_scale(threat_scaled, eval_params_tuned::THREAT_MG_SCALE, eval_params_tuned::THREAT_EG_SCALE);
    }
    total += apply_scale(b.space, eval_params_tuned::SPACE_MG_SCALE, eval_params_tuned::SPACE_EG_SCALE);
    b.endgame_terms = evaluate_endgame_terms(pos, &b);
    total += apply_scale(
      b.endgame_terms,
      eval_params_tuned::ENDGAME_KING_ACTIVITY_MG_SCALE,
      eval_params_tuned::ENDGAME_KING_ACTIVITY_EG_SCALE);
    const int tuned_tempo = (eval_params::TEMPO_BONUS * eval_params_tuned::TEMPO_SCALE) / 100;
    const int tempo_sign = pos.side_to_move() == WHITE ? 1 : -1;
    total.mg += tuned_tempo * tempo_sign;
    b.tempo = tuned_tempo * tempo_sign;

    const int phase = use_incremental ? std::clamp(pos.phase(), 0, eval_params::MAX_PHASE)
                                      : std::clamp(popcount(pos.pieces(WHITE, KNIGHT) | pos.pieces(BLACK, KNIGHT))
                                                     + popcount(pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP))
                                                     + 2 * popcount(pos.pieces(WHITE, ROOK) | pos.pieces(BLACK, ROOK))
                                                     + 4 * popcount(pos.pieces(WHITE, QUEEN) | pos.pieces(BLACK, QUEEN)),
                                                   0,
                                                   eval_params::MAX_PHASE);
    b.phase = phase;

    int blended = (total.mg * phase + total.eg * (eval_params::MAX_PHASE - phase)) / eval_params::MAX_PHASE;
    blended = (blended * b.endgame_scale) / 128;
    b.total_white_pov = blended;

    if (out) {
        *out = b;
    }

    return pos.side_to_move() == WHITE ? blended : -blended;
}

}  // namespace makaira
