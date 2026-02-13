#include "lc0_features112.h"

#include "bitboard.h"

#include <vector>

namespace makaira::lc0 {

namespace {

void set_plane_all(InputPlanes112& out, int plane, float value) {
    const int base = plane * k_squares;
    for (int i = 0; i < k_squares; ++i) {
        out[base + i] = value;
    }
}

void fill_plane_bb(InputPlanes112& out, int plane, Bitboard bb, float value = 1.0f) {
    const int base = plane * k_squares;
    while (bb) {
        const Square sq = pop_lsb(bb);
        out[base + static_cast<int>(sq)] = value;
    }
}

std::vector<Position> reconstruct_history(const Position& pos, int plies) {
    std::vector<Position> states;
    states.reserve(plies);

    Position cur = pos;
    states.push_back(cur);

    for (int i = 1; i < plies; ++i) {
        if (cur.history().empty()) {
            break;
        }
        cur.unmake_move();
        states.push_back(cur);
    }

    while (static_cast<int>(states.size()) < plies) {
        states.push_back(states.back());
    }

    return states;
}

}  // namespace

InputPlanes112 extract_features_112(const Position& pos) {
    InputPlanes112 out{};

    // 8 history plies * 13 planes = 104 planes.
    const auto states = reconstruct_history(pos, 8);

    for (int h = 0; h < 8; ++h) {
        const Position& s = states[h];
        const Color ours = s.side_to_move();
        const Color theirs = ~ours;

        const int base = h * 13;

        fill_plane_bb(out, base + 0, s.pieces(ours, PAWN));
        fill_plane_bb(out, base + 1, s.pieces(ours, KNIGHT));
        fill_plane_bb(out, base + 2, s.pieces(ours, BISHOP));
        fill_plane_bb(out, base + 3, s.pieces(ours, ROOK));
        fill_plane_bb(out, base + 4, s.pieces(ours, QUEEN));
        fill_plane_bb(out, base + 5, s.pieces(ours, KING));

        fill_plane_bb(out, base + 6, s.pieces(theirs, PAWN));
        fill_plane_bb(out, base + 7, s.pieces(theirs, KNIGHT));
        fill_plane_bb(out, base + 8, s.pieces(theirs, BISHOP));
        fill_plane_bb(out, base + 9, s.pieces(theirs, ROOK));
        fill_plane_bb(out, base + 10, s.pieces(theirs, QUEEN));
        fill_plane_bb(out, base + 11, s.pieces(theirs, KING));

        if (s.is_repetition()) {
            set_plane_all(out, base + 12, 1.0f);
        }
    }

    const Position& cur = states[0];
    const Color stm = cur.side_to_move();
    const int cr = cur.castling_rights();

    const bool we_can_ooo = stm == WHITE ? (cr & WHITE_OOO) != 0 : (cr & BLACK_OOO) != 0;
    const bool we_can_oo = stm == WHITE ? (cr & WHITE_OO) != 0 : (cr & BLACK_OO) != 0;
    const bool they_can_ooo = stm == WHITE ? (cr & BLACK_OOO) != 0 : (cr & WHITE_OOO) != 0;
    const bool they_can_oo = stm == WHITE ? (cr & BLACK_OO) != 0 : (cr & WHITE_OO) != 0;

    if (we_can_ooo) set_plane_all(out, 104, 1.0f);
    if (we_can_oo) set_plane_all(out, 105, 1.0f);
    if (they_can_ooo) set_plane_all(out, 106, 1.0f);
    if (they_can_oo) set_plane_all(out, 107, 1.0f);

    if (stm == BLACK) {
        set_plane_all(out, 108, 1.0f);
    }

    set_plane_all(out, 109, static_cast<float>(cur.halfmove_clock()));
    // 110 is zero by default.
    set_plane_all(out, 111, 1.0f);

    return out;
}

}  // namespace makaira::lc0
