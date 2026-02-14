#include "lc0_features112.h"

#include "bitboard.h"

namespace makaira::lc0 {

namespace {

void set_plane_all(InputPlanes112& out, int plane, float value) {
    const int base = plane * k_squares;
    for (int i = 0; i < k_squares; ++i) {
        out[base + i] = value;
    }
}

void fill_plane_bb(InputPlanes112& out, int plane, Bitboard bb, bool mirror, float value = 1.0f) {
    const int base = plane * k_squares;
    while (bb) {
        const Square sq = pop_lsb(bb);
        const int idx = mirror ? (static_cast<int>(sq) ^ 56) : static_cast<int>(sq);
        out[base + idx] = value;
    }
}

}  // namespace

InputPlanes112 extract_features_112(const Position& pos) {
    InputPlanes112 out{};
    const Color current_stm = pos.side_to_move();
    const bool mirror_all = current_stm == BLACK;

    // 8 history plies * 13 planes = 104 planes.
    Position hist = pos;
    for (int h = 0; h < 8; ++h) {
        if (h > 0) {
            // For INPUT_CLASSICAL_112_PLANE lc0 leaves unavailable history
            // planes empty when history cannot be reconstructed.
            if (hist.history().empty()) {
                break;
            }
            hist.unmake_move();
        }

        const Position& s = hist;
        // lc0 Position history stores boards already transformed into the
        // side-to-move frame at each ply. In our absolute-board representation
        // we emulate INPUT_CLASSICAL_112_PLANE by applying the current
        // side-to-move frame to all history slices.
        const bool mirror = mirror_all;
        const Color ours = current_stm;
        const Color theirs = ~ours;

        const int base = h * 13;

        fill_plane_bb(out, base + 0, s.pieces(ours, PAWN), mirror);
        fill_plane_bb(out, base + 1, s.pieces(ours, KNIGHT), mirror);
        fill_plane_bb(out, base + 2, s.pieces(ours, BISHOP), mirror);
        fill_plane_bb(out, base + 3, s.pieces(ours, ROOK), mirror);
        fill_plane_bb(out, base + 4, s.pieces(ours, QUEEN), mirror);
        fill_plane_bb(out, base + 5, s.pieces(ours, KING), mirror);

        fill_plane_bb(out, base + 6, s.pieces(theirs, PAWN), mirror);
        fill_plane_bb(out, base + 7, s.pieces(theirs, KNIGHT), mirror);
        fill_plane_bb(out, base + 8, s.pieces(theirs, BISHOP), mirror);
        fill_plane_bb(out, base + 9, s.pieces(theirs, ROOK), mirror);
        fill_plane_bb(out, base + 10, s.pieces(theirs, QUEEN), mirror);
        fill_plane_bb(out, base + 11, s.pieces(theirs, KING), mirror);

        if (s.is_repetition()) {
            set_plane_all(out, base + 12, 1.0f);
        }
    }

    const Position& cur = pos;
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
