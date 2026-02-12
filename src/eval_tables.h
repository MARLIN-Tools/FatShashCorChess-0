#pragma once

#include "evaluator.h"
#include "types.h"

#include <array>

namespace makaira::eval_tables {

extern std::array<std::array<Score, SQ_NB>, PIECE_NB> PACKED_PSQT;
extern std::array<Bitboard, FILE_NB> FILE_MASK;
extern std::array<Bitboard, RANK_NB> RANK_MASK;
extern std::array<Bitboard, FILE_NB> ADJACENT_FILE_MASK;
extern std::array<std::array<Bitboard, SQ_NB>, COLOR_NB> FORWARD_MASK;
extern std::array<std::array<Bitboard, SQ_NB>, COLOR_NB> PASSED_MASK;

void init_eval_tables();

inline Square mirror_square(Square sq) {
    return static_cast<Square>(static_cast<int>(sq) ^ 56);
}

inline Score psqt(Piece pc, Square sq) {
    return PACKED_PSQT[pc][sq];
}

}  // namespace makaira::eval_tables