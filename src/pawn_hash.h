#pragma once

#include "evaluator.h"
#include "types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace makaira {

struct PawnHashEntry {
    Key key = 0;
    Score pawn_score{};
    std::array<Bitboard, COLOR_NB> passed{{0, 0}};
    std::array<int, COLOR_NB> shelter_mg{{0, 0}};
};

class PawnHashTable {
   public:
    explicit PawnHashTable(std::size_t entries = 1ULL << 16);

    void resize(std::size_t entries);
    void clear();

    const PawnHashEntry* probe(Key key) const;
    void store(const PawnHashEntry& entry);

   private:
    std::size_t index(Key key) const;

    std::vector<PawnHashEntry> table_{};
};

}  // namespace makaira