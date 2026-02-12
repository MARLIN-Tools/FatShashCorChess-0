#pragma once

#include "move.h"
#include "position.h"

#include <array>

namespace makaira {

enum class MovePickPhase : std::uint8_t {
    TT = 0,
    GOOD_CAPTURE = 1,
    QUIET = 2,
    BAD_CAPTURE = 3,
    END = 4
};

class MovePicker {
   public:
    MovePicker(const Position& pos, Move tt_move, bool qsearch_only);

    Move next(MovePickPhase* phase = nullptr);
    int generated_count() const { return generated_count_; }

   private:
    struct ScoredMove {
        Move move{};
        int score = 0;
    };

    static bool better(const ScoredMove& lhs, const ScoredMove& rhs);
    Move pick_next_from_bucket(std::array<ScoredMove, 256>& bucket, int count, int& index, MovePickPhase phase, MovePickPhase* out);
    static int capture_score(const Position& pos, Move move);

    Move tt_move_{};
    bool qsearch_only_ = false;
    bool tt_done_ = false;
    int generated_count_ = 0;

    std::array<ScoredMove, 256> good_captures_{};
    std::array<ScoredMove, 256> quiets_{};
    std::array<ScoredMove, 256> bad_captures_{};
    int good_count_ = 0;
    int quiet_count_ = 0;
    int bad_count_ = 0;
    int good_idx_ = 0;
    int quiet_idx_ = 0;
    int bad_idx_ = 0;
};

}  // namespace makaira

