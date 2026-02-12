#include "bitboard.h"
#include "evaluator.h"
#include "hce_evaluator.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "zobrist.h"

#include <iostream>

int main() {
    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HCEEvaluator evaluator;
    makaira::Searcher searcher(evaluator);

    {
        makaira::Position pos;
        if (!pos.set_startpos()) {
            std::cerr << "[FAIL] startpos FEN\n";
            return 1;
        }

        makaira::SearchLimits limits;
        limits.depth = 3;

        const auto result = searcher.search(pos, limits);
        if (result.best_move.is_none()) {
            std::cerr << "[FAIL] startpos bestmove is none\n";
            return 1;
        }

        if (!pos.make_move(result.best_move)) {
            std::cerr << "[FAIL] startpos bestmove is illegal\n";
            return 1;
        }
        pos.unmake_move();

        std::cout << "[PASS] startpos depth3 bestmove " << makaira::move_to_uci(result.best_move) << "\n";
    }

    {
        makaira::Position pos;
        if (!pos.set_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")) {
            std::cerr << "[FAIL] checkmate FEN\n";
            return 1;
        }

        makaira::SearchLimits limits;
        limits.depth = 2;

        const auto result = searcher.search(pos, limits);
        if (!result.best_move.is_none()) {
            std::cerr << "[FAIL] checkmated side produced a move\n";
            return 1;
        }

        if (result.score > -makaira::VALUE_MATE + 1) {
            std::cerr << "[FAIL] checkmate score not detected: " << result.score << "\n";
            return 1;
        }

        std::cout << "[PASS] checkmate detection score " << result.score << "\n";
    }

    {
        makaira::Position pos;
        if (!pos.set_startpos()) {
            std::cerr << "[FAIL] null move startpos setup\n";
            return 1;
        }

        const auto key_before = pos.key();
        const auto pawn_key_before = pos.pawn_key();
        const auto stm_before = pos.side_to_move();
        const auto ep_before = pos.ep_square();
        const auto hm_before = pos.halfmove_clock();
        const auto fm_before = pos.fullmove_number();

        pos.make_null_move();
        if (pos.side_to_move() == stm_before) {
            std::cerr << "[FAIL] null move did not flip side\n";
            return 1;
        }

        pos.unmake_null_move();
        if (pos.key() != key_before || pos.pawn_key() != pawn_key_before || pos.side_to_move() != stm_before
            || pos.ep_square() != ep_before || pos.halfmove_clock() != hm_before || pos.fullmove_number() != fm_before) {
            std::cerr << "[FAIL] null move unmake mismatch\n";
            return 1;
        }

        std::cout << "[PASS] null move make/unmake\n";
    }

    std::cout << "All search smoke tests passed.\n";
    return 0;
}
