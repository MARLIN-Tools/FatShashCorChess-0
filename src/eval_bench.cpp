#include "bitboard.h"
#include "hce_evaluator.h"
#include "movegen.h"
#include "position.h"
#include "zobrist.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    makaira::attacks::init();
    makaira::init_zobrist();

    const int positions_target = argc > 1 ? std::max(1, std::atoi(argv[1])) : 512;
    const int evals_per_position = argc > 2 ? std::max(1, std::atoi(argv[2])) : 64;

    makaira::HCEEvaluator eval;
    std::mt19937_64 rng(0xBADC0DEULL);

    std::vector<makaira::Position> positions;
    positions.reserve(positions_target);

    for (int i = 0; i < positions_target; ++i) {
        makaira::Position pos;
        pos.set_startpos();

        const int plies = 8 + static_cast<int>(rng() % 20ULL);
        for (int p = 0; p < plies; ++p) {
            makaira::MoveList moves;
            makaira::generate_legal(pos, moves);
            if (moves.count == 0) {
                break;
            }
            const int idx = static_cast<int>(rng() % static_cast<std::uint64_t>(moves.count));
            if (!pos.make_move(moves[idx])) {
                break;
            }
        }

        positions.push_back(pos);
    }

    volatile int sink = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& pos : positions) {
        for (int i = 0; i < evals_per_position; ++i) {
            sink += eval.static_eval(pos);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    const auto st = eval.stats();
    const std::uint64_t total_evals = static_cast<std::uint64_t>(positions.size()) * static_cast<std::uint64_t>(evals_per_position);
    const double secs = ms > 0 ? static_cast<double>(ms) / 1000.0 : 0.001;
    const std::uint64_t evals_per_sec = static_cast<std::uint64_t>(static_cast<double>(total_evals) / secs);

    std::cout << "evals " << total_evals << "\n";
    std::cout << "time_ms " << ms << "\n";
    std::cout << "evals_per_sec " << evals_per_sec << "\n";
    std::cout << "eval_calls " << st.eval_calls << "\n";
    std::cout << "pawn_hash_hits " << st.pawn_hash_hits << "\n";
    std::cout << "pawn_hash_misses " << st.pawn_hash_misses << "\n";
    if (st.pawn_hash_hits + st.pawn_hash_misses > 0) {
        const double hit_rate = 100.0 * static_cast<double>(st.pawn_hash_hits)
                              / static_cast<double>(st.pawn_hash_hits + st.pawn_hash_misses);
        std::cout << "pawn_hash_hit_rate_pct " << hit_rate << "\n";
    }

    if (sink == 42) {
        std::cerr << "";
    }

    return 0;
}