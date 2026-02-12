#include "bitboard.h"
#include "perft.h"
#include "position.h"
#include "zobrist.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct PerftCase {
    std::string name;
    std::string fen;
    std::vector<std::pair<int, std::uint64_t>> checks;
};

int main() {
    makaira::attacks::init();
    makaira::init_zobrist();

    const std::vector<PerftCase> cases = {
      {
        "startpos",
        makaira::CHESS_STARTPOS_FEN,
        {{1, 20ULL}, {2, 400ULL}, {3, 8902ULL}, {4, 197281ULL}, {5, 4865609ULL}},
      },
      {
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        {{1, 48ULL}, {2, 2039ULL}, {3, 97862ULL}, {4, 4085603ULL}},
      },
      {
        "position3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        {{1, 14ULL}, {2, 191ULL}, {3, 2812ULL}, {4, 43238ULL}},
      },
      {
        "position4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        {{1, 6ULL}, {2, 264ULL}, {3, 9467ULL}, {4, 422333ULL}},
      },
      {
        "position5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        {{1, 44ULL}, {2, 1486ULL}, {3, 62379ULL}, {4, 2103487ULL}},
      },
      {
        "position6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        {{1, 46ULL}, {2, 2079ULL}, {3, 89890ULL}, {4, 3894594ULL}},
      },
    };

    bool ok = true;
    for (const auto& test : cases) {
        makaira::Position pos;
        if (!pos.set_from_fen(test.fen)) {
            std::cerr << "[FAIL] invalid FEN in test: " << test.name << "\n";
            return 1;
        }

        for (const auto& [depth, expected] : test.checks) {
            const std::uint64_t got = makaira::perft(pos, depth);
            if (got != expected) {
                ok = false;
                std::cerr << "[FAIL] " << test.name << " depth " << depth << " expected " << expected
                          << " got " << got << "\n";
            } else {
                std::cout << "[PASS] " << test.name << " depth " << depth << " nodes " << got << "\n";
            }
        }
    }

    if (!ok) {
        return 1;
    }

    std::cout << "All perft checks passed.\n";
    return 0;
}
