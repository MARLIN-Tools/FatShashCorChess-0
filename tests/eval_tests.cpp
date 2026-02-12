#include "bitboard.h"
#include "eval_tables.h"
#include "hce_evaluator.h"
#include "movegen.h"
#include "position.h"
#include "zobrist.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

char swap_piece_color(char c) {
    if (c >= 'a' && c <= 'z') {
        return static_cast<char>(c - 'a' + 'A');
    }
    if (c >= 'A' && c <= 'Z') {
        return static_cast<char>(c - 'A' + 'a');
    }
    return c;
}

std::string mirror_fen(const std::string& fen) {
    std::istringstream iss(fen);
    std::string board, stm, castling, ep;
    std::string hm, fm;
    iss >> board >> stm >> castling >> ep >> hm >> fm;

    std::vector<std::string> ranks;
    std::string cur;
    for (char c : board) {
        if (c == '/') {
            ranks.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    ranks.push_back(cur);

    std::array<std::array<char, 8>, 8> grid{};
    for (int r = 0; r < 8; ++r) {
        int file = 0;
        for (char c : ranks[r]) {
            if (c >= '1' && c <= '8') {
                const int n = c - '0';
                for (int i = 0; i < n; ++i) {
                    grid[r][file++] = '.';
                }
            } else {
                grid[r][file++] = c;
            }
        }
    }

    std::array<std::array<char, 8>, 8> rot{};
    for (int r = 0; r < 8; ++r) {
        for (int f = 0; f < 8; ++f) {
            const char c = grid[7 - r][7 - f];
            rot[r][f] = c == '.' ? '.' : swap_piece_color(c);
        }
    }

    std::string mirrored_board;
    for (int r = 0; r < 8; ++r) {
        if (!mirrored_board.empty()) {
            mirrored_board.push_back('/');
        }
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            const char c = rot[r][f];
            if (c == '.') {
                ++empty;
            } else {
                if (empty > 0) {
                    mirrored_board.push_back(static_cast<char>('0' + empty));
                    empty = 0;
                }
                mirrored_board.push_back(c);
            }
        }
        if (empty > 0) {
            mirrored_board.push_back(static_cast<char>('0' + empty));
        }
    }

    std::string mirrored_stm = (stm == "w") ? "b" : "w";
    std::string mirrored_castling = "-";
    if (castling != "-") {
        mirrored_castling.clear();
        for (char c : castling) {
            if (c == 'K') mirrored_castling.push_back('k');
            else if (c == 'Q') mirrored_castling.push_back('q');
            else if (c == 'k') mirrored_castling.push_back('K');
            else if (c == 'q') mirrored_castling.push_back('Q');
        }
        if (mirrored_castling.empty()) {
            mirrored_castling = "-";
        }
    }

    std::string mirrored_ep = "-";
    if (ep != "-") {
        const char file = ep[0];
        const char rank = ep[1];
        const char mirrored_rank = static_cast<char>('1' + ('8' - rank));
        mirrored_ep = std::string{file, mirrored_rank};
    }

    std::ostringstream oss;
    oss << mirrored_board << ' ' << mirrored_stm << ' ' << mirrored_castling << ' ' << mirrored_ep << ' '
        << (hm.empty() ? "0" : hm) << ' ' << (fm.empty() ? "1" : fm);
    return oss.str();
}

}  // namespace

int main() {
    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HCEEvaluator eval;

    {
        makaira::Position pos;
        if (!pos.set_startpos()) {
            std::cerr << "[FAIL] startpos setup\n";
            return 1;
        }

        const int s1 = eval.static_eval(pos);
        const int s2 = eval.static_eval(pos);
        if (s1 != s2) {
            std::cerr << "[FAIL] determinism mismatch\n";
            return 1;
        }
        std::cout << "[PASS] determinism\n";
    }

    {
        const std::vector<std::string> fens = {
          makaira::CHESS_STARTPOS_FEN,
          "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        };

        for (const auto& fen : fens) {
            makaira::Position a;
            makaira::Position b;
            if (!a.set_from_fen(fen) || !b.set_from_fen(mirror_fen(fen))) {
                std::cerr << "[FAIL] symmetry setup\n";
                return 1;
            }

            const int sa = eval.static_eval(a);
            const int sb = eval.static_eval(b);
            if (std::abs(sa - sb) > 4) {
                std::cerr << "[FAIL] symmetry mismatch for fen: " << fen << " got " << sa << " and " << sb << "\n";
                return 1;
            }
        }
        std::cout << "[PASS] symmetry\n";
    }

    {
        makaira::Position pos;
        if (!pos.set_startpos()) {
            std::cerr << "[FAIL] incremental startpos\n";
            return 1;
        }

        std::mt19937_64 rng(0xC0FFEEULL);
        std::string last_move = "(none)";

        for (int ply = 0; ply < 160; ++ply) {
            const int inc = eval.static_eval(pos);
            const int rec = eval.static_eval_recompute(pos);
            if (inc != rec) {
                int rec_mg = 0;
                int rec_eg = 0;
                for (int sq = makaira::SQ_A1; sq <= makaira::SQ_H8; ++sq) {
                    const auto s = static_cast<makaira::Square>(sq);
                    const auto pc = pos.piece_on(s);
                    if (pc == makaira::NO_PIECE) {
                        continue;
                    }
                    const auto ps = makaira::eval_tables::psqt(pc, s);
                    const int sign = makaira::color_of(pc) == makaira::WHITE ? 1 : -1;
                    rec_mg += sign * ps.mg;
                    rec_eg += sign * ps.eg;
                }
                std::cerr << "[FAIL] incremental mismatch at ply " << ply << " after move " << last_move
                          << " inc=" << inc << " rec=" << rec
                          << " inc_mg=" << (pos.mg_psqt(makaira::WHITE) - pos.mg_psqt(makaira::BLACK))
                          << " w_mg=" << pos.mg_psqt(makaira::WHITE)
                          << " b_mg=" << pos.mg_psqt(makaira::BLACK)
                          << " inc_eg=" << (pos.eg_psqt(makaira::WHITE) - pos.eg_psqt(makaira::BLACK))
                          << " rec_mg=" << rec_mg
                          << " rec_eg=" << rec_eg
                          << " wp_d2_mg=" << makaira::eval_tables::psqt(makaira::W_PAWN, makaira::square_from_string("d2")).mg
                          << " wp_d4_mg=" << makaira::eval_tables::psqt(makaira::W_PAWN, makaira::square_from_string("d4")).mg
                          << " phase=" << pos.phase()
                          << "\n";
                return 1;
            }

            makaira::MoveList moves;
            makaira::generate_legal(pos, moves);
            if (moves.count == 0) {
                break;
            }

            const int idx = static_cast<int>(rng() % static_cast<std::uint64_t>(moves.count));
            if (!pos.make_move(moves[idx])) {
                std::cerr << "[FAIL] could not make random legal move\n";
                return 1;
            }
            last_move = makaira::move_to_uci(moves[idx]);
        }
        std::cout << "[PASS] incremental_vs_recompute\n";
    }

    {
        makaira::Position pos;
        pos.set_from_fen("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1PN2/PPP1BPPP/R2Q1RK1 w kq - 0 10");

        for (int i = 0; i < 100; ++i) {
            eval.static_eval(pos);
        }

        const auto st = eval.stats();
        if (st.eval_calls < 100) {
            std::cerr << "[FAIL] eval calls counter\n";
            return 1;
        }
        if (st.pawn_hash_hits == 0) {
            std::cerr << "[FAIL] expected pawn hash hits\n";
            return 1;
        }

        std::cout << "[PASS] pawn_hash_stats\n";
    }

    std::cout << "All eval tests passed.\n";
    return 0;
}
