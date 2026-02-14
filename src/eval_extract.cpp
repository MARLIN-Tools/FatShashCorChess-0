#include "bitboard.h"
#include "eval_tables.h"
#include "hce_evaluator.h"
#include "position.h"
#include "zobrist.h"

#include <array>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool parse_line(const std::string& line, double& result, std::string& fen) {
    const std::size_t comma = line.find(',');
    if (comma == std::string::npos) {
        return false;
    }

    try {
        result = std::stod(line.substr(0, comma));
    } catch (...) {
        return false;
    }

    fen = line.substr(comma + 1);
    return !fen.empty();
}

constexpr int PSQT_BUCKETS = 32;
constexpr int PSQT_PIECES = 6;  // pawn..king

int psqt_index(makaira::PieceType pt, int bucket) {
    return (static_cast<int>(pt) - static_cast<int>(makaira::PAWN)) * PSQT_BUCKETS + bucket;
}

const char* piece_token(makaira::PieceType pt) {
    switch (pt) {
        case makaira::PAWN: return "pawn";
        case makaira::KNIGHT: return "knight";
        case makaira::BISHOP: return "bishop";
        case makaira::ROOK: return "rook";
        case makaira::QUEEN: return "queen";
        case makaira::KING: return "king";
        default: return "x";
    }
}

void write_psqt_header(std::ostream& out) {
    for (makaira::PieceType pt : {makaira::PAWN, makaira::KNIGHT, makaira::BISHOP, makaira::ROOK, makaira::QUEEN, makaira::KING}) {
        for (int b = 0; b < PSQT_BUCKETS; ++b) {
            out << ",psqt_" << piece_token(pt) << "_b" << std::setw(2) << std::setfill('0') << b;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: fatshashcorchess0_eval_extract <input.csv> <output.csv>\n";
        std::cerr << "Input format: result,fen  where result in {0,0.5,1}\n";
        return 1;
    }

    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HCEEvaluator eval;

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Failed to open input: " << argv[1] << "\n";
        return 1;
    }

    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "Failed to open output: " << argv[2] << "\n";
        return 1;
    }

    out << "result,stm,phase,"
           "matpsqt_mg,matpsqt_eg,"
           "pawn_mg,pawn_eg,"
           "pawn_passed_mg,pawn_passed_eg,"
           "pawn_isolated_mg,pawn_isolated_eg,"
           "pawn_doubled_mg,pawn_doubled_eg,"
           "pawn_backward_mg,pawn_backward_eg,"
           "pawn_candidate_mg,pawn_candidate_eg,"
           "pawn_connected_mg,pawn_connected_eg,"
           "pawn_supported_mg,pawn_supported_eg,"
           "pawn_outside_mg,pawn_outside_eg,"
           "pawn_blocked_mg,pawn_blocked_eg,"
           "mob_mg,mob_eg,"
           "king_mg,king_eg,"
           "king_shelter_mg,king_shelter_eg,"
           "king_storm_mg,king_storm_eg,"
           "king_danger_mg,king_danger_eg,"
           "piece_mg,piece_eg,"
           "piece_bishop_pair_mg,piece_bishop_pair_eg,"
           "piece_rook_file_mg,piece_rook_file_eg,"
           "piece_rook_seventh_mg,piece_rook_seventh_eg,"
           "piece_knight_outpost_mg,piece_knight_outpost_eg,"
           "piece_bad_bishop_mg,piece_bad_bishop_eg,"
           "threat_mg,threat_eg,"
           "threat_hanging_mg,threat_hanging_eg,"
           "threat_pawn_mg,threat_pawn_eg,"
           "space_mg,space_eg,"
           "endgame_terms_mg,endgame_terms_eg,"
           "endgame_king_activity_mg,endgame_king_activity_eg,"
           "tempo,scale,eval_cp";
    write_psqt_header(out);
    out << '\n';

    std::string line;
    std::uint64_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        double result = 0.0;
        std::string fen;
        if (!parse_line(line, result, fen)) {
            continue;
        }

        makaira::Position pos;
        if (!pos.set_from_fen(fen)) {
            continue;
        }

        makaira::EvalBreakdown b{};
        const int eval_cp = eval.static_eval_trace(pos, &b);
        std::array<int, PSQT_PIECES * PSQT_BUCKETS> psqt_counts{};
        for (makaira::Color c : {makaira::WHITE, makaira::BLACK}) {
            const int sign = c == makaira::WHITE ? 1 : -1;
            for (makaira::PieceType pt : {makaira::PAWN, makaira::KNIGHT, makaira::BISHOP, makaira::ROOK, makaira::QUEEN, makaira::KING}) {
                makaira::Bitboard bb = pos.pieces(c, pt);
                while (bb) {
                    const makaira::Square sq = makaira::pop_lsb(bb);
                    const int bucket = makaira::eval_tables::psqt_bucket(sq, c);
                    psqt_counts[psqt_index(pt, bucket)] += sign;
                }
            }
        }

        out << result << ','
            << (pos.side_to_move() == makaira::WHITE ? 1 : -1) << ','
            << b.phase << ','
            << b.material_psqt.mg << ',' << b.material_psqt.eg << ','
            << b.pawns.mg << ',' << b.pawns.eg << ','
            << b.pawns_passed.mg << ',' << b.pawns_passed.eg << ','
            << b.pawns_isolated.mg << ',' << b.pawns_isolated.eg << ','
            << b.pawns_doubled.mg << ',' << b.pawns_doubled.eg << ','
            << b.pawns_backward.mg << ',' << b.pawns_backward.eg << ','
            << b.pawns_candidate.mg << ',' << b.pawns_candidate.eg << ','
            << b.pawns_connected.mg << ',' << b.pawns_connected.eg << ','
            << b.pawns_supported.mg << ',' << b.pawns_supported.eg << ','
            << b.pawns_outside.mg << ',' << b.pawns_outside.eg << ','
            << b.pawns_blocked.mg << ',' << b.pawns_blocked.eg << ','
            << b.mobility.mg << ',' << b.mobility.eg << ','
            << b.king_safety.mg << ',' << b.king_safety.eg << ','
            << b.king_shelter.mg << ',' << b.king_shelter.eg << ','
            << b.king_storm.mg << ',' << b.king_storm.eg << ','
            << b.king_danger.mg << ',' << b.king_danger.eg << ','
            << b.piece_features.mg << ',' << b.piece_features.eg << ','
            << b.piece_bishop_pair.mg << ',' << b.piece_bishop_pair.eg << ','
            << b.piece_rook_file.mg << ',' << b.piece_rook_file.eg << ','
            << b.piece_rook_seventh.mg << ',' << b.piece_rook_seventh.eg << ','
            << b.piece_knight_outpost.mg << ',' << b.piece_knight_outpost.eg << ','
            << b.piece_bad_bishop.mg << ',' << b.piece_bad_bishop.eg << ','
            << b.threats.mg << ',' << b.threats.eg << ','
            << b.threat_hanging.mg << ',' << b.threat_hanging.eg << ','
            << b.threat_pawn.mg << ',' << b.threat_pawn.eg << ','
            << b.space.mg << ',' << b.space.eg << ','
            << b.endgame_terms.mg << ',' << b.endgame_terms.eg << ','
            << b.endgame_king_activity.mg << ',' << b.endgame_king_activity.eg << ','
            << b.tempo << ','
            << b.endgame_scale << ','
            << eval_cp;
        for (int v : psqt_counts) {
            out << ',' << v;
        }
        out << '\n';
        ++rows;
    }

    std::cout << "rows " << rows << "\n";
    return 0;
}
