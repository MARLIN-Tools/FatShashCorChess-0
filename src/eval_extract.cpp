#include "bitboard.h"
#include "hce_evaluator.h"
#include "position.h"
#include "zobrist.h"

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: makaira_eval_extract <input.csv> <output.csv>\n";
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

    out << "result,stm,phase,matpsqt_mg,matpsqt_eg,pawn_mg,pawn_eg,mob_mg,mob_eg,king_mg,king_eg,"
           "piece_mg,piece_eg,threat_mg,threat_eg,space_mg,space_eg,tempo,scale,eval_cp\n";

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

        out << result << ','
            << (pos.side_to_move() == makaira::WHITE ? 1 : -1) << ','
            << b.phase << ','
            << b.material_psqt.mg << ',' << b.material_psqt.eg << ','
            << b.pawns.mg << ',' << b.pawns.eg << ','
            << b.mobility.mg << ',' << b.mobility.eg << ','
            << b.king_safety.mg << ',' << b.king_safety.eg << ','
            << b.piece_features.mg << ',' << b.piece_features.eg << ','
            << b.threats.mg << ',' << b.threats.eg << ','
            << b.space.mg << ',' << b.space.eg << ','
            << b.tempo << ','
            << b.endgame_scale << ','
            << eval_cp << '\n';
        ++rows;
    }

    std::cout << "rows " << rows << "\n";
    return 0;
}