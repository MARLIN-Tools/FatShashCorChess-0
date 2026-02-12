#include "bitboard.h"
#include "hce_evaluator.h"
#include "position.h"
#include "search.h"
#include "zobrist.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct BenchOptions {
    std::string fen_suite = "bench/fens.txt";
    int depth = 8;
    std::uint64_t nodes = 0;
    int hash_mb = 32;
};

bool parse_options(int argc, char** argv, BenchOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--fen-suite") {
            const char* v = next("--fen-suite");
            if (!v) {
                return false;
            }
            opt.fen_suite = v;
        } else if (arg == "--depth") {
            const char* v = next("--depth");
            if (!v) {
                return false;
            }
            opt.depth = std::max(1, std::stoi(v));
        } else if (arg == "--nodes") {
            const char* v = next("--nodes");
            if (!v) {
                return false;
            }
            opt.nodes = static_cast<std::uint64_t>(std::max<long long>(1LL, std::stoll(v)));
        } else if (arg == "--hash") {
            const char* v = next("--hash");
            if (!v) {
                return false;
            }
            opt.hash_mb = std::clamp(std::stoi(v), 1, 65536);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: makaira_search_bench [--fen-suite file] [--depth D] [--nodes N] [--hash MB]\n";
            return false;
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        }
    }

    return true;
}

std::vector<std::string> load_fens(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }

    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        out.push_back(line);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    BenchOptions opt;
    if (!parse_options(argc, argv, opt)) {
        return 1;
    }

    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HCEEvaluator evaluator;
    evaluator.clear_stats();

    makaira::Searcher searcher(evaluator);
    searcher.set_hash_size_mb(static_cast<std::size_t>(opt.hash_mb));

    std::vector<std::string> fens = load_fens(opt.fen_suite);
    if (fens.empty()) {
        std::cerr << "failed to load fen suite: " << opt.fen_suite << "\n";
        return 1;
    }

    makaira::SearchLimits limits;
    limits.depth = opt.depth;
    limits.nodes = opt.nodes;
    limits.move_overhead_ms = 0;

    makaira::SearchStats total{};
    int total_depth = 0;
    int total_seldepth = 0;

    const auto started = std::chrono::steady_clock::now();

    for (const std::string& fen : fens) {
        makaira::Position pos;
        if (!pos.set_from_fen(fen)) {
            std::cerr << "invalid fen in suite: " << fen << "\n";
            return 1;
        }

        searcher.clear_hash();
        const makaira::SearchResult result = searcher.search(pos, limits);

        total.nodes += result.stats.nodes;
        total.qnodes += result.stats.qnodes;
        total.tt_probes += result.stats.tt_probes;
        total.tt_hits += result.stats.tt_hits;
        total.beta_cutoffs += result.stats.beta_cutoffs;
        total.pvs_researches += result.stats.pvs_researches;
        total.movegen_calls += result.stats.movegen_calls;
        total.moves_generated += result.stats.moves_generated;
        total.move_pick_iterations += result.stats.move_pick_iterations;
        total.cutoff_tt += result.stats.cutoff_tt;
        total.cutoff_good_capture += result.stats.cutoff_good_capture;
        total.cutoff_quiet += result.stats.cutoff_quiet;
        total.cutoff_bad_capture += result.stats.cutoff_bad_capture;

        total_depth += result.depth;
        total_seldepth += result.seldepth;
    }

    const auto ended = std::chrono::steady_clock::now();
    const auto elapsed_ms =
      std::max<std::int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count());

    const std::uint64_t nps = (total.nodes * 1000ULL) / static_cast<std::uint64_t>(elapsed_ms);
    const double tt_hit_rate = total.tt_probes == 0 ? 0.0 : (100.0 * static_cast<double>(total.tt_hits)) / static_cast<double>(total.tt_probes);
    const double qratio = total.nodes == 0 ? 0.0 : (100.0 * static_cast<double>(total.qnodes)) / static_cast<double>(total.nodes);
    const double moves_per_gen = total.movegen_calls == 0 ? 0.0 : static_cast<double>(total.moves_generated) / static_cast<double>(total.movegen_calls);

    const makaira::EvalStats est = evaluator.stats();

    std::cout << "positions " << fens.size() << "\n";
    std::cout << "depth_limit " << opt.depth << "\n";
    std::cout << "node_limit " << opt.nodes << "\n";
    std::cout << "hash_mb " << opt.hash_mb << "\n";
    std::cout << "elapsed_ms " << elapsed_ms << "\n";
    std::cout << "nodes " << total.nodes << "\n";
    std::cout << "nps " << nps << "\n";
    std::cout << "avg_depth " << (fens.empty() ? 0 : total_depth / static_cast<int>(fens.size())) << "\n";
    std::cout << "avg_seldepth " << (fens.empty() ? 0 : total_seldepth / static_cast<int>(fens.size())) << "\n";
    std::cout << "tt_probes " << total.tt_probes << "\n";
    std::cout << "tt_hits " << total.tt_hits << "\n";
    std::cout << "tt_hit_rate_pct " << tt_hit_rate << "\n";
    std::cout << "qnodes " << total.qnodes << "\n";
    std::cout << "qnodes_ratio_pct " << qratio << "\n";
    std::cout << "movegen_calls " << total.movegen_calls << "\n";
    std::cout << "moves_generated " << total.moves_generated << "\n";
    std::cout << "moves_per_movegen " << moves_per_gen << "\n";
    std::cout << "move_pick_iterations " << total.move_pick_iterations << "\n";
    std::cout << "beta_cutoffs " << total.beta_cutoffs << "\n";
    std::cout << "cutoff_tt " << total.cutoff_tt << "\n";
    std::cout << "cutoff_good_capture " << total.cutoff_good_capture << "\n";
    std::cout << "cutoff_quiet " << total.cutoff_quiet << "\n";
    std::cout << "cutoff_bad_capture " << total.cutoff_bad_capture << "\n";
    std::cout << "eval_calls " << est.eval_calls << "\n";
    std::cout << "pawn_hash_hits " << est.pawn_hash_hits << "\n";
    std::cout << "pawn_hash_misses " << est.pawn_hash_misses << "\n";

    return 0;
}
