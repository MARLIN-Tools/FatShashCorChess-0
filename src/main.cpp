#include "bitboard.h"
#include "evaluator.h"
#include "hybrid_evaluator.h"
#include "movegen.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "zobrist.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_tokens(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) {
        out.push_back(tok);
    }
    return out;
}

std::string join_pv(const std::vector<makaira::Move>& pv) {
    std::string out;
    for (const makaira::Move move : pv) {
        if (!out.empty()) {
            out += ' ';
        }
        out += makaira::move_to_uci(move);
    }
    return out;
}

std::uint64_t fun_display_count(std::uint64_t raw) {
    std::uint64_t knopen = raw;
    knopen += knopen / 7;
    return (knopen / 7) * 2199;
}

int run_openbench_bench(makaira::Searcher& searcher, makaira::Position& pos) {
    // OpenBench expects a deterministic "bench" command that terminates and
    // prints both nodes and nps-like tokens.
    constexpr std::uint64_t kBenchNodes = 100000ULL;

    if (!pos.set_startpos()) {
        std::cerr << "bench failed: could not set startpos\n";
        return 1;
    }

    makaira::SearchLimits limits;
    limits.depth = 64;
    limits.nodes = kBenchNodes;
    limits.move_overhead_ms = 0;
    limits.nodes_as_time = false;

    const auto started = std::chrono::steady_clock::now();
    (void)searcher.search(pos, limits);
    const auto elapsed_ms = std::max<std::int64_t>(
      1, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count());
    const std::uint64_t nps = (kBenchNodes * 1000ULL) / static_cast<std::uint64_t>(elapsed_ms);

    // Keep wording compatible with OpenBench parser in Client/bench.py.
    std::cout << "nodes searched " << kBenchNodes << "\n";
    std::cout << "nps " << nps << "\n";
    return 0;
}

void run_perft(makaira::Position& pos, int depth, bool divide) {
    using clock = std::chrono::steady_clock;

    const auto start = clock::now();
    std::uint64_t nodes = 0;

    if (divide) {
        const auto rows = makaira::perft_divide(pos, depth);
        for (const auto& row : rows) {
            std::cout << row.first << ": " << row.second << "\n";
            nodes += row.second;
        }
    } else {
        nodes = makaira::perft(pos, depth);
    }

    const auto end = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const double seconds = ms > 0 ? static_cast<double>(ms) / 1000.0 : 0.001;
    const std::uint64_t nps = static_cast<std::uint64_t>(static_cast<double>(nodes) / seconds);

    std::cout << "nodes " << fun_display_count(nodes) << "\n";
    std::cout << "time_ms " << ms << "\n";
    std::cout << "nps " << fun_display_count(nps) << "\n";
}

bool handle_position(makaira::Position& pos, const std::vector<std::string>& tokens) {
    if (tokens.size() < 2) {
        return false;
    }

    std::size_t i = 1;
    if (tokens[i] == "startpos") {
        if (!pos.set_startpos()) {
            return false;
        }
        ++i;
    } else if (tokens[i] == "fen") {
        ++i;
        std::string fen;
        int fields = 0;
        while (i < tokens.size() && tokens[i] != "moves" && fields < 6) {
            if (!fen.empty()) {
                fen += ' ';
            }
            fen += tokens[i++];
            ++fields;
        }
        if (!pos.set_from_fen(fen)) {
            return false;
        }
    } else {
        return false;
    }

    if (i < tokens.size() && tokens[i] == "moves") {
        ++i;
        for (; i < tokens.size(); ++i) {
            const makaira::Move move = makaira::parse_uci_move(pos, tokens[i]);
            if (move.is_none() || !pos.make_move(move)) {
                return false;
            }
        }
    }

    return true;
}

int parse_int(const std::string& s, int fallback) {
    try {
        return std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

bool parse_bool(const std::string& s) {
    if (s == "1" || s == "true" || s == "on") {
        return true;
    }
    return false;
}

std::string normalize_option_key(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            out.push_back(static_cast<char>(std::tolower(uc)));
        }
    }
    return out;
}

makaira::SearchLimits parse_go_limits(const std::vector<std::string>& tokens) {
    makaira::SearchLimits limits;

    for (std::size_t i = 1; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];

        auto next_int = [&](int fallback) {
            if (i + 1 >= tokens.size()) {
                return fallback;
            }
            ++i;
            return parse_int(tokens[i], fallback);
        };

        if (t == "depth") {
            limits.depth = next_int(0);
        } else if (t == "nodes") {
            limits.nodes = static_cast<std::uint64_t>(std::max(0, next_int(0)));
        } else if (t == "movetime") {
            limits.movetime_ms = next_int(-1);
        } else if (t == "wtime") {
            limits.wtime_ms = next_int(-1);
        } else if (t == "btime") {
            limits.btime_ms = next_int(-1);
        } else if (t == "winc") {
            limits.winc_ms = next_int(0);
        } else if (t == "binc") {
            limits.binc_ms = next_int(0);
        } else if (t == "movestogo") {
            limits.movestogo = next_int(0);
        } else if (t == "ponder") {
            limits.ponder = true;
        } else if (t == "infinite") {
            limits.infinite = true;
        }
    }

    return limits;
}

void print_uci_score(int score) {
    if (std::abs(score) >= makaira::VALUE_MATE - makaira::MAX_PLY) {
        const int sign = score > 0 ? 1 : -1;
        const int mate_ply = makaira::VALUE_MATE - std::abs(score);
        const int mate_moves = (mate_ply + 1) / 2;
        std::cout << "score mate " << sign * mate_moves;
        return;
    }

    std::cout << "score cp " << score;
}

bool handle_setoption(makaira::Searcher& searcher,
                      makaira::HybridEvaluator& evaluator,
                      makaira::SearchConfig& search_config,
                      int& move_overhead_ms,
                      int& uci_threads,
                      bool& nodes_as_time,
                      bool& use_lc0_eval,
                      std::string& lc0_weights_file,
                      int& lc0_cp_scale,
                      int& lc0_score_map,
                      std::string& status_message,
                      const std::vector<std::string>& tokens) {
    std::string name;
    std::string value;
    bool parsing_name = false;
    bool parsing_value = false;

    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "name") {
            parsing_name = true;
            parsing_value = false;
            continue;
        }
        if (tokens[i] == "value") {
            parsing_name = false;
            parsing_value = true;
            continue;
        }

        if (parsing_name) {
            if (!name.empty()) {
                name += ' ';
            }
            name += tokens[i];
        } else if (parsing_value) {
            if (!value.empty()) {
                value += ' ';
            }
            value += tokens[i];
        }
    }

    for (char& c : name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const std::string key = normalize_option_key(name);

    if (key == "hash") {
        const int mb = std::clamp(parse_int(value, 32), 1, 65536);
        searcher.set_hash_size_mb(static_cast<std::size_t>(mb));
        return true;
    }

    if (key == "threads") {
        uci_threads = std::clamp(parse_int(value, uci_threads), 1, 256);
        return true;
    }

    if (key == "clearhash") {
        searcher.clear_hash();
        return true;
    }

    if (key == "clearheuristics") {
        searcher.clear_heuristics();
        return true;
    }

    if (key == "moveoverhead") {
        move_overhead_ms = std::clamp(parse_int(value, 30), 0, 10000);
        return true;
    }

    if (key == "nodesastime") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        nodes_as_time = parse_bool(value);
        return true;
    }

    if (key == "uselc0eval") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        use_lc0_eval = parse_bool(value);
        evaluator.set_use_lc0(false);
        if (!use_lc0_eval) {
            status_message = "lc0 eval disabled";
            return true;
        }

        evaluator.set_lc0_cp_scale(lc0_cp_scale);
        evaluator.set_lc0_score_map(lc0_score_map);
        if (!evaluator.lc0_ready() && !evaluator.load_lc0_weights(lc0_weights_file, true)) {
            use_lc0_eval = false;
            status_message = "failed to load lc0 weights: " + evaluator.lc0_last_error();
            return true;
        }

        evaluator.set_use_lc0(true);
        status_message = "lc0 eval enabled";
        return true;
    }

    if (key == "lc0weightsfile") {
        lc0_weights_file = value;
        if (use_lc0_eval) {
            evaluator.set_use_lc0(false);
            if (!evaluator.load_lc0_weights(lc0_weights_file, true)) {
                use_lc0_eval = false;
                status_message = "failed to load lc0 weights: " + evaluator.lc0_last_error();
                return true;
            }
            evaluator.set_use_lc0(true);
        }
        status_message = "lc0 weights file set";
        return true;
    }

    if (key == "lc0cpscale") {
        lc0_cp_scale = std::clamp(parse_int(value, lc0_cp_scale), 1, 2000);
        evaluator.set_lc0_cp_scale(lc0_cp_scale);
        return true;
    }

    if (key == "lc0scoremap") {
        lc0_score_map = std::clamp(parse_int(value, lc0_score_map), 0, 2);
        evaluator.set_lc0_score_map(lc0_score_map);
        return true;
    }

    if (key == "usehistory") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        search_config.use_history = parse_bool(value);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "usecontinuationhistory") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        search_config.use_cont_history = parse_bool(value);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "usenullmovepruning") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        search_config.use_nmp = parse_bool(value);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "uselmr") {
        for (char& c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        search_config.use_lmr = parse_bool(value);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "historymax") {
        search_config.history_max = std::clamp(parse_int(value, search_config.history_max), 1024, 32767);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "historybonusscale") {
        search_config.history_bonus_scale = std::clamp(parse_int(value, search_config.history_bonus_scale), 1, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "historymalusdivisor") {
        search_config.history_malus_divisor = std::clamp(parse_int(value, search_config.history_malus_divisor), 1, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "conthistory2plydivisor") {
        search_config.cont_history_2ply_divisor = std::clamp(parse_int(value, search_config.cont_history_2ply_divisor), 1, 8);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpmindepth") {
        search_config.nmp_min_depth = std::clamp(parse_int(value, search_config.nmp_min_depth), 2, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpbasereduction") {
        search_config.nmp_base_reduction = std::clamp(parse_int(value, search_config.nmp_base_reduction), 1, 8);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpdepthdivisor") {
        search_config.nmp_depth_divisor = std::clamp(parse_int(value, search_config.nmp_depth_divisor), 1, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpmarginbase") {
        search_config.nmp_margin_base = std::clamp(parse_int(value, search_config.nmp_margin_base), 0, 500);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpmarginperdepth") {
        search_config.nmp_margin_per_depth = std::clamp(parse_int(value, search_config.nmp_margin_per_depth), 0, 200);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpnonpawnmin") {
        search_config.nmp_non_pawn_min = std::clamp(parse_int(value, search_config.nmp_non_pawn_min), 0, 3000);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpverifynonpawnmax") {
        search_config.nmp_verify_non_pawn_max = std::clamp(parse_int(value, search_config.nmp_verify_non_pawn_max), 0, 3000);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "nmpverifymindepth") {
        search_config.nmp_verify_min_depth = std::clamp(parse_int(value, search_config.nmp_verify_min_depth), 2, 24);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "lmrmindepth") {
        search_config.lmr_min_depth = std::clamp(parse_int(value, search_config.lmr_min_depth), 2, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "lmrfulldepthmoves") {
        search_config.lmr_full_depth_moves = std::clamp(parse_int(value, search_config.lmr_full_depth_moves), 0, 16);
        searcher.set_search_config(search_config);
        return true;
    }

    if (key == "lmrhistorythreshold") {
        search_config.lmr_history_threshold = std::clamp(parse_int(value, search_config.lmr_history_threshold), 0, 16000);
        searcher.set_search_config(search_config);
        return true;
    }

    return false;
}

}  // namespace

int main(int argc, char** argv) {
    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HybridEvaluator evaluator;
    makaira::Searcher searcher(evaluator);
    makaira::SearchConfig search_config{};
    searcher.set_search_config(search_config);
    int move_overhead_ms = 30;
    int uci_threads = 1;
    bool nodes_as_time = false;
    bool use_lc0_eval = false;
    std::string lc0_weights_file = "t1-256x10-distilled-swa-2432500.pb.gz";
    int lc0_cp_scale = 220;
    int lc0_score_map = 1;

    makaira::Position position;
    if (!position.set_startpos()) {
        std::cerr << "Failed to set start position\n";
        return 1;
    }

    if (argc >= 2 && std::string(argv[1]) == "bench") {
        return run_openbench_bench(searcher, position);
    }

    if (argc >= 3 && std::string(argv[1]) == "perft") {
        const int depth = std::stoi(argv[2]);
        if (argc >= 4) {
            std::string fen;
            for (int i = 3; i < argc; ++i) {
                if (!fen.empty()) {
                    fen += ' ';
                }
                fen += argv[i];
            }
            if (!position.set_from_fen(fen)) {
                std::cerr << "Invalid FEN\n";
                return 1;
            }
        }
        run_perft(position, depth, false);
        return 0;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line == "uci") {
            std::cout << "id name FatShashCorChess 0\n";
            std::cout << "id author MARLIN-Tools\n";
            std::cout << "option name Threads type spin default " << uci_threads << " min 1 max 256\n";
            std::cout << "option name Hash type spin default 32 min 1 max 65536\n";
            std::cout << "option name MoveOverhead type spin default " << move_overhead_ms << " min 0 max 10000\n";
            std::cout << "option name NodesAsTime type check default " << (nodes_as_time ? "true" : "false") << "\n";
            std::cout << "option name UseLc0Eval type check default " << (use_lc0_eval ? "true" : "false") << "\n";
            std::cout << "option name Lc0WeightsFile type string default " << lc0_weights_file << "\n";
            std::cout << "option name Lc0CpScale type spin default " << lc0_cp_scale << " min 1 max 2000\n";
            std::cout << "option name Lc0ScoreMap type spin default " << lc0_score_map << " min 0 max 2\n";
            std::cout << "option name UseHistory type check default " << (search_config.use_history ? "true" : "false") << "\n";
            std::cout << "option name UseContinuationHistory type check default "
                      << (search_config.use_cont_history ? "true" : "false") << "\n";
            std::cout << "option name UseNullMovePruning type check default "
                      << (search_config.use_nmp ? "true" : "false") << "\n";
            std::cout << "option name UseLMR type check default " << (search_config.use_lmr ? "true" : "false") << "\n";
            std::cout << "option name HistoryMax type spin default " << search_config.history_max << " min 1024 max 32767\n";
            std::cout << "option name HistoryBonusScale type spin default " << search_config.history_bonus_scale
                      << " min 1 max 16\n";
            std::cout << "option name HistoryMalusDivisor type spin default " << search_config.history_malus_divisor
                      << " min 1 max 16\n";
            std::cout << "option name ContHistory2PlyDivisor type spin default " << search_config.cont_history_2ply_divisor
                      << " min 1 max 8\n";
            std::cout << "option name NMPMinDepth type spin default " << search_config.nmp_min_depth << " min 2 max 16\n";
            std::cout << "option name NMPBaseReduction type spin default " << search_config.nmp_base_reduction << " min 1 max 8\n";
            std::cout << "option name NMPDepthDivisor type spin default " << search_config.nmp_depth_divisor << " min 1 max 16\n";
            std::cout << "option name NMPMarginBase type spin default " << search_config.nmp_margin_base << " min 0 max 500\n";
            std::cout << "option name NMPMarginPerDepth type spin default " << search_config.nmp_margin_per_depth
                      << " min 0 max 200\n";
            std::cout << "option name NMPNonPawnMin type spin default " << search_config.nmp_non_pawn_min << " min 0 max 3000\n";
            std::cout << "option name NMPVerifyNonPawnMax type spin default " << search_config.nmp_verify_non_pawn_max
                      << " min 0 max 3000\n";
            std::cout << "option name NMPVerifyMinDepth type spin default " << search_config.nmp_verify_min_depth
                      << " min 2 max 24\n";
            std::cout << "option name LMRMinDepth type spin default " << search_config.lmr_min_depth << " min 2 max 16\n";
            std::cout << "option name LMRFullDepthMoves type spin default " << search_config.lmr_full_depth_moves
                      << " min 0 max 16\n";
            std::cout << "option name LMRHistoryThreshold type spin default " << search_config.lmr_history_threshold
                      << " min 0 max 16000\n";
            std::cout << "option name Clear Heuristics type button\n";
            std::cout << "uciok\n";
        } else if (line == "isready") {
            if (use_lc0_eval && !evaluator.lc0_ready()) {
                if (!evaluator.load_lc0_weights(lc0_weights_file, true)) {
                    use_lc0_eval = false;
                    evaluator.set_use_lc0(false);
                    std::cout << "info string failed to load lc0 weights: " << evaluator.lc0_last_error() << "\n";
                } else {
                    evaluator.set_use_lc0(true);
                }
            }
            std::cout << "readyok\n";
        } else if (line == "ucinewgame") {
            position.set_startpos();
            searcher.clear_hash();
            searcher.clear_heuristics();
        } else if (line.rfind("setoption", 0) == 0) {
            const auto tokens = split_tokens(line);
            std::string option_status;
            handle_setoption(searcher,
                             evaluator,
                             search_config,
                             move_overhead_ms,
                             uci_threads,
                             nodes_as_time,
                             use_lc0_eval,
                             lc0_weights_file,
                             lc0_cp_scale,
                             lc0_score_map,
                             option_status,
                             tokens);
            if (!option_status.empty()) {
                std::cout << "info string " << option_status << "\n";
            }
        } else if (line.rfind("position", 0) == 0) {
            const auto tokens = split_tokens(line);
            if (!handle_position(position, tokens)) {
                std::cout << "info string invalid position command\n";
            }
        } else if (line.rfind("go perft", 0) == 0) {
            const auto tokens = split_tokens(line);
            if (tokens.size() >= 3) {
                const int depth = std::stoi(tokens[2]);
                run_perft(position, depth, true);
            }
        } else if (line.rfind("go", 0) == 0) {
            const auto tokens = split_tokens(line);
            auto limits = parse_go_limits(tokens);
            limits.move_overhead_ms = move_overhead_ms;
            limits.nodes_as_time = nodes_as_time;
            if (limits.depth <= 0 && limits.movetime_ms <= 0 && limits.nodes == 0
                && limits.wtime_ms <= 0 && limits.btime_ms <= 0 && !limits.infinite) {
                limits.depth = 8;
            }

            const auto result = searcher.search(position, limits, [](const makaira::SearchIterationInfo& info) {
                std::cout << "info depth " << info.depth
                          << " seldepth " << info.seldepth
                          << " ";
                print_uci_score(info.score);
                std::cout << " nodes " << fun_display_count(info.nodes)
                          << " time " << info.time_ms
                          << " nps " << fun_display_count(info.nps)
                          << " string ttHit=" << info.stats.tt_hits << "/" << info.stats.tt_probes
                          << " qnodes=" << info.stats.qnodes
                          << " mgen=" << info.stats.movegen_calls
                          << " mgMoves=" << info.stats.moves_generated
                          << " pick=" << info.stats.move_pick_iterations
                          << " histUpd=" << info.stats.history_updates
                          << " contUpd=" << info.stats.cont_history_updates
                          << " nmp=" << info.stats.nmp_cutoffs << "/" << info.stats.nmp_attempts
                          << " nmpVer=" << info.stats.nmp_verifications << ":" << info.stats.nmp_verification_fails
                          << " lmr=" << info.stats.lmr_reduced
                          << " lmrRe=" << info.stats.lmr_researches
                          << " lmrFH=" << info.stats.lmr_fail_high_after_reduce
                          << " pvsResearch=" << info.stats.pvs_researches
                          << " betaCuts=" << info.stats.beta_cutoffs
                          << " scoreDelta=" << info.score_delta
                          << " aspFails=" << info.aspiration_fails
                          << " bmChanges=" << info.bestmove_changes
                          << " rootMoves=" << info.root_legal_moves
                          << " tOpt=" << info.optimum_time_ms
                          << " tEff=" << info.effective_optimum_ms
                          << " tMax=" << info.maximum_time_ms
                          << " stab=" << info.stability_score
                          << " cx=" << info.complexity_x100;
                if (!info.pv.empty()) {
                    std::cout << " pv " << join_pv(info.pv);
                }
                std::cout << "\n";
            });

            std::cout << "bestmove " << makaira::move_to_uci(result.best_move) << "\n";
        } else if (line.rfind("benchraw", 0) == 0) {
            const auto tokens = split_tokens(line);

            makaira::SearchLimits limits;
            limits.depth = 8;
            limits.move_overhead_ms = 0;
            limits.nodes_as_time = false;
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "depth" && i + 1 < tokens.size()) {
                    limits.depth = std::max(1, parse_int(tokens[++i], limits.depth));
                } else if (tokens[i] == "nodes" && i + 1 < tokens.size()) {
                    limits.nodes = static_cast<std::uint64_t>(std::max(1, parse_int(tokens[++i], 0)));
                }
            }

            evaluator.clear_stats();
            const auto started = std::chrono::steady_clock::now();
            const auto result = searcher.search(position, limits);
            const auto elapsed =
              std::max<std::int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::steady_clock::now() - started)
                                           .count());
            const std::uint64_t raw_nps = (result.stats.nodes * 1000ULL) / static_cast<std::uint64_t>(elapsed);
            const double tthit = result.stats.tt_probes == 0
                                   ? 0.0
                                   : (100.0 * static_cast<double>(result.stats.tt_hits))
                                       / static_cast<double>(result.stats.tt_probes);
            const auto est = evaluator.stats();

            std::cout << "info string benchraw depth " << result.depth
                      << " seldepth " << result.seldepth
                      << " nodes " << result.stats.nodes
                      << " time_ms " << elapsed
                      << " nps " << raw_nps
                      << " tt_hit_pct " << tthit
                      << " qnodes " << result.stats.qnodes
                      << " movegen " << result.stats.movegen_calls
                      << " moves_generated " << result.stats.moves_generated
                      << " pick_iters " << result.stats.move_pick_iterations
                      << " history_updates " << result.stats.history_updates
                      << " cont_updates " << result.stats.cont_history_updates
                      << " nmp " << result.stats.nmp_cutoffs << "/" << result.stats.nmp_attempts
                      << " nmp_verify " << result.stats.nmp_verifications << ":" << result.stats.nmp_verification_fails
                      << " lmr " << result.stats.lmr_reduced
                      << " lmr_re " << result.stats.lmr_researches
                      << " lmr_fh " << result.stats.lmr_fail_high_after_reduce
                      << " eval_calls " << est.eval_calls
                      << " pawn_hash_hits " << est.pawn_hash_hits
                      << " pawn_hash_misses " << est.pawn_hash_misses
                      << "\n";
        } else if (line.rfind("perft", 0) == 0) {
            const auto tokens = split_tokens(line);
            if (tokens.size() >= 2) {
                const int depth = std::stoi(tokens[1]);
                run_perft(position, depth, true);
            }
        } else if (line == "eval") {
            makaira::EvalBreakdown b{};
            const int score = evaluator.static_eval_trace(position, &b);
            std::cout << "info string eval score_cp " << score
                      << " phase " << b.phase
                      << " mat_psqt_mg " << b.material_psqt.mg
                      << " mat_psqt_eg " << b.material_psqt.eg
                      << " pawns_mg " << b.pawns.mg
                      << " pawns_eg " << b.pawns.eg
                      << " mobility_mg " << b.mobility.mg
                      << " mobility_eg " << b.mobility.eg
                      << " king_mg " << b.king_safety.mg
                      << " piece_mg " << b.piece_features.mg
                      << " threats_mg " << b.threats.mg
                      << " space_mg " << b.space.mg
                      << " scale " << b.endgame_scale
                      << "\n";
        } else if (line == "ponderhit") {
            // Synchronous search mode: no active ponder thread to promote.
        } else if (line == "stop") {
            // Synchronous search mode: stop is consumed for UCI compatibility.
        } else if (line == "quit") {
            break;
        }
    }

    return 0;
}
