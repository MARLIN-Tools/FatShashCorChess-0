#include "bitboard.h"
#include "evaluator.h"
#include "hce_evaluator.h"
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
                      makaira::SearchConfig& search_config,
                      int& move_overhead_ms,
                      bool& nodes_as_time,
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
    std::string value_lc = value;
    for (char& c : value_lc) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    auto apply_cfg = [&]() {
        searcher.set_search_config(search_config);
        return true;
    };

    auto set_bool_cfg = [&](const char* option_name, bool& field) {
        if (name != option_name) {
            return false;
        }
        field = parse_bool(value_lc);
        return apply_cfg();
    };

    auto set_spin_cfg = [&](const char* option_name, int& field, int min_v, int max_v) {
        if (name != option_name) {
            return false;
        }
        field = std::clamp(parse_int(value, field), min_v, max_v);
        return apply_cfg();
    };

    if (name == "hash") {
        const int mb = std::clamp(parse_int(value, 32), 1, 65536);
        searcher.set_hash_size_mb(static_cast<std::size_t>(mb));
        return true;
    }

    if (name == "clear hash") {
        searcher.clear_hash();
        return true;
    }

    if (name == "clear heuristics") {
        searcher.clear_heuristics();
        return true;
    }

    if (name == "move overhead") {
        move_overhead_ms = std::clamp(parse_int(value, 30), 0, 10000);
        return true;
    }

    if (name == "nodes as time") {
        nodes_as_time = parse_bool(value_lc);
        return true;
    }

    if (set_bool_cfg("use history", search_config.use_history)) return true;
    if (set_bool_cfg("use continuation history", search_config.use_cont_history)) return true;
    if (set_bool_cfg("use null move pruning", search_config.use_nmp)) return true;
    if (set_bool_cfg("use lmr", search_config.use_lmr)) return true;
    if (set_bool_cfg("use see", search_config.use_see)) return true;
    if (set_bool_cfg("use qdelta", search_config.use_qdelta)) return true;
    if (set_bool_cfg("use rfp", search_config.use_rfp)) return true;
    if (set_bool_cfg("userfp", search_config.use_rfp)) return true;
    if (set_bool_cfg("use razoring", search_config.use_razoring)) return true;
    if (set_bool_cfg("userazoring", search_config.use_razoring)) return true;
    if (set_bool_cfg("use futility", search_config.use_futility)) return true;
    if (set_bool_cfg("usefutility", search_config.use_futility)) return true;
    if (set_bool_cfg("use lmp", search_config.use_lmp)) return true;
    if (set_bool_cfg("uselmp", search_config.use_lmp)) return true;
    if (set_bool_cfg("use history pruning", search_config.use_history_pruning)) return true;
    if (set_bool_cfg("usehistorypruning", search_config.use_history_pruning)) return true;
    if (set_bool_cfg("use probcut", search_config.use_probcut)) return true;
    if (set_bool_cfg("useprobcut", search_config.use_probcut)) return true;
    if (set_bool_cfg("use singular", search_config.use_singular)) return true;
    if (set_bool_cfg("usesingular", search_config.use_singular)) return true;
    if (set_bool_cfg("use capture history", search_config.use_capture_history)) return true;

    if (set_spin_cfg("history max", search_config.history_max, 1024, 32767)) return true;
    if (set_spin_cfg("nmp min depth", search_config.nmp_min_depth, 2, 16)) return true;
    if (set_spin_cfg("nmp margin base", search_config.nmp_margin_base, 0, 500)) return true;
    if (set_spin_cfg("nmp margin per depth", search_config.nmp_margin_per_depth, 0, 200)) return true;
    if (set_spin_cfg("lmr min depth", search_config.lmr_min_depth, 2, 16)) return true;
    if (set_spin_cfg("lmr full depth moves", search_config.lmr_full_depth_moves, 0, 32)) return true;
    if (set_spin_cfg("see q threshold cp", search_config.see_q_threshold_cp, -3000, 3000)) return true;
    if (set_spin_cfg("q delta margin cp", search_config.q_delta_margin_cp, 0, 3000)) return true;
    if (set_spin_cfg("rfp max depth", search_config.rfp_max_depth, 1, 16)) return true;
    if (set_spin_cfg("rfp margin base cp", search_config.rfp_margin_base_cp, 0, 3000)) return true;
    if (set_spin_cfg("rfp margin per depth cp", search_config.rfp_margin_per_depth_cp, 0, 1000)) return true;
    if (set_spin_cfg("razor max depth", search_config.razor_max_depth, 1, 8)) return true;
    if (set_spin_cfg("razor margin d1 cp", search_config.razor_margin_d1_cp, 0, 3000)) return true;
    if (set_spin_cfg("razor margin d2 cp", search_config.razor_margin_d2_cp, 0, 3000)) return true;
    if (set_spin_cfg("razor margin d3 cp", search_config.razor_margin_d3_cp, 0, 3000)) return true;
    if (set_spin_cfg("futility max depth", search_config.futility_max_depth, 1, 8)) return true;
    if (set_spin_cfg("futility margin base cp", search_config.futility_margin_base_cp, 0, 3000)) return true;
    if (set_spin_cfg("futility margin per depth cp", search_config.futility_margin_per_depth_cp, 0, 1000)) return true;
    if (set_spin_cfg("zugzwang non pawn material cp", search_config.zugzwang_non_pawn_material_cp, 0, 5000)) return true;
    if (set_spin_cfg("lmp max depth", search_config.lmp_max_depth, 1, 8)) return true;
    if (set_spin_cfg("lmp d1", search_config.lmp_d1, 1, 128)) return true;
    if (set_spin_cfg("lmp d2", search_config.lmp_d2, 1, 128)) return true;
    if (set_spin_cfg("lmp d3", search_config.lmp_d3, 1, 128)) return true;
    if (set_spin_cfg("lmp d4", search_config.lmp_d4, 1, 128)) return true;
    if (set_spin_cfg("histprune max depth", search_config.histprune_max_depth, 1, 12)) return true;
    if (set_spin_cfg("histprune min moves", search_config.histprune_min_moves, 1, 64)) return true;
    if (set_spin_cfg("histprune threshold cp", search_config.histprune_threshold_cp, -32768, 32767)) return true;
    if (set_spin_cfg("probcut min depth", search_config.probcut_min_depth, 1, 32)) return true;
    if (set_spin_cfg("probcut reduction", search_config.probcut_reduction, 1, 8)) return true;
    if (set_spin_cfg("probcut margin cp", search_config.probcut_margin_cp, 0, 2000)) return true;
    if (set_spin_cfg("probcut see threshold cp", search_config.probcut_see_threshold_cp, -3000, 3000)) return true;
    if (set_spin_cfg("singular min depth", search_config.singular_min_depth, 1, 32)) return true;
    if (set_spin_cfg("singular reduction", search_config.singular_reduction, 1, 8)) return true;
    if (set_spin_cfg("singular margin cp", search_config.singular_margin_cp, 0, 2000)) return true;
    if (set_spin_cfg("max extensions per pv line", search_config.max_extensions_per_pv_line, 0, 8)) return true;

    return false;
}

}  // namespace

int main(int argc, char** argv) {
    makaira::attacks::init();
    makaira::init_zobrist();

    makaira::HCEEvaluator evaluator;
    makaira::Searcher searcher(evaluator);
    makaira::SearchConfig search_config{};
    searcher.set_search_config(search_config);
    int move_overhead_ms = 30;
    bool nodes_as_time = false;

    makaira::Position position;
    if (!position.set_startpos()) {
        std::cerr << "Failed to set start position\n";
        return 1;
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
        if (line == "uci") {
            std::cout << "id name fatshashcorchess 0\n";
            std::cout << "id author MARLIN-Tools\n";
            std::cout << "option name Hash type spin default 32 min 1 max 65536\n";
            std::cout << "option name Move Overhead type spin default 30 min 0 max 10000\n";
            std::cout << "option name Nodes as Time type check default false\n";
            std::cout << "option name Use History type check default true\n";
            std::cout << "option name Use Continuation History type check default true\n";
            std::cout << "option name Use Null Move Pruning type check default true\n";
            std::cout << "option name Use LMR type check default true\n";
            std::cout << "option name Use SEE type check default true\n";
            std::cout << "option name Use QDelta type check default true\n";
            std::cout << "option name Use RFP type check default false\n";
            std::cout << "option name UseRFP type check default false\n";
            std::cout << "option name Use Razoring type check default false\n";
            std::cout << "option name UseRazoring type check default false\n";
            std::cout << "option name Use Futility type check default false\n";
            std::cout << "option name UseFutility type check default false\n";
            std::cout << "option name Use LMP type check default false\n";
            std::cout << "option name UseLMP type check default false\n";
            std::cout << "option name Use History Pruning type check default false\n";
            std::cout << "option name UseHistoryPruning type check default false\n";
            std::cout << "option name Use ProbCut type check default false\n";
            std::cout << "option name UseProbCut type check default false\n";
            std::cout << "option name Use Singular type check default false\n";
            std::cout << "option name UseSingular type check default false\n";
            std::cout << "option name Use Capture History type check default true\n";
            std::cout << "option name History Max type spin default 20815 min 1024 max 32767\n";
            std::cout << "option name NMP Min Depth type spin default 2 min 2 max 16\n";
            std::cout << "option name NMP Margin Base type spin default 61 min 0 max 500\n";
            std::cout << "option name NMP Margin Per Depth type spin default 18 min 0 max 200\n";
            std::cout << "option name LMR Min Depth type spin default 2 min 2 max 16\n";
            std::cout << "option name LMR Full Depth Moves type spin default 2 min 0 max 16\n";
            std::cout << "option name SEE Q Threshold CP type spin default 0 min -3000 max 3000\n";
            std::cout << "option name Q Delta Margin CP type spin default 120 min 0 max 3000\n";
            std::cout << "option name RFP Max Depth type spin default 6 min 1 max 16\n";
            std::cout << "option name RFP Margin Base CP type spin default 90 min 0 max 3000\n";
            std::cout << "option name RFP Margin Per Depth CP type spin default 40 min 0 max 1000\n";
            std::cout << "option name Razor Max Depth type spin default 3 min 1 max 8\n";
            std::cout << "option name Razor Margin D1 CP type spin default 350 min 0 max 3000\n";
            std::cout << "option name Razor Margin D2 CP type spin default 500 min 0 max 3000\n";
            std::cout << "option name Razor Margin D3 CP type spin default 650 min 0 max 3000\n";
            std::cout << "option name Futility Max Depth type spin default 4 min 1 max 8\n";
            std::cout << "option name Futility Margin Base CP type spin default 80 min 0 max 3000\n";
            std::cout << "option name Futility Margin Per Depth CP type spin default 60 min 0 max 1000\n";
            std::cout << "option name Zugzwang Non Pawn Material CP type spin default 1200 min 0 max 5000\n";
            std::cout << "option name LMP Max Depth type spin default 4 min 1 max 8\n";
            std::cout << "option name LMP D1 type spin default 10 min 1 max 128\n";
            std::cout << "option name LMP D2 type spin default 14 min 1 max 128\n";
            std::cout << "option name LMP D3 type spin default 18 min 1 max 128\n";
            std::cout << "option name LMP D4 type spin default 22 min 1 max 128\n";
            std::cout << "option name HistPrune Max Depth type spin default 5 min 1 max 12\n";
            std::cout << "option name HistPrune Min Moves type spin default 6 min 1 max 64\n";
            std::cout << "option name HistPrune Threshold CP type spin default -4096 min -32768 max 32767\n";
            std::cout << "option name ProbCut Min Depth type spin default 9 min 1 max 32\n";
            std::cout << "option name ProbCut Reduction type spin default 3 min 1 max 8\n";
            std::cout << "option name ProbCut Margin CP type spin default 200 min 0 max 2000\n";
            std::cout << "option name ProbCut SEE Threshold CP type spin default 0 min -3000 max 3000\n";
            std::cout << "option name Singular Min Depth type spin default 10 min 1 max 32\n";
            std::cout << "option name Singular Reduction type spin default 3 min 1 max 8\n";
            std::cout << "option name Singular Margin CP type spin default 80 min 0 max 2000\n";
            std::cout << "option name Max Extensions Per PV Line type spin default 1 min 0 max 8\n";
            std::cout << "option name Clear Heuristics type button\n";
            std::cout << "uciok\n";
        } else if (line == "isready") {
            std::cout << "readyok\n";
        } else if (line == "ucinewgame") {
            position.set_startpos();
            searcher.clear_hash();
            searcher.clear_heuristics();
        } else if (line.rfind("setoption", 0) == 0) {
            const auto tokens = split_tokens(line);
            handle_setoption(searcher, search_config, move_overhead_ms, nodes_as_time, tokens);
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
            evaluator.set_profile_mode(false);
            limits.move_overhead_ms = move_overhead_ms;
            limits.nodes_as_time = nodes_as_time;
            if (limits.depth <= 0 && limits.movetime_ms <= 0 && limits.nodes == 0
                && limits.wtime_ms <= 0 && limits.btime_ms <= 0 && !limits.infinite) {
                limits.depth = 8;
            }

            const auto result = searcher.search(position, limits, [](const makaira::SearchIterationInfo& info) {
                std::cout << "info depth " << info.depth << " seldepth " << info.seldepth << " ";
                print_uci_score(info.score);
                std::cout << " nodes " << fun_display_count(info.nodes) << " time " << info.time_ms << " nps "
                          << fun_display_count(info.nps);
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

            evaluator.set_profile_mode(false);
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
                      << " nodes_pv " << result.stats.nodes_pv
                      << " nodes_cut " << result.stats.nodes_cut
                      << " nodes_all " << result.stats.nodes_all
                      << " qnodes " << result.stats.qnodes
                      << " qnodes_in_check " << result.stats.qnodes_in_check
                      << " qnodes_standpat_used " << result.stats.qnodes_standpat_used
                      << " movegen " << result.stats.movegen_calls
                      << " moves_generated " << result.stats.moves_generated
                      << " pick_iters " << result.stats.move_pick_iterations
                      << " tt_cutoffs " << result.stats.tt_cutoffs
                      << " tt_hit_exact " << result.stats.hit_exact
                      << " tt_hit_lower " << result.stats.hit_lower
                      << " tt_hit_upper " << result.stats.hit_upper
                      << " tt_move_used " << result.stats.tt_move_used
                      << " fh_tt " << result.stats.fh_tt
                      << " fh_goodcap " << result.stats.fh_goodcap
                      << " fh_quiet " << result.stats.fh_quiet
                      << " fh_badcap " << result.stats.fh_badcap
                      << " fh_promo " << result.stats.fh_promo
                      << " fh_check " << result.stats.fh_check
                      << " history_updates " << result.stats.history_updates
                      << " cont_updates " << result.stats.cont_history_updates
                      << " nmp " << result.stats.nmp_cutoffs << "/" << result.stats.nmp_attempts
                      << " nmp_verify " << result.stats.nmp_verifications << ":" << result.stats.nmp_verification_fails
                      << " lmr " << result.stats.lmr_reduced
                      << " lmr_re " << result.stats.lmr_researches
                      << " lmr_fh " << result.stats.lmr_fail_high_after_reduce
                      << " rfp_hits " << result.stats.rfp_hits
                      << " razor_hits " << result.stats.razor_hits
                      << " futility_prunes " << result.stats.futility_prunes
                      << " lmp_prunes " << result.stats.lmp_prunes
                      << " hist_prunes " << result.stats.hist_prunes
                      << " probcut_hits " << result.stats.probcut_hits
                      << " see_calls " << result.stats.see_calls
                      << " see_prunes_q " << result.stats.see_prunes_q
                      << " q_delta_prunes " << result.stats.q_delta_prunes
                      << " mate_lost_events " << result.stats.mate_lost_events
                      << " pv_instability_events " << result.stats.pv_instability_events
                      << " draw_alarm_events " << result.stats.draw_alarm_events
                      << " eval_calls " << est.eval_calls
                      << " eval_ns_total " << est.eval_ns_total
                      << " eval_ns_pawn " << est.eval_ns_pawn
                      << " eval_ns_attack_maps " << est.eval_ns_attack_maps
                      << " eval_ns_mobility " << est.eval_ns_mobility
                      << " eval_ns_king " << est.eval_ns_king
                      << " eval_ns_threats " << est.eval_ns_threats
                      << " eval_ns_space " << est.eval_ns_space
                      << " pawn_hash_hits " << est.pawn_hash_hits
                      << " pawn_hash_misses " << est.pawn_hash_misses
                      << "\n";
        } else if (line.rfind("profile_search", 0) == 0) {
            const auto tokens = split_tokens(line);

            makaira::SearchLimits limits;
            limits.depth = 8;
            limits.nodes = 10000;
            limits.move_overhead_ms = 0;
            limits.nodes_as_time = false;
            limits.profile_mode = true;
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "depth" && i + 1 < tokens.size()) {
                    limits.depth = std::max(1, parse_int(tokens[++i], limits.depth));
                } else if (tokens[i] == "nodes" && i + 1 < tokens.size()) {
                    limits.nodes = static_cast<std::uint64_t>(std::max(1, parse_int(tokens[++i], static_cast<int>(limits.nodes))));
                }
            }

            evaluator.set_profile_mode(true);
            evaluator.clear_stats();
            const auto started = std::chrono::steady_clock::now();
            const auto result = searcher.search(position, limits);
            const auto elapsed =
              std::max<std::int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::steady_clock::now() - started)
                                           .count());
            const std::uint64_t raw_nps = (result.stats.nodes * 1000ULL) / static_cast<std::uint64_t>(elapsed);
            const auto est = evaluator.stats();

            std::cout << "info string profile_search depth " << result.depth
                      << " seldepth " << result.seldepth
                      << " nodes " << result.stats.nodes
                      << " time_ms " << elapsed
                      << " nps " << raw_nps
                      << " nodes_pv " << result.stats.nodes_pv
                      << " nodes_cut " << result.stats.nodes_cut
                      << " nodes_all " << result.stats.nodes_all
                      << " tt " << result.stats.tt_hits << "/" << result.stats.tt_probes
                      << " tt_cutoffs " << result.stats.tt_cutoffs
                      << " qnodes " << result.stats.qnodes
                      << " q_in_check " << result.stats.qnodes_in_check
                      << " q_standpat " << result.stats.qnodes_standpat_used
                      << " fh_tt " << result.stats.fh_tt
                      << " fh_goodcap " << result.stats.fh_goodcap
                      << " fh_quiet " << result.stats.fh_quiet
                      << " fh_badcap " << result.stats.fh_badcap
                      << " fh_promo " << result.stats.fh_promo
                      << " fh_check " << result.stats.fh_check
                      << " rfp " << result.stats.rfp_hits
                      << " razor " << result.stats.razor_hits
                      << " futility " << result.stats.futility_prunes
                      << " lmp " << result.stats.lmp_prunes
                      << " histp " << result.stats.hist_prunes
                      << " probcut " << result.stats.probcut_hits
                      << " see_calls " << result.stats.see_calls
                      << " see_prunes_q " << result.stats.see_prunes_q
                      << " q_delta_prunes " << result.stats.q_delta_prunes
                      << " alarm_mate_lost " << result.stats.mate_lost_events
                      << " alarm_pv_instability " << result.stats.pv_instability_events
                      << " alarm_draw " << result.stats.draw_alarm_events
                      << " eval_calls " << est.eval_calls
                      << " eval_ns_total " << est.eval_ns_total
                      << " eval_ns_pawn " << est.eval_ns_pawn
                      << " eval_ns_attack_maps " << est.eval_ns_attack_maps
                      << " eval_ns_mobility " << est.eval_ns_mobility
                      << " eval_ns_king " << est.eval_ns_king
                      << " eval_ns_threats " << est.eval_ns_threats
                      << " eval_ns_space " << est.eval_ns_space
                      << " pawn_hash_hits " << est.pawn_hash_hits
                      << " pawn_hash_misses " << est.pawn_hash_misses
                      << "\n";
            evaluator.set_profile_mode(false);
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

