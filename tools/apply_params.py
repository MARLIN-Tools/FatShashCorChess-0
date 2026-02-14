#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

HEADER_TEMPLATE = """#pragma once

namespace makaira::eval_params_tuned {{

inline constexpr int MATERIAL_PSQT_MG_SCALE = {MATERIAL_PSQT_MG_SCALE};
inline constexpr int MATERIAL_PSQT_EG_SCALE = {MATERIAL_PSQT_EG_SCALE};
inline constexpr int PAWN_MG_SCALE = {PAWN_MG_SCALE};
inline constexpr int PAWN_EG_SCALE = {PAWN_EG_SCALE};
inline constexpr int PAWN_PASSED_MG_SCALE = {PAWN_PASSED_MG_SCALE};
inline constexpr int PAWN_PASSED_EG_SCALE = {PAWN_PASSED_EG_SCALE};
inline constexpr int PAWN_ISOLATED_MG_SCALE = {PAWN_ISOLATED_MG_SCALE};
inline constexpr int PAWN_ISOLATED_EG_SCALE = {PAWN_ISOLATED_EG_SCALE};
inline constexpr int PAWN_DOUBLED_MG_SCALE = {PAWN_DOUBLED_MG_SCALE};
inline constexpr int PAWN_DOUBLED_EG_SCALE = {PAWN_DOUBLED_EG_SCALE};
inline constexpr int PAWN_BACKWARD_MG_SCALE = {PAWN_BACKWARD_MG_SCALE};
inline constexpr int PAWN_BACKWARD_EG_SCALE = {PAWN_BACKWARD_EG_SCALE};
inline constexpr int PAWN_CANDIDATE_MG_SCALE = {PAWN_CANDIDATE_MG_SCALE};
inline constexpr int PAWN_CANDIDATE_EG_SCALE = {PAWN_CANDIDATE_EG_SCALE};
inline constexpr int PAWN_CONNECTED_MG_SCALE = {PAWN_CONNECTED_MG_SCALE};
inline constexpr int PAWN_CONNECTED_EG_SCALE = {PAWN_CONNECTED_EG_SCALE};
inline constexpr int PAWN_SUPPORTED_MG_SCALE = {PAWN_SUPPORTED_MG_SCALE};
inline constexpr int PAWN_SUPPORTED_EG_SCALE = {PAWN_SUPPORTED_EG_SCALE};
inline constexpr int PAWN_OUTSIDE_MG_SCALE = {PAWN_OUTSIDE_MG_SCALE};
inline constexpr int PAWN_OUTSIDE_EG_SCALE = {PAWN_OUTSIDE_EG_SCALE};
inline constexpr int PAWN_BLOCKED_MG_SCALE = {PAWN_BLOCKED_MG_SCALE};
inline constexpr int PAWN_BLOCKED_EG_SCALE = {PAWN_BLOCKED_EG_SCALE};
inline constexpr int MOBILITY_MG_SCALE = {MOBILITY_MG_SCALE};
inline constexpr int MOBILITY_EG_SCALE = {MOBILITY_EG_SCALE};
inline constexpr int KING_MG_SCALE = {KING_MG_SCALE};
inline constexpr int KING_EG_SCALE = {KING_EG_SCALE};
inline constexpr int KING_SHELTER_MG_SCALE = {KING_SHELTER_MG_SCALE};
inline constexpr int KING_SHELTER_EG_SCALE = {KING_SHELTER_EG_SCALE};
inline constexpr int KING_STORM_MG_SCALE = {KING_STORM_MG_SCALE};
inline constexpr int KING_STORM_EG_SCALE = {KING_STORM_EG_SCALE};
inline constexpr int KING_DANGER_MG_SCALE = {KING_DANGER_MG_SCALE};
inline constexpr int KING_DANGER_EG_SCALE = {KING_DANGER_EG_SCALE};
inline constexpr int PIECE_MG_SCALE = {PIECE_MG_SCALE};
inline constexpr int PIECE_EG_SCALE = {PIECE_EG_SCALE};
inline constexpr int PIECE_BISHOP_PAIR_MG_SCALE = {PIECE_BISHOP_PAIR_MG_SCALE};
inline constexpr int PIECE_BISHOP_PAIR_EG_SCALE = {PIECE_BISHOP_PAIR_EG_SCALE};
inline constexpr int PIECE_ROOK_FILE_MG_SCALE = {PIECE_ROOK_FILE_MG_SCALE};
inline constexpr int PIECE_ROOK_FILE_EG_SCALE = {PIECE_ROOK_FILE_EG_SCALE};
inline constexpr int PIECE_ROOK_SEVENTH_MG_SCALE = {PIECE_ROOK_SEVENTH_MG_SCALE};
inline constexpr int PIECE_ROOK_SEVENTH_EG_SCALE = {PIECE_ROOK_SEVENTH_EG_SCALE};
inline constexpr int PIECE_KNIGHT_OUTPOST_MG_SCALE = {PIECE_KNIGHT_OUTPOST_MG_SCALE};
inline constexpr int PIECE_KNIGHT_OUTPOST_EG_SCALE = {PIECE_KNIGHT_OUTPOST_EG_SCALE};
inline constexpr int PIECE_BAD_BISHOP_MG_SCALE = {PIECE_BAD_BISHOP_MG_SCALE};
inline constexpr int PIECE_BAD_BISHOP_EG_SCALE = {PIECE_BAD_BISHOP_EG_SCALE};
inline constexpr int THREAT_MG_SCALE = {THREAT_MG_SCALE};
inline constexpr int THREAT_EG_SCALE = {THREAT_EG_SCALE};
inline constexpr int THREAT_HANGING_MG_SCALE = {THREAT_HANGING_MG_SCALE};
inline constexpr int THREAT_HANGING_EG_SCALE = {THREAT_HANGING_EG_SCALE};
inline constexpr int THREAT_PAWN_MG_SCALE = {THREAT_PAWN_MG_SCALE};
inline constexpr int THREAT_PAWN_EG_SCALE = {THREAT_PAWN_EG_SCALE};
inline constexpr int SPACE_MG_SCALE = {SPACE_MG_SCALE};
inline constexpr int SPACE_EG_SCALE = {SPACE_EG_SCALE};
inline constexpr int ENDGAME_KING_ACTIVITY_MG_SCALE = {ENDGAME_KING_ACTIVITY_MG_SCALE};
inline constexpr int ENDGAME_KING_ACTIVITY_EG_SCALE = {ENDGAME_KING_ACTIVITY_EG_SCALE};
inline constexpr int TEMPO_SCALE = {TEMPO_SCALE};

}}  // namespace makaira::eval_params_tuned
"""

PSQT_HEADER_PREAMBLE = """#pragma once

#include <array>

namespace makaira::eval_psqt_tuned {{

// [piece_type][bucket], piece_type index uses PieceType enum values.
inline constexpr std::array<std::array<int, 32>, 7> PSQT_BUCKET_MG = {{{{
{PSQT_BUCKET_MG}
}}}};

inline constexpr std::array<std::array<int, 32>, 7> PSQT_BUCKET_EG = {{{{
{PSQT_BUCKET_EG}
}}}};

}}  // namespace makaira::eval_psqt_tuned
"""


def format_psqt_table(rows):
    lines = []
    for row in rows:
        vals = ", ".join(str(int(v)) for v in row)
        lines.append(f"  {{{vals}}},")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate eval_params_tuned.h from texel_fit output")
    ap.add_argument("weights", type=Path, help="JSON from tools/texel_fit.py")
    ap.add_argument("--out", type=Path, default=Path("src/eval_params_tuned.h"))
    ap.add_argument("--psqt-out", type=Path, default=Path("src/eval_psqt_tuned.h"))
    args = ap.parse_args()

    payload = json.loads(args.weights.read_text(encoding="utf-8"))
    scales = payload.get("scales", {})

    defaults = {
        "MATERIAL_PSQT_MG_SCALE": 100,
        "MATERIAL_PSQT_EG_SCALE": 100,
        "PAWN_MG_SCALE": 100,
        "PAWN_EG_SCALE": 100,
        "PAWN_PASSED_MG_SCALE": 100,
        "PAWN_PASSED_EG_SCALE": 100,
        "PAWN_ISOLATED_MG_SCALE": 100,
        "PAWN_ISOLATED_EG_SCALE": 100,
        "PAWN_DOUBLED_MG_SCALE": 100,
        "PAWN_DOUBLED_EG_SCALE": 100,
        "PAWN_BACKWARD_MG_SCALE": 100,
        "PAWN_BACKWARD_EG_SCALE": 100,
        "PAWN_CANDIDATE_MG_SCALE": 100,
        "PAWN_CANDIDATE_EG_SCALE": 100,
        "PAWN_CONNECTED_MG_SCALE": 100,
        "PAWN_CONNECTED_EG_SCALE": 100,
        "PAWN_SUPPORTED_MG_SCALE": 100,
        "PAWN_SUPPORTED_EG_SCALE": 100,
        "PAWN_OUTSIDE_MG_SCALE": 100,
        "PAWN_OUTSIDE_EG_SCALE": 100,
        "PAWN_BLOCKED_MG_SCALE": 100,
        "PAWN_BLOCKED_EG_SCALE": 100,
        "MOBILITY_MG_SCALE": 100,
        "MOBILITY_EG_SCALE": 100,
        "KING_MG_SCALE": 100,
        "KING_EG_SCALE": 100,
        "KING_SHELTER_MG_SCALE": 100,
        "KING_SHELTER_EG_SCALE": 100,
        "KING_STORM_MG_SCALE": 100,
        "KING_STORM_EG_SCALE": 100,
        "KING_DANGER_MG_SCALE": 100,
        "KING_DANGER_EG_SCALE": 100,
        "PIECE_MG_SCALE": 100,
        "PIECE_EG_SCALE": 100,
        "PIECE_BISHOP_PAIR_MG_SCALE": 100,
        "PIECE_BISHOP_PAIR_EG_SCALE": 100,
        "PIECE_ROOK_FILE_MG_SCALE": 100,
        "PIECE_ROOK_FILE_EG_SCALE": 100,
        "PIECE_ROOK_SEVENTH_MG_SCALE": 100,
        "PIECE_ROOK_SEVENTH_EG_SCALE": 100,
        "PIECE_KNIGHT_OUTPOST_MG_SCALE": 100,
        "PIECE_KNIGHT_OUTPOST_EG_SCALE": 100,
        "PIECE_BAD_BISHOP_MG_SCALE": 100,
        "PIECE_BAD_BISHOP_EG_SCALE": 100,
        "THREAT_MG_SCALE": 100,
        "THREAT_EG_SCALE": 100,
        "THREAT_HANGING_MG_SCALE": 100,
        "THREAT_HANGING_EG_SCALE": 100,
        "THREAT_PAWN_MG_SCALE": 100,
        "THREAT_PAWN_EG_SCALE": 100,
        "SPACE_MG_SCALE": 100,
        "SPACE_EG_SCALE": 100,
        "ENDGAME_KING_ACTIVITY_MG_SCALE": 100,
        "ENDGAME_KING_ACTIVITY_EG_SCALE": 100,
        "TEMPO_SCALE": 100,
    }

    defaults.update({k: int(v) for k, v in scales.items() if k in defaults})

    text = HEADER_TEMPLATE.format(**defaults)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")

    default_row = [0] * 32
    psqt_mg = payload.get("psqt_bucket_mg", [default_row[:] for _ in range(7)])
    psqt_eg = payload.get("psqt_bucket_eg", [default_row[:] for _ in range(7)])

    if len(psqt_mg) != 7 or any(len(r) != 32 for r in psqt_mg):
        raise ValueError("psqt_bucket_mg must be 7x32")
    if len(psqt_eg) != 7 or any(len(r) != 32 for r in psqt_eg):
        raise ValueError("psqt_bucket_eg must be 7x32")

    psqt_text = PSQT_HEADER_PREAMBLE.format(
        PSQT_BUCKET_MG=format_psqt_table(psqt_mg),
        PSQT_BUCKET_EG=format_psqt_table(psqt_eg),
    )
    args.psqt_out.write_text(psqt_text, encoding="utf-8")
    print(f"wrote {args.psqt_out}")


if __name__ == "__main__":
    main()
