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
inline constexpr int MOBILITY_MG_SCALE = {MOBILITY_MG_SCALE};
inline constexpr int MOBILITY_EG_SCALE = {MOBILITY_EG_SCALE};
inline constexpr int KING_MG_SCALE = {KING_MG_SCALE};
inline constexpr int KING_EG_SCALE = {KING_EG_SCALE};
inline constexpr int PIECE_MG_SCALE = {PIECE_MG_SCALE};
inline constexpr int PIECE_EG_SCALE = {PIECE_EG_SCALE};
inline constexpr int THREAT_MG_SCALE = {THREAT_MG_SCALE};
inline constexpr int THREAT_EG_SCALE = {THREAT_EG_SCALE};
inline constexpr int SPACE_MG_SCALE = {SPACE_MG_SCALE};
inline constexpr int SPACE_EG_SCALE = {SPACE_EG_SCALE};
inline constexpr int TEMPO_SCALE = {TEMPO_SCALE};

}}  // namespace makaira::eval_params_tuned
"""


def main():
    ap = argparse.ArgumentParser(description="Generate eval_params_tuned.h from texel_fit output")
    ap.add_argument("weights", type=Path, help="JSON from tools/texel_fit.py")
    ap.add_argument("--out", type=Path, default=Path("src/eval_params_tuned.h"))
    args = ap.parse_args()

    payload = json.loads(args.weights.read_text(encoding="utf-8"))
    scales = payload.get("scales", {})

    defaults = {
        "MATERIAL_PSQT_MG_SCALE": 100,
        "MATERIAL_PSQT_EG_SCALE": 100,
        "PAWN_MG_SCALE": 100,
        "PAWN_EG_SCALE": 100,
        "MOBILITY_MG_SCALE": 100,
        "MOBILITY_EG_SCALE": 100,
        "KING_MG_SCALE": 100,
        "KING_EG_SCALE": 100,
        "PIECE_MG_SCALE": 100,
        "PIECE_EG_SCALE": 100,
        "THREAT_MG_SCALE": 100,
        "THREAT_EG_SCALE": 100,
        "SPACE_MG_SCALE": 100,
        "SPACE_EG_SCALE": 100,
        "TEMPO_SCALE": 100,
    }

    defaults.update({k: int(v) for k, v in scales.items() if k in defaults})

    text = HEADER_TEMPLATE.format(**defaults)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()