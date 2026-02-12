#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_dataset(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty dataset")

    feature_cols = [c for c in reader.fieldnames if c not in {"result", "eval_cp"}]
    x = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
    y = np.array([float(r["result"]) for r in rows], dtype=np.float64)

    # 0/0.5/1 labels are mapped to 0..1 target probabilities.
    return x, y, feature_cols


def train_logistic(x, y, lr=0.01, l2=1e-4, epochs=2000):
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for epoch in range(epochs):
        z = x @ w + b
        p = sigmoid(z)

        # Logistic loss gradient with L2.
        g = (p - y)
        gw = (x.T @ g) / n + l2 * w
        gb = np.mean(g)

        w -= lr * gw
        b -= lr * gb

        if epoch % 200 == 0:
            loss = -np.mean(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))
            print(f"epoch={epoch} loss={loss:.6f}")

    return w, b


def to_scale(val):
    return int(round(100.0 * float(val)))


def build_scale_map(feature_cols, weights):
    # Default all scales to 100.
    scales = {
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

    mapping = {
        "matpsqt_mg": "MATERIAL_PSQT_MG_SCALE",
        "matpsqt_eg": "MATERIAL_PSQT_EG_SCALE",
        "pawn_mg": "PAWN_MG_SCALE",
        "pawn_eg": "PAWN_EG_SCALE",
        "mob_mg": "MOBILITY_MG_SCALE",
        "mob_eg": "MOBILITY_EG_SCALE",
        "king_mg": "KING_MG_SCALE",
        "king_eg": "KING_EG_SCALE",
        "piece_mg": "PIECE_MG_SCALE",
        "piece_eg": "PIECE_EG_SCALE",
        "threat_mg": "THREAT_MG_SCALE",
        "threat_eg": "THREAT_EG_SCALE",
        "space_mg": "SPACE_MG_SCALE",
        "space_eg": "SPACE_EG_SCALE",
        "tempo": "TEMPO_SCALE",
    }

    for c, w in zip(feature_cols, weights):
        if c in mapping:
            scales[mapping[c]] = int(np.clip(to_scale(w), 25, 400))

    return scales


def main():
    ap = argparse.ArgumentParser(description="Fit Makaira eval component scales with logistic regression")
    ap.add_argument("dataset", type=Path, help="CSV from makaira_eval_extract")
    ap.add_argument("--out", type=Path, default=Path("tools/texel_weights.json"))
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()

    x, y, feature_cols = load_dataset(args.dataset)
    # Standardize for stable optimization.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    xs = (x - mean) / std

    w, b = train_logistic(xs, y, lr=args.lr, l2=args.l2, epochs=args.epochs)

    # Convert back to unstandardized weights.
    w_unscaled = w / std
    b_unscaled = b - np.dot(mean, w_unscaled)

    scales = build_scale_map(feature_cols, w_unscaled)

    payload = {
        "bias": float(b_unscaled),
        "features": feature_cols,
        "weights": [float(v) for v in w_unscaled],
        "scales": scales,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()