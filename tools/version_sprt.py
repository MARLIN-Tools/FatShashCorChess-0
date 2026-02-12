#!/usr/bin/env python3
"""
Build multiple Makaira commit binaries and run pairwise SPRT matches.

Requires:
  - Python 3
  - python-chess (`pip install chess`)
  - CMake + build toolchain
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from typing import Iterable, List, Tuple

import chess
import chess.engine


@dataclasses.dataclass
class EngineVersion:
    commit: str
    short: str
    label: str
    binary: pathlib.Path


@dataclasses.dataclass
class SprtConfig:
    alpha: float = 0.05
    beta: float = 0.05
    elo0: float = 0.0
    elo1: float = 5.0
    min_games: int = 20
    max_games: int = 200


@dataclasses.dataclass
class SprtResult:
    decision: str
    games: int
    score: float
    llr: float
    wins: int
    draws: int
    losses: int


def run(cmd: List[str], cwd: pathlib.Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def git_lines(repo: pathlib.Path, args: List[str]) -> List[str]:
    cp = run(["git", *args], cwd=repo)
    return [x.strip() for x in cp.stdout.splitlines() if x.strip()]


def collect_commits(repo: pathlib.Path, rev: str, first_parent: bool, max_commits: int) -> List[str]:
    args = ["rev-list", "--reverse"]
    if first_parent:
        args.append("--first-parent")
    args.append(rev)
    commits = git_lines(repo, args)
    if max_commits > 0:
        commits = commits[-max_commits:]
    return commits


def commit_subject(repo: pathlib.Path, commit: str) -> str:
    return run(["git", "show", "-s", "--format=%h %s", commit], cwd=repo).stdout.strip()


def binary_name(commit: str) -> str:
    return f"makaira_{commit[:8]}.exe" if os.name == "nt" else f"makaira_{commit[:8]}"


def build_commit(repo: pathlib.Path, commit: str, out_dir: pathlib.Path, jobs: int, keep_worktrees: bool) -> pathlib.Path | None:
    bins_dir = out_dir / "bin"
    wts_dir = out_dir / "worktrees"
    logs_dir = out_dir / "logs"
    bins_dir.mkdir(parents=True, exist_ok=True)
    wts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_bin = bins_dir / binary_name(commit)
    if out_bin.exists():
        return out_bin

    wt = wts_dir / commit[:12]
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)

    try:
        run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=repo)

        build_dir = wt / "build-rel"
        run(["cmake", "-S", str(wt), "-B", str(build_dir)], cwd=repo)
        run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs)], cwd=repo)

        candidates = [
            build_dir / "Release" / "makaira.exe",
            build_dir / "makaira.exe",
            build_dir / "makaira",
        ]
        src_bin = next((p for p in candidates if p.exists()), None)
        if not src_bin:
            return None

        shutil.copy2(src_bin, out_bin)
        return out_bin
    except subprocess.CalledProcessError as exc:
        log_path = logs_dir / f"build_{commit[:8]}.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"CMD: {' '.join(exc.cmd)}\n")
            f.write("---- STDOUT ----\n")
            f.write(exc.stdout or "")
            f.write("\n---- STDERR ----\n")
            f.write(exc.stderr or "")
        return None
    finally:
        if not keep_worktrees:
            try:
                run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, check=False)
            except Exception:
                pass


def logistic_expectation(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(samples: List[float], elo0: float, elo1: float) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    var = max(var, 1e-6)
    p0 = logistic_expectation(elo0)
    p1 = logistic_expectation(elo1)
    return n * (((mean - p0) ** 2 - (mean - p1) ** 2) / (2.0 * var))


def sprt_bound_upper(alpha: float, beta: float) -> float:
    return math.log((1.0 - beta) / alpha)


def sprt_bound_lower(alpha: float, beta: float) -> float:
    return math.log(beta / (1.0 - alpha))


def play_one_game(
    white: chess.engine.SimpleEngine,
    black: chess.engine.SimpleEngine,
    start_fen: str,
    movetime_ms: int,
    max_plies: int,
) -> str:
    board = chess.Board(start_fen)
    limit = chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0))

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white if board.turn == chess.WHITE else black
        try:
            result = engine.play(board, limit)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            # Side to move produced invalid/terminated response: forfeit loss.
            return "0-1" if board.turn == chess.WHITE else "1-0"

        if result.move is None or result.move == chess.Move.null():
            # Null move / no move in normal play is treated as forfeit.
            return "0-1" if board.turn == chess.WHITE else "1-0"

        if result.move not in board.legal_moves:
            # Defensive check: illegal move from engine is immediate loss.
            return "0-1" if board.turn == chess.WHITE else "1-0"

        board.push(result.move)

    if not board.is_game_over(claim_draw=True):
        return "1/2-1/2"
    return board.result(claim_draw=True)


def match_sprt(
    challenger: EngineVersion,
    opponent: EngineVersion,
    openings: List[str],
    movetime_ms: int,
    max_plies: int,
    sprt: SprtConfig,
    hash_mb: int,
    threads: int,
    seed: int,
) -> SprtResult:
    rng = random.Random(seed)
    opening_pool = openings[:]
    rng.shuffle(opening_pool)
    if not opening_pool:
        opening_pool = [chess.STARTING_FEN]

    outcomes: List[float] = []
    wins = draws = losses = 0
    upper = sprt_bound_upper(sprt.alpha, sprt.beta)
    lower = sprt_bound_lower(sprt.alpha, sprt.beta)

    with chess.engine.SimpleEngine.popen_uci(str(challenger.binary)) as eng_c, chess.engine.SimpleEngine.popen_uci(
        str(opponent.binary)
    ) as eng_o:
        # Optional common options for fairness.
        for eng in (eng_c, eng_o):
            try:
                eng.configure({"Hash": hash_mb, "Threads": threads})
            except Exception:
                pass

        for g in range(sprt.max_games):
            fen = opening_pool[g % len(opening_pool)]
            challenger_white = (g % 2) == 0
            white = eng_c if challenger_white else eng_o
            black = eng_o if challenger_white else eng_c

            result = play_one_game(white, black, fen, movetime_ms, max_plies)

            if result == "1-0":
                score = 1.0 if challenger_white else 0.0
            elif result == "0-1":
                score = 0.0 if challenger_white else 1.0
            else:
                score = 0.5

            outcomes.append(score)
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

            llr = sprt_llr(outcomes, sprt.elo0, sprt.elo1)
            if len(outcomes) >= sprt.min_games:
                if llr >= upper:
                    return SprtResult("H1", len(outcomes), sum(outcomes), llr, wins, draws, losses)
                if llr <= lower:
                    return SprtResult("H0", len(outcomes), sum(outcomes), llr, wins, draws, losses)

    llr = sprt_llr(outcomes, sprt.elo0, sprt.elo1)
    return SprtResult("INCONCLUSIVE", len(outcomes), sum(outcomes), llr, wins, draws, losses)


def load_fens(paths: Iterable[pathlib.Path]) -> List[str]:
    out: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Makaira versions and run pairwise SPRT matches.")
    ap.add_argument("--repo", default=".", help="Path to Makaira git repo")
    ap.add_argument("--rev", default="main", help="Revision to enumerate commits from")
    ap.add_argument("--max-commits", type=int, default=0, help="Use last N commits (0 = all)")
    ap.add_argument("--first-parent", action="store_true", default=True, help="Use first-parent history")
    ap.add_argument("--out", default="sprt", help="Output dir")
    ap.add_argument("--jobs", type=int, default=8, help="Build parallel jobs")
    ap.add_argument("--keep-worktrees", action="store_true", help="Keep temporary worktrees")
    ap.add_argument("--movetime-ms", type=int, default=30, help="Per-move time in ms")
    ap.add_argument("--max-plies", type=int, default=220, help="Max plies per game before draw adjudication")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--max-games", type=int, default=120)
    ap.add_argument("--hash", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    commits = collect_commits(repo, args.rev, args.first_parent, args.max_commits)
    if not commits:
        print("No commits found.", file=sys.stderr)
        return 1

    versions: List[EngineVersion] = []
    print(f"Building {len(commits)} revisions...")
    for c in commits:
        b = build_commit(repo, c, out_dir, args.jobs, args.keep_worktrees)
        if not b:
            print(f"[SKIP] {c[:8]} build failed")
            continue
        label = commit_subject(repo, c)
        print(f"[OK] {label} -> {b.name}")
        versions.append(EngineVersion(commit=c, short=c[:8], label=label, binary=b))

    if len(versions) < 2:
        print("Need at least 2 buildable versions.", file=sys.stderr)
        return 1

    openings = load_fens(
        [
            repo / "bench" / "fens.txt",
            repo / "bench" / "tactical_fens.txt",
        ]
    )
    if not openings:
        openings = [chess.STARTING_FEN]

    sprt_cfg = SprtConfig(
        alpha=args.alpha,
        beta=args.beta,
        elo0=args.elo0,
        elo1=args.elo1,
        min_games=args.min_games,
        max_games=args.max_games,
    )

    wins = {v.short: 0 for v in versions}
    losses = {v.short: 0 for v in versions}
    inconc = {v.short: 0 for v in versions}
    points = {v.short: 0.0 for v in versions}
    games = {v.short: 0 for v in versions}

    print("\nRunning pairwise SPRT matches...")
    pair_results: List[Tuple[str, str, SprtResult]] = []
    for i in range(len(versions)):
        for j in range(i + 1, len(versions)):
            a = versions[i]
            b = versions[j]
            seed = args.seed + i * 1000 + j
            res = match_sprt(
                challenger=a,
                opponent=b,
                openings=openings,
                movetime_ms=args.movetime_ms,
                max_plies=args.max_plies,
                sprt=sprt_cfg,
                hash_mb=args.hash,
                threads=args.threads,
                seed=seed,
            )
            pair_results.append((a.short, b.short, res))
            points[a.short] += res.score
            games[a.short] += res.games
            points[b.short] += float(res.games) - res.score
            games[b.short] += res.games
            if res.decision == "H1":
                wins[a.short] += 1
                losses[b.short] += 1
            elif res.decision == "H0":
                wins[b.short] += 1
                losses[a.short] += 1
            else:
                inconc[a.short] += 1
                inconc[b.short] += 1

            print(
                f"{a.short} vs {b.short}: {res.decision} "
                f"({res.wins}-{res.draws}-{res.losses}, games={res.games}, llr={res.llr:.3f})"
            )

    print("\nScoreboard (pairwise SPRT)")
    ranked = sorted(
        versions,
        key=lambda v: (
            wins[v.short] - losses[v.short],
            (points[v.short] / games[v.short]) if games[v.short] else 0.5,
            wins[v.short],
            -losses[v.short],
        ),
        reverse=True,
    )
    for v in ranked:
        score_rate = (points[v.short] / games[v.short]) if games[v.short] else 0.5
        print(
            f"{v.short}: W={wins[v.short]} L={losses[v.short]} I={inconc[v.short]} "
            f"score={wins[v.short]-losses[v.short]} "
            f"pts={points[v.short]:.1f}/{games[v.short]} ({score_rate*100.0:.2f}%) | {v.label}"
        )

    best = ranked[0]
    print(f"\nBEST_BY_SPRT_SCORE: {best.short} | {best.label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
