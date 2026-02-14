#!/usr/bin/env python3
"""
Build multiple FatShashCorChess 0 commit binaries and run pairwise SPRT matches via fastchess.

Requires:
  - Python 3
  - CMake + build toolchain
  - fastchess executable
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Iterable, List, Tuple


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
    return f"fatshashcorchess0_{commit[:8]}.exe" if os.name == "nt" else f"fatshashcorchess0_{commit[:8]}"


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
            build_dir / "Release" / "fatshashcorchess0.exe",
            build_dir / "fatshashcorchess0.exe",
            build_dir / "fatshashcorchess0",
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


def find_fastchess(explicit_path: str | None) -> pathlib.Path | None:
    if explicit_path:
        p = pathlib.Path(explicit_path).expanduser().resolve()
        if p.exists():
            return p
        return None

    env_path = os.getenv("FASTCHESS")
    if env_path:
        p = pathlib.Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    for exe in ("fastchess.exe", "fastchess"):
        which = shutil.which(exe)
        if which:
            return pathlib.Path(which).resolve()

    home = pathlib.Path.home()
    roots = [home / "Downloads", home / "Desktop"]
    patterns = ["fastchess.exe", "fastchess*.exe"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for found in root.rglob(pat):
                if found.is_file():
                    return found.resolve()
    return None


def normalize_fen(fen_like: str) -> str | None:
    # Keep only the FEN fields and ensure 6-field FEN.
    fields = fen_like.split()
    if len(fields) < 4:
        return None
    if len(fields) == 4:
        fields.extend(["0", "1"])
    elif len(fields) == 5:
        fields.append("1")
    return " ".join(fields[:6])


def load_fens(paths: Iterable[pathlib.Path]) -> List[str]:
    out: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept EPD/FEN lines and drop trailing inline comments.
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if ";" in line:
                line = line.split(";", 1)[0].strip()
            if not line:
                continue
            fen = normalize_fen(line)
            if fen:
                out.append(fen)
    return out


def write_openings_file(openings: List[str], path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for fen in openings:
            f.write(fen + "\n")
    return path


RESULTS_RE = re.compile(
    r"Games:\s*(?P<games>\d+),\s*Wins:\s*(?P<wins>\d+),\s*Losses:\s*(?P<losses>\d+),\s*Draws:\s*(?P<draws>\d+),\s*Points:\s*(?P<points>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LLR_RE = re.compile(
    r"LLR:\s*(?P<llr>[-+]?\d+(?:\.\d+)?)\s*\([^)]+\)\s*\((?P<lower>[-+]?\d+(?:\.\d+)?),\s*(?P<upper>[-+]?\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)


def parse_fastchess_result(output: str, fallback_cfg: SprtConfig) -> SprtResult:
    m_res = None
    for m in RESULTS_RE.finditer(output):
        m_res = m
    if not m_res:
        raise RuntimeError("Could not parse fastchess results block.")

    games = int(m_res.group("games"))
    wins = int(m_res.group("wins"))
    losses = int(m_res.group("losses"))
    draws = int(m_res.group("draws"))
    points = float(m_res.group("points"))

    llr = 0.0
    lower = sprt_bound_lower(fallback_cfg.alpha, fallback_cfg.beta)
    upper = sprt_bound_upper(fallback_cfg.alpha, fallback_cfg.beta)

    m_llr = None
    for m in LLR_RE.finditer(output):
        m_llr = m
    if m_llr:
        llr = float(m_llr.group("llr"))
        lower = float(m_llr.group("lower"))
        upper = float(m_llr.group("upper"))

    decision = "INCONCLUSIVE"
    if "SPRT: H1" in output:
        decision = "H1"
    elif "SPRT: H0" in output:
        decision = "H0"
    else:
        if llr >= upper:
            decision = "H1"
        elif llr <= lower:
            decision = "H0"

    return SprtResult(
        decision=decision,
        games=games,
        score=points,
        llr=llr,
        wins=wins,
        draws=draws,
        losses=losses,
    )


def match_sprt_fastchess(
    fastchess: pathlib.Path,
    challenger: EngineVersion,
    opponent: EngineVersion,
    openings_file: pathlib.Path,
    tc: str,
    movetime_ms: int,
    sprt: SprtConfig,
    hash_mb: int,
    threads: int,
    seed: int,
    games_per_match: int,
    pgn_dir: pathlib.Path | None,
) -> SprtResult:
    rounds = max(1, (games_per_match + 1) // 2)

    engine_a_args = [
        f"name={challenger.short}",
        f"cmd={str(challenger.binary)}",
        f"option.Hash={hash_mb}",
    ]
    engine_b_args = [
        f"name={opponent.short}",
        f"cmd={str(opponent.binary)}",
        f"option.Hash={hash_mb}",
    ]
    if threads > 1:
        engine_a_args.append(f"option.Threads={threads}")
        engine_b_args.append(f"option.Threads={threads}")

    cmd = [
        str(fastchess),
        "-engine",
        *engine_a_args,
        "-engine",
        *engine_b_args,
    ]

    each = ["-each", "proto=uci"]
    if movetime_ms > 0:
        st_seconds = max(0.001, movetime_ms / 1000.0)
        each.append(f"st={st_seconds:.3f}")
    else:
        each.append(f"tc={tc}")

    cmd += each
    cmd += [
        "-rounds",
        str(rounds),
        "-repeat",
        "-openings",
        f"file={str(openings_file)}",
        "format=epd",
        "order=random",
        "-srand",
        str(seed),
        "-sprt",
        f"elo0={sprt.elo0}",
        f"elo1={sprt.elo1}",
        f"alpha={sprt.alpha}",
        f"beta={sprt.beta}",
    ]

    if pgn_dir:
        pgn_dir.mkdir(parents=True, exist_ok=True)
        pgn = pgn_dir / f"{challenger.short}_vs_{opponent.short}.pgn"
        cmd.extend(["-pgnout", f"file={str(pgn)}"])

    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = (cp.stdout or "") + "\n" + (cp.stderr or "")

    if cp.returncode != 0 and "Finished match" not in output:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(f"fastchess failed ({cp.returncode}):\n{tail}")

    return parse_fastchess_result(output, sprt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FatShashCorChess 0 versions and run pairwise SPRT matches using fastchess.")
    ap.add_argument("--repo", default=".", help="Path to FatShashCorChess 0 git repo")
    ap.add_argument("--rev", default="main", help="Revision to enumerate commits from")
    ap.add_argument("--max-commits", type=int, default=0, help="Use last N commits (0 = all)")
    ap.add_argument("--first-parent", action="store_true", default=True, help="Use first-parent history")
    ap.add_argument("--out", default="sprt", help="Output dir")
    ap.add_argument("--jobs", type=int, default=8, help="Build parallel jobs")
    ap.add_argument("--keep-worktrees", action="store_true", help="Keep temporary worktrees")
    ap.add_argument("--fastchess", default="", help="Path to fastchess binary (auto-detect if omitted)")
    ap.add_argument("--tc", default="10+0.1", help="fastchess time control (used when --movetime-ms <= 0)")
    ap.add_argument("--movetime-ms", type=int, default=-1, help="Legacy per-move time in ms; overrides --tc when >0")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument(
        "--games-per-match",
        type=int,
        default=1000,
        help="Total games per pair (uses rounds=ceil(games/2) with color-repeat).",
    )
    ap.add_argument("--min-games", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--hash", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openings", nargs="*", default=[], help="Opening files (FEN/EPD); defaults to bench/fens.txt")
    ap.add_argument("--save-pgn", action="store_true", help="Save PGN for each pair match")
    args = ap.parse_args()

    if args.games_per_match <= 0:
        print("games-per-match must be > 0.", file=sys.stderr)
        return 1

    if args.min_games > 0 or args.max_games > 0:
        print("Note: fastchess controls game count via --games-per-match; min/max are ignored.", file=sys.stderr)

    repo = pathlib.Path(args.repo).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fastchess = find_fastchess(args.fastchess or None)
    if not fastchess:
        print("Could not find fastchess. Set --fastchess or FASTCHESS env var.", file=sys.stderr)
        return 1
    print(f"Using fastchess: {fastchess}")

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

    opening_paths = [pathlib.Path(p).resolve() for p in args.openings] if args.openings else [repo / "bench" / "fens.txt"]
    openings = load_fens(opening_paths)
    if not openings:
        print("No valid opening FEN/EPD entries found.", file=sys.stderr)
        return 1
    openings_file = write_openings_file(openings, out_dir / "openings_fastchess.epd")

    sprt_cfg = SprtConfig(
        alpha=args.alpha,
        beta=args.beta,
        elo0=args.elo0,
        elo1=args.elo1,
        min_games=args.games_per_match,
        max_games=args.games_per_match,
    )

    wins = {v.short: 0 for v in versions}
    losses = {v.short: 0 for v in versions}
    inconc = {v.short: 0 for v in versions}
    points = {v.short: 0.0 for v in versions}
    games = {v.short: 0 for v in versions}

    print("\nRunning pairwise fastchess SPRT matches...")
    for i in range(len(versions)):
        for j in range(i + 1, len(versions)):
            a = versions[i]
            b = versions[j]
            seed = args.seed + i * 1000 + j
            try:
                res = match_sprt_fastchess(
                    fastchess=fastchess,
                    challenger=a,
                    opponent=b,
                    openings_file=openings_file,
                    tc=args.tc,
                    movetime_ms=args.movetime_ms,
                    sprt=sprt_cfg,
                    hash_mb=args.hash,
                    threads=args.threads,
                    seed=seed,
                    games_per_match=args.games_per_match,
                    pgn_dir=(out_dir / "pgn") if args.save_pgn else None,
                )
            except RuntimeError as exc:
                print(f"{a.short} vs {b.short}: ERROR\n{exc}", file=sys.stderr)
                continue

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
