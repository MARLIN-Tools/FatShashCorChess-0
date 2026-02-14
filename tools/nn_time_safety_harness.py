#!/usr/bin/env python3
"""
Short-TC NN time-safety harness using fastchess.

Default run:
  - st=2 (2 seconds per move)
  - 10 games (5 pairs)
  - fixed openings file
  - reports timeout count, avg/p95 move time from PGN comments,
    and hard-stop polling stats from benchraw.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
from typing import Dict, List, Tuple


def find_fastchess(explicit: str | None) -> pathlib.Path:
    if explicit:
        p = pathlib.Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"fastchess not found: {p}")
    for exe in ("fastchess.exe", "fastchess"):
        which = shutil.which(exe)
        if which:
            return pathlib.Path(which).resolve()
    raise FileNotFoundError("fastchess not found in PATH; pass --fastchess")


def percentile_ms(values_ms: List[float], p: float) -> float:
    if not values_ms:
        return 0.0
    v = sorted(values_ms)
    idx = int(max(0, min(len(v) - 1, round((len(v) - 1) * p))))
    return v[idx]


def normalize_fen_line(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    fields = s.split()
    if len(fields) < 4:
        return None
    if len(fields) == 4:
        fields.extend(["0", "1"])
    elif len(fields) == 5:
        fields.append("1")
    return " ".join(fields[:6])


def sanitize_openings(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    fens: List[str] = []
    for line in lines:
        fen = normalize_fen_line(line)
        if fen:
            fens.append(fen)
    if not fens:
        raise RuntimeError(f"No valid FENs found in openings file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(fens) + "\n", encoding="utf-8")
    return dst


def parse_pgn_metrics(pgn_path: pathlib.Path, nn_name: str) -> Dict[str, float]:
    text = pgn_path.read_text(encoding="utf-8", errors="replace")
    games = [g for g in text.split("\n\n[Event ") if g.strip()]
    timeout_total = 0
    timeout_nn = 0
    timeout_opp = 0
    move_times_ms: List[float] = []

    # Move comments from fastchess PGN usually contain "... 1.974s, ..."
    for sec in re.findall(r"\{[^{}]*?([0-9]+(?:\.[0-9]+)?)s,\s*[^{}]*\}", text):
        try:
            move_times_ms.append(float(sec) * 1000.0)
        except ValueError:
            pass

    for raw in games:
        g = raw
        if not g.startswith("[Event "):
            g = "[Event " + g
        white_m = re.search(r'^\[White "([^"]+)"\]', g, re.MULTILINE)
        black_m = re.search(r'^\[Black "([^"]+)"\]', g, re.MULTILINE)
        result_m = re.search(r'^\[Result "([^"]+)"\]', g, re.MULTILINE)
        term_m = re.search(r'^\[Termination "([^"]+)"\]', g, re.MULTILINE)
        if not white_m or not black_m or not result_m:
            continue
        white = white_m.group(1)
        black = black_m.group(1)
        result = result_m.group(1)
        termination = (term_m.group(1).lower() if term_m else "")
        if "time forfeit" not in termination:
            continue
        timeout_total += 1
        loser = ""
        if result == "1-0":
            loser = black
        elif result == "0-1":
            loser = white
        if loser == nn_name:
            timeout_nn += 1
        else:
            timeout_opp += 1

    avg_ms = (sum(move_times_ms) / len(move_times_ms)) if move_times_ms else 0.0
    p95_ms = percentile_ms(move_times_ms, 0.95)
    return {
        "pgn_games": float(len(games)),
        "timeouts_total": float(timeout_total),
        "timeouts_nn": float(timeout_nn),
        "timeouts_opp": float(timeout_opp),
        "move_samples": float(len(move_times_ms)),
        "move_time_avg_ms": avg_ms,
        "move_time_p95_ms": p95_ms,
    }


def parse_kv_line(line: str) -> Dict[str, str]:
    parts = line.strip().split()
    out: Dict[str, str] = {}
    i = 0
    while i + 1 < len(parts):
        key = parts[i]
        val = parts[i + 1]
        if key in {"info", "string", "benchraw"}:
            i += 1
            continue
        out[key] = val
        i += 2
    return out


def run_benchraw(
  engine: pathlib.Path, uci_options: List[Tuple[str, str]], depth: int, nodes: int
) -> Dict[str, str]:
    cmds = ["uci", "isready"]
    for name, value in uci_options:
        cmds.append(f"setoption name {name} value {value}")
    cmds += ["isready", "position startpos", f"benchraw depth {depth} nodes {max(1, nodes)}", "quit"]
    payload = "\n".join(cmds) + "\n"
    cp = subprocess.run([str(engine)], input=payload, text=True, capture_output=True, check=False)
    output = (cp.stdout or "") + "\n" + (cp.stderr or "")
    bench_lines = [ln for ln in output.splitlines() if "info string benchraw" in ln]
    if not bench_lines:
        raise RuntimeError("benchraw output not found")
    return parse_kv_line(bench_lines[-1])


def run_timed_probe(engine: pathlib.Path, uci_options: List[Tuple[str, str]], movetime_ms: int) -> Dict[str, str]:
    cmds = ["uci", "isready"]
    for name, value in uci_options:
        cmds.append(f"setoption name {name} value {value}")
    cmds += ["isready", "position startpos", f"go movetime {max(1, movetime_ms)}", "quit"]
    payload = "\n".join(cmds) + "\n"
    cp = subprocess.run([str(engine)], input=payload, text=True, capture_output=True, check=False)
    output = (cp.stdout or "") + "\n" + (cp.stderr or "")
    info_lines = [ln for ln in output.splitlines() if ln.startswith("info depth ")]
    for ln in reversed(info_lines):
        if " hStop=" in ln:
            out: Dict[str, str] = {}
            m1 = re.search(r"\bhStop=([0-9]+)", ln)
            m2 = re.search(r"\bhStopAvgGap=([0-9]+(?:\.[0-9]+)?)", ln)
            m3 = re.search(r"\bhStopMaxGap=([0-9]+)", ln)
            m4 = re.search(r"\bhStopMaxMsGap=([0-9]+)", ln)
            if m1:
                out["hard_stop_checks"] = m1.group(1)
            if m2:
                out["hard_stop_avg_nodes_gap"] = m2.group(1)
            if m3:
                out["hard_stop_max_nodes_gap"] = m3.group(1)
            if m4:
                out["hard_stop_max_ms_gap"] = m4.group(1)
            return out
    return {}


def run_fastchess(
  fastchess: pathlib.Path,
  engine: pathlib.Path,
  openings: pathlib.Path,
  out_pgn: pathlib.Path,
  st_seconds: float,
  tc: str,
  rounds: int,
  hash_mb: int,
  threads: int,
  nn_name: str,
  opp_name: str,
  nn_options: List[Tuple[str, str]],
) -> str:
    out_pgn.parent.mkdir(parents=True, exist_ok=True)
    if out_pgn.exists():
        out_pgn.unlink()

    engine_a = [
        "name=" + nn_name,
        "cmd=" + str(engine),
        f"option.Hash={hash_mb}",
        f"option.Threads={threads}",
    ] + [f"option.{k}={v}" for k, v in nn_options]
    engine_b = [
        "name=" + opp_name,
        "cmd=" + str(engine),
        f"option.Hash={hash_mb}",
        f"option.Threads={threads}",
        "option.UseLc0Eval=false",
    ]

    cmd = [
        str(fastchess),
        "-engine",
        *engine_a,
        "-engine",
        *engine_b,
        "-each",
        "proto=uci",
        "-rounds",
        str(rounds),
        "-repeat",
        "-openings",
        f"file={str(openings)}",
        "format=epd",
        "order=sequential",
        "-concurrency",
        "1",
        "-pgnout",
        f"file={str(out_pgn)}",
    ]
    if tc:
        cmd.insert(cmd.index("-rounds"), f"tc={tc}")
    else:
        cmd.insert(cmd.index("-rounds"), f"st={st_seconds:.3f}")

    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = (cp.stdout or "") + "\n" + (cp.stderr or "")
    if cp.returncode != 0 and "Finished match" not in output:
        tail = "\n".join(output.splitlines()[-60:])
        raise RuntimeError(f"fastchess failed ({cp.returncode}):\n{tail}")
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description="Run short-TC NN time-safety harness.")
    ap.add_argument("--engine", default="build-rel/Release/fatshashcorchess0.exe")
    ap.add_argument("--fastchess", default="")
    ap.add_argument("--openings", default="bench/fens.txt")
    ap.add_argument("--out-dir", default="sprt_tmp/time_safety")
    ap.add_argument("--st", type=float, default=2.0, help="seconds per move")
    ap.add_argument("--tc", default="", help="fastchess tc (e.g. 10+0.1); overrides --st if set")
    ap.add_argument("--games", type=int, default=10, help="must be even")
    ap.add_argument("--hash", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--bench-depth", type=int, default=8)
    ap.add_argument("--bench-nodes", type=int, default=256)
    ap.add_argument("--probe-movetime-ms", type=int, default=2000)
    args = ap.parse_args()

    if args.games <= 0 or args.games % 2 != 0:
        raise RuntimeError("--games must be a positive even number")

    engine = pathlib.Path(args.engine).resolve()
    if not engine.exists():
        raise FileNotFoundError(f"Engine binary not found: {engine}")
    openings = pathlib.Path(args.openings).resolve()
    if not openings.exists():
        raise FileNotFoundError(f"Openings file not found: {openings}")
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    openings_clean = sanitize_openings(openings, out_dir / "openings_clean.epd")
    pgn_path = out_dir / "nn_vs_hce_st.pgn"
    log_path = out_dir / "fastchess_output.log"
    fastchess = find_fastchess(args.fastchess or None)

    nn_name = "NN_ORT_SAFE"
    opp_name = "HCE"
    nn_options = [
        ("UseLc0Eval", "true"),
        ("MoveOverhead", "100"),
        ("Lc0Backend", "2"),
        ("Lc0ExecBackend", "4"),
        ("Lc0BatchMax", "4"),
        ("Lc0BatchWaitUs", "200"),
        ("Lc0EvalThreads", "1"),
        ("TimePollTargetUs", "1000"),
        ("TimePollEmergencyUs", "250"),
        ("TimePollMinNodes", "8"),
    ]

    fc_out = run_fastchess(
      fastchess=fastchess,
      engine=engine,
      openings=openings_clean,
      out_pgn=pgn_path,
      st_seconds=args.st,
      tc=args.tc.strip(),
      rounds=args.games // 2,
      hash_mb=args.hash,
      threads=args.threads,
      nn_name=nn_name,
      opp_name=opp_name,
      nn_options=nn_options,
    )
    log_path.write_text(fc_out, encoding="utf-8")

    pgn_metrics = parse_pgn_metrics(pgn_path, nn_name)
    hard_stop_metrics = run_timed_probe(engine, nn_options, movetime_ms=args.probe_movetime_ms)
    bench_metrics = run_benchraw(engine, nn_options, depth=args.bench_depth, nodes=args.bench_nodes)

    print(f"engine={engine}")
    print(f"fastchess={fastchess}")
    print(f"openings={openings_clean}")
    print(f"pgn={pgn_path}")
    print(f"log={log_path}")
    print(
      f"games={int(pgn_metrics['pgn_games'])} "
      f"timeouts_total={int(pgn_metrics['timeouts_total'])} "
      f"timeouts_nn={int(pgn_metrics['timeouts_nn'])} "
      f"timeouts_opp={int(pgn_metrics['timeouts_opp'])}"
    )
    print(
      f"move_samples={int(pgn_metrics['move_samples'])} "
      f"move_time_avg_ms={pgn_metrics['move_time_avg_ms']:.2f} "
      f"move_time_p95_ms={pgn_metrics['move_time_p95_ms']:.2f}"
    )
    print(
      "hard_stop_checks="
      + hard_stop_metrics.get("hard_stop_checks", "0")
      + " hard_stop_avg_nodes_gap="
      + hard_stop_metrics.get("hard_stop_avg_nodes_gap", "0")
      + " hard_stop_max_ms_gap="
      + hard_stop_metrics.get("hard_stop_max_ms_gap", "0")
    )
    print(
      "nn_eval_latency_avg_us="
      + bench_metrics.get("nn_eval_latency_avg_us", "0")
      + " nn_eval_latency_p95_est_us="
      + bench_metrics.get("nn_eval_latency_p95_est_us", "0")
      + " nn_eval_latency_max_us="
      + bench_metrics.get("nn_eval_latency_max_us", "0")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
