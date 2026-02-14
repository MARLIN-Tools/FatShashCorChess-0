#!/usr/bin/env python3
"""
Run a small profiling matrix for lc0 eval backends.

Example:
  python tools/lc0_profile_matrix.py --engine build-lc0/Release/fatshashcorchess0_eval_bench.exe --weights t1-256x10-distilled-swa-2432500.pb.gz
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def run_case(
    engine: Path,
    positions: int,
    evals: int,
    backend: str,
    weights: Path,
    batch: int,
    wait_us: int,
    threads: int,
    exec_backend: int,
    batch_policy: int,
    cache_policy: int,
    backend_strict: int,
) -> dict[str, str]:
    cmd = [
        str(engine),
        str(positions),
        str(evals),
        "--backend",
        backend,
        "--weights",
        str(weights),
        "--lc0-batch-max",
        str(batch),
        "--lc0-batch-wait-us",
        str(wait_us),
        "--lc0-eval-threads",
        str(threads),
        "--lc0-exec-backend",
        str(exec_backend),
        "--lc0-batch-policy",
        str(batch_policy),
        "--lc0-cache-policy",
        str(cache_policy),
        "--lc0-backend-strict",
        str(backend_strict),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stdout}\n{cp.stderr}")
    return parse_kv(cp.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="build-lc0/Release/fatshashcorchess0_eval_bench.exe")
    ap.add_argument("--weights", default="t1-256x10-distilled-swa-2432500.pb.gz")
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--evals", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-list", default="1,2,4,8,16,32")
    ap.add_argument("--wait-us", type=int, default=1000)
    ap.add_argument("--exec-backends", default="0,1,2", help="lc0 exec backend ids to test (0=scalar,1=int8,2=onednn)")
    ap.add_argument("--batch-policy", type=int, default=0, choices=[0, 1])
    ap.add_argument("--cache-policy", type=int, default=1, choices=[0, 1])
    ap.add_argument("--backend-strict", type=int, default=0, choices=[0, 1])
    ap.add_argument("--report-md", default="", help="optional markdown report output path")
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    weights = Path(args.weights).resolve()

    if not engine.exists():
        raise FileNotFoundError(f"engine not found: {engine}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    rows: list[dict[str, str]] = []
    print("backend,exec_backend,batch,threads,evals_per_sec,nn_avg_batch_size,nn_avg_infer_us,nn_avg_queue_wait_us,eval_cache_hit_rate_pct")

    # Baseline HCE row
    hce = run_case(
        engine, args.positions, args.evals, "hce", weights, 1, args.wait_us, 1, 0, args.batch_policy, args.cache_policy, args.backend_strict
    )
    print(f"hce,0,1,1,{hce.get('evals_per_sec','')},{hce.get('nn_avg_batch_size','')},{hce.get('nn_avg_infer_us','')},{hce.get('nn_avg_queue_wait_us','')},{hce.get('eval_cache_hit_rate_pct','')}")
    rows.append({
        "backend": "hce",
        "exec_backend": "0",
        "batch": "1",
        "threads": "1",
        "evals_per_sec": hce.get("evals_per_sec", ""),
        "nn_avg_batch_size": hce.get("nn_avg_batch_size", ""),
        "nn_avg_infer_us": hce.get("nn_avg_infer_us", ""),
        "nn_avg_queue_wait_us": hce.get("nn_avg_queue_wait_us", ""),
        "eval_cache_hit_rate_pct": hce.get("eval_cache_hit_rate_pct", ""),
    })

    exec_backends = [int(x.strip()) for x in args.exec_backends.split(",") if x.strip()]
    for exec_backend in exec_backends:
        for b in [int(x.strip()) for x in args.batch_list.split(",") if x.strip()]:
            sync = run_case(
                engine,
                args.positions,
                args.evals,
                "lc0_sync",
                weights,
                b,
                args.wait_us,
                1,
                exec_backend,
                args.batch_policy,
                args.cache_policy,
                args.backend_strict,
            )
            print(
                f"lc0_sync,{exec_backend},{b},1,{sync.get('evals_per_sec','')},{sync.get('nn_avg_batch_size','')},"
                f"{sync.get('nn_avg_infer_us','')},{sync.get('nn_avg_queue_wait_us','')},{sync.get('eval_cache_hit_rate_pct','')}"
            )
            rows.append({
                "backend": "lc0_sync",
                "exec_backend": str(exec_backend),
                "batch": str(b),
                "threads": "1",
                "evals_per_sec": sync.get("evals_per_sec", ""),
                "nn_avg_batch_size": sync.get("nn_avg_batch_size", ""),
                "nn_avg_infer_us": sync.get("nn_avg_infer_us", ""),
                "nn_avg_queue_wait_us": sync.get("nn_avg_queue_wait_us", ""),
                "eval_cache_hit_rate_pct": sync.get("eval_cache_hit_rate_pct", ""),
            })
            async_res = run_case(
                engine,
                args.positions,
                args.evals,
                "lc0_async",
                weights,
                b,
                args.wait_us,
                args.threads,
                exec_backend,
                args.batch_policy,
                args.cache_policy,
                args.backend_strict,
            )
            print(
                f"lc0_async,{exec_backend},{b},{args.threads},{async_res.get('evals_per_sec','')},{async_res.get('nn_avg_batch_size','')},"
                f"{async_res.get('nn_avg_infer_us','')},{async_res.get('nn_avg_queue_wait_us','')},{async_res.get('eval_cache_hit_rate_pct','')}"
            )
            rows.append({
                "backend": "lc0_async",
                "exec_backend": str(exec_backend),
                "batch": str(b),
                "threads": str(args.threads),
                "evals_per_sec": async_res.get("evals_per_sec", ""),
                "nn_avg_batch_size": async_res.get("nn_avg_batch_size", ""),
                "nn_avg_infer_us": async_res.get("nn_avg_infer_us", ""),
                "nn_avg_queue_wait_us": async_res.get("nn_avg_queue_wait_us", ""),
                "eval_cache_hit_rate_pct": async_res.get("eval_cache_hit_rate_pct", ""),
            })

    if args.report_md:
        report_path = Path(args.report_md).resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = (Path("reports") / f"nn_baseline_{stamp}.md").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NN Baseline Profile",
        "",
        f"- engine: `{engine}`",
        f"- weights: `{weights}`",
        f"- positions: {args.positions}",
        f"- evals_per_position: {args.evals}",
        f"- async_threads: {args.threads}",
        f"- batch_policy: {args.batch_policy}",
        f"- cache_policy: {args.cache_policy}",
        f"- backend_strict: {args.backend_strict}",
        "",
        "| backend | exec_backend | batch | threads | evals_per_sec | nn_avg_batch_size | nn_avg_infer_us | nn_avg_queue_wait_us | eval_cache_hit_rate_pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['exec_backend']} | {row['batch']} | {row['threads']} | "
            f"{row['evals_per_sec']} | {row['nn_avg_batch_size']} | {row['nn_avg_infer_us']} | "
            f"{row['nn_avg_queue_wait_us']} | {row['eval_cache_hit_rate_pct']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_md={report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
