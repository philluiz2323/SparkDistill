"""CLI: run a checkpoint against the benchmark basket, emit scores.

    python -m eval.harness --checkpoint outputs/qwen3.5-4b-phase1 --out eval/results/candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.benchmarks import BENCHMARKS, run_benchmark


def run_harness(model_path: str, benchmarks: list[str], work_dir: Path, limit: int | None = None) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in benchmarks:
        benchmark = BENCHMARKS[key]
        scores[key] = run_benchmark(benchmark, model_path, work_dir, limit=limit)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local path or HF hub id of the checkpoint to score")
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        choices=sorted(BENCHMARKS),
        default=None,
        help="benchmark to run (repeatable). Default: the full basket",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("eval/results/_work"))
    parser.add_argument("--limit", type=int, default=None, help="cap examples per benchmark (cheap re-verification)")
    parser.add_argument("--out", type=Path, required=True, help="where to write the resulting scores json")
    args = parser.parse_args(argv)

    benchmarks = args.benchmarks or sorted(BENCHMARKS)
    scores = run_harness(args.checkpoint, benchmarks, args.work_dir, limit=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoint": args.checkpoint, "scores": scores}, indent=2))
    print(f"wrote scores for {args.checkpoint} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
