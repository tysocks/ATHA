#!/usr/bin/env python3
"""Run the ATHA Workstream 6.3 performance benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from atha.benchmarks import run_benchmark_suite, write_benchmark_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--include-slow", action="store_true", help="Include the full-engine FFSC case.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    results = run_benchmark_suite(output_dir=args.output_dir, include_slow=args.include_slow)
    report_path = args.report or args.output_dir / "benchmark_report.json"
    write_benchmark_report(report_path, results)
    print(f"Benchmark suite ({len(results)} cases)")
    for result in results:
        solves = "n/a" if result.algebraic_solve_count is None else str(result.algebraic_solve_count)
        skips = "n/a" if result.algebraic_solve_skip_count is None else str(result.algebraic_solve_skip_count)
        residual = "n/a" if result.max_abs_normalized_residual is None else f"{result.max_abs_normalized_residual:.3g}"
        print(
            f"  [{result.size}] {result.id}: {result.wall_time_s:.3f}s "
            f"solves={solves} skipped={skips} maxR={residual} "
            f"outputs={result.output_bytes} B"
        )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
