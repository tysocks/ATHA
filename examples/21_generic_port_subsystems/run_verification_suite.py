#!/usr/bin/env python3
"""Run the Workstream 6.2 subsystem verification suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from atha.validation.verification_suite import (
    run_verification_suite,
    verification_cases,
    write_verification_suite_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    parser.add_argument("--include-slow", action="store_true", help="Include full-engine mission cases.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON suite report path.")
    args = parser.parse_args()

    cases = verification_cases(include_slow=args.include_slow)
    report = run_verification_suite(cases, output_dir=args.output_dir)
    report_path = args.report or args.output_dir / "verification_suite_report.json"
    write_verification_suite_report(report_path, report)
    print(f"Verification suite: {'PASS' if report.passed else 'FAIL'} ({len(report.results)} cases)")
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.spec.id} (level {result.spec.level})")
        for error in result.errors:
            print(f"         error: {error}")
    print(f"Report: {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
