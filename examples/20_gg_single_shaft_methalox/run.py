"""Methalox single-shaft gas-generator transient example."""

from __future__ import annotations

import csv
from pathlib import Path

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main():
    result = run_config_folder(CONFIG_PATH, progress=True).require_summary()
    metrics = _csv_metrics(result.csv)
    print("\nMethalox gas-generator single-shaft generic profile")
    print(f"  Solver status   : {result.solver_status}")
    print(f"  Solver source   : {result.solver_source}")
    print(f"  Final mdot      : {metrics.get('final_mdot', float('nan')):.3f} kg/s")
    print(f"  Peak thrust     : {metrics.get('peak_thrust', float('nan')):.1f} N")
    print(f"  Final thrust    : {metrics.get('final_thrust', float('nan')):.1f} N")
    print(f"  Shaft range     : {metrics.get('min_shaft_rpm', float('nan')):.0f}-{metrics.get('max_shaft_rpm', float('nan')):.0f} rpm")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")
    if result.acceptance_report is not None:
        print(f"  Acceptance      : {result.acceptance_report} ({'PASS' if result.acceptance_passed else 'FAIL'})")
    if result.regression_report is not None:
        print(f"  Regression      : {result.regression_report} ({'PASS' if result.regression_passed else 'FAIL'})")


def _csv_metrics(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    thrust = [float(row["THRUST"]) for row in rows if row.get("THRUST")]
    shaft = [float(row["SHAFT_RPM"]) for row in rows if row.get("SHAFT_RPM")]
    return {
        "final_mdot": float(rows[-1].get("MDOT_TOTAL", "nan")),
        "peak_thrust": max(thrust) if thrust else float("nan"),
        "final_thrust": float(rows[-1].get("THRUST", "nan")),
        "min_shaft_rpm": min(shaft) if shaft else float("nan"),
        "max_shaft_rpm": max(shaft) if shaft else float("nan"),
    }


if __name__ == "__main__":
    main()
