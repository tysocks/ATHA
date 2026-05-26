"""Single LOX pump map transient example."""

from __future__ import annotations

import csv
from pathlib import Path

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main() -> None:
    result = run_config_folder(CONFIG_PATH, progress=True).require_summary()
    metrics = _csv_metrics(result.csv)
    print("\nSingle LOX pump map transient")
    print(f"  Solver status    : {result.solver_status}")
    print(f"  Solver source    : {result.solver_source}")
    print(f"  Final mdot       : {metrics.get('final_mdot', float('nan')):.4f} kg/s")
    print(f"  Final pump dP    : {metrics.get('final_dp_mpa', float('nan')):.4f} MPa")
    print(f"  Max pump speed   : {metrics.get('max_rpm', float('nan')):.0f} rpm")
    print(f"  Final valve pos. : {metrics.get('final_valve_position', float('nan')):.4f}")
    print(f"  CSV              : {result.csv}")
    print(f"  Plot             : {result.plot}")
    if result.acceptance_report is not None:
        status = "PASS" if result.acceptance_passed else "FAIL"
        print(f"  Acceptance       : {result.acceptance_report} ({status})")
    if result.regression_report is not None:
        status = "PASS" if result.regression_passed else "FAIL"
        print(f"  Regression       : {result.regression_report} ({status})")


def _csv_metrics(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    speeds = [float(row["PUMP_RPM"]) for row in rows if row.get("PUMP_RPM")]
    tracking_rows = [row for row in rows if float(row.get("TARGET_MDOT") or 0.0) > 0.0]
    final = tracking_rows[-1] if tracking_rows else rows[-1]
    return {
        "final_mdot": float(final.get("MDOT", "nan")),
        "final_dp_mpa": float(final.get("PUMP_DP", "nan")),
        "max_rpm": max(speeds) if speeds else float("nan"),
        "final_valve_position": float(final.get("VALVE_POSITION", "nan")),
    }


if __name__ == "__main__":
    main()
