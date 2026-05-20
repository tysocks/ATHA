"""5 kN LOX/ethanol two-shaft gas-generator generic DAE example."""

from __future__ import annotations

import csv
from pathlib import Path

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main() -> None:
    result = run_config_folder(CONFIG_PATH, progress=True).require_summary()
    metrics = _csv_metrics(result.csv)

    print("\n5 kN LOX/ethanol two-shaft gas-generator generic profile")
    print(f"  Solver status    : {result.solver_status}")
    print(f"  Peak thrust      : {metrics.get('peak_thrust', float('nan')):.1f} N")
    print(f"  Final thrust     : {metrics.get('final_thrust', float('nan')):.1f} N")
    print(f"  Peak mdot        : {metrics.get('peak_mdot', float('nan')):.3f} kg/s")
    print(f"  Final mdot       : {metrics.get('final_mdot', float('nan')):.3f} kg/s")
    print(f"  Final OF         : {metrics.get('final_of', float('nan')):.3f}")
    print(
        "  Shaft ranges     : "
        f"LOX {metrics.get('min_lox_rpm', float('nan')):.0f}-{metrics.get('max_lox_rpm', float('nan')):.0f} rpm, "
        f"fuel {metrics.get('min_fuel_rpm', float('nan')):.0f}-{metrics.get('max_fuel_rpm', float('nan')):.0f} rpm"
    )
    print(f"  CSV              : {result.csv}")
    print(f"  HDF5             : {result.hdf5}")
    print(f"  Manifest         : {result.manifest}")
    print(f"  Plot             : {result.plot}")
    if result.acceptance_report is not None:
        print(f"  Acceptance       : {result.acceptance_report} ({'PASS' if result.acceptance_passed else 'FAIL'})")
    if result.regression_report is not None:
        print(f"  Regression       : {result.regression_report} ({'PASS' if result.regression_passed else 'FAIL'})")


def _csv_metrics(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    thrust = _series(rows, "THRUST")
    mdot = _series(rows, "MDOT_TOTAL")
    lox_rpm = _series(rows, "LOX_SHAFT_RPM")
    fuel_rpm = _series(rows, "FUEL_SHAFT_RPM")
    return {
        "peak_thrust": max(thrust) if thrust else float("nan"),
        "final_thrust": _final(rows, "THRUST"),
        "peak_mdot": max(mdot) if mdot else float("nan"),
        "final_mdot": _final(rows, "MDOT_TOTAL"),
        "final_of": _final(rows, "OF"),
        "min_lox_rpm": min(lox_rpm) if lox_rpm else float("nan"),
        "max_lox_rpm": max(lox_rpm) if lox_rpm else float("nan"),
        "min_fuel_rpm": min(fuel_rpm) if fuel_rpm else float("nan"),
        "max_fuel_rpm": max(fuel_rpm) if fuel_rpm else float("nan"),
    }


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key)]


def _final(rows: list[dict[str, str]], key: str) -> float:
    return float(rows[-1].get(key, "nan"))


if __name__ == "__main__":
    main()
