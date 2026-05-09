from __future__ import annotations

import argparse
from pathlib import Path

from atha.runner import run_config_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atha-run", description="Run an ATHA config folder or analysis YAML.")
    parser.add_argument("config", help="Config folder or analysis.yaml path")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    args = parser.parse_args(argv)

    result = run_config_folder(Path(args.config), output_dir=Path(args.output_dir))
    summary = result.require_summary()
    print(f"ATHA run complete: {result.name} ({result.analysis_type})")
    if result.csv is not None:
        print(f"CSV: {result.csv}")
    hdf5 = getattr(summary, "hdf5", None)
    if hdf5 is not None:
        print(f"HDF5: {hdf5}")
    manifest = getattr(summary, "manifest", None)
    if manifest is not None:
        print(f"Manifest: {manifest}")
    linearization = getattr(summary, "linearization", None)
    if linearization is not None:
        print(f"Linearization: {linearization}")
    acceptance_report = getattr(summary, "acceptance_report", None)
    if acceptance_report is not None:
        status = "PASS" if getattr(summary, "acceptance_passed", False) else "FAIL"
        print(f"Acceptance: {acceptance_report} ({status})")
    monte_carlo_file = getattr(summary, "monte_carlo_file", None)
    if monte_carlo_file is not None:
        print(f"Monte Carlo: {monte_carlo_file}")
    sweep_plot = getattr(summary, "sweep_plot", None)
    if sweep_plot is not None:
        print(f"Sweep plot: {sweep_plot}")
    plot = result.plot
    if plot is not None:
        print(f"Plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
