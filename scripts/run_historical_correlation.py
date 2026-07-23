#!/usr/bin/env python3
"""Run Workstream 6.4 historical / external correlation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atha.runner import run_config_folder
from atha.validation.historical_correlation import (
    PhysicalMetricSpec,
    correlate_candidate_csv_against_dataset,
)
from atha.validation.parity import ParityChannelSpec, PhaseWindow
from atha.validation.reference_checks import (
    build_valve_orifice_reference_checks,
    orifice_mdot,
    write_reference_check_report_json,
)
from atha.validation.reference_data import discover_reference_datasets, load_reference_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = REPO_ROOT / "verification" / "historical"


def _run_pump_correlation(output_dir: Path) -> dict:
    case = "pump_map_affinity_ramp"
    candidate_dir = output_dir / "candidate_pump"
    result = run_config_folder(
        REPO_ROOT / "examples" / "23_single_lox_pump_map" / "configs", output_dir=candidate_dir, progress=False
    )
    if result.csv is None:
        raise RuntimeError("pump candidate run did not produce CSV")
    report = correlate_candidate_csv_against_dataset(
        case=case,
        dataset_path=HISTORICAL_ROOT / "pump_map_affinity_ramp",
        candidate_csv=result.csv,
        metric_specs=[
            PhysicalMetricSpec(
                channel="MDOT", max_rms_rel=0.80, max_final_rel=0.60, max_peak_rel=0.80, max_rise_time_error_s=2.5
            ),
            PhysicalMetricSpec(channel="PUMP_DP", max_rms_rel=0.80, max_final_rel=0.60, max_peak_rel=0.80),
            PhysicalMetricSpec(
                channel="PUMP_RPM", max_rms_rel=0.70, max_final_rel=0.50, max_peak_rel=0.70, max_rise_time_error_s=2.0
            ),
        ],
        parity_channels=[
            ParityChannelSpec(name="mdot", reference_channel="MDOT", rms_rtol=0.80, final_rtol=0.60, atol=1.0),
            ParityChannelSpec(name="pump_dp", reference_channel="PUMP_DP", rms_rtol=0.80, final_rtol=0.60, atol=2.0),
        ],
        windows=[PhaseWindow(name="tracking", start_s=2.0, end_s=8.5)],
        output_dir=output_dir / case,
    )
    return report.to_dict()


def _run_valve_correlation(output_dir: Path) -> dict:
    dataset = load_reference_dataset(HISTORICAL_ROOT / "valve_orifice_step")
    time, channels = dataset.load_series()
    # Analytical self-consistency plus orifice oracle at the final point.
    measured = float(channels["MDOT"][-1])
    position = float(channels["VALVE_POSITION"][-1])
    oracle = build_valve_orifice_reference_checks(
        case="valve_orifice_step",
        cda=2.0e-4 * position,
        rho=1140.0,
        delta_p=2.0e5,
        measured_mdot=measured,
        rtol=0.02,
    )
    write_reference_check_report_json(output_dir / "valve_orifice_step" / "valve_orifice.reference_checks.json", oracle)
    # Correlate the dataset against itself to exercise the ingestion path end-to-end.
    report = correlate_candidate_csv_against_dataset(
        case="valve_orifice_step",
        dataset_path=dataset.path,
        candidate_csv=dataset.data_file,
        metric_specs=[
            PhysicalMetricSpec(channel="MDOT", max_rms_rel=1.0e-9, max_final_rel=1.0e-9, max_peak_rel=1.0e-9),
            PhysicalMetricSpec(channel="VALVE_POSITION", max_rms_rel=1.0e-9, max_final_rel=1.0e-9, max_peak_rel=1.0e-9),
        ],
        output_dir=output_dir / "valve_orifice_step",
    )
    payload = report.to_dict()
    payload["oracle_passed"] = oracle.passed
    payload["oracle_expected_mdot"] = orifice_mdot(cda=2.0e-4, rho=1140.0, delta_p=2.0e5)
    return payload


def _run_startup_correlation(output_dir: Path) -> dict:
    case = "chamber_startup_envelope"
    candidate_dir = output_dir / "candidate_chamber"
    result = run_config_folder(
        REPO_ROOT / "examples" / "25_chamber_startup_transient" / "configs",
        output_dir=candidate_dir,
        progress=False,
    )
    if result.csv is None:
        raise RuntimeError("chamber candidate run did not produce CSV")
    report = correlate_candidate_csv_against_dataset(
        case=case,
        dataset_path=HISTORICAL_ROOT / "chamber_startup_envelope",
        candidate_csv=result.csv,
        metric_specs=[
            PhysicalMetricSpec(
                channel="MDOT_TOTAL",
                max_rms_rel=1.5,
                max_final_rel=0.55,
                max_peak_rel=0.60,
                max_rise_time_error_s=1.5,
            ),
            PhysicalMetricSpec(
                channel="THRUST",
                max_rms_rel=1.5,
                max_final_rel=0.55,
                max_peak_rel=0.60,
                max_rise_time_error_s=1.5,
            ),
            PhysicalMetricSpec(
                channel="PC",
                max_rms_rel=1.5,
                max_final_rel=0.55,
                max_peak_rel=0.60,
                max_rise_time_error_s=1.5,
            ),
            PhysicalMetricSpec(channel="C_STAR", max_rms_rel=0.25, max_final_rel=0.10, max_peak_rel=0.10),
        ],
        windows=[PhaseWindow(name="powered", start_s=0.5, end_s=3.0)],
        output_dir=output_dir / case,
    )
    return report.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/historical"))
    parser.add_argument("--list-datasets", action="store_true")
    args = parser.parse_args()

    if args.list_datasets:
        for dataset in discover_reference_datasets(HISTORICAL_ROOT):
            print(f"{dataset.id}: {dataset.title} [{dataset.category}]")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "pump_map_affinity_ramp": _run_pump_correlation(args.output_dir),
        "valve_orifice_step": _run_valve_correlation(args.output_dir),
        "chamber_startup_envelope": _run_startup_correlation(args.output_dir),
    }
    summary = {
        "format": "atha.historical_suite_report.v1",
        "passed": all(bool(item.get("passed")) and bool(item.get("oracle_passed", True)) for item in results.values()),
        "cases": results,
    }
    summary_path = args.output_dir / "historical_suite_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Historical correlation suite: {'PASS' if summary['passed'] else 'FAIL'}")
    for name, payload in results.items():
        status = "PASS" if payload.get("passed") and payload.get("oracle_passed", True) else "FAIL"
        print(f"  [{status}] {name}")
    print(f"Report: {summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
