"""Workstream 6.4 historical correlation unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atha.validation.historical_correlation import (
    PhysicalMetricSpec,
    build_correlation_report,
    compute_rise_time_s,
    compute_settling_time_s,
    correlate_candidate_csv_against_dataset,
)
from atha.validation.reference_data import discover_reference_datasets, load_reference_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = REPO_ROOT / "verification" / "historical"


def test_discover_historical_datasets() -> None:
    datasets = discover_reference_datasets(HISTORICAL)
    ids = {dataset.id for dataset in datasets}
    assert {"pump_map_affinity_ramp", "valve_orifice_step", "chamber_startup_envelope"} <= ids


def test_load_valve_dataset_and_self_correlate(tmp_path: Path) -> None:
    dataset = load_reference_dataset(HISTORICAL / "valve_orifice_step")
    time, channels = dataset.load_series()
    assert time.size > 10
    assert "MDOT" in channels
    report = correlate_candidate_csv_against_dataset(
        case="valve_self",
        dataset_path=dataset.path,
        candidate_csv=dataset.data_file,
        metric_specs=[PhysicalMetricSpec(channel="MDOT", max_rms_rel=1.0e-9, max_final_rel=1.0e-9, max_peak_rel=1.0e-9)],
        output_dir=tmp_path,
    )
    assert report.passed
    assert (tmp_path / "valve_self.correlation.json").exists()


def test_rise_and_settling_time_helpers() -> None:
    time = np.linspace(0.0, 1.0, 101)
    values = 1.0 - np.exp(-time / 0.2)
    rise = compute_rise_time_s(time, values)
    settle = compute_settling_time_s(time, values, band_fraction=0.05)
    assert rise is not None and rise > 0.0
    assert settle is not None and settle >= 0.0


def test_build_correlation_report_detects_bias() -> None:
    dataset = load_reference_dataset(HISTORICAL / "valve_orifice_step")
    time, channels = dataset.load_series()
    biased = {key: values * 1.5 for key, values in channels.items()}
    report = build_correlation_report(
        case="biased_valve",
        dataset=dataset,
        candidate_time=time,
        candidate_channels=biased,
        metric_specs=[PhysicalMetricSpec(channel="MDOT", max_rms_rel=0.05, max_final_rel=0.05, max_peak_rel=0.05)],
    )
    assert report.passed is False
