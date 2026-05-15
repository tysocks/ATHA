from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atha.runner import run_config_folder
from atha.validation.regression import EXAMPLE_REGRESSION_WINDOWS, build_regression_report_from_file


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    ("example", "csv_name"),
    [
        ("15_valve_volume_transient", "valve_volume_transient.csv"),
        ("16_two_valve_transient_chain", "two_valve_transient_chain.csv"),
        ("17_tca_propellant_valve_transient", "tca_propellant_valve_transient.csv"),
        ("18_tca_mdot_controller", "tca_mdot_controller.csv"),
    ],
)
def test_examples_15_to_18_regression_windows(tmp_path, example: str, csv_name: str):
    result = run_config_folder(Path("examples") / example / "configs", output_dir=tmp_path)
    report = build_regression_report_from_file(
        result.csv,
        case=example,
        windows=EXAMPLE_REGRESSION_WINDOWS[example],
    )

    assert result.csv.name == csv_name
    assert report.passed is True


@pytest.mark.integration
@pytest.mark.slow
def test_example_20_gg_single_shaft_shutdown_transient(tmp_path):
    result = run_config_folder(Path("examples") / "20_gg_single_shaft_methalox" / "configs", output_dir=tmp_path)
    summary = result.require_summary()

    assert result.analysis_type == "gg_single_shaft_transient"
    assert summary.csv.exists()
    assert np.nanmax(summary.thrust) > 1.0e5
    assert summary.thrust[-1] < 0.5 * np.nanmax(summary.thrust)
    assert summary.mdot_total[-1] < 0.5 * np.nanmax(summary.mdot_total)
