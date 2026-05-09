from __future__ import annotations

from pathlib import Path

from atha.analysis.pressure_fed import PressureFedTCASummary, run_pressure_fed_tca


TwoValveChainSummary = PressureFedTCASummary


def run_two_valve_chain(config_path: str | Path, output_dir: str | Path = "outputs") -> PressureFedTCASummary:
    return run_pressure_fed_tca(config_path, output_dir=output_dir)
