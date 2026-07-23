"""Subsystem verification suite gates (Workstream 6.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from atha.validation.verification_suite import run_verification_case, verification_cases

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.mark.parametrize(
    "case_id",
    [
        "valve_pipe_volume",
        "pump_pipe_valve",
        "pump_shaft_turbine",
        "injector_chamber_nozzle",
        "chamber_nozzle",
        "preburner_turbine",
        "regen_channel",
    ],
)
def test_subsystem_verification_case(case_id: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    specs = {spec.id: spec for spec in verification_cases(include_slow=False)}
    spec = specs[case_id]
    output_dir = tmp_path_factory.mktemp(case_id)
    result = run_verification_case(spec, output_dir=output_dir)
    assert not result.errors, result.errors
    assert result.acceptance_passed is True, f"acceptance failed for {case_id}"
    if result.reference_passed is not None:
        assert result.reference_passed is True, f"reference checks failed for {case_id}"


def test_verification_registry_has_subsystem_cases() -> None:
    subsystem_cases = verification_cases(level=2, include_slow=False)
    assert len(subsystem_cases) >= 7
