"""Optional slow full-engine verification gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atha.validation.verification_suite import run_verification_case, verification_cases

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_ffsc_canonical_mdot_tracking(tmp_path: Path) -> None:
    specs = {spec.id: spec for spec in verification_cases(include_slow=True)}
    spec = specs["ffsc_dae_acceptance"]
    result = run_verification_case(spec, output_dir=tmp_path)
    assert not result.errors, result.errors
    assert result.acceptance_report is not None
    payload = json.loads(result.acceptance_report.read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["final_mdot_tracking"]["passed"] is True
    assert checks["tail_mdot_rms_tracking"]["passed"] is True
