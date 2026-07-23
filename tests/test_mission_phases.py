"""Workstream 6.5 mission-phase advance guard tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from atha.config.mission_phases import (
    evaluate_advance_guard,
    parse_advance_guard,
    resolve_phase_name,
    resolve_phase_name_with_guards,
    update_forced_phase_ends,
)
from atha.runner.solver_driver import ExecutionPhase


def _phases() -> list[ExecutionPhase]:
    return [
        ExecutionPhase(name="prestart", start_s=0.0, end_s=1.0),
        ExecutionPhase(
            name="startup",
            start_s=1.0,
            end_s=3.0,
            advance_when={"path": "chamber.P", "op": ">=", "value": 4.0e6},
        ),
        ExecutionPhase(name="CLC", start_s=3.0, end_s=10.0),
    ]


def test_parse_and_evaluate_advance_guard() -> None:
    guard = parse_advance_guard({"path": "chamber.P", "op": ">=", "value": 4.0e6})
    assert guard is not None
    assert evaluate_advance_guard(guard, {"chamber.P": 4.1e6}) is True
    assert evaluate_advance_guard(guard, {"chamber.P": 3.9e6}) is False


def test_timed_resolve_unchanged_without_forced_ends() -> None:
    phases = _phases()
    assert resolve_phase_name(phases, 0.5) == "prestart"
    assert resolve_phase_name_with_guards(phases, 1.5) == "startup"
    assert resolve_phase_name_with_guards(phases, 4.0) == "CLC"


def test_early_advance_pulls_next_phase_forward() -> None:
    phases = _phases()
    forced = update_forced_phase_ends(phases, 1.6, {"chamber.P": 4.5e6}, {})
    assert forced["startup"] == pytest.approx(1.6)
    assert resolve_phase_name_with_guards(phases, 1.6, forced_end_times=forced) == "CLC"
    assert resolve_phase_name_with_guards(phases, 2.5, forced_end_times=forced) == "CLC"
    # Without forced ends, startup would still own t=2.5.
    assert resolve_phase_name(phases, 2.5) == "startup"


def test_update_forced_phase_ends_is_idempotent() -> None:
    phases = _phases()
    first = update_forced_phase_ends(phases, 1.2, {"chamber.P": 5.0e6}, {})
    second = update_forced_phase_ends(phases, 1.8, {"chamber.P": 5.0e6}, first)
    assert second == first


def test_mapping_phase_objects_supported() -> None:
    phases = [
        SimpleNamespace(name="a", start_s=0.0, end_s=1.0, advance_when=None),
        {"name": "b", "start_s": 1.0, "end_s": 2.0, "advance_when": {"path": "x", "op": ">", "value": 0.5}},
    ]
    forced = update_forced_phase_ends(phases, 1.1, {"x": 0.9}, {})
    assert forced["b"] == pytest.approx(1.1)
