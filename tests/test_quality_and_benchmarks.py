"""Unit tests for Workstream 6.3 quality / performance helpers."""

from __future__ import annotations

from pathlib import Path

from atha.assembly import EngineAssembler
from atha.benchmarks import BENCHMARK_CASES
from atha.config import load_analysis_config
from atha.runner.dae_execution import _command_maps_equal


def test_source_catalog_is_cached() -> None:
    loaded = load_analysis_config(Path("examples/21_generic_port_subsystems/chamber_nozzle/analysis.yaml"))
    assembler = EngineAssembler(loaded)
    first = assembler.source_catalog()
    second = assembler.source_catalog()
    assert first is second
    assert "mdot.total" in first.sources


def test_command_maps_equal_tolerates_identical_numeric_maps() -> None:
    left = {"valve.command": 0.5, "controller.x.error": 1.0}
    right = {"valve.command": 0.5, "controller.x.error": 1.0}
    assert _command_maps_equal(left, right)
    assert not _command_maps_equal(left, {"valve.command": 0.51})


def test_benchmark_registry_has_three_sizes() -> None:
    sizes = {case.size for case in BENCHMARK_CASES}
    assert sizes == {"fast", "medium", "slow"}
