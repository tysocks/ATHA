from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from atha.assembly import EngineAssembler
from atha.config import load_analysis_config


@dataclass
class PortNetworkDiagnosticsSummary:
    config_path: Path
    component_count: int
    connection_count: int
    variable_count: int
    residual_count: int
    is_square: bool
    solve_attempted: bool
    solve_success: bool | None = None
    max_normalized_residual: tuple[str, float] | None = None
    residuals_json: Path | None = None


def run_port_network_diagnostics(
    config_path: str | Path,
    output_dir: str | Path = "outputs",
) -> PortNetworkDiagnosticsSummary:
    """Build and optionally solve the generic YAML port network.

    This analysis is a Phase-20 diagnostic entrypoint. It exercises automatic
    port unknown generation, connection residual assembly, boundary anchors,
    and component residual contracts without requiring a topology-specific
    Python runner.
    """

    path = Path(config_path)
    loaded = load_analysis_config(path)
    problem = EngineAssembler(loaded).port_network_problem()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    run = loaded.analysis_config.analysis
    solve_attempted = bool(run.get("solve", True))
    solve_success = None
    max_residual: tuple[str, float] | None = None
    residuals: dict[str, float] = {}
    normalized: dict[str, float] = {}
    if solve_attempted:
        solution = problem.solve(0.0, None, {})
        solve_success = solution.success
        max_residual = solution.max_normalized_residual
        residuals = solution.residuals
        normalized = solution.normalized_residuals

    output_name = str(run.get("diagnostics_output", f"{loaded.analysis_config.name}.port_network.json"))
    residuals_json = output_dir / output_name
    payload: dict[str, Any] = {
        "analysis": loaded.analysis_config.name,
        "engine": loaded.engine.name,
        "component_count": len(loaded.engine.components),
        "connection_count": len(loaded.engine.connections),
        "variable_count": problem.n_algebraic,
        "residual_count": len(problem.residual_names),
        "is_square": problem.n_algebraic == len(problem.residual_names),
        "variables": problem.variable_names,
        "residuals": problem.residual_names,
        "solve_attempted": solve_attempted,
        "solve_success": solve_success,
        "max_normalized_residual": list(max_residual) if max_residual is not None else None,
        "residual_values": residuals,
        "normalized_residuals": normalized,
    }
    residuals_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PortNetworkDiagnosticsSummary(
        config_path=path,
        component_count=len(loaded.engine.components),
        connection_count=len(loaded.engine.connections),
        variable_count=problem.n_algebraic,
        residual_count=len(problem.residual_names),
        is_square=problem.n_algebraic == len(problem.residual_names),
        solve_attempted=solve_attempted,
        solve_success=solve_success,
        max_normalized_residual=max_residual,
        residuals_json=residuals_json,
    )
