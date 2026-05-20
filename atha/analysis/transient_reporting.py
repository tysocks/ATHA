from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.output.provenance import build_run_provenance


def transient_integration_segments(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    time_cfg = run.get("time", {})
    if not isinstance(time_cfg, Mapping):
        return []
    phases = time_cfg.get("phases", [])
    if isinstance(phases, list) and phases:
        segments = []
        for phase in phases:
            if not isinstance(phase, Mapping):
                continue
            start = phase.get("start_s", phase.get("start"))
            end = phase.get("end_s", phase.get("end"))
            if start is None or end is None:
                continue
            segments.append(
                {
                    "start_s": float(start),
                    "end_s": float(end),
                    "reason": f"phase:{phase.get('name', 'unnamed')}",
                }
            )
        if segments:
            return segments
    if "start_s" in time_cfg and "end_s" in time_cfg:
        return [{"start_s": float(time_cfg["start_s"]), "end_s": float(time_cfg["end_s"]), "reason": "transient"}]
    return []


def transient_solver_options(loaded: Any) -> dict[str, Any]:
    solver_cfg = loaded.analysis_config.solver.get("dae", loaded.analysis_config.solver.get("transient", {}))
    return {
        "method": solver_cfg.get("method"),
        "rtol": _optional_float(solver_cfg.get("rtol")),
        "atol": _optional_float(solver_cfg.get("atol")),
        "max_step": _optional_float(solver_cfg.get("max_step")),
    }


def transient_run_provenance(
    *,
    loaded: Any,
    config_path: str | Path,
    run: Mapping[str, Any],
    residuals: Mapping[str, float] | None = None,
    residual_history: Mapping[str, np.ndarray] | None = None,
    acceptance_report: str | Path | None = None,
    acceptance_passed: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    time_cfg = run.get("time", {}) if isinstance(run.get("time", {}), Mapping) else {}
    solver_options = transient_solver_options(loaded)
    return build_run_provenance(
        config_path=config_path,
        analysis_name=loaded.analysis_config.name,
        analysis_type=str(run.get("type", "")),
        solver_options=solver_options,
        solve_policy={"runner": "transient", "checked": True},
        residual_tolerance=_optional_float(solver_options.get("atol")),
        residuals=residuals,
        residual_history=residual_history,
        integration_segments=transient_integration_segments(run),
        time_start_s=_optional_float(time_cfg.get("start_s")),
        time_end_s=_optional_float(time_cfg.get("end_s")),
        acceptance_report=acceptance_report,
        acceptance_passed=acceptance_passed,
        extra=extra,
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
