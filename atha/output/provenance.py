from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


RUN_PROVENANCE_SCHEMA = "atha.run_provenance.v1"


def build_run_provenance(
    *,
    config_path: str | Path,
    analysis_name: str,
    analysis_type: str,
    solver_options: Mapping[str, Any] | None = None,
    solve_policy: Mapping[str, Any] | None = None,
    residual_tolerance: float | None = None,
    residual_history: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    residuals: Mapping[str, float] | None = None,
    integration_segments: Sequence[Mapping[str, Any]] | None = None,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    acceptance_report: str | Path | None = None,
    acceptance_passed: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable run provenance payload for any ATHA analysis runner."""

    path = Path(config_path)
    largest_name, largest_value = _largest_residual(residual_history=residual_history, residuals=residuals)
    payload: dict[str, Any] = {
        "schema_version": RUN_PROVENANCE_SCHEMA,
        "git_commit": git_commit(path.parent if path.is_file() else path),
        "config_hash": config_hash(path),
        "config_path": str(path),
        "analysis_name": str(analysis_name),
        "analysis_type": str(analysis_type),
        "solver_options": dict(solver_options or {}),
        "solve_policy": dict(solve_policy or {}),
        "residual_tolerance": None if residual_tolerance is None else float(residual_tolerance),
        "largest_residual_over_time": {"name": largest_name, "value": float(largest_value)},
        "integration_segments": [dict(segment) for segment in integration_segments or ()],
        "time_start_s": None if time_start_s is None else float(time_start_s),
        "time_end_s": None if time_end_s is None else float(time_end_s),
    }
    if acceptance_report is not None:
        payload["acceptance_report"] = str(acceptance_report)
        payload["acceptance_passed"] = bool(acceptance_passed)
    if extra:
        payload.update(dict(extra))
    return payload


def config_hash(config_path: str | Path) -> str:
    root = Path(config_path)
    root = root.parent if root.is_file() else root
    suffixes = {".yaml", ".yml", ".json", ".csv"}
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(cwd: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def _largest_residual(
    *,
    residual_history: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    residuals: Mapping[str, float] | None = None,
) -> tuple[str, float]:
    largest_name = ""
    largest_value = 0.0
    for name, series in (residual_history or {}).items():
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        candidate = float(np.nanmax(np.abs(values)))
        if abs(candidate) >= abs(largest_value):
            largest_name = name
            largest_value = candidate
    for name, value in (residuals or {}).items():
        candidate = abs(float(value))
        if candidate >= abs(largest_value):
            largest_name = name
            largest_value = candidate
    return largest_name, largest_value
