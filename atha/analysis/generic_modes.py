from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.analysis.linearization import PerturbationConfig, finite_difference_state_space, write_linearization_json
from atha.analysis.reduced_cycles import build_reduced_cycle_provider
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources
from atha.output.provenance import RUN_PROVENANCE_SCHEMA, build_run_provenance, config_hash, git_commit
from atha.runner.context import AnalysisContext
from atha.runner.dae_execution import DAEExecutionProblem, DAEExecutionResult, DAEPoint
from atha.validation.acceptance import build_generic_port_acceptance_report, write_acceptance_report_json
from atha.validation.regression import (
    build_regression_report_from_file,
    regression_windows_from_config,
    write_regression_report_json,
)


@dataclass
class GenericDAESummary:
    config_path: Path
    solver_status: str
    mode: str
    csv: Path | None = None
    plot: Path | None = None
    hdf5: Path | None = None
    manifest: Path | None = None
    residuals_csv: Path | None = None
    residuals_json: Path | None = None
    linearization: Path | None = None
    acceptance_report: Path | None = None
    acceptance_passed: bool | None = None
    regression_report: Path | None = None
    regression_passed: bool | None = None
    time: np.ndarray | None = None
    state_names: list[str] | None = None
    algebraic_names: list[str] | None = None
    residual_names: list[str] | None = None
    integration_segments: list[dict[str, Any]] | None = None


def run_generic_steady(context: AnalysisContext) -> GenericDAESummary:
    problem = DAEExecutionProblem(
        context.loaded,
        context.execution_plan,
        reduced_cycle_provider=build_reduced_cycle_provider(context.loaded),
        progress_callback=context.progress_callback,
    )
    point = problem.trim_initial_conditions()
    output = _run_output(context.loaded.analysis_config.analysis, "steady_diagnostics.json")
    path = context.output_dir / str(output.get("diagnostics", output.get("residuals_json", "steady_diagnostics.json")))
    payload = _point_payload(point, problem)
    payload["trim"] = _trim_payload(point, problem)
    payload["provenance"] = _run_provenance(context, problem, point=point)
    _write_json(path, payload)
    return GenericDAESummary(
        config_path=context.config_path,
        solver_status="solved generic steady port network",
        mode="steady",
        residuals_json=path,
        state_names=problem.state_names,
        algebraic_names=problem.algebraic_names,
        residual_names=problem.network_problem.residual_names,
        integration_segments=[{"start_s": context.execution_plan.time_start_s, "end_s": context.execution_plan.time_start_s, "reason": "steady"}],
    )


def run_generic_profile(context: AnalysisContext) -> GenericDAESummary:
    reduced_cycle_provider = build_reduced_cycle_provider(context.loaded)
    problem = DAEExecutionProblem(
        context.loaded,
        context.execution_plan,
        reduced_cycle_provider=reduced_cycle_provider,
        progress_callback=context.progress_callback,
    )
    trim_point = problem.trim_initial_conditions() if context.execution_plan.trim_enabled else None
    sample_times = _sample_times(context)
    result = problem.integrate(sample_times)
    integration_segments = [
        {"start_s": segment.start_s, "end_s": segment.end_s, "reason": segment.reason}
        for segment in result.segments
    ]
    output = _run_output(context.loaded.analysis_config.analysis, "profile.csv")
    residuals_json = context.output_dir / str(output.get("diagnostics", "profile_diagnostics.json"))
    acceptance_path, acceptance_passed = _write_acceptance_if_configured(context, result)
    result_payload = _result_payload(result)
    provenance = _run_provenance(
        context,
        problem,
        result=result,
        trim_point=trim_point,
        acceptance_path=acceptance_path,
        acceptance_passed=acceptance_passed,
    )
    result_payload["provenance"] = provenance
    if trim_point is not None:
        result_payload["trim"] = _trim_payload(trim_point, problem)
    if acceptance_path is not None:
        result_payload["acceptance_report"] = str(acceptance_path)
        result_payload["acceptance_passed"] = bool(acceptance_passed)
    _write_json(residuals_json, result_payload)
    artifacts = None
    if context.loaded.telemetry is not None:
        validate_telemetry_sources(context.loaded.telemetry, context.source_catalog(_result_sources(result)))
        artifacts, _headers, _columns = OutputProcessor(
            output_dir=context.output_dir,
            telemetry_config=context.loaded.telemetry,
            run_output=output,
            metadata={
                "analysis": context.loaded.analysis_config.name,
                "analysis_type": context.analysis_type,
                "integration_segments": integration_segments,
                "trim": _trim_payload(trim_point, problem) if trim_point is not None else None,
                "provenance": provenance,
            },
        ).write(
            _samples_from_result(result),
            residuals={name: float(values[-1]) for name, values in result.residual_history.items()} if result.residual_history else None,
            state_history={name: result.X[:, i] for i, name in enumerate(result.state_names)},
            algebraic_history={name: result.Z[:, i] for i, name in enumerate(result.algebraic_names)},
            residual_history=result.residual_history,
            boundary_history=result.boundary_history,
        )
    regression_path, regression_passed = _write_regression_if_configured(context, artifacts.csv if artifacts else None)
    return GenericDAESummary(
        config_path=context.config_path,
        solver_status="solved generic DAE profile",
        mode="profile",
        csv=artifacts.csv if artifacts else None,
        plot=artifacts.plot if artifacts else None,
        hdf5=artifacts.hdf5 if artifacts else None,
        manifest=artifacts.manifest if artifacts else None,
        residuals_csv=artifacts.residuals_csv if artifacts else None,
        residuals_json=artifacts.residuals_json if artifacts and artifacts.residuals_json else residuals_json,
        acceptance_report=acceptance_path,
        acceptance_passed=acceptance_passed,
        regression_report=regression_path,
        regression_passed=regression_passed,
        time=result.time,
        state_names=result.state_names,
        algebraic_names=result.algebraic_names,
        residual_names=result.residual_names,
        integration_segments=integration_segments,
    )


def run_generic_linearization(context: AnalysisContext) -> GenericDAESummary:
    problem = DAEExecutionProblem(
        context.loaded,
        context.execution_plan,
        reduced_cycle_provider=build_reduced_cycle_provider(context.loaded),
        progress_callback=context.progress_callback,
    )
    x0 = problem.initial_state()
    u0 = np.zeros(0, dtype=float)
    t0 = context.execution_plan.time_start_s
    linearization_cfg = context.analysis.get("linearization", {}) if isinstance(context.analysis.get("linearization", {}), dict) else {}
    output_labels = _linearization_outputs(linearization_cfg, problem, t0, x0)

    def dynamics(x: np.ndarray, _u: np.ndarray) -> np.ndarray:
        return problem.rhs(t0, x.copy())

    def outputs(x: np.ndarray, _u: np.ndarray) -> np.ndarray:
        point = problem.evaluate(t0, x.copy())
        values = _point_values(point)
        return np.asarray([float(values.get(label, 0.0)) for label in output_labels], dtype=float)

    linearization = finite_difference_state_space(
        dynamics,
        outputs,
        x0,
        u0,
        state_labels=problem.state_names,
        input_labels=[],
        output_labels=output_labels,
        perturbations=_perturbation_config(linearization_cfg.get("perturbations", {})),
    )
    output = _run_output(context.loaded.analysis_config.analysis, "linearization.json")
    path = write_linearization_json(
        context.output_dir / str(linearization_cfg.get("output", output.get("linearization", "linearization.json"))),
        linearization,
    )
    return GenericDAESummary(
        config_path=context.config_path,
        solver_status="linearized generic DAE problem",
        mode="linearization",
        linearization=path,
        state_names=problem.state_names,
        algebraic_names=problem.algebraic_names,
        residual_names=problem.network_problem.residual_names,
    )


def _sample_times(context: AnalysisContext) -> np.ndarray | None:
    if context.loaded.telemetry is None:
        return None
    return telemetry_times(context.loaded.telemetry, context.execution_plan.time_start_s, context.execution_plan.time_end_s)


def _run_output(analysis: Mapping[str, Any], default_csv: str) -> dict[str, Any]:
    raw = analysis.get("output", {})
    output = dict(raw) if isinstance(raw, Mapping) else {}
    output.setdefault("csv", default_csv)
    output.setdefault("plot", Path(str(output["csv"])).with_suffix(".png").name)
    output.setdefault("hdf5", Path(str(output["csv"])).with_suffix(".h5").name)
    output.setdefault("manifest", Path(str(output["csv"])).with_suffix(".manifest.json").name)
    return output


def _samples_from_result(result: DAEExecutionResult) -> list[dict[str, float]]:
    samples = []
    for point in result.points:
        sample = {"time": float(point.time)}
        sample.update(_point_values(point))
        samples.append(sample)
    return samples


def _point_values(point: DAEPoint) -> dict[str, float]:
    values: dict[str, float] = {}
    for mapping in (point.states, point.algebraics, point.commands, point.measurements):
        values.update({key: float(value) for key, value in mapping.items() if _is_number(value)})
    values.update({f"target.{key}": float(value) for key, value in point.targets.items() if _is_number(value)})
    values.update({f"targets.{key}": float(value) for key, value in point.targets.items() if _is_number(value)})
    values.update({f"boundaries.{key}": float(value) for key, value in point.boundaries.items() if _is_number(value)})
    values.update({key: float(value) for key, value in point.boundaries.items() if _is_number(value)})
    values.update({f"timings.{key}": float(value) for key, value in point.timings.items() if _is_number(value)})
    for key, value in point.timings.items():
        if _is_number(value):
            values.setdefault(key, float(value))
    values.update({f"residuals.{key}": float(value) for key, value in point.normalized_residuals.items()})
    return values


def _point_payload(point: DAEPoint, problem: DAEExecutionProblem) -> dict[str, Any]:
    return {
        "time": point.time,
        "state_names": problem.state_names,
        "algebraic_names": problem.algebraic_names,
        "residual_names": problem.network_problem.residual_names,
        "states": point.states,
        "algebraics": point.algebraics,
        "residuals": point.residuals,
        "normalized_residuals": point.normalized_residuals,
        "commands": point.commands,
        "targets": point.targets,
        "boundaries": point.boundaries,
        "measurements": point.measurements,
    }


def _trim_payload(point: DAEPoint | None, problem: DAEExecutionProblem) -> dict[str, Any] | None:
    if point is None:
        return None
    name, value = _largest_abs(point.normalized_residuals)
    return {
        "time": point.time,
        "balances": [
            {
                "name": balance.name,
                "variable": balance.variable,
                "value": point.algebraics.get(balance.variable),
                "residual": point.residuals.get(f"balances.{balance.name}.residual"),
                "normalized_residual": point.normalized_residuals.get(f"balances.{balance.name}.residual"),
                "expression": balance.residual.expression,
            }
            for balance in problem.balances
        ],
        "largest_normalized_residual": {"name": name, "value": value},
        "algebraics": point.algebraics,
    }


def _result_payload(result: DAEExecutionResult) -> dict[str, Any]:
    return {
        "time": result.time.tolist(),
        "integration_segments": [
            {"start_s": segment.start_s, "end_s": segment.end_s, "reason": segment.reason}
            for segment in result.segments
        ],
        "state_names": result.state_names,
        "algebraic_names": result.algebraic_names,
        "residual_names": result.residual_names,
        "states": {name: result.X[:, i].tolist() for i, name in enumerate(result.state_names)},
        "algebraics": {name: result.Z[:, i].tolist() for i, name in enumerate(result.algebraic_names)},
        "residual_history": {name: values.tolist() for name, values in result.residual_history.items()},
        "command_history": {name: values.tolist() for name, values in result.command_history.items()},
        "target_history": {name: values.tolist() for name, values in result.target_history.items()},
        "boundary_history": {name: values.tolist() for name, values in result.boundary_history.items()},
        "measurement_history": {name: values.tolist() for name, values in result.measurement_history.items()},
    }


def _run_provenance(
    context: AnalysisContext,
    problem: DAEExecutionProblem,
    *,
    result: DAEExecutionResult | None = None,
    point: DAEPoint | None = None,
    trim_point: DAEPoint | None = None,
    acceptance_path: Path | None = None,
    acceptance_passed: bool | None = None,
) -> dict[str, Any]:
    integration = context.execution_plan.integration
    segments = [
        {"start_s": float(segment.start_s), "end_s": float(segment.end_s), "reason": segment.reason}
        for segment in result.segments
    ] if result is not None else [
        {"start_s": float(context.execution_plan.time_start_s), "end_s": float(context.execution_plan.time_start_s), "reason": "steady"}
    ]
    payload = build_run_provenance(
        config_path=context.config_path,
        analysis_name=context.loaded.analysis_config.name,
        analysis_type=context.analysis_type,
        solver_options={
            "method": integration.method,
            "rtol": float(integration.rtol),
            "atol": float(integration.atol),
            "max_step": None if integration.max_step is None else float(integration.max_step),
            "segment_at_samples": bool(integration.segment_at_samples),
            "segment_at_controller_samples": bool(integration.segment_at_controller_samples),
        },
        solve_policy={
            "allow_non_square": bool(problem.solve_policy.allow_non_square),
            "checked": bool(problem.solve_policy.checked),
            "residual_tolerance": float(problem.solve_policy.residual_tolerance),
            "max_nfev": problem.solve_policy.max_nfev,
            "strict_sources": bool(problem.solve_policy.strict_sources),
            "corrector": problem.solve_policy.corrector,
            "corrector_iterations": int(problem.solve_policy.corrector_iterations),
            "preconditioner": problem.solve_policy.preconditioner,
        },
        residual_tolerance=float(problem.solve_policy.residual_tolerance),
        residual_history=result.residual_history if result is not None else None,
        residuals=point.normalized_residuals if point is not None else None,
        integration_segments=segments,
        time_start_s=float(context.execution_plan.time_start_s),
        time_end_s=float(context.execution_plan.time_end_s),
        acceptance_report=acceptance_path,
        acceptance_passed=acceptance_passed,
    )
    if trim_point is not None:
        payload["trim"] = _trim_payload(trim_point, problem)
    return payload


def _config_hash(config_path: Path) -> str:
    return config_hash(config_path)


def _git_commit(cwd: Path) -> str:
    return git_commit(cwd)


def _largest_residual_over_time(result: DAEExecutionResult) -> tuple[str, float]:
    largest_name = ""
    largest_value = 0.0
    for name, series in result.residual_history.items():
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        candidate = float(np.nanmax(np.abs(values)))
        if abs(candidate) >= abs(largest_value):
            largest_name = name
            largest_value = candidate
    return largest_name, largest_value


def _result_sources(result: DAEExecutionResult) -> set[str]:
    paths = {"time"}
    paths.update(result.state_names)
    paths.update(result.algebraic_names)
    paths.update(result.residual_names)
    paths.update(f"residuals.{name}" for name in result.residual_names)
    for history in (
        result.command_history,
        result.target_history,
        result.boundary_history,
        result.measurement_history,
    ):
        paths.update(history)
    return paths


def _write_acceptance_if_configured(context: AnalysisContext, result: DAEExecutionResult) -> tuple[Path | None, bool | None]:
    cfg = context.analysis.get("acceptance", {})
    if not isinstance(cfg, Mapping):
        return None, None
    report_name = cfg.get("report")
    if not report_name:
        return None, None
    values = _acceptance_values(result)
    residuals = {name: float(series[-1]) for name, series in result.residual_history.items() if series.size}
    shaft_paths = tuple(str(path) for path in cfg.get("shaft_paths", []) if isinstance(path, str)) if isinstance(cfg.get("shaft_paths", []), list) else ()
    report = build_generic_port_acceptance_report(
        case=str(cfg.get("case", context.loaded.analysis_config.name)),
        time=result.time,
        values=values,
        residuals=residuals,
        tolerances=cfg.get("tolerances", {}) if isinstance(cfg.get("tolerances", {}), Mapping) else {},
        shaft_paths=shaft_paths,
        required_paths=tuple(str(path) for path in cfg.get("required_paths", []) if isinstance(path, str)) if isinstance(cfg.get("required_paths", []), list) else (),
        evaluation_end_s=float(cfg["evaluation_end_s"]) if cfg.get("evaluation_end_s") is not None else None,
    )
    path = write_acceptance_report_json(context.output_dir / str(report_name), report)
    return path, report.passed


def _write_regression_if_configured(context: AnalysisContext, csv_path: Path | None) -> tuple[Path | None, bool | None]:
    if csv_path is None:
        return None, None
    cfg = context.analysis.get("regression", {})
    if not isinstance(cfg, Mapping):
        return None, None
    windows = regression_windows_from_config(cfg.get("windows"))
    if not windows:
        return None, None
    report = build_regression_report_from_file(
        csv_path,
        case=str(cfg.get("case", context.loaded.analysis_config.name)),
        windows=windows,
    )
    report_name = cfg.get("report", Path(str(csv_path.name)).with_suffix(".regression.json").name)
    path = write_regression_report_json(context.output_dir / str(report_name), report)
    return path, report.passed


def _acceptance_values(result: DAEExecutionResult) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for i, name in enumerate(result.state_names):
        values[name] = result.X[:, i]
    for i, name in enumerate(result.algebraic_names):
        values[name] = result.Z[:, i]
    for history in (
        result.command_history,
        result.boundary_history,
        result.measurement_history,
    ):
        values.update(history)
    timing_keys: set[str] = set()
    for point in result.points:
        timing_keys.update(key for key, value in point.timings.items() if _is_number(value))
    for key in sorted(timing_keys):
        values.setdefault(key, np.asarray([float(point.timings.get(key, np.nan)) for point in result.points], dtype=float))
    values.update(result.target_history)
    values.update({f"target.{key}": series for key, series in result.target_history.items()})
    values.update({f"targets.{key}": series for key, series in result.target_history.items()})
    return values


def _linearization_outputs(cfg: Mapping[str, Any], problem: DAEExecutionProblem, t0: float, x0: np.ndarray) -> list[str]:
    outputs = cfg.get("outputs")
    if isinstance(outputs, list) and outputs:
        return [str(item) for item in outputs]
    point = problem.evaluate(t0, x0)
    values = _point_values(point)
    labels = [name for name in [*problem.algebraic_names, *problem.state_names] if name in values]
    return labels or sorted(values)


def _perturbation_config(raw: object) -> PerturbationConfig:
    if not isinstance(raw, Mapping):
        raw = {}
    return PerturbationConfig(
        state_default=float(raw.get("state_default", 1.0e-6)),
        input_default=float(raw.get("input_default", 1.0e-6)),
        minimum_absolute=float(raw.get("minimum_absolute", 1.0e-9)),
        per_state={str(k): float(v) for k, v in raw.get("per_state", {}).items()} if isinstance(raw.get("per_state", {}), Mapping) else None,
        per_input={str(k): float(v) for k, v in raw.get("per_input", {}).items()} if isinstance(raw.get("per_input", {}), Mapping) else None,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating))


def _largest_abs(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        return "", 0.0
    name = max(values, key=lambda key: abs(float(values[key])))
    return name, float(values[name])
