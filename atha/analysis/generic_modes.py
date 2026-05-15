from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.analysis.linearization import PerturbationConfig, finite_difference_state_space, write_linearization_json
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources
from atha.runner.context import AnalysisContext
from atha.runner.dae_execution import DAEExecutionProblem, DAEExecutionResult, DAEPoint


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
    time: np.ndarray | None = None
    state_names: list[str] | None = None
    algebraic_names: list[str] | None = None
    residual_names: list[str] | None = None


def run_generic_steady(context: AnalysisContext) -> GenericDAESummary:
    problem = DAEExecutionProblem(context.loaded, context.execution_plan)
    point = problem.evaluate(context.execution_plan.time_start_s, problem.initial_state())
    output = _run_output(context.loaded.analysis_config.analysis, "steady_diagnostics.json")
    path = context.output_dir / str(output.get("diagnostics", output.get("residuals_json", "steady_diagnostics.json")))
    _write_json(path, _point_payload(point, problem))
    return GenericDAESummary(
        config_path=context.config_path,
        solver_status="solved generic steady port network",
        mode="steady",
        residuals_json=path,
        state_names=problem.state_names,
        algebraic_names=problem.algebraic_names,
        residual_names=problem.network_problem.residual_names,
    )


def run_generic_profile(context: AnalysisContext) -> GenericDAESummary:
    problem = DAEExecutionProblem(context.loaded, context.execution_plan)
    sample_times = _sample_times(context)
    result = problem.integrate(sample_times)
    output = _run_output(context.loaded.analysis_config.analysis, "profile.csv")
    residuals_json = context.output_dir / str(output.get("diagnostics", "profile_diagnostics.json"))
    _write_json(residuals_json, _result_payload(result))
    artifacts = None
    if context.loaded.telemetry is not None:
        validate_telemetry_sources(context.loaded.telemetry, context.source_catalog(_result_sources(result)))
        artifacts, _headers, _columns = OutputProcessor(
            output_dir=context.output_dir,
            telemetry_config=context.loaded.telemetry,
            run_output=output,
            metadata={"analysis": context.loaded.analysis_config.name, "analysis_type": context.analysis_type},
        ).write(
            _samples_from_result(result),
            residuals={name: float(values[-1]) for name, values in result.residual_history.items()} if result.residual_history else None,
            state_history={name: result.X[:, i] for i, name in enumerate(result.state_names)},
            algebraic_history={name: result.Z[:, i] for i, name in enumerate(result.algebraic_names)},
            residual_history=result.residual_history,
            boundary_history=result.boundary_history,
        )
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
        time=result.time,
        state_names=result.state_names,
        algebraic_names=result.algebraic_names,
        residual_names=result.residual_names,
    )


def run_generic_linearization(context: AnalysisContext) -> GenericDAESummary:
    problem = DAEExecutionProblem(context.loaded, context.execution_plan)
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
    values.update({key: float(value) for key, value in point.timings.items() if _is_number(value)})
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


def _result_payload(result: DAEExecutionResult) -> dict[str, Any]:
    return {
        "time": result.time.tolist(),
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
