from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from atha.config import apply_path_overrides
from atha.runner.context import AnalysisContext
from atha.runner.solver_driver import SolverDriver


@dataclass
class GenericSweepSummary:
    config_path: Path
    solver_status: str
    mode: str
    cases: int
    csv: Path | None = None
    manifest: Path | None = None
    monte_carlo_file: Path | None = None
    sweep_plot: Path | None = None
    statistics: Path | None = None
    sensitivity: Path | None = None


def run_generic_sweep(context: AnalysisContext) -> GenericSweepSummary:
    analysis = context.analysis
    output = _run_output(analysis)
    output_dir = context.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_type = str(analysis.get("base_type", analysis.get("analysis_type", "profile")))
    cases = _build_cases(analysis, context.analysis_type)
    if not cases:
        raise ValueError("sweep/monte_carlo analysis did not define any perturbation cases")

    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    parameter_names: list[str] = sorted({path for case in cases for path in case["overrides"]})
    driver = SolverDriver(context.registry) if context.registry is not None else None
    if driver is None:
        raise ValueError("generic sweep requires an AnalysisRegistry on the AnalysisContext")

    for index, case in enumerate(cases):
        case_id = f"case_{index:04d}"
        overrides = {"analysis_config.analysis.type": base_type, **case["overrides"]}
        loaded = apply_path_overrides(context.loaded, overrides)
        spec = context.registry.get(base_type)
        plan = driver.build_execution_plan(loaded, base_type, spec.mode)
        child_context = AnalysisContext(
            loaded=loaded,
            config_path=context.config_path,
            output_dir=output_dir / "cases" / case_id,
            analysis_type=base_type,
            mode=spec.mode,
            execution_plan=plan,
            registry=context.registry,
        )
        summary = context.registry.run_context(child_context)
        metrics = _extract_metrics(summary, analysis.get("metrics", []))
        metric_names.update(metrics)
        row: dict[str, Any] = {
            "case_id": case_id,
            "sample_kind": case["kind"],
            "success": True,
            "analysis_type": base_type,
        }
        row.update({f"param.{name}": case["overrides"].get(name, "") for name in parameter_names})
        row.update({f"metric.{name}": value for name, value in metrics.items()})
        rows.append(row)

    csv_path = _write_cases_csv(output_dir / str(output["csv"]), rows)
    stats_path = _write_statistics(output_dir / str(output["statistics"]), rows, sorted(metric_names))
    sensitivity_path = _write_sensitivity(
        output_dir / str(output["sensitivity"]),
        rows,
        parameter_names,
        sorted(metric_names),
    )
    manifest_path = _write_manifest(
        output_dir / str(output["manifest"]),
        {
            "cases": csv_path,
            "statistics": stats_path,
            "sensitivity": sensitivity_path,
        },
        {
            "analysis": context.loaded.analysis_config.name,
            "analysis_type": context.analysis_type,
            "base_type": base_type,
            "case_count": len(rows),
        },
    )
    return GenericSweepSummary(
        config_path=context.config_path,
        solver_status=f"completed generic {context.analysis_type} with {len(rows)} cases",
        mode=context.mode,
        cases=len(rows),
        csv=csv_path,
        manifest=manifest_path,
        monte_carlo_file=csv_path if context.analysis_type == "monte_carlo" else None,
        sweep_plot=None,
        statistics=stats_path,
        sensitivity=sensitivity_path,
    )


def _build_cases(analysis: Mapping[str, Any], analysis_type: str) -> list[dict[str, Any]]:
    perturbations = analysis.get("perturbations", {})
    if not isinstance(perturbations, Mapping):
        raise ValueError("analysis.perturbations must be a mapping")
    cases: list[dict[str, Any]] = []
    sweep_items = perturbations.get("sweep", analysis.get("sweep", []))
    if isinstance(sweep_items, Mapping):
        sweep_items = sweep_items.get("parameters", [])
    sweep_specs = _normalize_parameter_specs(sweep_items)
    if sweep_specs:
        for combo in _cartesian_product(sweep_specs):
            cases.append({"kind": "sweep", "overrides": combo})

    mc_cfg = perturbations.get("monte_carlo", analysis.get("monte_carlo"))
    if isinstance(mc_cfg, Mapping):
        samples = int(mc_cfg.get("samples", 1))
        rng = np.random.default_rng(int(mc_cfg.get("seed", 1)))
        specs = _normalize_parameter_specs(mc_cfg.get("parameters", []))
        for _index in range(samples):
            overrides = {spec["path"]: _draw_distribution(spec, rng) for spec in specs}
            cases.append({"kind": "monte_carlo", "overrides": overrides})

    if not cases and analysis_type == "monte_carlo" and isinstance(analysis.get("monte_carlo"), Mapping):
        mc_cfg = analysis["monte_carlo"]
        rng = np.random.default_rng(int(mc_cfg.get("seed", 1)))
        specs = _normalize_parameter_specs(mc_cfg.get("parameters", []))
        for _index in range(int(mc_cfg.get("samples", 1))):
            cases.append({"kind": "monte_carlo", "overrides": {spec["path"]: _draw_distribution(spec, rng) for spec in specs}})
    return cases


def _normalize_parameter_specs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = raw.get("parameters", [])
    if not isinstance(raw, list):
        raise ValueError("perturbation parameters must be a list")
    specs = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"perturbation parameter {index} must be a mapping")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"perturbation parameter {index} requires a non-empty path")
        specs.append(dict(item))
    return specs


def _cartesian_product(specs: list[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    values_by_path = [(_path(spec), _values(spec)) for spec in specs]
    if not values_by_path:
        return []
    grids = np.meshgrid(*[np.arange(len(values)) for _path_name, values in values_by_path], indexing="ij")
    cases = []
    for indices in zip(*(grid.ravel() for grid in grids)):
        case = {}
        for (path, values), idx in zip(values_by_path, indices):
            case[path] = values[int(idx)]
        cases.append(case)
    return cases


def _path(spec: Mapping[str, Any]) -> str:
    return str(spec["path"])


def _values(spec: Mapping[str, Any]) -> list[Any]:
    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep parameter '{spec['path']}' values must be a non-empty list")
        return values
    if {"start", "stop", "points"} <= set(spec):
        return np.linspace(float(spec["start"]), float(spec["stop"]), int(spec["points"])).tolist()
    if "value" in spec:
        return [spec["value"]]
    raise ValueError(f"sweep parameter '{spec['path']}' requires values, value, or start/stop/points")


def _draw_distribution(spec: Mapping[str, Any], rng: np.random.Generator) -> Any:
    distribution = str(spec.get("distribution", "uniform")).lower()
    settings = spec.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError(f"distribution settings for '{spec['path']}' must be a mapping")
    if distribution == "uniform":
        return float(rng.uniform(float(settings.get("low", spec.get("low", 0.0))), float(settings.get("high", spec.get("high", 1.0)))))
    if distribution == "normal":
        return float(rng.normal(float(settings.get("mean", spec.get("mean", 0.0))), float(settings.get("std", spec.get("std", 1.0)))))
    if distribution == "lognormal":
        return float(rng.lognormal(float(settings.get("mean", spec.get("mean", 0.0))), float(settings.get("sigma", spec.get("sigma", 1.0)))))
    if distribution == "choice":
        choices = settings.get("values", spec.get("values"))
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"choice distribution for '{spec['path']}' requires values")
        return choices[int(rng.integers(0, len(choices)))]
    raise ValueError(f"unsupported distribution for '{spec['path']}': {distribution}")


def _extract_metrics(summary: Any, metric_specs: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rows = _read_summary_rows(summary)
    specs = metric_specs if isinstance(metric_specs, list) and metric_specs else []
    if specs:
        for spec in specs:
            if isinstance(spec, str):
                name = spec
                source = spec
                reducer = "final"
            elif isinstance(spec, Mapping):
                source = str(spec.get("source", spec.get("name", "")))
                name = str(spec.get("name", source))
                reducer = str(spec.get("reducer", "final"))
            else:
                raise ValueError("metrics entries must be strings or mappings")
            metrics[name] = _reduce_metric(rows, source, reducer)
        return metrics
    for key in sorted(rows[-1]) if rows else []:
        if key.lower() == "time":
            continue
        value = _to_float(rows[-1].get(key))
        if value is not None:
            metrics[key] = value
    if not metrics:
        residuals = _read_residual_payload(getattr(summary, "residuals_json", None))
        if residuals:
            values = [abs(float(value)) for value in residuals.values() if _to_float(value) is not None]
            metrics["max_abs_residual"] = max(values) if values else 0.0
    return metrics


def _read_summary_rows(summary: Any) -> list[dict[str, Any]]:
    csv_path = getattr(summary, "csv", None)
    if csv_path is None or not Path(csv_path).exists():
        return []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _reduce_metric(rows: list[dict[str, Any]], source: str, reducer: str) -> float:
    values = [_to_float(row.get(source)) for row in rows]
    numeric = np.asarray([value for value in values if value is not None], dtype=float)
    if numeric.size == 0:
        raise ValueError(f"metric source '{source}' was not found in child output")
    if reducer == "final":
        return float(numeric[-1])
    if reducer == "initial":
        return float(numeric[0])
    if reducer == "mean":
        return float(np.mean(numeric))
    if reducer == "min":
        return float(np.min(numeric))
    if reducer == "max":
        return float(np.max(numeric))
    raise ValueError(f"unsupported metric reducer for '{source}': {reducer}")


def _read_residual_payload(path: Any) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized = payload.get("normalized_residuals", payload.get("residuals", {})) if isinstance(payload, Mapping) else {}
    return {str(key): float(value) for key, value in normalized.items() if _to_float(value) is not None}


def _write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row})
    fixed = [name for name in ("case_id", "sample_kind", "success", "analysis_type") if name in headers]
    headers = fixed + [name for name in headers if name not in fixed]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_statistics(path: Path, rows: list[dict[str, Any]], metric_names: list[str]) -> Path:
    payload: dict[str, Any] = {"case_count": len(rows), "metrics": {}}
    for metric in metric_names:
        key = f"metric.{metric}"
        values = np.asarray([_to_float(row.get(key)) for row in rows if _to_float(row.get(key)) is not None], dtype=float)
        if values.size == 0:
            continue
        payload["metrics"][metric] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_sensitivity(path: Path, rows: list[dict[str, Any]], parameter_names: list[str], metric_names: list[str]) -> Path:
    records = []
    for parameter in parameter_names:
        x = np.asarray([_to_float(row.get(f"param.{parameter}")) for row in rows], dtype=object)
        if any(value is None for value in x):
            continue
        x_numeric = np.asarray(x, dtype=float)
        if np.allclose(x_numeric, x_numeric[0]):
            continue
        for metric in metric_names:
            y = np.asarray([_to_float(row.get(f"metric.{metric}")) for row in rows], dtype=object)
            if any(value is None for value in y):
                continue
            y_numeric = np.asarray(y, dtype=float)
            if np.allclose(y_numeric, y_numeric[0]):
                continue
            corr = float(np.corrcoef(x_numeric, y_numeric)[0, 1])
            records.append({"parameter": parameter, "metric": metric, "pearson_r": corr})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["parameter", "metric", "pearson_r"])
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_manifest(path: Path, artifacts: Mapping[str, Path | None], metadata: Mapping[str, Any]) -> Path:
    payload = {
        "format": "atha.sweep_manifest.v1",
        "artifacts": {name: str(value) for name, value in artifacts.items() if value is not None},
        "metadata": dict(metadata),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_output(analysis: Mapping[str, Any]) -> dict[str, str]:
    raw = analysis.get("output", {})
    output = dict(raw) if isinstance(raw, Mapping) else {}
    output.setdefault("csv", "sweep_cases.csv")
    output.setdefault("statistics", Path(str(output["csv"])).with_suffix(".statistics.json").name)
    output.setdefault("sensitivity", Path(str(output["csv"])).with_suffix(".sensitivity.csv").name)
    output.setdefault("manifest", Path(str(output["csv"])).with_suffix(".manifest.json").name)
    return {str(key): str(value) for key, value in output.items()}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
