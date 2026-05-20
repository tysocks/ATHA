from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    category: str
    passed: bool
    value: float
    limit: float
    units: str = ""
    message: str = ""


@dataclass(frozen=True)
class AcceptanceReport:
    case: str
    passed: bool
    checks: list[AcceptanceCheck]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.acceptance_report.v1",
            "case": self.case,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
            "metadata": self.metadata,
        }


def build_ffsc_reduced_acceptance_report(
    *,
    time: np.ndarray,
    mdot_total: np.ndarray,
    target_mdot_total: np.ndarray,
    of_ratio: np.ndarray,
    target_of: np.ndarray,
    thrust: np.ndarray,
    lox_shaft_rpm: np.ndarray,
    methane_shaft_rpm: np.ndarray,
    residuals: Mapping[str, float],
    linearization_path: Path | None,
    tolerances: Mapping[str, float] | None = None,
) -> AcceptanceReport:
    """Evaluate the reduced FFSC DAE acceptance gate.

    The checks are intentionally explicit and machine-readable. They are not a
    substitute for the future full port-variable validation, but they do catch
    the failure modes this reduced acceptance model is meant to guard against:
    numerical residual failures, missing controller response, dead shaft
    dynamics, target-tracking collapse, missing linearization, and loss of the
    150 kN / 40 kg/s endpoint.
    """

    tolerances = dict(tolerances or {})
    checks: list[AcceptanceCheck] = []
    final_mdot_target = float(target_mdot_total[-1])
    final_of_target = float(target_of[-1])
    final_thrust_target = float(tolerances.get("final_thrust_target", 150000.0))
    mdot_rel_limit = float(tolerances.get("final_mdot_rel", 0.08))
    thrust_rel_limit = float(tolerances.get("final_thrust_rel", 0.08))
    of_abs_limit = float(tolerances.get("final_of_abs", 0.45))
    rms_mdot_rel_limit = float(tolerances.get("mdot_tracking_rms_rel", 0.28))
    rms_of_abs_limit = float(tolerances.get("of_tracking_rms_abs", 1.25))
    min_shaft_delta = float(tolerances.get("min_shaft_speed_delta_rpm", 100.0))
    residual_limit = float(tolerances.get("max_normalized_residual", 1.0e-6))

    final_mdot_rel = abs(float(mdot_total[-1]) - final_mdot_target) / max(abs(final_mdot_target), 1.0e-12)
    final_thrust_rel = abs(float(thrust[-1]) - final_thrust_target) / max(abs(final_thrust_target), 1.0e-12)
    final_of_abs = abs(float(of_ratio[-1]) - final_of_target)
    tail = _tail_mask(time, seconds=float(tolerances.get("tracking_tail_s", 10.0)))
    mdot_rms_rel = _rms(mdot_total[tail] - target_mdot_total[tail]) / max(np.mean(np.abs(target_mdot_total[tail])), 1.0)
    of_rms_abs = _rms(of_ratio[tail] - target_of[tail])
    lox_delta = float(np.nanmax(lox_shaft_rpm) - np.nanmin(lox_shaft_rpm))
    methane_delta = float(np.nanmax(methane_shaft_rpm) - np.nanmin(methane_shaft_rpm))
    max_residual = max((abs(float(value)) for value in residuals.values()), default=0.0)

    checks.extend(
        [
            _check("final_mdot_tracking", "physical_model", final_mdot_rel, mdot_rel_limit, "rel"),
            _check("final_thrust_tracking", "physical_model", final_thrust_rel, thrust_rel_limit, "rel"),
            _check("final_of_tracking", "controller", final_of_abs, of_abs_limit, "OF"),
            _check("tail_mdot_rms_tracking", "controller", mdot_rms_rel, rms_mdot_rel_limit, "rel"),
            _check("tail_of_rms_tracking", "controller", of_rms_abs, rms_of_abs_limit, "OF"),
            _check("lox_shaft_response", "physical_model", lox_delta, min_shaft_delta, "rpm", greater_is_pass=True),
            _check("methane_shaft_response", "physical_model", methane_delta, min_shaft_delta, "rpm", greater_is_pass=True),
            _check("max_normalized_residual", "numerical", max_residual, residual_limit),
        ]
    )
    linearization_exists = linearization_path is not None and Path(linearization_path).exists()
    checks.append(
        AcceptanceCheck(
            name="linearization_artifact",
            category="linearization",
            passed=linearization_exists,
            value=1.0 if linearization_exists else 0.0,
            limit=1.0,
            message="linearization JSON exists" if linearization_exists else "linearization JSON was not written",
        )
    )
    finite_outputs = all(np.all(np.isfinite(values)) for values in (mdot_total, of_ratio, thrust, lox_shaft_rpm, methane_shaft_rpm))
    checks.append(
        AcceptanceCheck(
            name="finite_outputs",
            category="telemetry",
            passed=finite_outputs,
            value=1.0 if finite_outputs else 0.0,
            limit=1.0,
            message="all acceptance arrays are finite" if finite_outputs else "one or more acceptance arrays contain NaN/Inf",
        )
    )
    return AcceptanceReport(
        case="ffsc_reduced_dae_acceptance",
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={
            "final_mdot_target": final_mdot_target,
            "final_of_target": final_of_target,
            "final_thrust_target": final_thrust_target,
            "time_start_s": float(time[0]),
            "time_end_s": float(time[-1]),
        },
    )


def build_generic_port_acceptance_report(
    *,
    case: str,
    time: np.ndarray,
    values: Mapping[str, np.ndarray],
    residuals: Mapping[str, float],
    tolerances: Mapping[str, float] | None = None,
    shaft_paths: tuple[str, ...] = (),
    required_paths: tuple[str, ...] = (),
    evaluation_end_s: float | None = None,
) -> AcceptanceReport:
    """Evaluate a generic-port profile acceptance gate from telemetry arrays."""

    tolerances = dict(tolerances or {})
    checks: list[AcceptanceCheck] = []
    tracking_mask = _evaluation_mask(time, evaluation_end_s)
    mdot = _series(values, "mdot.total")
    target_mdot = _series(values, "target.mdot_total", fallback="targets.mdot_total")
    of_ratio = _series(values, "chamber.OF")
    target_of = _series(values, "target.OF", fallback="targets.OF")
    thrust = _series(values, "nozzle.thrust")
    residual_limit = float(tolerances.get("max_normalized_residual", 1.0e-6))
    max_residual = max((abs(float(value)) for value in residuals.values()), default=0.0)

    if mdot.size and target_mdot.size:
        mdot_track = mdot[tracking_mask]
        target_mdot_track = target_mdot[tracking_mask]
        time_track = time[tracking_mask]
        final_mdot_rel = abs(float(mdot_track[-1]) - float(target_mdot_track[-1])) / max(abs(float(target_mdot_track[-1])), 1.0e-12)
        checks.append(_check("final_mdot_tracking", "generic_port", final_mdot_rel, float(tolerances.get("final_mdot_rel", 0.2)), "rel"))
        tail = _tail_mask(time_track, seconds=float(tolerances.get("tracking_tail_s", 10.0)))
        checks.append(
            _check(
                "tail_mdot_rms_tracking",
                "generic_port",
                _rms(mdot_track[tail] - target_mdot_track[tail]) / max(np.mean(np.abs(target_mdot_track[tail])), 1.0),
                float(tolerances.get("mdot_tracking_rms_rel", 0.35)),
                "rel",
            )
        )
    if of_ratio.size and target_of.size:
        of_track = of_ratio[tracking_mask]
        target_of_track = target_of[tracking_mask]
        checks.append(
            _check(
                "final_of_tracking",
                "generic_port",
                abs(float(of_track[-1]) - float(target_of_track[-1])),
                float(tolerances.get("final_of_abs", 0.5)),
                "OF",
            )
        )
    if thrust.size:
        checks.append(
            _check(
                "powered_thrust",
                "generic_port",
                float(np.nanmax(thrust)),
                float(tolerances.get("min_peak_thrust", 1.0e5)),
                "N",
                greater_is_pass=True,
            )
        )
        if "shutdown_final_thrust_fraction" in tolerances:
            fraction = float(thrust[-1]) / max(float(np.nanmax(thrust)), 1.0e-12)
            checks.append(_check("shutdown_thrust_decay", "generic_port", fraction, float(tolerances["shutdown_final_thrust_fraction"]), "fraction"))
    for path in shaft_paths:
        series = _series(values, path)
        if series.size:
            checks.append(
                _check(
                    f"{path}_response",
                    "generic_port",
                    float(np.nanmax(series) - np.nanmin(series)),
                    float(tolerances.get("min_shaft_speed_delta_rpm", 100.0)),
                    "rpm",
                    greater_is_pass=True,
                )
            )
    for path in required_paths:
        series = _series(values, path)
        finite = bool(series.size and np.any(np.isfinite(series)))
        checks.append(
            AcceptanceCheck(
                name=f"{path}_available",
                category="generic_port",
                passed=finite,
                value=1.0 if finite else 0.0,
                limit=1.0,
                message=f"{path} finite history exists" if finite else f"{path} history missing or non-finite",
            )
        )
    checks.append(_check("max_normalized_residual", "numerical", max_residual, residual_limit))
    finite_outputs = all(np.any(np.isfinite(series)) for series in values.values())
    checks.append(
        AcceptanceCheck(
            name="finite_outputs",
            category="telemetry",
            passed=finite_outputs,
            value=1.0 if finite_outputs else 0.0,
            limit=1.0,
            message="all generic-port arrays are finite" if finite_outputs else "one or more generic-port arrays contain NaN/Inf",
        )
    )
    return AcceptanceReport(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={
            "time_start_s": float(time[0]) if time.size else 0.0,
            "time_end_s": float(time[-1]) if time.size else 0.0,
            "evaluation_end_s": None if evaluation_end_s is None else float(evaluation_end_s),
            "solver": "generic_port_profile",
        },
    )


def write_acceptance_report_json(path: str | Path, report: AcceptanceReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _check(
    name: str,
    category: str,
    value: float,
    limit: float,
    units: str = "",
    *,
    greater_is_pass: bool = False,
) -> AcceptanceCheck:
    passed = value >= limit if greater_is_pass else value <= limit
    op = ">=" if greater_is_pass else "<="
    return AcceptanceCheck(
        name=name,
        category=category,
        passed=bool(passed),
        value=float(value),
        limit=float(limit),
        units=units,
        message=f"{value:.6g} {op} {limit:.6g}",
    )


def _tail_mask(time: np.ndarray, *, seconds: float) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    start = max(float(time[-1]) - max(float(seconds), 0.0), float(time[0]))
    mask = time >= start
    if not np.any(mask):
        mask[-1] = True
    return mask


def _evaluation_mask(time: np.ndarray, evaluation_end_s: float | None) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if time.size == 0 or evaluation_end_s is None:
        return np.ones(time.shape, dtype=bool)
    mask = time <= float(evaluation_end_s) + 1.0e-12
    if not np.any(mask):
        mask[0] = True
    return mask


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def _series(values: Mapping[str, np.ndarray], key: str, *, fallback: str | None = None) -> np.ndarray:
    if key in values:
        return np.asarray(values[key], dtype=float)
    if fallback and fallback in values:
        return np.asarray(values[fallback], dtype=float)
    return np.zeros(0, dtype=float)
