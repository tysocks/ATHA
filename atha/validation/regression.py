from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.output.comparison import load_time_series


@dataclass(frozen=True)
class MetricWindow:
    name: str
    channel: str
    reducer: str
    expected: float | None = None
    atol: float | None = None
    rtol: float | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class RegressionCheck:
    name: str
    channel: str
    reducer: str
    value: float
    passed: bool
    expected: float | None = None
    atol: float | None = None
    rtol: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    message: str = ""


@dataclass(frozen=True)
class RegressionReport:
    case: str
    passed: bool
    checks: list[RegressionCheck]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.regression_report.v1",
            "case": self.case,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
            "metadata": self.metadata,
        }


EXAMPLE_REGRESSION_WINDOWS: dict[str, list[MetricWindow]] = {
    "15_valve_volume_transient": [
        MetricWindow("final_valve_position", "VALVE_POSITION", "final", expected=1.0, atol=2.0e-3),
        MetricWindow("final_pressure_bar", "P_DOWNSTREAM", "final", expected=1.25875, atol=2.0e-2),
        MetricWindow("final_mdot_in", "MDOT_IN", "final", expected=3.5937e-3, rtol=0.08),
        MetricWindow("final_mdot_out", "MDOT_OUT", "final", expected=2.3162e-3, rtol=0.08),
    ],
    "16_two_valve_transient_chain": [
        MetricWindow("final_valve_a_position", "VALVE_A_POSITION", "final", expected=1.0, atol=3.0e-3),
        MetricWindow("final_valve_b_position", "VALVE_B_POSITION", "final", expected=0.9994, atol=3.0e-3),
        MetricWindow("final_pc_bar", "PC", "final", expected=13.1833, atol=0.25),
        MetricWindow("final_thrust", "THRUST", "final", expected=141.173, rtol=0.08),
    ],
    "17_tca_propellant_valve_transient": [
        MetricWindow("final_methane_valve", "METHANE_VALVE_POSITION", "final", expected=1.0, atol=3.0e-3),
        MetricWindow("final_lox_valve", "LOX_VALVE_POSITION", "final", expected=0.9994, atol=3.0e-3),
        MetricWindow("final_of", "OF", "final", expected=3.1583, atol=0.25),
        MetricWindow("final_thrust", "THRUST", "final", expected=150018.0, rtol=0.08),
    ],
    "18_tca_mdot_controller": [
        MetricWindow("final_target_mdot", "TARGET_MDOT_TOTAL", "final", expected=20.0, atol=1.0e-9),
        MetricWindow("final_mdot_total", "MDOT_TOTAL", "final", expected=21.0203, rtol=0.12),
        MetricWindow("max_mdot_total", "MDOT_TOTAL", "max", expected=30.3500, rtol=0.08),
        MetricWindow("final_lox_valve", "LOX_VALVE_POSITION", "final", expected=0.05287, rtol=0.25),
    ],
}


def build_regression_report(
    *,
    case: str,
    time: np.ndarray,
    channels: Mapping[str, np.ndarray],
    windows: list[MetricWindow],
    metadata: Mapping[str, Any] | None = None,
) -> RegressionReport:
    checks = [_evaluate_window(window, channels) for window in windows]
    return RegressionReport(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={
            "time_start_s": float(time[0]) if time.size else None,
            "time_end_s": float(time[-1]) if time.size else None,
            **dict(metadata or {}),
        },
    )


def build_regression_report_from_file(
    path: str | Path,
    *,
    case: str | None = None,
    windows: list[MetricWindow] | None = None,
    time_column: str = "TIME",
    metadata: Mapping[str, Any] | None = None,
) -> RegressionReport:
    time, channels = load_time_series(path, time_column=time_column)
    case_name = case or Path(path).stem
    metric_windows = windows or EXAMPLE_REGRESSION_WINDOWS.get(case_name)
    if not metric_windows:
        raise ValueError(f"no regression windows supplied for case {case_name!r}")
    return build_regression_report(
        case=case_name,
        time=time,
        channels=channels,
        windows=metric_windows,
        metadata={"source": str(path), **dict(metadata or {})},
    )


def write_regression_report_json(path: str | Path, report: RegressionReport) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _evaluate_window(window: MetricWindow, channels: Mapping[str, np.ndarray]) -> RegressionCheck:
    if window.channel not in channels:
        return RegressionCheck(
            name=window.name,
            channel=window.channel,
            reducer=window.reducer,
            value=float("nan"),
            passed=False,
            message=f"channel {window.channel!r} not found",
        )
    value = _reduce(np.asarray(channels[window.channel], dtype=float), window.reducer)
    passed, message = _window_passed(value, window)
    return RegressionCheck(
        name=window.name,
        channel=window.channel,
        reducer=window.reducer,
        value=value,
        passed=passed,
        expected=window.expected,
        atol=window.atol,
        rtol=window.rtol,
        minimum=window.minimum,
        maximum=window.maximum,
        message=message,
    )


def _reduce(values: np.ndarray, reducer: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if reducer == "final":
        return float(finite[-1])
    if reducer == "initial":
        return float(finite[0])
    if reducer == "min":
        return float(np.min(finite))
    if reducer == "max":
        return float(np.max(finite))
    if reducer == "mean":
        return float(np.mean(finite))
    if reducer == "rms":
        return float(np.sqrt(np.mean(finite**2)))
    raise ValueError(f"unsupported regression reducer: {reducer}")


def _window_passed(value: float, window: MetricWindow) -> tuple[bool, str]:
    if not np.isfinite(value):
        return False, "value is not finite"
    if window.minimum is not None and value < window.minimum:
        return False, f"{value:.6g} < minimum {window.minimum:.6g}"
    if window.maximum is not None and value > window.maximum:
        return False, f"{value:.6g} > maximum {window.maximum:.6g}"
    if window.expected is None:
        return True, "no expected value supplied"
    atol = float(window.atol or 0.0)
    rtol = float(window.rtol or 0.0)
    limit = max(atol, rtol * max(abs(float(window.expected)), 1.0))
    error = abs(value - float(window.expected))
    if error <= limit:
        return True, f"abs_error {error:.6g} <= {limit:.6g}"
    return False, f"abs_error {error:.6g} > {limit:.6g}"
