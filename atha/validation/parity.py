from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atha.output.comparison import load_time_series


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class ParityChannelSpec:
    name: str
    reference_channel: str
    candidate_channel: str | None = None
    category: str = "plant"
    atol: float | None = None
    rtol: float | None = None
    rms_atol: float | None = None
    rms_rtol: float | None = None
    final_atol: float | None = None
    final_rtol: float | None = None
    windows: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class ParityCheck:
    name: str
    category: str
    reference_channel: str
    candidate_channel: str
    window: str
    metric: str
    value: float
    limit: float
    passed: bool
    sample_count: int
    message: str = ""


@dataclass(frozen=True)
class ParityReport:
    case: str
    passed: bool
    checks: list[ParityCheck]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.parity_report.v1",
            "case": self.case,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
            "metadata": self.metadata,
        }


def build_parity_report(
    *,
    case: str,
    reference_time: np.ndarray,
    reference_channels: Mapping[str, np.ndarray],
    candidate_time: np.ndarray,
    candidate_channels: Mapping[str, np.ndarray],
    channels: Sequence[ParityChannelSpec],
    windows: Sequence[PhaseWindow] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ParityReport:
    """Compare a candidate transient against an oracle on common sample times."""

    ref_time = np.asarray(reference_time, dtype=float)
    cand_time = np.asarray(candidate_time, dtype=float)
    if ref_time.size == 0 or cand_time.size == 0:
        raise ValueError("reference and candidate time arrays must be non-empty")
    phase_windows = tuple(windows or (PhaseWindow("all"),))
    checks: list[ParityCheck] = []
    for spec in channels:
        candidate_name = spec.candidate_channel or spec.reference_channel
        selected_windows = _selected_windows(spec, phase_windows)
        if spec.reference_channel not in reference_channels or candidate_name not in candidate_channels:
            checks.extend(_missing_checks(spec, candidate_name, selected_windows or (PhaseWindow("all"),)))
            continue
        ref_series = np.asarray(reference_channels[spec.reference_channel], dtype=float)
        cand_series = np.asarray(candidate_channels[candidate_name], dtype=float)
        if ref_series.size != ref_time.size:
            raise ValueError(f"reference channel {spec.reference_channel!r} length does not match reference time")
        if cand_series.size != cand_time.size:
            raise ValueError(f"candidate channel {candidate_name!r} length does not match candidate time")
        for window in selected_windows:
            window_time, ref_values, cand_values = _windowed_common_series(ref_time, ref_series, cand_time, cand_series, window)
            checks.extend(_evaluate_spec(spec, candidate_name, window, window_time, ref_values, cand_values))
    return ParityReport(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={
            "reference_time_start_s": float(ref_time[0]),
            "reference_time_end_s": float(ref_time[-1]),
            "candidate_time_start_s": float(cand_time[0]),
            "candidate_time_end_s": float(cand_time[-1]),
            "windows": [window.__dict__ for window in phase_windows],
            **dict(metadata or {}),
        },
    )


def build_parity_report_from_files(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    case: str,
    channels: Sequence[ParityChannelSpec],
    windows: Sequence[PhaseWindow] | None = None,
    time_column: str = "TIME",
    metadata: Mapping[str, Any] | None = None,
) -> ParityReport:
    ref_time, ref_channels = load_time_series(reference_path, time_column=time_column)
    cand_time, cand_channels = load_time_series(candidate_path, time_column=time_column)
    return build_parity_report(
        case=case,
        reference_time=ref_time,
        reference_channels=ref_channels,
        candidate_time=cand_time,
        candidate_channels=cand_channels,
        channels=channels,
        windows=windows,
        metadata={
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            **dict(metadata or {}),
        },
    )


def write_parity_report_json(path: str | Path, report: ParityReport) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_parity_delta_csv(
    path: str | Path,
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    channels: Sequence[ParityChannelSpec],
    windows: Sequence[PhaseWindow] | None = None,
    time_column: str = "TIME",
) -> Path:
    """Write reference, candidate, and error histories on the reference time base."""

    ref_time, ref_channels = load_time_series(reference_path, time_column=time_column)
    cand_time, cand_channels = load_time_series(candidate_path, time_column=time_column)
    phase_windows = tuple(windows or (PhaseWindow("all"),))
    rows_by_key: dict[tuple[str, float], dict[str, float | str]] = {}
    fieldnames = ["window", "time_s"]
    channel_fields: list[str] = []
    for spec in channels:
        candidate_name = spec.candidate_channel or spec.reference_channel
        ref_field = f"{spec.name}.reference"
        cand_field = f"{spec.name}.candidate"
        err_field = f"{spec.name}.error"
        channel_fields.extend([ref_field, cand_field, err_field])
        if spec.reference_channel not in ref_channels or candidate_name not in cand_channels:
            continue
        ref_series = np.asarray(ref_channels[spec.reference_channel], dtype=float)
        cand_series = np.asarray(cand_channels[candidate_name], dtype=float)
        for window in _selected_windows(spec, phase_windows):
            time, reference, candidate = _windowed_common_series(ref_time, ref_series, cand_time, cand_series, window)
            for index, t_value in enumerate(time):
                key = (window.name, float(t_value))
                row = rows_by_key.setdefault(key, {"window": window.name, "time_s": float(t_value)})
                row[ref_field] = float(reference[index])
                row[cand_field] = float(candidate[index])
                row[err_field] = float(candidate[index] - reference[index])
    fieldnames.extend(dict.fromkeys(channel_fields))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for _key, row in sorted(rows_by_key.items(), key=lambda item: (item[0][0], item[0][1])):
            writer.writerow(row)
    return out_path


def parity_channel_specs_from_config(raw: object) -> list[ParityChannelSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("parity channels must be a list of mappings")
    specs: list[ParityChannelSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"parity channel {index} must be a mapping")
        try:
            name = str(item["name"])
            reference = str(item.get("reference_channel", item["channel"]))
        except KeyError as exc:
            raise ValueError(f"parity channel {index} missing required key {exc.args[0]!r}") from exc
        windows = item.get("windows", ())
        if isinstance(windows, str):
            window_names = (windows,)
        elif isinstance(windows, list):
            window_names = tuple(str(value) for value in windows)
        else:
            window_names = ()
        specs.append(
            ParityChannelSpec(
                name=name,
                reference_channel=reference,
                candidate_channel=str(item["candidate_channel"]) if item.get("candidate_channel") is not None else None,
                category=str(item.get("category", "plant")),
                atol=_optional_float(item.get("atol")),
                rtol=_optional_float(item.get("rtol")),
                rms_atol=_optional_float(item.get("rms_atol")),
                rms_rtol=_optional_float(item.get("rms_rtol")),
                final_atol=_optional_float(item.get("final_atol")),
                final_rtol=_optional_float(item.get("final_rtol")),
                windows=window_names,
                required=bool(item.get("required", True)),
            )
        )
    return specs


def phase_windows_from_config(raw: object) -> list[PhaseWindow]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("phase windows must be a list of mappings")
    windows: list[PhaseWindow] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"phase window {index} must be a mapping")
        name = str(item.get("name", f"window_{index}"))
        start = item.get("start_s", item.get("start"))
        end = item.get("end_s", item.get("end"))
        windows.append(
            PhaseWindow(
                name=name,
                start_s=_optional_float(start),
                end_s=_optional_float(end),
            )
        )
    return windows


def _evaluate_spec(
    spec: ParityChannelSpec,
    candidate_channel: str,
    window: PhaseWindow,
    time: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> list[ParityCheck]:
    if time.size == 0:
        return [
            ParityCheck(
                name=spec.name,
                category=spec.category,
                reference_channel=spec.reference_channel,
                candidate_channel=candidate_channel,
                window=window.name,
                metric="sample_count",
                value=0.0,
                limit=1.0,
                passed=False,
                sample_count=0,
                message="no common samples in window",
            )
        ]
    error = candidate - reference
    scale = max(float(np.nanmax(np.abs(reference))), 1.0)
    checks = [
        _metric_check(spec, candidate_channel, window, "max_abs_error", float(np.nanmax(np.abs(error))), _limit(spec.atol, spec.rtol, scale), time.size),
        _metric_check(spec, candidate_channel, window, "rms_error", float(np.sqrt(np.nanmean(error**2))), _limit(spec.rms_atol, spec.rms_rtol, scale), time.size),
        _metric_check(spec, candidate_channel, window, "final_abs_error", abs(float(error[-1])), _limit(spec.final_atol, spec.final_rtol, max(abs(float(reference[-1])), 1.0)), time.size),
    ]
    return [check for check in checks if np.isfinite(check.limit)]


def _metric_check(
    spec: ParityChannelSpec,
    candidate_channel: str,
    window: PhaseWindow,
    metric: str,
    value: float,
    limit: float,
    sample_count: int,
) -> ParityCheck:
    passed = bool(value <= limit)
    return ParityCheck(
        name=spec.name,
        category=spec.category,
        reference_channel=spec.reference_channel,
        candidate_channel=candidate_channel,
        window=window.name,
        metric=metric,
        value=float(value),
        limit=float(limit),
        passed=passed,
        sample_count=int(sample_count),
        message=f"{value:.6g} <= {limit:.6g}" if passed else f"{value:.6g} > {limit:.6g}",
    )


def _missing_checks(
    spec: ParityChannelSpec,
    candidate_channel: str,
    windows: Sequence[PhaseWindow],
) -> list[ParityCheck]:
    if not spec.required:
        return []
    return [
        ParityCheck(
            name=spec.name,
            category=spec.category,
            reference_channel=spec.reference_channel,
            candidate_channel=candidate_channel,
            window=window.name,
            metric="available",
            value=0.0,
            limit=1.0,
            passed=False,
            sample_count=0,
            message="reference or candidate channel missing",
        )
        for window in windows
    ]


def _selected_windows(spec: ParityChannelSpec, phase_windows: Sequence[PhaseWindow]) -> tuple[PhaseWindow, ...]:
    if not spec.windows:
        return tuple(phase_windows)
    return tuple(window for window in phase_windows if window.name in spec.windows)


def _windowed_common_series(
    reference_time: np.ndarray,
    reference_values: np.ndarray,
    candidate_time: np.ndarray,
    candidate_values: np.ndarray,
    window: PhaseWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(float(reference_time[0]), float(candidate_time[0]))
    end = min(float(reference_time[-1]), float(candidate_time[-1]))
    if window.start_s is not None:
        start = max(start, float(window.start_s))
    if window.end_s is not None:
        end = min(end, float(window.end_s))
    mask = (reference_time >= start - 1.0e-12) & (reference_time <= end + 1.0e-12)
    time = reference_time[mask]
    if time.size == 0:
        return time, np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    reference = reference_values[mask]
    candidate = np.interp(time, candidate_time, candidate_values)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    return time[finite], reference[finite], candidate[finite]


def _limit(atol: float | None, rtol: float | None, scale: float) -> float:
    if atol is None and rtol is None:
        return float("nan")
    return max(float(atol or 0.0), float(rtol or 0.0) * max(abs(float(scale)), 1.0))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
