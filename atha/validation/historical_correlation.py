"""Historical / external correlation reports with physically meaningful metrics.

Extends the existing parity and comparison helpers with rise time, overshoot,
settling time, RMS error, peak error, final steady-state error, and impulse /
integrated mass-flow style metrics used for Workstream 6.4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atha.output.comparison import load_time_series
from atha.validation.parity import (
    ParityChannelSpec,
    PhaseWindow,
    build_parity_report,
    write_parity_delta_csv,
    write_parity_report_json,
)
from atha.validation.reference_data import ReferenceDataset, load_reference_dataset


@dataclass(frozen=True)
class PhysicalMetricSpec:
    channel: str
    reference_channel: str | None = None
    rise_time_fraction: float = 0.10  # 10% -> 90% band around final
    settling_band_fraction: float = 0.05
    max_rms_rel: float | None = 0.25
    max_final_rel: float | None = 0.15
    max_peak_rel: float | None = 0.35
    max_rise_time_error_s: float | None = None
    max_settling_time_error_s: float | None = None
    max_integrated_rel: float | None = None


@dataclass(frozen=True)
class PhysicalMetricCheck:
    name: str
    channel: str
    metric: str
    value: float
    limit: float
    passed: bool
    units: str = ""
    message: str = ""


@dataclass
class CorrelationReport:
    case: str
    dataset_id: str
    passed: bool
    checks: list[PhysicalMetricCheck]
    parity_passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.historical_correlation_report.v1",
            "case": self.case,
            "dataset_id": self.dataset_id,
            "passed": self.passed,
            "parity_passed": self.parity_passed,
            "checks": [check.__dict__ for check in self.checks],
            "metadata": self.metadata,
        }


def compute_rise_time_s(time: np.ndarray, values: np.ndarray, *, low: float = 0.1, high: float = 0.9) -> float | None:
    """10–90% rise time relative to the final value (from initial)."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.size < 2:
        return None
    start = float(values[0])
    final = float(values[-1])
    span = final - start
    if abs(span) < 1.0e-12:
        return 0.0
    low_thr = start + low * span
    high_thr = start + high * span
    if span >= 0.0:
        low_idx = np.where(values >= low_thr)[0]
        high_idx = np.where(values >= high_thr)[0]
    else:
        low_idx = np.where(values <= low_thr)[0]
        high_idx = np.where(values <= high_thr)[0]
    if low_idx.size == 0 or high_idx.size == 0:
        return None
    t_low = float(time[low_idx[0]])
    t_high = float(time[high_idx[0]])
    if t_high < t_low:
        return None
    return t_high - t_low


def compute_settling_time_s(time: np.ndarray, values: np.ndarray, *, band_fraction: float = 0.05) -> float | None:
    """First time after which the signal stays within ``band_fraction`` of final."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.size == 0:
        return None
    final = float(values[-1])
    band = max(abs(final) * float(band_fraction), 1.0e-12)
    outside = np.where(np.abs(values - final) > band)[0]
    if outside.size == 0:
        return 0.0
    last_outside = int(outside[-1])
    if last_outside >= time.size - 1:
        return None
    return float(time[last_outside + 1] - time[0])


def compute_overshoot(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    final = float(values[-1])
    peak = float(np.nanmax(values)) if final >= float(values[0]) else float(np.nanmin(values))
    return peak - final


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_integrated_error(time: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    err = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    if time.size < 2:
        return float(err[-1]) if err.size else 0.0
    return _trapz(err, time)


def build_correlation_report(
    *,
    case: str,
    dataset: ReferenceDataset,
    candidate_time: np.ndarray,
    candidate_channels: Mapping[str, np.ndarray],
    metric_specs: Sequence[PhysicalMetricSpec],
    parity_channels: Sequence[ParityChannelSpec] | None = None,
    windows: Sequence[PhaseWindow] | None = None,
    evaluation_window: PhaseWindow | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CorrelationReport:
    """Compare a candidate transient against a reference dataset with physical metrics."""

    ref_time, ref_channels = dataset.load_series()
    checks: list[PhysicalMetricCheck] = []
    for spec in metric_specs:
        ref_name = spec.reference_channel or spec.channel
        if ref_name not in ref_channels or spec.channel not in candidate_channels:
            checks.append(
                PhysicalMetricCheck(
                    name=f"{spec.channel}_available",
                    channel=spec.channel,
                    metric="available",
                    value=0.0,
                    limit=1.0,
                    passed=False,
                    message="missing reference or candidate channel",
                )
            )
            continue
        time, reference, candidate = _aligned_series(
            ref_time,
            ref_channels[ref_name],
            np.asarray(candidate_time, dtype=float),
            np.asarray(candidate_channels[spec.channel], dtype=float),
            window=evaluation_window,
        )
        if time.size == 0:
            checks.append(
                PhysicalMetricCheck(
                    name=f"{spec.channel}_samples",
                    channel=spec.channel,
                    metric="sample_count",
                    value=0.0,
                    limit=1.0,
                    passed=False,
                    message="no overlapping samples in evaluation window",
                )
            )
            continue
        checks.extend(_metric_checks_for_channel(spec, time, reference, candidate))

    parity_passed = None
    if parity_channels:
        parity = build_parity_report(
            case=case,
            reference_time=ref_time,
            reference_channels=ref_channels,
            candidate_time=np.asarray(candidate_time, dtype=float),
            candidate_channels=candidate_channels,
            channels=parity_channels,
            windows=windows,
            metadata={"dataset_id": dataset.id},
        )
        parity_passed = parity.passed
        for item in parity.checks:
            checks.append(
                PhysicalMetricCheck(
                    name=f"parity.{item.name}.{item.metric}",
                    channel=item.candidate_channel,
                    metric=item.metric,
                    value=item.value,
                    limit=item.limit,
                    passed=item.passed,
                    message=item.message,
                )
            )

    return CorrelationReport(
        case=case,
        dataset_id=dataset.id,
        passed=all(check.passed for check in checks),
        checks=checks,
        parity_passed=parity_passed,
        metadata={
            "dataset_path": str(dataset.path),
            "source": dataset.source,
            "provenance": dataset.provenance,
            "allowed_use": dataset.allowed_use,
            "notes": list(dataset.notes),
            **dict(metadata or {}),
        },
    )


def correlate_candidate_csv_against_dataset(
    *,
    case: str,
    dataset_path: str | Path,
    candidate_csv: str | Path,
    metric_specs: Sequence[PhysicalMetricSpec],
    parity_channels: Sequence[ParityChannelSpec] | None = None,
    windows: Sequence[PhaseWindow] | None = None,
    evaluation_window: PhaseWindow | None = None,
    time_column: str = "TIME",
    output_dir: str | Path | None = None,
) -> CorrelationReport:
    """High-level helper: load dataset + candidate CSV and emit optional artifacts."""

    dataset = load_reference_dataset(dataset_path)
    cand_time, cand_channels = load_time_series(candidate_csv, time_column=time_column)
    active_window = evaluation_window
    if active_window is None and windows:
        active_window = windows[0]
    report = build_correlation_report(
        case=case,
        dataset=dataset,
        candidate_time=cand_time,
        candidate_channels=cand_channels,
        metric_specs=metric_specs,
        parity_channels=parity_channels,
        windows=windows,
        evaluation_window=active_window,
        metadata={"candidate_csv": str(candidate_csv)},
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_correlation_report_json(out / f"{case}.correlation.json", report)
        if parity_channels:
            ref_time, ref_channels = dataset.load_series()
            # Write a temporary reference CSV for delta overlay reuse.
            temp_ref = out / f"{dataset.id}.aligned_reference.csv"
            _write_series_csv(temp_ref, ref_time, ref_channels)
            write_parity_delta_csv(
                out / f"{case}.parity_delta.csv",
                temp_ref,
                candidate_csv,
                channels=parity_channels,
                windows=windows,
                time_column=time_column,
            )
            parity = build_parity_report(
                case=case,
                reference_time=ref_time,
                reference_channels=ref_channels,
                candidate_time=cand_time,
                candidate_channels=cand_channels,
                channels=parity_channels,
                windows=windows,
                metadata={"dataset_id": dataset.id},
            )
            write_parity_report_json(out / f"{case}.parity.json", parity)
    return report


def write_correlation_report_json(path: str | Path, report: CorrelationReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _metric_checks_for_channel(
    spec: PhysicalMetricSpec,
    time: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> list[PhysicalMetricCheck]:
    checks: list[PhysicalMetricCheck] = []
    err = candidate - reference
    ref_scale = max(float(np.nanmean(np.abs(reference))), 1.0e-12)
    if spec.max_rms_rel is not None:
        rms_rel = float(np.sqrt(np.nanmean(err**2))) / ref_scale
        checks.append(_check(spec.channel, "rms_rel", rms_rel, spec.max_rms_rel, "rel"))
    if spec.max_final_rel is not None:
        denom = max(abs(float(reference[-1])), 1.0e-6 * ref_scale, 1.0e-12)
        final_rel = abs(float(err[-1])) / denom
        checks.append(_check(spec.channel, "final_rel", final_rel, spec.max_final_rel, "rel"))
    if spec.max_peak_rel is not None:
        peak_rel = abs(float(np.nanmax(candidate)) - float(np.nanmax(reference))) / max(
            abs(float(np.nanmax(reference))), 1.0e-12
        )
        checks.append(_check(spec.channel, "peak_rel", peak_rel, spec.max_peak_rel, "rel"))
    rise_ref = compute_rise_time_s(time, reference)
    rise_cand = compute_rise_time_s(time, candidate)
    if spec.max_rise_time_error_s is not None and rise_ref is not None and rise_cand is not None:
        checks.append(
            _check(spec.channel, "rise_time_error_s", abs(rise_cand - rise_ref), spec.max_rise_time_error_s, "s")
        )
    settle_ref = compute_settling_time_s(time, reference, band_fraction=spec.settling_band_fraction)
    settle_cand = compute_settling_time_s(time, candidate, band_fraction=spec.settling_band_fraction)
    if spec.max_settling_time_error_s is not None and settle_ref is not None and settle_cand is not None:
        checks.append(
            _check(
                spec.channel,
                "settling_time_error_s",
                abs(settle_cand - settle_ref),
                spec.max_settling_time_error_s,
                "s",
            )
        )
    if spec.max_integrated_rel is not None:
        integrated = abs(compute_integrated_error(time, reference, candidate))
        ref_impulse = abs(_trapz(reference, time)) if time.size > 1 else abs(float(reference[-1]))
        integrated_rel = integrated / max(ref_impulse, 1.0e-12)
        checks.append(_check(spec.channel, "integrated_rel", integrated_rel, spec.max_integrated_rel, "rel"))
    # Always emit overshoot diagnostics (informational pass using generous limit when unspecified).
    overshoot = compute_overshoot(candidate)
    checks.append(
        PhysicalMetricCheck(
            name=f"{spec.channel}_overshoot",
            channel=spec.channel,
            metric="overshoot",
            value=float(overshoot),
            limit=float("inf"),
            passed=True,
            units="same_as_channel",
            message=f"overshoot={overshoot:.6g} (diagnostic)",
        )
    )
    return checks


def _check(channel: str, metric: str, value: float, limit: float, units: str) -> PhysicalMetricCheck:
    passed = bool(value <= limit)
    return PhysicalMetricCheck(
        name=f"{channel}_{metric}",
        channel=channel,
        metric=metric,
        value=float(value),
        limit=float(limit),
        passed=passed,
        units=units,
        message=f"{value:.6g} <= {limit:.6g}" if passed else f"{value:.6g} > {limit:.6g}",
    )


def _aligned_series(
    ref_time: np.ndarray,
    reference: np.ndarray,
    cand_time: np.ndarray,
    candidate: np.ndarray,
    *,
    window: PhaseWindow | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(float(ref_time[0]), float(cand_time[0]))
    end = min(float(ref_time[-1]), float(cand_time[-1]))
    if window is not None:
        if window.start_s is not None:
            start = max(start, float(window.start_s))
        if window.end_s is not None:
            end = min(end, float(window.end_s))
    mask = (ref_time >= start - 1.0e-12) & (ref_time <= end + 1.0e-12)
    time = ref_time[mask]
    ref = reference[mask]
    cand = np.interp(time, cand_time, candidate)
    finite = np.isfinite(ref) & np.isfinite(cand)
    return time[finite], ref[finite], cand[finite]


def _write_series_csv(path: Path, time: np.ndarray, channels: Mapping[str, np.ndarray]) -> Path:
    headers = ["TIME", *channels.keys()]
    data = np.column_stack([time, *[np.asarray(channels[name], dtype=float) for name in channels]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="")
    return path
