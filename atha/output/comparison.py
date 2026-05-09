from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChannelComparison:
    channel: str
    rmse: float
    max_error: float
    steady_state_bias: float
    overshoot: float
    settling_time_s: float | None = None


def compare_time_series(
    reference_time: np.ndarray,
    reference: Mapping[str, np.ndarray],
    actual_time: np.ndarray,
    actual: Mapping[str, np.ndarray],
    *,
    settling_tolerance: float | Mapping[str, float] | None = None,
) -> list[ChannelComparison]:
    """Compare actual channels against reference channels using interpolation."""

    reference_time = np.asarray(reference_time, dtype=float)
    actual_time = np.asarray(actual_time, dtype=float)
    results: list[ChannelComparison] = []
    for channel, reference_values in reference.items():
        if channel not in actual:
            continue
        ref = np.asarray(reference_values, dtype=float)
        act = np.interp(reference_time, actual_time, np.asarray(actual[channel], dtype=float))
        err = act - ref
        final_ref = float(ref[-1]) if ref.size else 0.0
        final_act = float(act[-1]) if act.size else 0.0
        peak_ref = float(np.max(ref)) if ref.size else 0.0
        peak_act = float(np.max(act)) if act.size else 0.0
        settling_time = _settling_time(reference_time, err, ref, channel, settling_tolerance)
        results.append(
            ChannelComparison(
                channel=str(channel),
                rmse=float(np.sqrt(np.mean(err**2))) if err.size else 0.0,
                max_error=float(np.max(np.abs(err))) if err.size else 0.0,
                steady_state_bias=final_act - final_ref,
                overshoot=peak_act - peak_ref,
                settling_time_s=settling_time,
            )
        )
    return results


def write_comparison_report_json(path: str | Path, comparisons: list[ChannelComparison]) -> Path:
    out_path = Path(path)
    payload = {
        "format": "atha.comparison_report.v1",
        "channels": [comparison.__dict__ for comparison in comparisons],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _settling_time(
    reference_time: np.ndarray,
    error: np.ndarray,
    reference_values: np.ndarray,
    channel: str,
    settling_tolerance: float | Mapping[str, float] | None,
) -> float | None:
    if settling_tolerance is None or error.size == 0:
        return None
    if isinstance(settling_tolerance, Mapping):
        if channel not in settling_tolerance:
            return None
        tolerance = float(settling_tolerance[channel])
    else:
        tolerance = float(settling_tolerance)
    final_ref = abs(float(reference_values[-1])) if reference_values.size else 1.0
    threshold = max(tolerance * max(final_ref, 1.0), tolerance)
    outside = np.flatnonzero(np.abs(error) > threshold)
    if outside.size == 0:
        return float(reference_time[0])
    index = int(outside[-1]) + 1
    if index >= reference_time.size:
        return None
    return float(reference_time[index])
