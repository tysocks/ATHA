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


def load_time_series_csv(path: str | Path, *, time_column: str = "TIME") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a telemetry CSV into a time vector and numeric channel arrays."""

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = list(data.dtype.names or [])
    if time_column not in names:
        raise ValueError(f"time column {time_column!r} not found in CSV {path}")
    time = np.asarray(data[time_column], dtype=float)
    channels = {name: np.asarray(data[name], dtype=float) for name in names if name != time_column}
    return time, channels


def load_time_series_hdf5(path: str | Path, *, time_channel: str = "TIME") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load telemetry channels from an ATHA HDF5 telemetry file."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load HDF5 telemetry") from exc

    channels: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        if "telemetry" not in h5:
            raise ValueError(f"HDF5 file does not contain a telemetry group: {path}")
        group = h5["telemetry"]
        for name, dataset in group.items():
            channels[str(name)] = np.asarray(dataset, dtype=float)
    if time_channel not in channels:
        raise ValueError(f"time channel {time_channel!r} not found in HDF5 {path}")
    time = channels.pop(time_channel)
    return time, channels


def load_time_series(path: str | Path, *, time_column: str = "TIME") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load ATHA CSV or HDF5 telemetry by file suffix."""

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_time_series_csv(path, time_column=time_column)
    if suffix in {".h5", ".hdf5"}:
        return load_time_series_hdf5(path, time_channel=time_column)
    raise ValueError(f"Unsupported telemetry file type: {suffix}")


def compare_time_series_files(
    reference_path: str | Path,
    actual_path: str | Path,
    *,
    channels: list[str] | None = None,
    time_column: str = "TIME",
    settling_tolerance: float | Mapping[str, float] | None = None,
) -> list[ChannelComparison]:
    """Compare two ATHA telemetry files using common channels."""

    reference_time, reference = load_time_series(reference_path, time_column=time_column)
    actual_time, actual = load_time_series(actual_path, time_column=time_column)
    if channels is not None:
        selected = set(channels)
        reference = {name: values for name, values in reference.items() if name in selected}
        actual = {name: values for name, values in actual.items() if name in selected}
    return compare_time_series(
        reference_time,
        reference,
        actual_time,
        actual,
        settling_tolerance=settling_tolerance,
    )


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
