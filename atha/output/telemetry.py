from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np


_UNIT_SCALE = {
    ("Pa", "MPa"): 1.0e-6,
    ("Pa", "bar"): 1.0e-5,
    ("N", "kN"): 1.0e-3,
}


def build_telemetry_rows(config, samples: list[Mapping[str, Any]]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Build telemetry arrays from sampled contexts.

    Each sample is a flat mapping of source names to values. Channel aliases
    from telemetry YAML become the exported column names.
    """

    if config is None:
        raise ValueError("telemetry config is required for telemetry export")
    headers: list[str] = []
    columns: dict[str, np.ndarray] = {}
    for channel in config.channels:
        alias = str(channel["alias"])
        source = str(channel["source"])
        scale = _scale_for_channel(source, channel.get("units"))
        headers.append(alias)
        columns[alias] = np.asarray([float(sample[source]) * scale for sample in samples], dtype=float)
    return headers, columns


def validate_telemetry_sources(config, available_sources: set[str]) -> None:
    """Validate telemetry channel sources against a source catalog."""

    if config is None:
        return
    missing = [str(channel["source"]) for channel in config.channels if str(channel["source"]) not in available_sources]
    if missing:
        raise ValueError(f"Telemetry source(s) are not available: {missing}")


def write_telemetry_csv(path: str | Path, headers: list[str], columns: Mapping[str, np.ndarray]) -> Path:
    out_path = Path(path)
    data = np.column_stack([columns[name] for name in headers])
    np.savetxt(out_path, data, delimiter=",", header=",".join(headers), comments="")
    return out_path


def write_telemetry_hdf5(
    path: str | Path,
    headers: list[str],
    columns: Mapping[str, np.ndarray],
    *,
    telemetry_config: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    state_history: Mapping[str, np.ndarray] | None = None,
    algebraic_history: Mapping[str, np.ndarray] | None = None,
    residual_history: Mapping[str, np.ndarray] | None = None,
    boundary_history: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write generic telemetry arrays and metadata to HDF5."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 telemetry export") from exc

    out_path = Path(path)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["format"] = "atha.telemetry.v1"
        for key, value in (metadata or {}).items():
            h5.attrs[str(key)] = _hdf5_attr(value)
        data_group = h5.create_group("telemetry")
        channel_units = _channel_units_by_alias(telemetry_config)
        channel_sources = _channel_sources_by_alias(telemetry_config)
        for name in headers:
            dataset = data_group.create_dataset(name, data=np.asarray(columns[name], dtype=float))
            if name in channel_units:
                dataset.attrs["units"] = channel_units[name]
            if name in channel_sources:
                dataset.attrs["source"] = channel_sources[name]
        _write_group(h5, "states", state_history)
        _write_group(h5, "algebraics", algebraic_history)
        _write_group(h5, "residuals", residual_history)
        _write_group(h5, "boundaries", boundary_history)
    return out_path


def write_output_manifest(path: str | Path, artifacts: Mapping[str, str | Path | None], metadata: Mapping[str, Any] | None = None) -> Path:
    """Write a small machine-readable manifest of generated run artifacts."""

    out_path = Path(path)
    payload = {
        "format": "atha.output_manifest.v1",
        "artifacts": {name: str(value) for name, value in artifacts.items() if value is not None},
        "metadata": dict(metadata or {}),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _scale_for_channel(source: str, units: Any) -> float:
    if units is None:
        return 1.0
    source_units = _default_source_units(source)
    return _UNIT_SCALE.get((source_units, str(units)), 1.0)


def _default_source_units(source: str) -> str | None:
    if source.endswith(".P") or source.endswith("_P") or source == "chamber.P":
        return "Pa"
    if source.endswith(".thrust") or source == "nozzle.thrust":
        return "N"
    return None


def _channel_units_by_alias(config: Any | None) -> dict[str, str]:
    if config is None:
        return {}
    return {
        str(channel["alias"]): str(channel["units"])
        for channel in config.channels
        if "alias" in channel and "units" in channel
    }


def _channel_sources_by_alias(config: Any | None) -> dict[str, str]:
    if config is None:
        return {}
    return {
        str(channel["alias"]): str(channel["source"])
        for channel in config.channels
        if "alias" in channel and "source" in channel
    }


def _hdf5_attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        return value
    return json.dumps(value, sort_keys=True)


def _write_group(h5: Any, name: str, values: Mapping[str, np.ndarray] | None) -> None:
    group = h5.create_group(name)
    if values is None:
        return
    for key, value in values.items():
        group.create_dataset(str(key), data=np.asarray(value, dtype=float))
