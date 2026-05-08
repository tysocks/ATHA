from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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


def write_telemetry_csv(path: str | Path, headers: list[str], columns: Mapping[str, np.ndarray]) -> Path:
    out_path = Path(path)
    data = np.column_stack([columns[name] for name in headers])
    np.savetxt(out_path, data, delimiter=",", header=",".join(headers), comments="")
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
