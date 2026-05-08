from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from atha.config.schema import ConfigError, MapConfig
from atha.maps import PerformanceMap


def build_performance_maps(configs: Dict[str, MapConfig]) -> Dict[str, PerformanceMap]:
    """Build every configured map into a runtime ``PerformanceMap``."""

    return {name: build_performance_map(config) for name, config in configs.items()}


def build_performance_map(config: MapConfig) -> PerformanceMap:
    """Build a runtime ``PerformanceMap`` from a modular Map YAML config."""

    source_type = str(config.source.get("type", "")).lower()
    extrapolation = str(config.interpolation.get("extrapolation", "clamp"))

    if source_type == "constant":
        values = config.source.get("values", {})
        if not isinstance(values, dict):
            raise ConfigError(f"Map '{config.name}' constant source requires values mapping")
        return PerformanceMap.constant(**{str(k): float(v) for k, v in values.items()})

    if source_type == "csv":
        csv_path = _resolve_data_path(config, str(config.source.get("path", "")))
        axes, outputs = _read_csv_map(config, csv_path)
        if config.kind in {"structured_grid", "structured"}:
            return _structured_from_rows(config, axes, outputs, extrapolation)
        if config.kind == "scattered":
            return PerformanceMap.from_scattered(axes, outputs, extrapolation)
        if _is_structured(axes):
            return _structured_from_rows(config, axes, outputs, extrapolation)
        return PerformanceMap.from_scattered(axes, outputs, extrapolation)

    if source_type in {"hdf5", "h5"}:
        hdf5_path = _resolve_data_path(config, str(config.source.get("path", "")))
        if _is_atha_hdf5_map(config):
            return PerformanceMap.load(str(hdf5_path))
        axes, outputs = _read_hdf5_map(config, hdf5_path)
        if config.kind in {"structured_grid", "structured"}:
            return PerformanceMap.from_arrays(axes, outputs, extrapolation)
        return PerformanceMap.from_scattered(axes, outputs, extrapolation)

    raise ConfigError(
        f"Map '{config.name}' source.type must be csv, hdf5, or constant; got {source_type!r}"
    )


def _resolve_data_path(config: MapConfig, ref: str) -> Path:
    if not ref:
        raise ConfigError(f"Map '{config.name}' source.path is required")
    path = Path(ref).expanduser()
    if not path.is_absolute():
        if config.path is None:
            path = Path.cwd() / path
        else:
            path = config.path.parent / path
    return path.resolve()


def _axis_name(item: dict) -> str:
    return str(item["name"])


def _source_key(item: dict) -> str:
    key = item.get("column", item.get("dataset", item.get("name")))
    if not isinstance(key, str) or not key:
        raise ConfigError(f"Map axis/output entry requires name plus column or dataset: {item}")
    return key


def _read_csv_map(config: MapConfig, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    axes = {_axis_name(item): [] for item in config.axes}
    outputs = {_axis_name(item): [] for item in config.outputs}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in config.axes:
                axes[_axis_name(item)].append(float(row[_source_key(item)]))
            for item in config.outputs:
                outputs[_axis_name(item)].append(float(row[_source_key(item)]))

    return (
        {name: np.asarray(values, dtype=float) for name, values in axes.items()},
        {name: np.asarray(values, dtype=float) for name, values in outputs.items()},
    )


def _read_hdf5_map(config: MapConfig, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    import h5py

    group_name = config.source.get("group")
    with h5py.File(path, "r") as f:
        group = f[group_name] if group_name else f
        axes = {
            _axis_name(item): np.asarray(group[_source_key(item)][()], dtype=float)
            for item in config.axes
        }
        outputs = {
            _axis_name(item): np.asarray(group[_source_key(item)][()], dtype=float)
            for item in config.outputs
        }
    return axes, outputs


def _is_atha_hdf5_map(config: MapConfig) -> bool:
    return not config.axes and not config.outputs


def _is_structured(axes: Dict[str, np.ndarray]) -> bool:
    if not axes:
        return True
    unique_counts = [len(np.unique(values)) for values in axes.values()]
    first_len = len(next(iter(axes.values())))
    return int(np.prod(unique_counts)) == first_len


def _structured_from_rows(
    config: MapConfig,
    axes: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    extrapolation: str,
) -> PerformanceMap:
    axis_names = list(axes.keys())
    unique_axes = {name: np.unique(axes[name]) for name in axis_names}
    shape = tuple(len(unique_axes[name]) for name in axis_names)
    grid_outputs = {name: np.empty(shape, dtype=float) for name in outputs}

    n_rows = len(next(iter(axes.values()))) if axes else 1
    for row_idx in range(n_rows):
        grid_idx = tuple(
            int(np.searchsorted(unique_axes[name], axes[name][row_idx]))
            for name in axis_names
        )
        for out_name, values in outputs.items():
            grid_outputs[out_name][grid_idx] = float(values[row_idx])

    expected = int(np.prod(shape)) if shape else 1
    if expected != n_rows:
        raise ConfigError(
            f"Map '{config.name}' is declared structured but rows do not form a complete grid"
        )
    return PerformanceMap.from_arrays(unique_axes, grid_outputs, extrapolation)
