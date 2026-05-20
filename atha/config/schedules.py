from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from atha.config.schema import BoundaryConditionsConfig, ConfigError, OperatingConditionsConfig
from atha.thermo.properties import flatten_fluid_state, fluid_state_from_spec, is_fluid_state_spec


def evaluate_boundary_conditions(config: BoundaryConditionsConfig, t: float) -> Dict[str, Any]:
    """Evaluate boundary-condition values at time ``t``."""

    values: Dict[str, Any] = {}
    for name, spec in config.conditions.items():
        evaluated = _evaluate_value_or_schedule(spec, t, f"boundary condition '{name}'")
        if is_fluid_state_spec(evaluated):
            values.update(flatten_fluid_state(name, fluid_state_from_spec(evaluated)))
        else:
            values[name] = evaluated
    return values


def evaluate_operating_targets(config: OperatingConditionsConfig, t: float) -> Dict[str, Any]:
    """Evaluate operating target setpoints at time ``t`` keyed by target name."""

    values: Dict[str, Any] = {}
    for name, spec in config.targets.items():
        if isinstance(spec, Mapping) and "schedule" in spec:
            values[name] = evaluate_schedule(spec["schedule"], t, base_path=config.path)
        elif isinstance(spec, Mapping) and "value" in spec:
            values[name] = spec["value"]
        else:
            values[name] = spec
    return values


def evaluate_timing_events(config: Any, t: float) -> Dict[str, Any]:
    """Evaluate timing-event targets at time ``t``.

    Timing events are direct state/command targets, not controlled operating
    targets. Each event may define either a scalar ``value`` or any supported
    schedule under ``schedule``.
    """

    values: Dict[str, Any] = {}
    if config is None:
        return values
    for index, event in enumerate(config.events):
        target = event.get("target")
        if not isinstance(target, str) or not target:
            raise ConfigError(f"timings.events[{index}].target must be a non-empty string")
        if "schedule" in event:
            values[target] = evaluate_schedule(event["schedule"], t, base_path=config.path)
        elif "value" in event:
            values[target] = event["value"]
        else:
            raise ConfigError(f"timings.events[{index}] must contain value or schedule")
    return values


def evaluate_schedule(schedule: Any, t: float, base_path: Path | None = None) -> Any:
    """Evaluate a scalar schedule.

    Supported forms:
    - scalar value
    - ``{"type": "constant", "value": x}``
    - ``{"type": "step", "time": t0, "initial": a, "final": b}``
    - ``{"type": "ramp", "t_start": a, "t_end": b, "y_start": c, "y_end": d}``
    - ``{"type": "table", "values": [[t0, y0], ...]}``
    - ``{"type": "profile", "source": {"type": "json", "path": "targets.json"}}``
    - ``{"type": "runbox", ...}``
    """

    if not isinstance(schedule, Mapping):
        return schedule
    stype = str(schedule.get("type", "constant"))
    if stype == "constant":
        return schedule.get("value")
    if stype == "step":
        t0 = float(schedule.get("time", 0.0))
        return schedule.get("final") if t >= t0 else schedule.get("initial")
    if stype == "ramp":
        t_start = float(schedule["t_start"])
        t_end = float(schedule["t_end"])
        y_start = float(schedule["y_start"])
        y_end = float(schedule["y_end"])
        if t_end <= t_start:
            raise ConfigError("ramp schedule requires t_end > t_start")
        frac = min(max((t - t_start) / (t_end - t_start), 0.0), 1.0)
        return y_start + frac * (y_end - y_start)
    if stype == "table":
        values = schedule.get("values", [])
        if not values:
            raise ConfigError("table schedule requires at least one [time, value] row")
        times = np.asarray([float(row[0]) for row in values], dtype=float)
        vals = np.asarray([float(row[1]) for row in values], dtype=float)
        return float(np.interp(float(t), times, vals))
    if stype == "profile":
        return _evaluate_profile(schedule, t, base_path=base_path)
    if stype == "runbox":
        return _evaluate_runbox(schedule, t, base_path=base_path)
    raise ConfigError(f"Unknown schedule type: {stype}")


def schedule_breakpoints(schedule: Any, base_path: Path | None = None) -> list[float]:
    """Return known discontinuity or corner times for a schedule.

    The solver driver uses these times to split integrations cleanly instead of
    allowing a stiff integrator to step across commanded events.
    """

    if not isinstance(schedule, Mapping):
        return []
    stype = str(schedule.get("type", "constant"))
    if stype == "constant":
        return []
    if stype == "step":
        return [float(schedule.get("time", 0.0))]
    if stype == "ramp":
        return [float(schedule["t_start"]), float(schedule["t_end"])]
    if stype == "table":
        values = schedule.get("values", [])
        return [float(row[0]) for row in values]
    if stype == "profile":
        rows = _load_profile_rows(schedule, base_path)
        time_column = str(schedule.get("time_column", "time_s"))
        return [float(row[time_column]) for row in rows if time_column in row]
    if stype == "runbox":
        setpoint = _load_runbox_setpoint(schedule.get("setpoint", {}), base_path)
        bounds = schedule.get("bounds", {})
        _ = (setpoint, bounds)
        n = int(schedule.get("points_per_side", 8))
        dwell_s = float(schedule.get("dwell_s", 0.25))
        if n < 2 or dwell_s <= 0.0:
            return []
        count = 4 * n - 2
        return [i * dwell_s for i in range(count + 1)]
    return []


def collect_config_breakpoints(*configs: Any, t_start: float, t_end: float) -> list[float]:
    """Collect schedule breakpoints from boundary, timing, and target configs."""

    points = {float(t_start), float(t_end)}
    for config in configs:
        if config is None:
            continue
        for schedule in _iter_config_schedules(config):
            for point in schedule_breakpoints(schedule, base_path=getattr(config, "path", None)):
                if t_start <= point <= t_end:
                    points.add(float(point))
    return sorted(points)


def _evaluate_value_or_schedule(spec: Any, t: float, label: str) -> Any:
    if isinstance(spec, Mapping):
        if is_fluid_state_spec(spec):
            return {
                key: _evaluate_value_or_schedule(value, t, f"{label}.{key}")
                for key, value in spec.items()
            }
        if "schedule" in spec:
            return evaluate_schedule(spec["schedule"], t)
        if "value" in spec:
            return spec["value"]
        if "type" in spec:
            return evaluate_schedule(spec, t)
        raise ConfigError(f"{label} must contain value or schedule")
    return spec


def _iter_config_schedules(config: Any):
    if hasattr(config, "conditions"):
        for spec in config.conditions.values():
            yield from _iter_schedules(spec)
    if hasattr(config, "targets"):
        for spec in config.targets.values():
            yield from _iter_schedules(spec)
    if hasattr(config, "events"):
        for event in config.events:
            yield from _iter_schedules(event)


def _iter_schedules(value: Any):
    if isinstance(value, Mapping):
        if "schedule" in value:
            yield value["schedule"]
        elif "type" in value:
            yield value
        for nested in value.values():
            if isinstance(nested, (Mapping, list)):
                yield from _iter_schedules(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_schedules(item)


def _evaluate_runbox(schedule: Mapping[str, Any], t: float, base_path: Path | None = None) -> Dict[str, float]:
    setpoint = _load_runbox_setpoint(schedule.get("setpoint", {}), base_path)
    bounds = schedule.get("bounds", {})
    if not isinstance(setpoint, Mapping) or not isinstance(bounds, Mapping):
        raise ConfigError("runbox schedule requires setpoint and bounds mappings")

    mdot_total = float(setpoint.get("mdot_total"))
    if "OF" in setpoint:
        of_design = float(setpoint["OF"])
    else:
        mdot_lox = float(setpoint["mdot_lox"])
        mdot_fuel = float(setpoint["mdot_fuel"])
        of_design = mdot_lox / mdot_fuel

    mdot_frac = bounds.get("mdot_total_fraction")
    of_frac = bounds.get("of_fraction")
    if not isinstance(mdot_frac, list) or len(mdot_frac) != 2:
        raise ConfigError("runbox bounds.mdot_total_fraction must be [low, high]")
    if not isinstance(of_frac, list) or len(of_frac) != 2:
        raise ConfigError("runbox bounds.of_fraction must be [low, high]")

    mdot_low = mdot_total * float(mdot_frac[0])
    mdot_high = mdot_total * float(mdot_frac[1])
    of_low = of_design * float(of_frac[0])
    of_high = of_design * float(of_frac[1])
    n = int(schedule.get("points_per_side", 8))
    dwell_s = float(schedule.get("dwell_s", 0.25))
    if n < 2:
        raise ConfigError("runbox points_per_side must be >= 2")
    if dwell_s <= 0.0:
        raise ConfigError("runbox dwell_s must be > 0")

    bottom = [(m, of_low) for m in np.linspace(mdot_low, mdot_high, n)]
    right = [(mdot_high, of) for of in np.linspace(of_low, of_high, n)[1:]]
    top = [(m, of_high) for m in np.linspace(mdot_high, mdot_low, n)[1:]]
    left = [(mdot_low, of) for of in np.linspace(of_high, of_low, n)[1:]]
    points = bottom + right + top + left
    closed = points + [points[0]]
    times = np.arange(len(closed), dtype=float) * dwell_s
    tc = min(max(float(t), 0.0), float(times[-1]))
    mdots = np.asarray([p[0] for p in closed], dtype=float)
    ofs = np.asarray([p[1] for p in closed], dtype=float)
    mdot = float(np.interp(tc, times, mdots))
    of = float(np.interp(tc, times, ofs))
    mdot_fuel = mdot / (1.0 + of)
    mdot_lox = mdot - mdot_fuel

    return {
        "mdot_total": mdot,
        "OF": of,
        "mdot_lox": mdot_lox,
        "mdot_fuel": mdot_fuel,
        "duration": float(times[-1]),
    }


def _evaluate_profile(schedule: Mapping[str, Any], t: float, base_path: Path | None = None) -> Dict[str, float]:
    rows = _load_profile_rows(schedule, base_path)
    if len(rows) < 1:
        raise ConfigError("profile schedule requires at least one row")

    time_column = str(schedule.get("time_column", "time_s"))
    if time_column not in rows[0]:
        raise ConfigError(f"profile schedule missing time column: {time_column}")

    times = np.asarray([float(row[time_column]) for row in rows], dtype=float)
    if np.any(np.diff(times) < 0.0):
        raise ConfigError("profile schedule times must be monotonically increasing")

    output_map = schedule.get("outputs")
    if output_map is None:
        output_map = {
            key: key
            for key in rows[0]
            if key != time_column and isinstance(_coerce_source_value(rows[0][key]), (int, float))
        }
    if not isinstance(output_map, Mapping) or not output_map:
        raise ConfigError("profile schedule requires at least one output")

    tc = min(max(float(t), float(times[0])), float(times[-1]))
    values: Dict[str, float] = {}
    for output_name, column_name in output_map.items():
        column = str(column_name)
        if column not in rows[0]:
            raise ConfigError(f"profile schedule missing output column: {column}")
        samples = np.asarray([float(row[column]) for row in rows], dtype=float)
        values[str(output_name)] = float(np.interp(tc, times, samples))
    values["duration"] = float(times[-1] - times[0])
    return values


def _load_profile_rows(schedule: Mapping[str, Any], base_path: Path | None) -> list[Dict[str, Any]]:
    if "values" in schedule:
        values = schedule["values"]
        if not isinstance(values, list):
            raise ConfigError("profile schedule values must be a list")
        return [_coerce_row(row) for row in values]

    source = schedule.get("source")
    if source is None:
        raise ConfigError("profile schedule requires values or source")
    if isinstance(source, str):
        source_spec: Mapping[str, Any] = {"path": source}
    elif isinstance(source, Mapping):
        source_spec = source
    else:
        raise ConfigError("profile schedule source must be a path string or mapping")

    path_value = source_spec.get("path")
    if path_value is None:
        raise ConfigError("profile schedule source requires path")
    source_path = _resolve_source_path(str(path_value), base_path)
    source_type = str(source_spec.get("type") or source_path.suffix.lstrip(".")).lower()
    if source_type == "json":
        rows = _load_json_profile(source_path, source_spec)
    elif source_type == "csv":
        rows = _load_csv_profile(source_path)
    else:
        raise ConfigError(f"Unsupported profile schedule source type: {source_type}")
    return rows


def _load_json_profile(path: Path, source_spec: Mapping[str, Any]) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    key = source_spec.get("key")
    if key is not None:
        if not isinstance(data, Mapping) or key not in data:
            raise ConfigError(f"profile JSON key not found: {key}")
        data = data[key]
    elif isinstance(data, Mapping) and "targets" in data:
        data = data["targets"]
    if not isinstance(data, list):
        raise ConfigError("profile JSON must resolve to a list of rows")
    return [_coerce_row(row) for row in data]


def _load_csv_profile(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [_coerce_row(row) for row in rows]


def _coerce_row(row: Any) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ConfigError("profile rows must be mappings")
    return {str(key): _coerce_source_value(value) for key, value in row.items() if value not in (None, "")}


def _load_runbox_setpoint(setpoint: Any, base_path: Path | None) -> Dict[str, Any]:
    if not isinstance(setpoint, Mapping):
        raise ConfigError("runbox setpoint must be a mapping")

    source = setpoint.get("source")
    inline = {key: value for key, value in setpoint.items() if key != "source"}
    if source is None:
        return dict(inline)

    if isinstance(source, str):
        source_spec: Mapping[str, Any] = {"path": source}
    elif isinstance(source, Mapping):
        source_spec = source
    else:
        raise ConfigError("runbox setpoint.source must be a path string or mapping")

    path_value = source_spec.get("path")
    if path_value is None:
        raise ConfigError("runbox setpoint.source requires path")

    source_path = _resolve_source_path(str(path_value), base_path)
    source_type = str(source_spec.get("type") or source_path.suffix.lstrip(".")).lower()
    if source_type == "json":
        loaded = _load_json_setpoint(source_path, source_spec)
    elif source_type == "csv":
        loaded = _load_csv_setpoint(source_path, source_spec)
    else:
        raise ConfigError(f"Unsupported runbox setpoint source type: {source_type}")

    loaded.update(inline)
    return loaded


def _resolve_source_path(path_value: str, base_path: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_path is not None:
        return base_path.parent / path
    return path


def _load_json_setpoint(path: Path, source_spec: Mapping[str, Any]) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    key = source_spec.get("key")
    if key is not None:
        if not isinstance(data, Mapping) or key not in data:
            raise ConfigError(f"runbox JSON setpoint key not found: {key}")
        data = data[key]
    if not isinstance(data, Mapping):
        raise ConfigError("runbox JSON setpoint must resolve to an object")
    return dict(data)


def _load_csv_setpoint(path: Path, source_spec: Mapping[str, Any]) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ConfigError("runbox CSV setpoint requires at least one data row")

    row_name = source_spec.get("row")
    if row_name is None:
        row = rows[0]
    else:
        key_column = str(source_spec.get("key_column", "name"))
        matches = [candidate for candidate in rows if candidate.get(key_column) == str(row_name)]
        if not matches:
            raise ConfigError(f"runbox CSV setpoint row not found: {row_name}")
        row = matches[0]
    return {key: _coerce_source_value(value) for key, value in row.items() if value not in (None, "")}


def _coerce_source_value(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value
