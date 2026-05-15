from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar

from atha.config.schema import (
    AnalysisConfig,
    BoundaryConditionsConfig,
    ConfigError,
    ControllerConfig,
    EngineConfig,
    MapConfig,
    OperatingConditionsConfig,
    TelemetryConfig,
    TimingConfig,
    TransientConfig,
)


T = TypeVar("T")


@dataclass(frozen=True)
class LoadedAnalysisConfig:
    """Resolved configuration bundle rooted at one Analysis YAML file."""

    analysis_config: AnalysisConfig
    engine: EngineConfig
    maps: Dict[str, MapConfig]
    transients: Dict[str, TransientConfig]
    boundary_conditions: Optional[BoundaryConditionsConfig] = None
    operating_conditions: Optional[OperatingConditionsConfig] = None
    timings: Optional[TimingConfig] = None
    controllers: Optional[ControllerConfig] = None
    telemetry: Optional[TelemetryConfig] = None


def load_analysis_config(path: str | Path) -> LoadedAnalysisConfig:
    """Load an Analysis YAML and all referenced modular config files.

    The Analysis YAML is the only required entrypoint. Referenced paths are
    resolved relative to the YAML file that contains the reference.
    """

    analysis_path = Path(path).expanduser().resolve()
    analysis_data = _read_yaml_mapping(analysis_path)
    analysis = AnalysisConfig.from_yaml(analysis_data, path=analysis_path)

    engine_path = _resolve_ref(analysis.engine, analysis_path)
    engine = EngineConfig.from_yaml(_read_yaml_mapping(engine_path), path=engine_path)

    maps = {
        name: _load_ref(ref, analysis_path, MapConfig.from_yaml)
        for name, ref in analysis.maps.items()
    }
    transients = _load_transients(analysis.transients, analysis_path)

    _validate_engine_references(engine, maps, transients)
    from atha.components.registry import validate_engine_config

    validate_engine_config(engine, transients)

    loaded = LoadedAnalysisConfig(
        analysis_config=analysis,
        engine=engine,
        maps=maps,
        transients=transients,
        boundary_conditions=_load_optional_ref(analysis.boundary_conditions, analysis_path, BoundaryConditionsConfig.from_yaml),
        operating_conditions=_load_optional_ref(analysis.operating_conditions, analysis_path, OperatingConditionsConfig.from_yaml),
        timings=_load_optional_ref(analysis.timings, analysis_path, TimingConfig.from_yaml),
        controllers=_load_optional_ref(analysis.controllers, analysis_path, ControllerConfig.from_yaml),
        telemetry=_load_optional_ref(analysis.telemetry, analysis_path, TelemetryConfig.from_yaml),
    )
    _validate_timing_targets(loaded.timings)
    _validate_controller_outputs(loaded.controllers, transients)
    return loaded


def load_config_folder(path: str | Path) -> LoadedAnalysisConfig:
    """Load a config folder by resolving ``analysis.yaml`` inside it."""

    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "analysis.yaml"
    return load_analysis_config(candidate)


def _read_yaml_mapping(path: Path, include_stack: tuple[Path, ...] = ()) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load ATHA YAML configs") from exc

    path = path.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if path in include_stack:
        chain = " -> ".join(str(item) for item in (*include_stack, path))
        raise ConfigError(f"YAML include cycle detected: {chain}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    return _expand_yaml_includes(dict(data), path, (*include_stack, path))


def _expand_yaml_includes(data: Dict[str, Any], path: Path, include_stack: tuple[Path, ...]) -> Mapping[str, Any]:
    include_value = data.pop("$include", data.pop("include", None))
    if include_value is None:
        return data
    if isinstance(include_value, (str, Path)):
        include_refs = [str(include_value)]
    elif isinstance(include_value, list):
        include_refs = include_value
    else:
        raise ConfigError(f"YAML include in {path} must be a path string or list of path strings")

    merged: Dict[str, Any] = {}
    for index, ref in enumerate(include_refs):
        if not isinstance(ref, str) or not ref:
            raise ConfigError(f"YAML include[{index}] in {path} must be a non-empty path string")
        include_path = Path(ref).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        included = _read_yaml_mapping(include_path, include_stack=include_stack)
        merged = _merge_yaml_mappings(merged, dict(included))
    return _merge_yaml_mappings(merged, data)


def _merge_yaml_mappings(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_yaml_mappings(dict(merged[key]), value)
        elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = [*merged[key], *value]
        else:
            merged[key] = value
    return merged


def _resolve_ref(ref: str, source_path: Path) -> Path:
    ref_path = Path(ref).expanduser()
    if not ref_path.is_absolute():
        ref_path = source_path.parent / ref_path
    return ref_path.resolve()


def _load_ref(ref: str, source_path: Path, factory: Callable[[Mapping[str, Any], Optional[Path]], T]) -> T:
    path = _resolve_ref(ref, source_path)
    return factory(_read_yaml_mapping(path), path)


def _load_optional_ref(
    ref: Optional[str],
    source_path: Path,
    factory: Callable[[Mapping[str, Any], Optional[Path]], T],
) -> Optional[T]:
    if ref is None:
        return None
    if not isinstance(ref, str) or not ref:
        raise ConfigError("Optional config references must be non-empty path strings")
    return _load_ref(ref, source_path, factory)


def _load_transients(refs: Any, analysis_path: Path) -> Dict[str, TransientConfig]:
    if isinstance(refs, str):
        path = _resolve_ref(refs, analysis_path)
        data = _read_yaml_mapping(path)
        raw = data.get("transients", {})
        if isinstance(raw, list):
            configs = [TransientConfig.from_yaml(item, path=path) for item in raw]
            return {cfg.name: cfg for cfg in configs}
        if isinstance(raw, Mapping):
            configs = {}
            for name, item in raw.items():
                if not isinstance(item, Mapping):
                    raise ConfigError(f"transients.{name} must be a mapping")
                item_data = dict(item)
                item_data.setdefault("name", str(name))
                configs[str(name)] = TransientConfig.from_yaml(item_data, path=path)
            return configs
        raise ConfigError("transients YAML must contain a transients mapping or list")
    if isinstance(refs, Mapping):
        return {
            name: _load_ref(ref, analysis_path, TransientConfig.from_yaml)
            for name, ref in refs.items()
        }
    raise ConfigError("analysis.transients must be a path string or name-to-path mapping")


def _validate_engine_references(
    engine: EngineConfig,
    maps: Dict[str, MapConfig],
    transients: Dict[str, TransientConfig],
) -> None:
    for comp in engine.components.values():
        if comp.transient is not None and comp.transient not in transients:
            raise ConfigError(
                f"Component '{comp.name}' references transient '{comp.transient}', "
                "but Analysis YAML does not bind it"
            )
        for slot, binding in comp.maps.items():
            if binding.ref not in maps:
                raise ConfigError(
                    f"Component '{comp.name}' map slot '{slot}' references map "
                    f"'{binding.ref}', but Analysis YAML does not bind it"
                )
            available_outputs = set(maps[binding.ref].output_names)
            requested = []
            if binding.output is not None:
                requested.append(binding.output)
            requested.extend(binding.outputs.values())
            missing = [name for name in requested if name not in available_outputs]
            if missing:
                raise ConfigError(
                    f"Component '{comp.name}' map slot '{slot}' requests missing "
                    f"output(s) {missing} from map '{binding.ref}'. Available: "
                    f"{sorted(available_outputs)}"
                )


def _validate_timing_targets(timings: Optional[TimingConfig]) -> None:
    if timings is None:
        return
    for index, event in enumerate(timings.events):
        target = event.get("target")
        if not isinstance(target, str) or not target:
            raise ConfigError(f"timings.events[{index}].target must be a non-empty string")
        if "schedule" not in event and "value" not in event:
            raise ConfigError(f"timings.events[{index}] must contain value or schedule")


def _validate_controller_outputs(
    controllers: Optional[ControllerConfig],
    transients: Dict[str, TransientConfig],
) -> None:
    if controllers is None:
        return
    legal_paths = {cfg.command.get("path") for cfg in transients.values() if isinstance(cfg.command.get("path"), str)}
    from atha.config.controllers import controller_execution_order, controller_output_paths

    try:
        controller_execution_order(controllers.controllers)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    for name, controller in controllers.controllers.items():
        if not isinstance(controller, Mapping):
            raise ConfigError(f"controllers.{name} must be a mapping")
        controller_type = str(controller.get("type", "null"))
        outputs = controller_output_paths(controller)
        for output in outputs:
            if _is_controller_extension_path(output):
                continue
            if output not in legal_paths:
                raise ConfigError(
                    f"Controller '{name}' output '{output}' is not a known transient command path. "
                    f"Known command paths: {sorted(legal_paths)}"
                )
        _validate_controller_shape(name, controller_type, controller)


def _is_controller_extension_path(path: str) -> bool:
    return path.startswith("commands.") or path.startswith("controller.") or path.startswith("targets.")


def _validate_controller_shape(name: str, controller_type: str, controller: Mapping[str, Any]) -> None:
    common_allowed = {"active_phases"}
    allowed_by_type = {
        "null": {"type", "input", "output"} | common_allowed,
        "of_mass_flow_split": {"type", "inputs", "outputs"} | common_allowed,
        "gain_product": {"type", "inputs", "output"} | common_allowed,
        "proportional": {"type", "inputs", "output", "parameters"} | common_allowed,
        "pi": {"type", "inputs", "output", "parameters"} | common_allowed,
        "pid": {"type", "inputs", "output", "parameters"} | common_allowed,
        "scheduled_gain": {"type", "inputs", "output"} | common_allowed,
        "limiter": {"type", "input", "output", "parameters"} | common_allowed,
        "rate_limiter": {"type", "input", "output", "parameters"} | common_allowed,
        "selector": {"type", "inputs", "output", "mode", "index"} | common_allowed,
        "min": {"type", "inputs", "output"} | common_allowed,
        "max": {"type", "inputs", "output"} | common_allowed,
        "python_function": {"type", "function", "outputs", "parameters"} | common_allowed,
    }
    allowed = allowed_by_type.get(controller_type)
    if allowed is None:
        raise ConfigError(f"Unsupported controller '{name}' type: {controller_type}")
    unknown = sorted(set(controller) - allowed)
    if unknown:
        raise ConfigError(f"Controller '{name}' has unsupported key(s): {unknown}. Allowed: {sorted(allowed)}")
