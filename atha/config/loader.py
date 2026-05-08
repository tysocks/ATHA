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

    return LoadedAnalysisConfig(
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


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load ATHA YAML configs") from exc

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    return data


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
