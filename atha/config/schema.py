from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class ConfigError(ValueError):
    """Raised when a YAML configuration file is structurally invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    return value


@dataclass(frozen=True)
class MapBindingConfig:
    """Bind a component map slot to a logical map reference.

    ``ref`` is resolved through the top-level Analysis YAML ``maps`` section.
    ``output`` is optional and lets several component slots use different
    outputs from the same multi-output map asset.
    """

    ref: str
    output: Optional[str] = None
    outputs: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, value: Any, label: str) -> "MapBindingConfig":
        if isinstance(value, str):
            return cls(ref=value)
        data = _mapping(value, label)
        ref = data.get("ref")
        if not isinstance(ref, str) or not ref:
            raise ConfigError(f"{label}.ref must be a non-empty string")
        output = data.get("output")
        if output is not None and not isinstance(output, str):
            raise ConfigError(f"{label}.output must be a string")
        outputs = data.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise ConfigError(f"{label}.outputs must be a mapping")
        return cls(ref=ref, output=output, outputs={str(k): str(v) for k, v in outputs.items()})


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    maps: Dict[str, MapBindingConfig] = field(default_factory=dict)
    transient: Optional[str] = None
    initial_state: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, name: str, value: Any) -> "ComponentConfig":
        data = _mapping(value, f"components.{name}")
        ctype = data.get("type")
        if not isinstance(ctype, str) or not ctype:
            raise ConfigError(f"components.{name}.type must be a non-empty string")
        params = dict(_mapping(data.get("parameters", {}), f"components.{name}.parameters"))
        maps_raw = _mapping(data.get("maps", {}), f"components.{name}.maps")
        maps = {
            str(slot): MapBindingConfig.from_yaml(binding, f"components.{name}.maps.{slot}")
            for slot, binding in maps_raw.items()
        }
        transient = data.get("transient")
        if transient is not None and not isinstance(transient, str):
            raise ConfigError(f"components.{name}.transient must be a string")
        initial_raw = _mapping(data.get("initial_state", {}), f"components.{name}.initial_state")
        return cls(
            name=name,
            type=ctype,
            parameters=params,
            maps=maps,
            transient=transient,
            initial_state={str(k): float(v) for k, v in initial_raw.items()},
            metadata=dict(_mapping(data.get("metadata", {}), f"components.{name}.metadata")),
        )


@dataclass(frozen=True)
class ConnectionConfig:
    source: str
    target: str
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, value: Any, index: int) -> "ConnectionConfig":
        data = _mapping(value, f"connections[{index}]")
        source = data.get("from")
        target = data.get("to")
        domain = data.get("domain", "fluid")
        if not isinstance(source, str) or "." not in source:
            raise ConfigError(f"connections[{index}].from must be 'component.port'")
        if not isinstance(target, str) or "." not in target:
            raise ConfigError(f"connections[{index}].to must be 'component.port'")
        if domain not in {"fluid", "shaft", "thermal"}:
            raise ConfigError(f"connections[{index}].domain must be fluid, shaft, or thermal")
        return cls(source=source, target=target, domain=str(domain), metadata={k: v for k, v in data.items() if k not in {"from", "to", "domain"}})


@dataclass(frozen=True)
class EngineConfig:
    name: str
    components: Dict[str, ComponentConfig]
    connections: List[ConnectionConfig]
    units: str = "SI"
    version: Optional[str] = None
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "EngineConfig":
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("engine.name must be a non-empty string")
        comps_raw = _mapping(data.get("components", {}), "engine.components")
        components = {str(k): ComponentConfig.from_yaml(str(k), v) for k, v in comps_raw.items()}
        connections = [
            ConnectionConfig.from_yaml(v, i)
            for i, v in enumerate(_list(data.get("connections"), "engine.connections"))
        ]
        return cls(
            name=name,
            components=components,
            connections=connections,
            units=str(data.get("units", "SI")),
            version=data.get("version"),
            path=path,
        )


@dataclass(frozen=True)
class MapConfig:
    name: str
    kind: str
    source: Dict[str, Any]
    axes: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    interpolation: Dict[str, Any] = field(default_factory=dict)
    scaling: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "MapConfig":
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("map.name must be a non-empty string")
        outputs = [dict(_mapping(v, f"map.outputs[{i}]")) for i, v in enumerate(_list(data.get("outputs"), "map.outputs"))]
        return cls(
            name=name,
            kind=str(data.get("kind", "structured_grid")),
            source=dict(_mapping(data.get("source", {}), "map.source")),
            axes=[dict(_mapping(v, f"map.axes[{i}]")) for i, v in enumerate(_list(data.get("axes"), "map.axes"))],
            outputs=outputs,
            interpolation=dict(_mapping(data.get("interpolation", {}), "map.interpolation")),
            scaling=dict(_mapping(data.get("scaling", {}), "map.scaling")),
            path=path,
        )

    @property
    def output_names(self) -> List[str]:
        return [str(item["name"]) for item in self.outputs if "name" in item]


@dataclass(frozen=True)
class TransientConfig:
    name: str
    type: str
    state: Dict[str, Any] = field(default_factory=dict)
    command: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "TransientConfig":
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("transient.name must be a non-empty string")
        state = dict(_mapping(data.get("state", {}), "transient.state"))
        command_raw = data.get("command", {})
        command = {"path": command_raw} if isinstance(command_raw, str) else dict(_mapping(command_raw, "transient.command"))
        parameters = dict(_mapping(data.get("parameters", {}), "transient.parameters"))
        if "input" in data:
            command.setdefault("path", data["input"])
        if "output" in data:
            state.setdefault("path", data["output"])
        if "initial" in data:
            state.setdefault("initial", data["initial"])
        return cls(
            name=name,
            type=str(data.get("type", "first_order")),
            state=state,
            command=command,
            parameters=parameters,
            path=path,
        )


@dataclass(frozen=True)
class BoundaryConditionsConfig:
    name: str
    conditions: Dict[str, Any]
    time_unit: str = "s"
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "BoundaryConditionsConfig":
        return cls(
            name=str(data.get("name", path.stem if path else "boundary_conditions")),
            conditions=dict(_mapping(data.get("conditions", {}), "boundary_conditions.conditions")),
            time_unit=str(data.get("time_unit", "s")),
            path=path,
        )


@dataclass(frozen=True)
class OperatingConditionsConfig:
    name: str
    targets: Dict[str, Any]
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "OperatingConditionsConfig":
        return cls(
            name=str(data.get("name", path.stem if path else "operating_conditions")),
            targets=dict(_mapping(data.get("targets", {}), "operating_conditions.targets")),
            path=path,
        )


@dataclass(frozen=True)
class TimingConfig:
    name: str
    events: List[Dict[str, Any]]
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "TimingConfig":
        return cls(
            name=str(data.get("name", path.stem if path else "timings")),
            events=[dict(_mapping(v, f"timings.events[{i}]")) for i, v in enumerate(_list(data.get("events"), "timings.events"))],
            path=path,
        )


@dataclass(frozen=True)
class ControllerConfig:
    name: str
    controllers: Dict[str, Any]
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "ControllerConfig":
        return cls(
            name=str(data.get("name", path.stem if path else "controllers")),
            controllers=dict(_mapping(data.get("controllers", {}), "controllers.controllers")),
            path=path,
        )


@dataclass(frozen=True)
class TelemetryConfig:
    name: str
    channels: List[Dict[str, Any]]
    sample_rate_hz: Optional[float] = None
    exports: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "TelemetryConfig":
        rate = data.get("sample_rate_hz")
        return cls(
            name=str(data.get("name", path.stem if path else "telemetry")),
            channels=[dict(_mapping(v, f"telemetry.channels[{i}]")) for i, v in enumerate(_list(data.get("channels"), "telemetry.channels"))],
            sample_rate_hz=float(rate) if rate is not None else None,
            exports=dict(_mapping(data.get("exports", {}), "telemetry.exports")),
            path=path,
        )


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    engine: str
    maps: Dict[str, str] = field(default_factory=dict)
    transients: Any = field(default_factory=dict)
    boundary_conditions: Optional[str] = None
    operating_conditions: Optional[str] = None
    timings: Optional[str] = None
    controllers: Optional[str] = None
    telemetry: Optional[str] = None
    solver: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], path: Optional[Path] = None) -> "AnalysisConfig":
        name = data.get("name")
        engine = data.get("engine")
        if not isinstance(name, str) or not name:
            raise ConfigError("analysis name must be a non-empty string")
        if not isinstance(engine, str) or not engine:
            raise ConfigError("analysis.engine must be a non-empty path string")
        return cls(
            name=name,
            engine=engine,
            maps={str(k): str(v) for k, v in _mapping(data.get("maps", {}), "analysis.maps").items()},
            transients=_transient_refs(data.get("transients", {})),
            boundary_conditions=data.get("boundary_conditions"),
            operating_conditions=data.get("operating_conditions"),
            timings=data.get("timings"),
            controllers=data.get("controllers"),
            telemetry=data.get("telemetry"),
            solver=dict(_mapping(data.get("solver", {}), "analysis.solver")),
            analysis=dict(_mapping(data.get("analysis", {}), "analysis.analysis")),
            path=path,
        )


def _transient_refs(value: Any) -> Any:
    if isinstance(value, str):
        return value
    return {str(k): str(v) for k, v in _mapping(value, "analysis.transients").items()}
