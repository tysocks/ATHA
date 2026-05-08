"""YAML configuration schema and loading utilities."""

from atha.config.loader import LoadedAnalysisConfig, load_analysis_config
from atha.config.controllers import evaluate_controllers
from atha.config.maps import build_performance_map, build_performance_maps
from atha.config.schedules import (
    evaluate_boundary_conditions,
    evaluate_operating_targets,
    evaluate_schedule,
    evaluate_timing_events,
)
from atha.config.transients import SUPPORTED_TRANSIENT_TYPES, TransientBlock, TransientSystem
from atha.config.schema import (
    AnalysisConfig,
    BoundaryConditionsConfig,
    ComponentConfig,
    ConfigError,
    ConnectionConfig,
    ControllerConfig,
    EngineConfig,
    MapBindingConfig,
    MapConfig,
    OperatingConditionsConfig,
    TelemetryConfig,
    TimingConfig,
    TransientConfig,
)

__all__ = [
    "AnalysisConfig",
    "BoundaryConditionsConfig",
    "ComponentConfig",
    "ConfigError",
    "ConnectionConfig",
    "ControllerConfig",
    "EngineConfig",
    "LoadedAnalysisConfig",
    "MapBindingConfig",
    "MapConfig",
    "OperatingConditionsConfig",
    "TelemetryConfig",
    "TimingConfig",
    "TransientConfig",
    "TransientBlock",
    "TransientSystem",
    "SUPPORTED_TRANSIENT_TYPES",
    "load_analysis_config",
    "build_performance_map",
    "build_performance_maps",
    "evaluate_boundary_conditions",
    "evaluate_controllers",
    "evaluate_operating_targets",
    "evaluate_schedule",
    "evaluate_timing_events",
]
