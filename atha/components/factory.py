from __future__ import annotations

from typing import Any, Mapping

from atha.components.combustion_chamber import CombustionChamber
from atha.components.injector import MassFlowInjector
from atha.components.nozzle import Nozzle
from atha.components.registry import known_component_types
from atha.components.valve import Valve
from atha.config.schema import ComponentConfig, ConfigError
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.cantera_backend import CanteraBackend


def build_component_from_config(config: ComponentConfig, context: Mapping[str, Any]):
    """Instantiate a component from its YAML config.

    This is the first small piece of a ROCETS-like Configuration Processor. The
    registry is intentionally conservative; unsupported types fail loudly rather
    than falling back to ad hoc example logic.
    """

    params = _coerce_numbers(dict(config.parameters))
    ctype = config.type
    if ctype == "Valve":
        return Valve(config.name, **params)
    if ctype == "MassFlowInjector":
        return MassFlowInjector(config.name, **params)
    if ctype == "CombustionChamber":
        combustion = context["combustion"]
        return CombustionChamber(
            config.name,
            thermo=CanteraBackend(combustion["mechanism"], initial_X=combustion.get("chamber_initial_X")),
            fuel=combustion["fuel"],
            oxidizer=combustion["oxidizer"],
            **params,
        )
    if ctype == "Nozzle":
        efficiencies = params.pop("efficiencies", None)
        if efficiencies is not None:
            params["efficiencies"] = JANNAFEfficiencies(**efficiencies)
        return Nozzle(config.name, **params)
    raise ConfigError(f"Unsupported component type for factory construction: {ctype}. Known types: {known_component_types()}")


def _coerce_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _coerce_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numbers(v) for v in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value
