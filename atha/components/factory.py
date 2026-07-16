"""Legacy OOP component factory (non-canonical).

.. deprecated::
    The production ATHA path is YAML → ``EngineAssembler`` / ``PortNetworkBuilder``
    → residual/derivative contracts → ``DAEExecutionProblem``.

    This factory only constructs a small subset of OOP ``BaseComponent`` objects
    (Valve, MassFlowInjector, CombustionChamber, Nozzle) for the older
    ``Engine`` / ``EngineLayout`` solver path under ``atha.solver``. New models
    and examples should not call ``build_component_from_config``.
"""

from __future__ import annotations

import warnings
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
    """Instantiate a legacy OOP component from its YAML config.

    Prefer residual/derivative contracts and the generic-port DAE runner for all
    new work. This helper remains only for compatibility with the older
    ``EngineLayout`` solvers.
    """

    warnings.warn(
        "atha.components.factory.build_component_from_config is a legacy path; "
        "use PortNetworkBuilder / DAEExecutionProblem for new models.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    raise ConfigError(
        f"Unsupported component type for legacy factory construction: {ctype}. "
        f"Known registry types: {known_component_types()}. "
        "Use the generic-port residual/derivative contracts instead."
    )


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
