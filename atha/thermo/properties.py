from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from atha.thermo.ideal_gas import IdealGasBackend
from atha.thermo.interface import FluidState


FLUID_STATE_KEYS = {
    "fluid",
    "species",
    "model",
    "type",
    "P",
    "T",
    "h",
    "rho",
    "mdot",
    "gamma",
    "R",
    "cp",
    "cv",
    "mu",
    "k",
    "MW",
    "phase",
    "quality",
    "c_star",
}


def is_fluid_state_spec(value: Any) -> bool:
    """Return true when a boundary value describes a full fluid port state."""

    if not isinstance(value, Mapping):
        return False
    keys = set(value)
    if not keys & FLUID_STATE_KEYS:
        return False
    if "value" in keys or "schedule" in keys:
        return False
    return "fluid" in keys or "species" in keys or "model" in keys or "type" in keys


def fluid_state_from_spec(spec: Mapping[str, Any]) -> FluidState:
    """Build a `FluidState` from a YAML-friendly mapping.

    Supported model families:
    - `ideal_gas`
    - `incompressible_liquid` / `liquid`
    - `combustion_gas`
    """

    model = str(spec.get("model", spec.get("type", "ideal_gas"))).lower()
    fluid = str(spec.get("fluid", spec.get("species", "")))
    mdot = float(spec.get("mdot", 0.0))
    if model in {"ideal_gas", "gas"}:
        state = _ideal_gas_state(spec)
    elif model in {"incompressible_liquid", "liquid", "incompressible"}:
        state = _incompressible_liquid_state(spec)
    elif model in {"combustion_gas", "combustion", "hot_gas"}:
        state = _combustion_gas_state(spec)
    else:
        raise ValueError(f"Unsupported fluid property model: {model}")
    return replace(state, mdot=mdot, fluid=fluid, model=model)


def flatten_fluid_state(prefix: str, state: FluidState) -> dict[str, Any]:
    """Flatten a fluid state into `<prefix>.<field>` scalar paths."""

    values = {f"{prefix}.{key}": value for key, value in state.as_port_values().items()}
    values[prefix] = state
    return values


def _ideal_gas_state(spec: Mapping[str, Any]) -> FluidState:
    gamma = float(spec.get("gamma", 1.4))
    r_gas = float(spec.get("R", spec.get("gas_R", 287.0)))
    backend = IdealGasBackend(
        gamma=gamma,
        R=r_gas,
        mu=float(spec.get("mu", 1.8e-5)),
        k=float(spec.get("k", 0.026)),
        MW=float(spec.get("MW", 0.029)),
    )
    pressure = float(spec.get("P", spec.get("pressure", 101325.0)))
    if "T" in spec or "temperature" in spec:
        return backend.state_from_PT(pressure, float(spec.get("T", spec.get("temperature"))))
    if "h" in spec or "enthalpy" in spec:
        return backend.state_from_Ph(pressure, float(spec.get("h", spec.get("enthalpy"))))
    return backend.state_from_PT(pressure, 300.0)


def _incompressible_liquid_state(spec: Mapping[str, Any]) -> FluidState:
    pressure = float(spec.get("P", spec.get("pressure", 101325.0)))
    temperature = float(spec.get("T", spec.get("temperature", 298.15)))
    rho = float(spec.get("rho", spec.get("density", 1000.0)))
    cp = float(spec.get("cp", 4200.0))
    cv = float(spec.get("cv", cp))
    h = float(spec.get("h", spec.get("enthalpy", cp * temperature)))
    gamma = float(spec.get("gamma", cp / cv if cv else 1.0))
    return FluidState(
        P=pressure,
        T=temperature,
        h=h,
        rho=rho,
        s=float(spec.get("s", 0.0)),
        cp=cp,
        cv=cv,
        gamma=gamma,
        mu=float(spec.get("mu", 1.0e-3)),
        k=float(spec.get("k", 0.1)),
        MW=float(spec.get("MW", 0.018)),
        phase=str(spec.get("phase", "liquid")),
        quality=_optional_float(spec.get("quality")),
    )


def _combustion_gas_state(spec: Mapping[str, Any]) -> FluidState:
    gamma = float(spec.get("gamma", 1.25))
    r_gas = float(spec.get("R", spec.get("gas_R", 355.0)))
    cp = float(spec.get("cp", gamma * r_gas / max(gamma - 1.0, 1.0e-12)))
    temperature = float(spec.get("T", spec.get("temperature", 3500.0)))
    pressure = float(spec.get("P", spec.get("pressure", 101325.0)))
    rho = float(spec.get("rho", pressure / max(r_gas * temperature, 1.0e-12)))
    return FluidState(
        P=pressure,
        T=temperature,
        h=float(spec.get("h", spec.get("enthalpy", cp * temperature))),
        rho=rho,
        s=float(spec.get("s", 0.0)),
        cp=cp,
        cv=float(spec.get("cv", cp - r_gas)),
        gamma=gamma,
        mu=float(spec.get("mu", 4.0e-5)),
        k=float(spec.get("k", 0.08)),
        MW=float(spec.get("MW", 0.022)),
        phase=str(spec.get("phase", "gas")),
        quality=_optional_float(spec.get("quality")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)

