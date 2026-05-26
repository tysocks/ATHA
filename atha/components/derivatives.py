from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from atha.config.schema import ComponentConfig


@dataclass(frozen=True)
class DerivativeEvaluationContext:
    states: Mapping[str, float]
    algebraics: Mapping[str, float]
    measurements: Mapping[str, float] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    shaft_couplings: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)

    @property
    def values(self) -> dict[str, float]:
        return {**self.states, **self.algebraics, **self.measurements}

    def value(self, path: str, default: float = 0.0) -> float:
        for source in (self.states, self.algebraics, self.measurements, self.inputs):
            value = source.get(path)
            if isinstance(value, (int, float, np.floating)):
                return float(value)
        return float(default)


class ComponentDerivativeContract(Protocol):
    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        ...


class PipeDerivativeContract:
    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        inertance = float(component.parameters.get("inertance", component.parameters.get("L_inertance", 0.0)))
        if inertance > 0.0:
            return {f"{name}.mdot": context.value(f"{name}.dmdot_dt", 0.0)}
        tau = max(float(component.parameters.get("time_constant", component.parameters.get("tau", 0.0))), 0.0)
        if tau <= 0.0:
            return {f"{name}.mdot": 0.0}
        target = context.value(f"{name}.mdot_steady", context.value(f"{name}.inlet.mdot", context.value(f"{name}.mdot", 0.0)))
        current = context.value(f"{name}.mdot", 0.0)
        return {f"{name}.mdot": (target - current) / tau}


class FiniteVolumeDerivativeContract:
    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        volume = max(float(component.parameters.get("volume", 0.0)), 1.0e-12)
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 287.0)))
        temperature = context.value(f"{name}.T", float(component.parameters.get("T_adiabatic", component.parameters.get("initial_T", 300.0))))
        mdot_in = _sum_component_ports(context, name, ("fuel_inlet", "ox_inlet", "lox_inlet", "inlet"))
        mdot_out = _sum_component_ports(context, name, ("outlet",))
        gain = float(component.parameters.get("pressure_gain", gas_r * temperature / volume))
        derivative = gain * (mdot_in - mdot_out)
        pressure_floor = float(component.parameters.get("pressure_floor", 1.0))
        if context.value(f"{name}.P", pressure_floor) <= pressure_floor and derivative < 0.0:
            derivative = 0.0
        return {f"{name}.P": derivative}


class RotorDerivativeContract:
    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        inertia = max(float(component.parameters.get("moment_of_inertia", component.parameters.get("inertia", 1.0))), 1.0e-12)
        omega = context.value(f"{name}.omega", context.value(f"{name}.shaft.omega", 0.0))
        omega_abs = max(abs(omega), 1.0)
        coupling = context.shaft_couplings.get(name, {})
        drive_power = context.value(f"{name}.power_drive", 0.0)
        load_power = context.value(f"{name}.power_load", 0.0)
        drive_torque = context.value(f"{name}.tau_drive", 0.0)
        load_torque = context.value(f"{name}.tau_load", 0.0)
        for turbine in coupling.get("turbines", ()):
            power = max(context.value(f"{turbine}.power", 0.0), 0.0)
            drive_power += power
            drive_torque += context.value(f"{turbine}.tau_drive", power / omega_abs)
        for pump in coupling.get("pumps", ()):
            power = max(context.value(f"{pump}.power", 0.0), 0.0)
            load_power += power
            load_torque += context.value(f"{pump}.tau_load", power / omega_abs)
        friction = float(component.parameters.get("friction_coeff", component.parameters.get("friction", 0.0)))
        torque_balance = drive_torque - load_torque + (drive_power - load_power) / omega_abs - friction * omega
        return {f"{name}.omega": torque_balance / inertia}


def derivative_contract_for_type(type_name: str) -> ComponentDerivativeContract | None:
    return {
        "Pipe": PipeDerivativeContract(),
        "CombustionChamber": FiniteVolumeDerivativeContract(),
        "Preburner": FiniteVolumeDerivativeContract(),
        "GasGenerator": FiniteVolumeDerivativeContract(),
        "Rotor": RotorDerivativeContract(),
    }.get(type_name)


def _sum_component_ports(context: DerivativeEvaluationContext, component: str, ports: tuple[str, ...]) -> float:
    return sum(context.value(f"{component}.{port}.mdot") for port in ports)
