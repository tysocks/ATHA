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
    """Finite-volume combustor / preburner / gas-generator pressure ODE.

    Pressure uses the standard lumped mass-storage form.

    Enthalpy remains algebraically owned by ``energy_residual`` on the current
    DAE path. Integrating ``h`` while also syncing state-owned algebraics made
    mission-cycle residuals diverge; full dynamic enthalpy ownership is tracked
    in ``docs/MISSING_PHYSICS_BACKLOG.md``.
    """

    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        volume = max(float(component.parameters.get("volume", 0.0)), 1.0e-12)
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 287.0)))
        temperature = context.value(
            f"{name}.T",
            float(component.parameters.get("T_adiabatic", component.parameters.get("initial_T", 300.0))),
        )
        mdot_in = _sum_component_ports(context, name, ("fuel_inlet", "ox_inlet", "lox_inlet", "inlet"))
        mdot_out = _sum_component_ports(context, name, ("outlet",))
        gain = float(component.parameters.get("pressure_gain", gas_r * temperature / volume))
        dP_dt = gain * (mdot_in - mdot_out)
        pressure_floor = float(component.parameters.get("pressure_floor", 1.0))
        if context.value(f"{name}.P", pressure_floor) <= pressure_floor and dP_dt < 0.0:
            dP_dt = 0.0
        return {f"{name}.P": dP_dt}


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


class RegenChannelDerivativeContract:
    """Lumped wall-temperature ODE for regenerative cooling channels.

    Uses:
        dT_wall/dt = (Q_hot - Q_cool) / (m_wall * Cp_wall)

    Heat loads are taken from algebraic/measurement targets when available and
    otherwise reconstructed from the same Bartz / NTU-style parameters used by
    the residual contract.
    """

    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        wall_mass = max(float(component.parameters.get("wall_mass", 1.0)), 1.0e-12)
        wall_cp = max(float(component.parameters.get("wall_cp", 500.0)), 1.0e-12)
        q_hot, q_cool = _regen_heat_loads(component, context)
        return {f"{name}.T_wall": (q_hot - q_cool) / (wall_mass * wall_cp)}


class GasVolumeDerivativeContract:
    """Ideal-gas lumped volume pressure and enthalpy ODEs."""

    def derivatives(self, component: ComponentConfig, context: DerivativeEvaluationContext) -> dict[str, float]:
        name = component.name
        volume = max(float(component.parameters.get("volume", 1.0e-3)), 1.0e-12)
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 287.0)))
        gamma = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.4)))
        temperature = context.value(f"{name}.T", float(component.parameters.get("gas_T", component.parameters.get("initial_T", 300.0))))
        pressure = context.value(f"{name}.P", float(component.parameters.get("initial_P", 101325.0)))
        h = context.value(f"{name}.h", float(component.parameters.get("initial_h", 0.0)))
        mdot_in = _sum_component_ports(context, name, ("inlet", "inlet_a", "inlet_b"))
        mdot_out = _sum_component_ports(context, name, ("outlet", "outlet_a", "outlet_b"))
        mdot_net = mdot_in - mdot_out
        rho = max(pressure / max(gas_r * temperature, 1.0e-12), 1.0e-12)
        mass = max(rho * volume, 1.0e-12)
        dP_dt = (gamma * gas_r * temperature / volume) * mdot_net
        h_in = context.value(f"{name}.inlet.h", h)
        q_dot = context.value(f"{name}.heat.Q_dot", context.value(f"{name}.Q_dot", 0.0))
        dh_dt = (q_dot + mdot_in * h_in - mdot_out * h - volume * dP_dt) / mass
        return {f"{name}.P": dP_dt, f"{name}.h": dh_dt}


def derivative_contract_for_type(type_name: str) -> ComponentDerivativeContract | None:
    return {
        "Pipe": PipeDerivativeContract(),
        "CombustionChamber": FiniteVolumeDerivativeContract(),
        "Preburner": FiniteVolumeDerivativeContract(),
        "GasGenerator": FiniteVolumeDerivativeContract(),
        "Rotor": RotorDerivativeContract(),
        "RegenChannel": RegenChannelDerivativeContract(),
        "GasVolume": GasVolumeDerivativeContract(),
        "Volume": GasVolumeDerivativeContract(),
    }.get(type_name)


def _sum_component_ports(context: DerivativeEvaluationContext, component: str, ports: tuple[str, ...]) -> float:
    return sum(context.value(f"{component}.{port}.mdot") for port in ports)


def _inlet_enthalpy_flux(context: DerivativeEvaluationContext, component: str, fallback_h: float) -> float:
    total = 0.0
    for port in ("fuel_inlet", "ox_inlet", "lox_inlet", "inlet"):
        mdot = context.value(f"{component}.{port}.mdot", 0.0)
        if mdot == 0.0:
            continue
        h_port = context.value(f"{component}.{port}.h", fallback_h)
        total += mdot * h_port
    return total


def _regen_heat_loads(component: ComponentConfig, context: DerivativeEvaluationContext) -> tuple[float, float]:
    name = component.name
    q_hot = context.value(f"{name}.Q_hot", context.value(f"{name}.Q_hot_target", float("nan")))
    q_cool = context.value(f"{name}.Q_cool", context.value(f"{name}.Q_cool_target", float("nan")))
    if np.isfinite(q_hot) and np.isfinite(q_cool):
        return float(q_hot), float(q_cool)

    t_wall = context.value(f"{name}.T_wall", float(component.parameters.get("initial_T_wall", component.parameters.get("T_wall", 300.0))))
    t_gas = context.value(f"{name}.gas.T", float(component.parameters.get("gas_T", 3500.0)))
    p_gas = context.value(f"{name}.gas.P", float(component.parameters.get("Pc_design", component.parameters.get("gas_P", 1.0e7))))
    recovery = float(component.parameters.get("recovery_factor", 0.90))
    h_hot_design = float(component.parameters.get("h_hot_design", 5.0e4))
    pc_design = max(float(component.parameters.get("Pc_design", 1.0e7)), 1.0)
    a_hot = float(component.parameters.get("hot_area", component.parameters.get("A_hot", 0.0)))
    a_cool = float(component.parameters.get("cool_area", component.parameters.get("A_cool", 0.0)))
    h_hot = h_hot_design * (max(p_gas, 1.0) / pc_design) ** 0.8
    t_aw = recovery * t_gas
    q_hot = h_hot * a_hot * (t_aw - t_wall)

    mdot = abs(
        context.value(
            f"{name}.coolant_inlet.mdot",
            context.value(f"{name}.coolant_outlet.mdot", float(component.parameters.get("mdot_design", 1.0))),
        )
    )
    t_cool_in = context.value(
        f"{name}.coolant_inlet.T",
        context.value(f"{name}.T_bulk_in", float(component.parameters.get("coolant_T_in", 150.0))),
    )
    cp_cool = float(component.parameters.get("coolant_cp", 3500.0))
    h_cool = float(component.parameters.get("h_cool_design", 1.0e4))
    ntu = h_cool * a_cool / max(mdot * cp_cool, 1.0e-12)
    effectiveness = 1.0 - float(np.exp(-max(ntu, 0.0)))
    q_cool = effectiveness * mdot * cp_cool * (t_wall - t_cool_in)
    return float(q_hot), float(q_cool)
