from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from atha.config.schema import ComponentConfig
from atha.network.problem import NetworkResidual, NetworkVariable


@dataclass(frozen=True)
class ResidualEvaluationContext:
    """Inputs used by component residual providers."""

    z: Mapping[str, float]
    inputs: Mapping[str, Any]
    model: Mapping[str, Any] = field(default_factory=dict)

    def value(self, path: str, default: float = 0.0) -> float:
        if path in self.z:
            return float(self.z[path])
        if path in self.inputs:
            return float(self.inputs[path])
        return float(default)


class ComponentResidualContract(Protocol):
    """Residual-provider contract for reusable component models."""

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        ...

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        ...

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        ...


class ValveFlowContract:
    """Incompressible valve flow residual.

    Expected inputs:
      - ``<component>.inlet.P``
      - ``<component>.outlet.P``
      - ``<component>.inlet.rho``
      - ``<component>.position`` or ``<component>.A_frac``
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        return [NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        return [NetworkResidual(f"{component.name}.mdot_residual", units="kg/s", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        p_in = context.value(f"{name}.inlet.P")
        p_out = context.value(f"{name}.outlet.P")
        rho = max(context.value(f"{name}.inlet.rho", 1.0), 1.0e-12)
        position = context.value(f"{name}.position", context.value(f"{name}.A_frac", 1.0))
        cda = _valve_cda(component, context, position)
        d_p = p_in - p_out
        if d_p == 0.0:
            mdot_target = 0.0
        else:
            mdot_target = cda * (2.0 * rho * abs(d_p)) ** 0.5
            mdot_target = mdot_target if d_p > 0.0 else -mdot_target
        return {
            f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target,
            f"{name}.CdA": cda,
        }


class NozzleConductanceContract:
    """Simple nozzle conductance residual used by current pressure-fed studies."""

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", model.get("nozzle_mdot_scale", 1.0))), 1.0)
        return [NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", model.get("nozzle_mdot_scale", 1.0))), 1.0)
        return [NetworkResidual(f"{component.name}.mdot_residual", units="kg/s", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        p_in = context.value(f"{name}.inlet.P", context.value("chamber.P"))
        p_ambient = context.value(f"{name}.ambient.P", context.value("nozzle.ambient.P", 101325.0))
        conductance = float(context.model.get(f"{name}_conductance", context.model.get("nozzle_conductance", 0.0)))
        mdot_target = conductance * max(p_in - p_ambient, 0.0)
        return {f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target}


class InjectorPressureDropContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        return [NetworkVariable(f"{component.name}.outlet.P", units="Pa", scale=1.0e6, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [NetworkResidual(f"{component.name}.delta_P_residual", units="Pa", scale=1.0e6, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        delta_p = float(context.model.get(f"{name}_delta_P", component.parameters.get("delta_P_nominal", 0.0)))
        return {
            f"{name}.delta_P_residual": (
                context.value(f"{name}.inlet.P") - context.value(f"{name}.outlet.P") - delta_p
            )
        }


class FlowSplitterContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        return [
            NetworkVariable(f"{component.name}.outlet_a.mdot", units="kg/s", scale=scale, owner=component.name),
            NetworkVariable(f"{component.name}.outlet_b.mdot", units="kg/s", scale=scale, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        return [
            NetworkResidual(f"{component.name}.outlet_a.mdot_residual", units="kg/s", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_b.mdot_residual", units="kg/s", scale=scale, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        split = float(component.parameters.get("split_fraction", 0.5))
        mdot_in = context.value(f"{name}.inlet.mdot")
        return {
            f"{name}.outlet_a.mdot_residual": context.value(f"{name}.outlet_a.mdot") - split * mdot_in,
            f"{name}.outlet_b.mdot_residual": context.value(f"{name}.outlet_b.mdot") - (1.0 - split) * mdot_in,
        }


class PipeMomentumContract:
    """Quasi-steady incompressible pipe momentum residual.

    This is the Phase-22 port-solver contract. It closes pipe flow against
    pressure drop with either an explicit conductance or a simple Darcy-style
    resistance estimate. Dynamic pipe inertance remains a state derivative in a
    later DAE phase.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", component.parameters.get("mdot_design", 1.0))), 1.0)
        return [NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", component.parameters.get("mdot_design", 1.0))), 1.0)
        return [NetworkResidual(f"{component.name}.momentum_residual", units="kg/s", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        p_in = context.value(f"{name}.inlet.P")
        p_out = context.value(f"{name}.outlet.P")
        rho = max(context.value(f"{name}.inlet.rho", context.value(f"{name}.outlet.rho", 1000.0)), 1.0e-12)
        d_p = p_in - p_out
        conductance = _pipe_conductance(component, context.model, rho)
        target = conductance * abs(d_p) ** 0.5
        target = target if d_p >= 0.0 else -target
        return {f"{name}.momentum_residual": context.value(f"{name}.mdot") - target}


class FiniteVolumeCombustorContract:
    """Simplified chamber/preburner finite-volume closure.

    The contract closes the topology with mass balance, mixture-ratio,
    pressure, and temperature residuals. It deliberately uses configurable
    design/initial values rather than chemistry tables; high-fidelity
    thermochemistry is tracked as a future Phase-22/23 blocker.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        pressure_scale = max(float(component.parameters.get("initial_P", component.parameters.get("pressure_scale", 1.0e6))), 1.0)
        temperature_scale = max(float(component.parameters.get("initial_T", component.parameters.get("T_adiabatic", 1000.0))), 1.0)
        return [
            NetworkVariable(f"{component.name}.P", units="Pa", scale=pressure_scale, owner=component.name),
            NetworkVariable(f"{component.name}.OF", scale=max(float(component.parameters.get("design_MR", 4.0)), 1.0), owner=component.name),
            NetworkVariable(f"{component.name}.T", units="K", scale=temperature_scale, owner=component.name),
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=max(float(component.parameters.get("mdot_design", 1.0)), 1.0), owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        pressure_scale = max(float(component.parameters.get("initial_P", component.parameters.get("pressure_scale", 1.0e6))), 1.0)
        temperature_scale = max(float(component.parameters.get("initial_T", component.parameters.get("T_adiabatic", 1000.0))), 1.0)
        mdot_scale = max(float(component.parameters.get("mdot_design", 1.0)), 1.0)
        return [
            NetworkResidual(f"{component.name}.mass_balance_residual", units="kg/s", scale=mdot_scale, owner=component.name),
            NetworkResidual(f"{component.name}.OF_residual", scale=max(float(component.parameters.get("design_MR", 4.0)), 1.0), owner=component.name),
            NetworkResidual(f"{component.name}.pressure_residual", units="Pa", scale=pressure_scale, owner=component.name),
            NetworkResidual(f"{component.name}.temperature_residual", units="K", scale=temperature_scale, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        fuel = _port_sum(context, name, ("fuel_inlet",))
        oxidizer = _port_sum(context, name, ("ox_inlet", "lox_inlet"))
        other_in = _port_sum(context, name, ("inlet",))
        total_in = fuel + oxidizer + other_in
        target_of = oxidizer / max(fuel, 1.0e-12) if fuel > 0.0 else float(component.parameters.get("design_MR", 0.0))
        pressure_target = _first_context_value(
            context,
            (f"{name}.outlet.P", f"{name}.fuel_inlet.P", f"{name}.ox_inlet.P", f"{name}.lox_inlet.P"),
            float(component.parameters.get("initial_P", 101325.0)),
        )
        temperature_target = float(component.parameters.get("T_adiabatic", component.parameters.get("initial_T", 300.0)))
        return {
            f"{name}.mass_balance_residual": context.value(f"{name}.mdot") - total_in,
            f"{name}.OF_residual": context.value(f"{name}.OF") - target_of,
            f"{name}.pressure_residual": context.value(f"{name}.P") - pressure_target,
            f"{name}.temperature_residual": context.value(f"{name}.T") - temperature_target,
        }


class RegenThermalContract:
    """Minimal thermal residual contract for cooling/regen paths."""

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        return [
            NetworkVariable(f"{component.name}.T_wall", units="K", scale=1000.0, owner=component.name),
            NetworkVariable(f"{component.name}.Q_dot", units="W", scale=1.0e5, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        return [
            NetworkResidual(f"{component.name}.heat_balance_residual", units="W", scale=1.0e5, owner=component.name),
            NetworkResidual(f"{component.name}.wall_temperature_residual", units="K", scale=1000.0, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        q_hot = context.value(f"{name}.Q_hot", float(component.parameters.get("Q_hot", 0.0)))
        q_cool = context.value(f"{name}.Q_cool", float(component.parameters.get("Q_cool", 0.0)))
        target_t_wall = float(component.parameters.get("initial_T_wall", component.parameters.get("T_wall", 300.0)))
        return {
            f"{name}.heat_balance_residual": context.value(f"{name}.Q_dot") - (q_hot - q_cool),
            f"{name}.wall_temperature_residual": context.value(f"{name}.T_wall") - target_t_wall,
        }


class PumpHeadContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_delta_P_scale", component.parameters.get("delta_P_design", 1.0e6))), 1.0)
        return [NetworkVariable(f"{component.name}.delta_P", units="Pa", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_delta_P_scale", component.parameters.get("delta_P_design", 1.0e6))), 1.0)
        return [NetworkResidual(f"{component.name}.delta_P_residual", units="Pa", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        omega = context.value(f"{name}.shaft.omega", context.value(f"{name}.omega", _design_omega(component, 1.0)))
        omega_safe = max(abs(omega), 1.0e-12)
        mdot = abs(
            context.value(
                f"{name}.mdot",
                context.value(f"{name}.inlet.mdot", context.value(f"{name}.outlet.mdot", _nested_param(component, "pump_map", "mdot_design", 1.0))),
            )
        )
        diameter = float(component.parameters.get("diameter", 0.0))
        if diameter > 0.0:
            rho_design = max(
                float(_nested_param(component, "pump_map", "rho_design", context.value(f"{name}.inlet.rho", 1000.0))),
                1.0,
            )
            phi = mdot / max(rho_design * omega_safe * diameter**3, 1.0e-30)
            head_values = _evaluate_map(context, component, "head_map", {"phi": phi})
            psi = _first_map_value(head_values, ("psi", "head_coefficient"))
            if psi is not None:
                d_p_target = rho_design * psi * omega_safe**2 * diameter**2
            else:
                d_p_target = _first_map_value(head_values, ("pressure_rise", "head", "delta_P"))
                if d_p_target is None:
                    omega_design = max(_design_omega(component, omega_safe), 1.0e-12)
                    d_p_design = float(_nested_param(component, "pump_map", "dP_design", component.parameters.get("delta_P_design", 0.0)))
                    d_p_target = d_p_design * (omega_safe / omega_design) ** 2
            efficiency_values = _evaluate_map(context, component, "efficiency_map", {"phi": phi})
            eta = _first_map_value(efficiency_values, ("eta", "efficiency"))
        else:
            omega_design = max(_design_omega(component, omega_safe), 1.0e-12)
            mdot_design = max(float(_nested_param(component, "pump_map", "mdot_design", component.parameters.get("mdot_design", mdot if mdot else 1.0))), 1.0e-12)
            speed_ratio = omega_safe / omega_design
            flow_ratio = mdot / mdot_design
            head_values = _evaluate_map(
                context,
                component,
                "head_map",
                {
                    "speed_ratio": speed_ratio,
                    "flow_ratio": flow_ratio,
                    "corrected_speed": speed_ratio,
                    "corrected_flow": flow_ratio,
                },
            )
            d_p_target = _first_map_value(head_values, ("pressure_rise", "head", "delta_P"))
            if d_p_target is None:
                d_p_design = float(_nested_param(component, "pump_map", "dP_design", component.parameters.get("delta_P_design", 0.0)))
                d_p_target = d_p_design * speed_ratio**2
            efficiency_values = _evaluate_map(context, component, "efficiency_map", {"speed_ratio": speed_ratio, "flow_ratio": flow_ratio})
            eta = _first_map_value(efficiency_values, ("eta", "efficiency"))
        if eta is None:
            eta = float(_nested_param(component, "pump_map", "efficiency_design", component.parameters.get("efficiency_design", 0.74)))
        eta = min(max(float(eta), 1.0e-6), 1.0)
        return {
            f"{name}.delta_P_residual": context.value(f"{name}.delta_P") - d_p_target,
            f"{name}.efficiency": eta,
        }


class TurbinePowerContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_power_scale", component.parameters.get("power_design", 1.0))), 1.0)
        return [NetworkVariable(f"{component.name}.power", units="W", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_power_scale", component.parameters.get("power_design", 1.0))), 1.0)
        return [NetworkResidual(f"{component.name}.power_residual", units="W", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        mdot = abs(context.value(f"{name}.inlet.mdot"))
        mdot_design = max(float(_nested_param(component, "turbine_map", "mdot_design", component.parameters.get("mdot_design", mdot if mdot else 1.0))), 1.0e-12)
        p_in = context.value(f"{name}.inlet.P", 0.0)
        p_out = context.value(f"{name}.outlet.P", 0.0)
        pressure_ratio = context.value(
            f"{name}.pressure_ratio",
            p_in / max(p_out, 1.0) if p_in > 0.0 and p_out > 0.0 else float(_nested_param(component, "turbine_map", "PR_design", 1.0)),
        )
        corrected_flow_ratio = mdot / mdot_design
        map_values = _evaluate_map(
            context,
            component,
            "efficiency_map",
            {
                "pressure_ratio": pressure_ratio,
                "corrected_flow_ratio": corrected_flow_ratio,
                "corrected_flow": corrected_flow_ratio,
            },
        )
        eta = _first_map_value(map_values, ("efficiency", "eta"))
        if eta is None:
            eta = float(_nested_param(component, "turbine_map", "eta_design", component.parameters.get("efficiency", 1.0)))
        power_design = float(_nested_param(component, "turbine_map", "power_design", component.parameters.get("power_design", 0.0)))
        if power_design > 0.0:
            eta_design = max(float(_nested_param(component, "turbine_map", "eta_design", eta)), 1.0e-12)
            pr_design = max(float(_nested_param(component, "turbine_map", "PR_design", pressure_ratio)), 1.0e-12)
            power_target = power_design * corrected_flow_ratio * (eta / eta_design) * (pressure_ratio / pr_design)
        else:
            power_target = mdot * max(p_in - p_out, 0.0) * eta * 1.0e-3
        return {f"{name}.power_residual": context.value(f"{name}.power") - power_target}


class RotorTorqueBalanceContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        return [NetworkVariable(f"{component.name}.omega", units="rad/s", scale=1000.0, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [NetworkResidual(f"{component.name}.torque_balance", units="N*m", scale=100.0, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        tau_drive = context.value(f"{name}.tau_drive")
        tau_load = context.value(f"{name}.tau_load")
        friction = float(component.parameters.get("friction_coeff", 0.0))
        omega = context.value(f"{name}.omega", float(component.parameters.get("initial_omega", 0.0)))
        return {f"{name}.torque_balance": tau_drive - tau_load - friction * omega}


def residual_contract_for_type(type_name: str) -> ComponentResidualContract | None:
    return {
        "Pipe": PipeMomentumContract(),
        "Valve": ValveFlowContract(),
        "MassFlowInjector": InjectorPressureDropContract(),
        "FlowSplitter": FlowSplitterContract(),
        "CombustionChamber": FiniteVolumeCombustorContract(),
        "Preburner": FiniteVolumeCombustorContract(),
        "Nozzle": NozzleConductanceContract(),
        "Pump": PumpHeadContract(),
        "Turbine": TurbinePowerContract(),
        "Rotor": RotorTorqueBalanceContract(),
        "RegenChannel": RegenThermalContract(),
    }.get(type_name)


def _component_cda(component: ComponentConfig) -> float:
    if "CdA" in component.parameters:
        return float(component.parameters["CdA"])
    if "max_area" in component.parameters:
        return float(component.parameters["max_area"]) * float(component.parameters.get("discharge_coeff", 1.0))
    return 0.0


def _valve_cda(component: ComponentConfig, context: ResidualEvaluationContext, position: float) -> float:
    name = component.name
    position = max(min(float(position), 1.0), 0.0)
    cda_map = context.model.get(f"{name}.map.cda_map")
    if cda_map is not None:
        mapped = cda_map.evaluate(_valve_map_context(context, name, position))
        return max(float(mapped.get("CdA", mapped.get("cda", 0.0))), 0.0)
    cd_map = context.model.get(f"{name}.map.cd_map")
    if cd_map is not None:
        mapped = cd_map.evaluate(_valve_map_context(context, name, position))
        cd = float(mapped.get("Cd", mapped.get("cd", component.parameters.get("discharge_coeff", 1.0))))
        return max(cd * position * float(component.parameters.get("max_area", 1.0)), 0.0)
    return max(_component_cda(component) * position, 0.0)


def _valve_map_context(context: ResidualEvaluationContext, name: str, position: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for source in (context.inputs, context.z):
        for key, value in source.items():
            if isinstance(value, (int, float)):
                values[str(key)] = float(value)
    values.update(
        {
            "inlet.P": context.value(f"{name}.inlet.P"),
            "outlet.P": context.value(f"{name}.outlet.P"),
            "inlet.rho": context.value(f"{name}.inlet.rho", 1.0),
            "position": position,
            "valve.A_frac": position,
            f"{name}.position": position,
            f"{name}.A_frac": position,
        }
    )
    return values


def _pipe_conductance(component: ComponentConfig, model: Mapping[str, Any], rho: float) -> float:
    if f"{component.name}_conductance" in model:
        return float(model[f"{component.name}_conductance"])
    if "conductance" in component.parameters:
        return float(component.parameters["conductance"])
    diameter = float(component.parameters.get("diameter", 0.0))
    length = max(float(component.parameters.get("length", diameter if diameter else 1.0)), 1.0e-12)
    friction = max(float(component.parameters.get("friction_factor", 0.02)), 1.0e-12)
    if diameter <= 0.0:
        return float(component.parameters.get("fallback_conductance", 1.0e-6))
    area = 3.141592653589793 * diameter * diameter / 4.0
    resistance = friction * length / diameter
    return area * (2.0 * rho / max(resistance, 1.0e-12)) ** 0.5


def _port_sum(context: ResidualEvaluationContext, component: str, ports: tuple[str, ...]) -> float:
    return sum(context.value(f"{component}.{port}.mdot") for port in ports)


def _first_context_value(
    context: ResidualEvaluationContext,
    paths: tuple[str, ...],
    default: float,
) -> float:
    for path in paths:
        if path in context.z or path in context.inputs:
            return context.value(path)
    return default


def _nested_param(component: ComponentConfig, group: str, key: str, default: Any) -> Any:
    value = component.parameters.get(group, {})
    if isinstance(value, Mapping) and key in value:
        return value[key]
    return default


def _design_omega(component: ComponentConfig, default: float) -> float:
    raw = _nested_param(component, "pump_map", "speed_design", component.parameters.get("omega_design", default))
    raw = float(raw)
    if raw > 1000.0:
        return raw * 2.0 * 3.141592653589793 / 60.0
    return raw


def _evaluate_map(
    context: ResidualEvaluationContext,
    component: ComponentConfig,
    slot: str,
    values: Mapping[str, float],
) -> dict[str, float]:
    runtime_map = context.model.get(f"{component.name}.map.{slot}")
    if runtime_map is None:
        return {}
    result = runtime_map.evaluate(dict(values))
    output_name = context.model.get(f"{component.name}.map.{slot}.output")
    if isinstance(output_name, str) and output_name in result:
        return {output_name: float(result[output_name])}
    return {str(key): float(value) for key, value in result.items()}


def _first_map_value(values: Mapping[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in values:
            return float(values[name])
    return None
