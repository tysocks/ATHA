from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from atha.config.schema import ComponentConfig
from atha.network import NetworkResidual, NetworkVariable


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
        cda = float(context.model.get(f"{name}_CdA", _component_cda(component)))
        d_p = p_in - p_out
        if d_p == 0.0:
            mdot_target = 0.0
        else:
            mdot_target = cda * max(min(position, 1.0), 0.0) * (2.0 * rho * abs(d_p)) ** 0.5
            mdot_target = mdot_target if d_p > 0.0 else -mdot_target
        return {f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target}


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


class PumpHeadContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_delta_P_scale", component.parameters.get("delta_P_design", 1.0e6))), 1.0)
        return [NetworkVariable(f"{component.name}.delta_P", units="Pa", scale=scale, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_delta_P_scale", component.parameters.get("delta_P_design", 1.0e6))), 1.0)
        return [NetworkResidual(f"{component.name}.delta_P_residual", units="Pa", scale=scale, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        omega = context.value(f"{name}.shaft.omega", float(component.parameters.get("omega_design", 1.0)))
        omega_design = max(float(component.parameters.get("omega_design", omega)), 1.0e-12)
        d_p_design = float(component.parameters.get("delta_P_design", 0.0))
        d_p_target = d_p_design * (omega / omega_design) ** 2
        return {f"{name}.delta_P_residual": context.value(f"{name}.delta_P") - d_p_target}


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
        power_design = float(component.parameters.get("power_design", 0.0))
        mdot_design = max(float(component.parameters.get("mdot_design", mdot if mdot else 1.0)), 1.0e-12)
        power_target = power_design * mdot / mdot_design
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
        "Valve": ValveFlowContract(),
        "MassFlowInjector": InjectorPressureDropContract(),
        "FlowSplitter": FlowSplitterContract(),
        "Nozzle": NozzleConductanceContract(),
        "Pump": PumpHeadContract(),
        "Turbine": TurbinePowerContract(),
        "Rotor": RotorTorqueBalanceContract(),
    }.get(type_name)


def _component_cda(component: ComponentConfig) -> float:
    if "CdA" in component.parameters:
        return float(component.parameters["CdA"])
    if "max_area" in component.parameters:
        return float(component.parameters["max_area"]) * float(component.parameters.get("discharge_coeff", 1.0))
    return 0.0
