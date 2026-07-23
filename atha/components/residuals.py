from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

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


class BoundarySourceContract:
    """Fluid source boundary that anchors state properties without fixing flow.

    The downstream component network owns the mass-flow solution. The source
    only supplies thermodynamic state values at its outlet so detailed pump and
    feed-system solves can start from explicit tank/supply components instead
    of implicit boundary paths.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = component, model
        return []

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [
            NetworkResidual(f"{component.name}.outlet.P_residual", units="Pa", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.outlet.T_residual", units="K", scale=1000.0, owner=component.name),
            NetworkResidual(f"{component.name}.outlet.rho_residual", units="kg/m^3", scale=1000.0, owner=component.name),
            NetworkResidual(f"{component.name}.outlet.gamma_residual", scale=1.0, owner=component.name),
            NetworkResidual(f"{component.name}.outlet.h_residual", units="J/kg", scale=1.0e6, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        temperature = _boundary_property(component, context, "T", 300.0)
        cp = float(component.parameters.get("cp", component.parameters.get("specific_heat", 2000.0)))
        targets = {
            "P": _boundary_property(component, context, "P", 101325.0),
            "T": temperature,
            "rho": _boundary_property(component, context, "rho", 1.0),
            "gamma": _boundary_property(component, context, "gamma", 1.2),
            "h": _boundary_property(component, context, "h", cp * temperature),
        }
        return {
            f"{name}.outlet.{prop}_residual": context.value(f"{name}.outlet.{prop}") - value
            for prop, value in targets.items()
        }


class BoundarySinkContract:
    """Fluid sink boundary that anchors discharge pressure.

    Optional component parameters can also anchor T, rho, gamma, or h, but the
    default sink leaves those properties to upstream component closure and only
    imposes the pressure environment.
    """

    _optional_properties = {
        "T": ("K", 1000.0),
        "rho": ("kg/m^3", 1000.0),
        "gamma": ("", 1.0),
        "h": ("J/kg", 1.0e6),
    }

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = component, model
        return []

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        residuals = [NetworkResidual(f"{component.name}.inlet.P_residual", units="Pa", scale=1.0e6, owner=component.name)]
        for prop, (units, scale) in self._optional_properties.items():
            if prop in component.parameters or f"inlet_{prop}" in component.parameters:
                residuals.append(NetworkResidual(f"{component.name}.inlet.{prop}_residual", units=units, scale=scale, owner=component.name))
        return residuals

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        result = {
            f"{name}.inlet.P_residual": context.value(f"{name}.inlet.P") - _boundary_property(component, context, "P", 101325.0)
        }
        for prop in self._optional_properties:
            if prop in component.parameters or f"inlet_{prop}" in component.parameters:
                result[f"{name}.inlet.{prop}_residual"] = (
                    context.value(f"{name}.inlet.{prop}") - _boundary_property(component, context, prop, 0.0)
                )
        return result


class ValveFlowContract:
    """Valve flow residual with liquid and gas modes.

    Expected inputs:
      - ``<component>.inlet.P``
      - ``<component>.outlet.P``
      - ``<component>.inlet.rho``
      - ``<component>.position`` or ``<component>.A_frac``

    ``parameters.flow_model: compressible`` enables a choked/subcritical gas
    approximation using ``inlet.gamma`` and ``inlet.T``/``inlet.R`` when present.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        initial = float(component.parameters.get("initial_mdot", component.parameters.get("mdot_design", 0.0)))
        return [NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, initial=initial, owner=component.name)]

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
        flow_model = str(component.parameters.get("flow_model", component.parameters.get("model", "incompressible"))).lower()
        if bool(component.parameters.get("prescribed_mdot", False)):
            mdot_target = context.value(f"{name}.inlet.mdot", context.value(f"{name}.mdot"))
            choked = 0.0
        elif flow_model in {"compressible", "gas"}:
            mdot_target, choked = _compressible_orifice_mdot(
                cda,
                p_in,
                p_out,
                gamma=context.value(f"{name}.inlet.gamma", 1.4),
                r_gas=context.value(f"{name}.inlet.R", component.parameters.get("gas_R", 287.0)),
                temperature=context.value(f"{name}.inlet.T", component.parameters.get("T", 300.0)),
            )
            mdot_target = mdot_target if d_p >= 0.0 else -abs(mdot_target)
        elif d_p == 0.0:
            mdot_target = 0.0
            choked = 0.0
        else:
            mdot_target = cda * (2.0 * rho * abs(d_p)) ** 0.5
            mdot_target = mdot_target if d_p > 0.0 else -mdot_target
            choked = 0.0
        return {
            f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target,
            f"{name}.mdot_target": mdot_target,
            f"{name}.inlet.mdot_target": mdot_target,
            f"{name}.outlet.mdot_target": mdot_target,
            f"{name}.CdA": cda,
            f"{name}.choked": float(choked),
        }


class NozzleConductanceContract:
    """Nozzle choked-flow/thrust closure with conductance compatibility."""

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", model.get("nozzle_mdot_scale", 1.0))), 1.0)
        thrust_scale = max(float(component.parameters.get("thrust_scale", 1.0e5)), 1.0)
        mdot_initial = float(component.parameters.get("initial_mdot", component.parameters.get("mdot_design", scale)))
        p_initial = float(component.parameters.get("initial_P", component.parameters.get("chamber_pressure", 1.0e6)))
        throat_area = float(component.parameters.get("throat_area", 0.0))
        cf = float(component.parameters.get("thrust_coefficient", component.parameters.get("Cf", 1.5)))
        return [
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, initial=mdot_initial, owner=component.name),
            NetworkVariable(f"{component.name}.thrust", units="N", scale=thrust_scale, initial=cf * throat_area * max(p_initial - 101325.0, 0.0), owner=component.name),
            NetworkVariable(f"{component.name}.Cf", scale=2.0, initial=cf, owner=component.name),
            NetworkVariable(f"{component.name}.c_star", units="m/s", scale=2000.0, initial=float(component.parameters.get("c_star", 1500.0)), owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", model.get("nozzle_mdot_scale", 1.0))), 1.0)
        thrust_scale = max(float(component.parameters.get("thrust_scale", 1.0e5)), 1.0)
        return [
            NetworkResidual(f"{component.name}.mdot_residual", units="kg/s", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.thrust_residual", units="N", scale=thrust_scale, owner=component.name),
            NetworkResidual(f"{component.name}.Cf_residual", scale=2.0, owner=component.name),
            NetworkResidual(f"{component.name}.c_star_residual", units="m/s", scale=2000.0, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        p_in = context.value(f"{name}.inlet.P", context.value("chamber.P"))
        p_ambient = context.value(f"{name}.ambient.P", context.value("nozzle.ambient.P", 101325.0))
        conductance = float(context.model.get(f"{name}_conductance", context.model.get("nozzle_conductance", 0.0)))
        gamma = context.value(f"{name}.inlet.gamma", component.parameters.get("gamma", 1.22))
        rho = context.value(f"{name}.inlet.rho", 0.0)
        throat_area = float(component.parameters.get("throat_area", 0.0))
        discharge = float(component.parameters.get("discharge_coeff", component.parameters.get("Cd", 1.0)))
        configured_c_star = float(component.parameters.get("c_star", 0.0))
        if throat_area > 0.0 and configured_c_star > 0.0 and p_in > 0.0:
            mdot_target = discharge * p_in * throat_area / max(configured_c_star, 1.0e-12)
        elif throat_area > 0.0 and rho > 0.0 and p_in > 0.0:
            choked_coeff = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
            mdot_target = discharge * throat_area * (gamma * rho * p_in) ** 0.5 * choked_coeff
        else:
            mdot_target = conductance * max(p_in - p_ambient, 0.0)
        cf = float(component.parameters.get("thrust_coefficient", component.parameters.get("Cf", 1.5)))
        c_star = configured_c_star if configured_c_star > 0.0 else p_in * throat_area / max(mdot_target, 1.0e-12) if throat_area > 0.0 else 0.0
        thrust_target = cf * throat_area * max(p_in - p_ambient, 0.0) if throat_area > 0.0 else cf * mdot_target * c_star
        return {
            f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target,
            f"{name}.thrust_residual": context.value(f"{name}.thrust") - thrust_target,
            f"{name}.Cf_residual": context.value(f"{name}.Cf") - cf,
            f"{name}.c_star_residual": context.value(f"{name}.c_star") - c_star,
        }


class InjectorPressureDropContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        scale = max(float(component.parameters.get("mdot_design", 1.0)), 1.0)
        delta_p = float(component.parameters.get("delta_P_nominal", 0.0))
        return [
            NetworkVariable(f"{component.name}.outlet.P", units="Pa", scale=1.0e6, initial=float(component.parameters.get("initial_outlet_P", max(float(component.parameters.get("initial_inlet_P", 1.0e6)) - delta_p, 1.0))), owner=component.name),
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, initial=float(component.parameters.get("initial_mdot", component.parameters.get("mdot_design", 0.0))), owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        scale = max(float(component.parameters.get("mdot_design", 1.0)), 1.0)
        return [
            NetworkResidual(f"{component.name}.delta_P_residual", units="Pa", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.mdot_residual", units="kg/s", scale=scale, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        delta_p = float(context.model.get(f"{name}_delta_P", component.parameters.get("delta_P_nominal", 0.0)))
        rho = max(context.value(f"{name}.inlet.rho", component.parameters.get("rho", 1000.0)), 1.0e-12)
        cda = float(component.parameters.get("CdA", component.parameters.get("conductance", 0.0)))
        d_p_actual = context.value(f"{name}.inlet.P") - context.value(f"{name}.outlet.P")
        mdot_target = cda * (2.0 * rho * max(d_p_actual, 0.0)) ** 0.5 if cda > 0.0 else context.value(f"{name}.inlet.mdot", context.value(f"{name}.outlet.mdot", context.value(f"{name}.mdot")))
        return {
            f"{name}.delta_P_residual": (
                context.value(f"{name}.inlet.P") - context.value(f"{name}.outlet.P") - delta_p
            ),
            f"{name}.mdot_residual": context.value(f"{name}.mdot") - mdot_target,
            f"{name}.mdot_target": mdot_target,
            f"{name}.inlet.mdot_target": mdot_target,
            f"{name}.outlet.mdot_target": mdot_target,
            f"{name}.outlet.P_target": max(context.value(f"{name}.inlet.P") - delta_p, 1.0),
        }


class FlowSplitterContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        mdot_design = float(component.parameters.get("mdot_design", 1.0))
        split = float(component.parameters.get("split_fraction", 0.5))
        h_initial = float(component.parameters.get("initial_h", 0.0))
        return [
            NetworkVariable(f"{component.name}.outlet_a.mdot", units="kg/s", scale=scale, initial=split * mdot_design, owner=component.name),
            NetworkVariable(f"{component.name}.outlet_b.mdot", units="kg/s", scale=scale, initial=(1.0 - split) * mdot_design, owner=component.name),
            NetworkVariable(f"{component.name}.outlet_a.h", units="J/kg", scale=1.0e6, initial=h_initial, owner=component.name),
            NetworkVariable(f"{component.name}.outlet_b.h", units="J/kg", scale=1.0e6, initial=h_initial, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", 1.0)), 1.0)
        return [
            NetworkResidual(f"{component.name}.outlet_a.mdot_residual", units="kg/s", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_b.mdot_residual", units="kg/s", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_a.h_residual", units="J/kg", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_b.h_residual", units="J/kg", scale=1.0e6, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        mode = str(component.parameters.get("split_mode", component.parameters.get("mode", "fixed_fraction"))).lower()
        split = float(component.parameters.get("split_fraction", 0.5))
        mdot_in = context.value(f"{name}.inlet.mdot")
        if mode in {"hydraulic", "pressure", "demand", "downstream"}:
            demand_source = component.parameters.get("demand_source")
            demand_position_source = component.parameters.get("demand_position_source")
            if demand_position_source is not None:
                position = min(max(context.value(str(demand_position_source), 1.0), 0.0), 1.0)
                outlet_b = position * float(component.parameters.get("demand_design_mdot", (1.0 - split) * mdot_in))
            elif demand_source is not None:
                outlet_b = context.value(str(demand_source), (1.0 - split) * mdot_in)
            else:
                outlet_b = context.value(f"{name}.outlet_b.mdot", (1.0 - split) * mdot_in)
            outlet_b = min(max(outlet_b, 0.0), max(mdot_in, 0.0))
            outlet_a = mdot_in - outlet_b
        else:
            outlet_a = split * mdot_in
            outlet_b = (1.0 - split) * mdot_in
        return {
            f"{name}.outlet_a.mdot_residual": context.value(f"{name}.outlet_a.mdot") - outlet_a,
            f"{name}.outlet_b.mdot_residual": context.value(f"{name}.outlet_b.mdot") - outlet_b,
            f"{name}.outlet_a.h_residual": context.value(f"{name}.outlet_a.h") - context.value(f"{name}.inlet.h"),
            f"{name}.outlet_b.h_residual": context.value(f"{name}.outlet_b.h") - context.value(f"{name}.inlet.h"),
            f"{name}.outlet_a.mdot_target": outlet_a,
            f"{name}.outlet_b.mdot_target": outlet_b,
            f"{name}.outlet_a.h_target": context.value(f"{name}.inlet.h"),
            f"{name}.outlet_b.h_target": context.value(f"{name}.inlet.h"),
        }


class PipeMomentumContract:
    """Pipe pressure-drop and inertance residual.

    Steady pipes may set inertance to zero. Dynamic pipes can provide
    ``<pipe>.dmdot_dt`` from the DAE derivative path and nonzero
    ``parameters.inertance`` for a ROCETS-style inertial pressure term.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", component.parameters.get("mdot_design", 1.0))), 1.0)
        mdot_initial = float(component.parameters.get("initial_mdot", component.parameters.get("mdot_design", 0.0)))
        return [
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=scale, initial=mdot_initial, owner=component.name),
            NetworkVariable(f"{component.name}.dP_friction", units="Pa", scale=1.0e6, owner=component.name),
            NetworkVariable(f"{component.name}.dmdot_dt", units="kg/s^2", scale=scale, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        scale = max(float(model.get(f"{component.name}_mdot_scale", component.parameters.get("mdot_design", 1.0))), 1.0)
        return [
            NetworkResidual(f"{component.name}.momentum_residual", units="Pa", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.friction_residual", units="Pa", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.inertance_residual", units="kg/s^2", scale=scale, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        p_in = context.value(f"{name}.inlet.P")
        p_out = context.value(f"{name}.outlet.P")
        rho = max(context.value(f"{name}.inlet.rho", context.value(f"{name}.outlet.rho", 1000.0)), 1.0e-12)
        d_p = p_in - p_out
        mdot = context.value(f"{name}.mdot")
        d_p_friction_target = _pipe_friction_dp(component, context.model, rho, mdot)
        inertance = float(component.parameters.get("inertance", component.parameters.get("L_inertance", 0.0)))
        dmdot_dt_input = context.value(f"{name}.state_derivative.mdot", context.value(f"{name}.dmdot_dt", 0.0))
        if inertance > 0.0:
            dmdot_dt_target = (d_p - d_p_friction_target) / inertance
        else:
            dmdot_dt_target = dmdot_dt_input
        return {
            f"{name}.momentum_residual": d_p - context.value(f"{name}.dP_friction") - inertance * context.value(f"{name}.dmdot_dt"),
            f"{name}.friction_residual": context.value(f"{name}.dP_friction") - d_p_friction_target,
            f"{name}.inertance_residual": context.value(f"{name}.dmdot_dt") - dmdot_dt_input,
            f"{name}.dP_friction_target": d_p_friction_target,
            f"{name}.dmdot_dt_target": dmdot_dt_target,
        }


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
        temperature = float(component.parameters.get("initial_T", component.parameters.get("T_adiabatic", 300.0)))
        pressure = float(component.parameters.get("initial_P", 101325.0))
        cp = float(component.parameters.get("cp", component.parameters.get("gas_cp", 3500.0)))
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 355.0)))
        gamma = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.22)))
        return [
            NetworkVariable(f"{component.name}.P", units="Pa", scale=pressure_scale, initial=pressure, owner=component.name),
            NetworkVariable(f"{component.name}.OF", scale=max(float(component.parameters.get("design_MR", 4.0)), 1.0), initial=float(component.parameters.get("design_MR", 4.0)), owner=component.name),
            NetworkVariable(f"{component.name}.T", units="K", scale=temperature_scale, initial=temperature, owner=component.name),
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=max(float(component.parameters.get("mdot_design", 1.0)), 1.0), initial=float(component.parameters.get("initial_mdot", component.parameters.get("mdot_design", 1.0))), owner=component.name),
            NetworkVariable(f"{component.name}.h", units="J/kg", scale=1.0e6, initial=float(component.parameters.get("initial_h", cp * temperature)), owner=component.name),
            NetworkVariable(f"{component.name}.rho", units="kg/m^3", scale=100.0, initial=float(component.parameters.get("initial_rho", pressure / max(gas_r * temperature, 1.0e-12))), owner=component.name),
            NetworkVariable(f"{component.name}.gamma", scale=1.0, initial=gamma, owner=component.name),
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
            NetworkResidual(f"{component.name}.energy_residual", units="J/kg", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.density_residual", units="kg/m^3", scale=100.0, owner=component.name),
            NetworkResidual(f"{component.name}.gamma_residual", scale=1.0, owner=component.name),
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
        cp = float(component.parameters.get("cp", component.parameters.get("gas_cp", 3500.0)))
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 355.0)))
        gamma_target = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.22)))
        h_target = float(component.parameters.get("h_out", cp * temperature_target))
        volume = float(component.parameters.get("volume", 0.0))
        # When volume > 0, chamber/preburner/GG pressure is state-owned by the
        # finite-volume ODE. Do not algebraically pin P to an inlet/design
        # anchor — that fights mass-storage dynamics and collapses thrust.
        pressure = context.value(f"{name}.P", pressure_target)
        if volume > 0.0:
            pressure_residual = 0.0
            rho_target = pressure / max(gas_r * temperature_target, 1.0e-12)
        else:
            pressure_residual = pressure - pressure_target
            rho_target = pressure_target / max(gas_r * temperature_target, 1.0e-12)
        return {
            f"{name}.mass_balance_residual": context.value(f"{name}.mdot") - total_in,
            f"{name}.OF_residual": context.value(f"{name}.OF") - target_of,
            f"{name}.pressure_residual": pressure_residual,
            f"{name}.temperature_residual": context.value(f"{name}.T") - temperature_target,
            f"{name}.energy_residual": context.value(f"{name}.h") - h_target,
            f"{name}.density_residual": context.value(f"{name}.rho") - rho_target,
            f"{name}.gamma_residual": context.value(f"{name}.gamma") - gamma_target,
        }


class RegenThermalContract:
    """Thermal residual contract for regenerative cooling channels.

    Closes net heat load and exposes reconstructed ``Q_hot`` / ``Q_cool``
    targets so the DAE wall-temperature derivative can integrate without
    relying on the legacy OOP RegenChannel path.
    """

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        t_wall0 = float(component.parameters.get("initial_T_wall", component.parameters.get("T_wall", 300.0)))
        return [
            NetworkVariable(f"{component.name}.T_wall", units="K", scale=1000.0, initial=t_wall0, owner=component.name),
            NetworkVariable(f"{component.name}.Q_dot", units="W", scale=1.0e5, owner=component.name),
            NetworkVariable(f"{component.name}.Q_hot", units="W", scale=1.0e5, owner=component.name),
            NetworkVariable(f"{component.name}.Q_cool", units="W", scale=1.0e5, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [
            NetworkResidual(f"{component.name}.heat_balance_residual", units="W", scale=1.0e5, owner=component.name),
            NetworkResidual(f"{component.name}.wall_temperature_residual", units="K", scale=1000.0, owner=component.name),
            NetworkResidual(f"{component.name}.Q_hot_residual", units="W", scale=1.0e5, owner=component.name),
            NetworkResidual(f"{component.name}.Q_cool_residual", units="W", scale=1.0e5, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        t_wall = context.value(f"{name}.T_wall", float(component.parameters.get("initial_T_wall", component.parameters.get("T_wall", 300.0))))
        q_hot_target, q_cool_target, t_wall_eq = _regen_heat_targets(component, context, t_wall)
        return {
            f"{name}.heat_balance_residual": context.value(f"{name}.Q_dot") - (q_hot_target - q_cool_target),
            f"{name}.wall_temperature_residual": context.value(f"{name}.T_wall") - t_wall_eq,
            f"{name}.Q_hot_residual": context.value(f"{name}.Q_hot") - q_hot_target,
            f"{name}.Q_cool_residual": context.value(f"{name}.Q_cool") - q_cool_target,
            f"{name}.Q_hot_target": q_hot_target,
            f"{name}.Q_cool_target": q_cool_target,
            f"{name}.T_wall_target": t_wall_eq,
        }


class GasVolumeContract:
    """Ideal-gas volume algebraic anchors for pressure, temperature, and density."""

    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        pressure = float(component.parameters.get("initial_P", 101325.0))
        temperature = float(component.parameters.get("gas_T", component.parameters.get("initial_T", 300.0)))
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 287.0)))
        gamma = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.4)))
        cp = float(component.parameters.get("cp", component.parameters.get("gas_cp", gas_r * gamma / max(gamma - 1.0, 1.0e-6))))
        return [
            NetworkVariable(f"{component.name}.P", units="Pa", scale=max(pressure, 1.0e5), initial=pressure, owner=component.name),
            NetworkVariable(f"{component.name}.T", units="K", scale=max(temperature, 100.0), initial=temperature, owner=component.name),
            NetworkVariable(f"{component.name}.h", units="J/kg", scale=1.0e6, initial=float(component.parameters.get("initial_h", cp * temperature)), owner=component.name),
            NetworkVariable(f"{component.name}.rho", units="kg/m^3", scale=100.0, initial=pressure / max(gas_r * temperature, 1.0e-12), owner=component.name),
            NetworkVariable(f"{component.name}.gamma", scale=1.0, initial=gamma, owner=component.name),
            NetworkVariable(f"{component.name}.mdot", units="kg/s", scale=1.0, initial=0.0, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [
            NetworkResidual(f"{component.name}.pressure_residual", units="Pa", scale=1.0e5, owner=component.name),
            NetworkResidual(f"{component.name}.temperature_residual", units="K", scale=100.0, owner=component.name),
            NetworkResidual(f"{component.name}.energy_residual", units="J/kg", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.density_residual", units="kg/m^3", scale=100.0, owner=component.name),
            NetworkResidual(f"{component.name}.gamma_residual", scale=1.0, owner=component.name),
            NetworkResidual(f"{component.name}.mass_balance_residual", units="kg/s", scale=1.0, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        gas_r = float(component.parameters.get("gas_R", component.parameters.get("R", 287.0)))
        gamma = float(component.parameters.get("gamma", component.parameters.get("gas_gamma", 1.4)))
        temperature = float(component.parameters.get("gas_T", component.parameters.get("initial_T", 300.0)))
        pressure_target = _first_context_value(
            context,
            (f"{name}.inlet.P", f"{name}.outlet.P", f"{name}.P"),
            float(component.parameters.get("initial_P", 101325.0)),
        )
        cp = float(component.parameters.get("cp", component.parameters.get("gas_cp", gas_r * gamma / max(gamma - 1.0, 1.0e-6))))
        h_target = float(component.parameters.get("initial_h", cp * temperature))
        mdot_in = _port_sum(context, name, ("inlet", "inlet_a", "inlet_b"))
        mdot_out = _port_sum(context, name, ("outlet", "outlet_a", "outlet_b"))
        volume = float(component.parameters.get("volume", 0.0))
        pressure = context.value(f"{name}.P", pressure_target)
        if volume > 0.0:
            pressure_residual = 0.0
            rho_target = pressure / max(gas_r * temperature, 1.0e-12)
        else:
            pressure_residual = pressure - pressure_target
            rho_target = pressure_target / max(gas_r * temperature, 1.0e-12)
        return {
            f"{name}.pressure_residual": pressure_residual,
            f"{name}.temperature_residual": context.value(f"{name}.T") - temperature,
            f"{name}.energy_residual": context.value(f"{name}.h") - h_target,
            f"{name}.density_residual": context.value(f"{name}.rho") - rho_target,
            f"{name}.gamma_residual": context.value(f"{name}.gamma") - gamma,
            f"{name}.mass_balance_residual": context.value(f"{name}.mdot") - 0.5 * (mdot_in + mdot_out),
        }


class PumpHeadContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        pump_map = component.parameters.get("pump_map", {})
        if not isinstance(pump_map, Mapping):
            pump_map = {}
        scale = max(float(model.get(f"{component.name}_delta_P_scale", pump_map.get("dP_design", component.parameters.get("delta_P_design", 1.0e6)))), 1.0)
        power_scale = max(float(component.parameters.get("power_design", 1.0e6)), 1.0)
        mdot_design = float(pump_map.get("mdot_design", component.parameters.get("mdot_design", 1.0)))
        d_p_design = float(pump_map.get("dP_design", component.parameters.get("delta_P_design", scale)))
        rho_design = float(pump_map.get("rho_design", component.parameters.get("rho_design", 1000.0)))
        eta_design = min(max(float(pump_map.get("efficiency_design", component.parameters.get("efficiency_design", 0.74))), 1.0e-6), 1.0)
        omega_design = max(_design_omega(component, 1000.0), 1.0)
        power_initial = mdot_design * max(d_p_design, 0.0) / max(rho_design * eta_design, 1.0e-12)
        return [
            NetworkVariable(f"{component.name}.delta_P", units="Pa", scale=scale, initial=d_p_design, owner=component.name),
            NetworkVariable(f"{component.name}.power", units="W", scale=power_scale, initial=power_initial, owner=component.name),
            NetworkVariable(f"{component.name}.outlet.h", units="J/kg", scale=1.0e6, initial=d_p_design / max(rho_design * eta_design, 1.0e-12), owner=component.name),
            NetworkVariable(f"{component.name}.tau_load", units="N*m", scale=100.0, initial=power_initial / omega_design, owner=component.name),
            NetworkVariable(f"{component.name}.efficiency", scale=1.0, initial=eta_design, owner=component.name),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        pump_map = component.parameters.get("pump_map", {})
        if not isinstance(pump_map, Mapping):
            pump_map = {}
        scale = max(float(model.get(f"{component.name}_delta_P_scale", pump_map.get("dP_design", component.parameters.get("delta_P_design", 1.0e6)))), 1.0)
        power_scale = max(float(component.parameters.get("power_design", 1.0e6)), 1.0)
        return [
            NetworkResidual(f"{component.name}.delta_P_residual", units="Pa", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.power_residual", units="W", scale=power_scale, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_h_residual", units="J/kg", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.tau_load_residual", units="N*m", scale=100.0, owner=component.name),
            NetworkResidual(f"{component.name}.efficiency_residual", scale=1.0, owner=component.name),
        ]

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
        rho = max(context.value(f"{name}.inlet.rho", _nested_param(component, "pump_map", "rho_design", 1000.0)), 1.0e-12)
        h_in = context.value(f"{name}.inlet.h", 0.0)
        h_out_target = h_in + d_p_target / max(rho * eta, 1.0e-12)
        power_target = mdot * max(d_p_target, 0.0) / max(rho * eta, 1.0e-12)
        tau_target = power_target / max(omega_safe, 1.0)
        return {
            f"{name}.delta_P_residual": context.value(f"{name}.delta_P") - d_p_target,
            f"{name}.power_residual": context.value(f"{name}.power") - power_target,
            f"{name}.outlet_h_residual": context.value(f"{name}.outlet.h") - h_out_target,
            f"{name}.tau_load_residual": context.value(f"{name}.tau_load") - tau_target,
            f"{name}.efficiency_residual": context.value(f"{name}.efficiency") - eta,
            f"{name}.phi": float(phi) if diameter > 0.0 else 0.0,
            f"{name}.psi": float(d_p_target / max(rho * omega_safe**2 * diameter**2, 1.0e-30)) if diameter > 0.0 else 0.0,
            f"{name}.efficiency": eta,
            f"{name}.power_target": power_target,
            f"{name}.tau_load_target": tau_target,
            f"{name}.outlet.h_target": h_out_target,
        }


class TurbinePowerContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        turbine_map = component.parameters.get("turbine_map", {})
        if not isinstance(turbine_map, Mapping):
            turbine_map = {}
        scale = _turbine_power_scale(component, model, turbine_map)
        power_design = float(turbine_map.get("power_design", component.parameters.get("power_design", scale)))
        eta_design = float(turbine_map.get("eta_design", component.parameters.get("efficiency", 0.7)))
        pr_design = float(turbine_map.get("PR_design", component.parameters.get("pressure_ratio", 2.0)))
        omega_design = _turbine_design_omega(component, turbine_map, 1000.0)
        return [
            NetworkVariable(f"{component.name}.power", units="W", scale=scale, initial=power_design, owner=component.name),
            NetworkVariable(f"{component.name}.outlet.h", units="J/kg", scale=1.0e6, initial=float(component.parameters.get("initial_outlet_h", 0.0)), owner=component.name),
            NetworkVariable(f"{component.name}.tau_drive", units="N*m", scale=100.0, initial=power_design / omega_design, owner=component.name),
            NetworkVariable(f"{component.name}.efficiency", scale=1.0, initial=eta_design, owner=component.name),
            NetworkVariable(f"{component.name}.pressure_ratio", scale=5.0, initial=pr_design, owner=component.name, lower=1.0e-6),
        ]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        turbine_map = component.parameters.get("turbine_map", {})
        if not isinstance(turbine_map, Mapping):
            turbine_map = {}
        scale = _turbine_power_scale(component, model, turbine_map)
        return [
            NetworkResidual(f"{component.name}.power_residual", units="W", scale=scale, owner=component.name),
            NetworkResidual(f"{component.name}.outlet_h_residual", units="J/kg", scale=1.0e6, owner=component.name),
            NetworkResidual(f"{component.name}.tau_drive_residual", units="N*m", scale=100.0, owner=component.name),
            NetworkResidual(f"{component.name}.efficiency_residual", scale=1.0, owner=component.name),
            NetworkResidual(f"{component.name}.pressure_ratio_residual", scale=5.0, owner=component.name),
        ]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        mdot = abs(context.value(f"{name}.inlet.mdot"))
        mdot_design = max(float(_nested_param(component, "turbine_map", "mdot_design", component.parameters.get("mdot_design", mdot if mdot else 1.0))), 1.0e-12)
        p_in = context.value(f"{name}.inlet.P", 0.0)
        p_out = context.value(f"{name}.outlet.P", 0.0)
        pressure_ratio_target = p_in / max(p_out, 1.0) if p_in > 0.0 and p_out > 0.0 else float(_nested_param(component, "turbine_map", "PR_design", 1.0))
        pressure_ratio = max(pressure_ratio_target, 1.0e-6)
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
            gamma = context.value(f"{name}.inlet.gamma", component.parameters.get("gamma", 1.3))
            h_in = context.value(f"{name}.inlet.h", 0.0)
            delta_h_is = h_in * max(1.0 - pressure_ratio ** (-(gamma - 1.0) / max(gamma, 1.0e-12)), 0.0)
            power_target = mdot * eta * delta_h_is
        h_in = context.value(f"{name}.inlet.h", 0.0)
        h_out_target = h_in - power_target / max(mdot, 1.0e-12)
        omega = abs(context.value(f"{name}.shaft.omega", context.value(f"{name}.omega", 1.0)))
        tau_target = power_target / max(omega, 1.0)
        return {
            f"{name}.power_residual": context.value(f"{name}.power") - power_target,
            f"{name}.outlet_h_residual": context.value(f"{name}.outlet.h") - h_out_target,
            f"{name}.tau_drive_residual": context.value(f"{name}.tau_drive") - tau_target,
            f"{name}.efficiency_residual": context.value(f"{name}.efficiency") - eta,
            f"{name}.pressure_ratio_residual": context.value(f"{name}.pressure_ratio") - pressure_ratio_target,
            f"{name}.power_target": power_target,
            f"{name}.tau_drive_target": tau_target,
            f"{name}.outlet.h_target": h_out_target,
            f"{name}.efficiency": eta,
            f"{name}.pressure_ratio_target": pressure_ratio_target,
        }


class RotorTorqueBalanceContract:
    def variables(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkVariable]:
        _ = model
        initial = float(component.parameters.get("initial_omega", 0.0))
        if "initial_speed_rpm" in component.parameters:
            initial = float(component.parameters["initial_speed_rpm"]) * 3.141592653589793 / 30.0
        return [NetworkVariable(f"{component.name}.omega", units="rad/s", scale=1000.0, initial=initial, owner=component.name)]

    def residuals(self, component: ComponentConfig, model: Mapping[str, Any]) -> list[NetworkResidual]:
        _ = model
        return [NetworkResidual(f"{component.name}.torque_balance", units="N*m", scale=100.0, owner=component.name)]

    def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
        name = component.name
        tau_drive = context.value(f"{name}.tau_drive")
        tau_load = context.value(f"{name}.tau_load")
        friction = float(component.parameters.get("friction_coeff", component.parameters.get("friction", 0.0)))
        omega = context.value(f"{name}.omega", float(component.parameters.get("initial_omega", 0.0)))
        return {f"{name}.torque_balance": tau_drive - tau_load - friction * omega}


def residual_contract_for_type(type_name: str) -> ComponentResidualContract | None:
    return {
        "BoundarySource": BoundarySourceContract(),
        "BoundarySink": BoundarySinkContract(),
        "Pipe": PipeMomentumContract(),
        "Valve": ValveFlowContract(),
        "MassFlowInjector": InjectorPressureDropContract(),
        "FlowSplitter": FlowSplitterContract(),
        "CombustionChamber": FiniteVolumeCombustorContract(),
        "Preburner": FiniteVolumeCombustorContract(),
        "GasGenerator": FiniteVolumeCombustorContract(),
        "Nozzle": NozzleConductanceContract(),
        "Pump": PumpHeadContract(),
        "Turbine": TurbinePowerContract(),
        "Rotor": RotorTorqueBalanceContract(),
        "RegenChannel": RegenThermalContract(),
        "GasVolume": GasVolumeContract(),
        "Volume": GasVolumeContract(),
        "OrificeCompressible": ValveFlowContract(),
    }.get(type_name)


def _boundary_property(
    component: ComponentConfig,
    context: ResidualEvaluationContext,
    prop: str,
    default: float,
) -> float:
    name = component.name
    for key in (
        f"{name}.outlet.{prop}",
        f"{name}.inlet.{prop}",
        f"{name}.{prop}",
        f"{name}.boundary.{prop}",
        f"{name}.ambient.{prop}",
    ):
        boundary_key = f"boundaries.{key}"
        if boundary_key in context.inputs:
            return context.value(boundary_key, default)
    if prop in component.parameters:
        return float(component.parameters[prop])
    inlet_key = f"inlet_{prop}"
    outlet_key = f"outlet_{prop}"
    if inlet_key in component.parameters:
        return float(component.parameters[inlet_key])
    if outlet_key in component.parameters:
        return float(component.parameters[outlet_key])
    return float(default)


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


def _pipe_friction_dp(component: ComponentConfig, model: Mapping[str, Any], rho: float, mdot: float) -> float:
    conductance = _pipe_conductance(component, model, rho)
    if conductance > 0.0 and ("conductance" in component.parameters or f"{component.name}_conductance" in model):
        return (abs(mdot) / max(conductance, 1.0e-30)) ** 2 * (1.0 if mdot >= 0.0 else -1.0)
    diameter = float(component.parameters.get("diameter", 0.0))
    if diameter <= 0.0:
        return (abs(mdot) / max(conductance, 1.0e-30)) ** 2 * (1.0 if mdot >= 0.0 else -1.0)
    length = max(float(component.parameters.get("length", diameter)), 1.0e-12)
    friction = max(float(component.parameters.get("friction_factor", 0.02)), 1.0e-12)
    area = 3.141592653589793 * diameter * diameter / 4.0
    velocity = mdot / max(rho * area, 1.0e-30)
    return friction * length / diameter * rho * velocity * abs(velocity) / 2.0


def _compressible_orifice_mdot(
    cda: float,
    p_in: float,
    p_out: float,
    *,
    gamma: float,
    r_gas: float,
    temperature: float,
) -> tuple[float, float]:
    if cda <= 0.0 or p_in <= 0.0 or temperature <= 0.0:
        return 0.0, 0.0
    gamma = max(float(gamma), 1.0001)
    r_gas = max(float(r_gas), 1.0e-12)
    ratio = max(min(p_out / max(p_in, 1.0e-12), 1.0), 0.0)
    critical = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    common = cda * p_in * (gamma / (r_gas * temperature)) ** 0.5
    if ratio <= critical:
        choked_coeff = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return common * choked_coeff, 1.0
    term = ratio ** (2.0 / gamma) - ratio ** ((gamma + 1.0) / gamma)
    return common * (2.0 / (gamma - 1.0) * max(term, 0.0)) ** 0.5, 0.0


def _regen_heat_targets(
    component: ComponentConfig,
    context: ResidualEvaluationContext,
    t_wall: float,
) -> tuple[float, float, float]:
    """Return ``(Q_hot, Q_cool, T_wall_eq)`` for a regenerative channel."""

    name = component.name
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
    ntu = h_cool * a_cool / max(mdot * max(cp_cool, 1.0), 1.0e-12)
    effectiveness = 1.0 - float(np.exp(-max(ntu, 0.0)))
    g_cool = effectiveness * mdot * max(cp_cool, 1.0)
    q_cool = g_cool * (t_wall - t_cool_in)
    g_hot = h_hot * a_hot
    g_ref = max(g_hot + g_cool, 1.0)
    t_wall_eq = (g_hot * t_aw + g_cool * t_cool_in) / g_ref
    # If geometry is unspecified, fall back to an explicit wall-temperature anchor.
    if a_hot <= 0.0 and a_cool <= 0.0:
        t_wall_eq = float(component.parameters.get("initial_T_wall", component.parameters.get("T_wall", t_wall)))
        q_hot = float(component.parameters.get("Q_hot", 0.0))
        q_cool = float(component.parameters.get("Q_cool", 0.0))
    return float(q_hot), float(q_cool), float(t_wall_eq)


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
    aliases = {
        "dP_design": ("design_delta_P", "delta_P_design", "pressure_rise_design"),
        "eta_design": ("efficiency_design",),
        "PR_design": ("pressure_ratio_design", "pressure_ratio"),
    }
    for alias in aliases.get(key, ()):
        if alias in component.parameters:
            return component.parameters[alias]
    return default


def _design_omega(component: ComponentConfig, default: float) -> float:
    raw = _nested_param(component, "pump_map", "speed_design", component.parameters.get("omega_design", default))
    raw = float(raw)
    if raw > 1000.0:
        return raw * 2.0 * 3.141592653589793 / 60.0
    return raw


def _turbine_design_omega(component: ComponentConfig, turbine_map: Mapping[str, Any], default: float) -> float:
    raw = float(turbine_map.get("speed_design", component.parameters.get("omega_design", default)))
    if raw > 1000.0:
        return raw * 2.0 * 3.141592653589793 / 60.0
    return max(raw, 1.0)


def _turbine_power_scale(component: ComponentConfig, model: Mapping[str, Any], turbine_map: Mapping[str, Any]) -> float:
    configured = model.get(
        f"{component.name}_power_scale",
        turbine_map.get("power_design", component.parameters.get("power_design", component.parameters.get("power_scale", 1.0e6))),
    )
    return max(float(configured), 1.0)


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
