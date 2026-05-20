from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from atha.components.registry import component_residual_contract, component_spec
from atha.components.residuals import ResidualEvaluationContext
from atha.config import build_performance_maps, evaluate_boundary_conditions
from atha.config.loader import LoadedAnalysisConfig
from atha.config.schema import ComponentConfig, ConnectionConfig
from atha.network.problem import NetworkProblem, NetworkResidual, NetworkVariable


FLUID_VARIABLES = ("P", "mdot", "h", "T", "rho", "gamma")
SHAFT_VARIABLES = ("omega", "tau")
THERMAL_VARIABLES = ("T", "Q_dot")


@dataclass(frozen=True)
class Port:
    component: str
    name: str
    domain: str

    @property
    def path(self) -> str:
        return f"{self.component}.{self.name}"


@dataclass
class PortNetworkCatalog:
    ports: dict[str, Port] = field(default_factory=dict)
    variables: list[NetworkVariable] = field(default_factory=list)
    residuals: list[NetworkResidual] = field(default_factory=list)

    @property
    def variable_names(self) -> list[str]:
        return [variable.name for variable in self.variables]

    @property
    def residual_names(self) -> list[str]:
        return [residual.name for residual in self.residuals]

    def source_paths(self) -> set[str]:
        paths = set(self.ports)
        paths.update(self.variable_names)
        paths.update(self.residual_names)
        paths.update(f"residuals.{name}" for name in self.residual_names)
        return paths


class PortNetworkBuilder:
    """Build a ROCETS-like algebraic port network from YAML connections.

    This builder is intentionally topology-first. It creates fluid, shaft, and
    thermal port unknowns from `engine.yaml`, adds connection residuals, anchors
    variables to boundary values when provided, and calls registered component
    residual contracts. It is the generic foundation needed before replacing
    reduced-order compatibility analyses with a universal transient DAE loop.
    """

    def __init__(self, loaded: LoadedAnalysisConfig) -> None:
        self.loaded = loaded
        self._components = loaded.engine.components
        self._runtime_maps = build_performance_maps(loaded.maps) if loaded.maps else {}

    def catalog(self) -> PortNetworkCatalog:
        variables: dict[str, NetworkVariable] = {}
        residuals: dict[str, NetworkResidual] = {}
        ports: dict[str, Port] = {}
        for connection in self.loaded.engine.connections:
            for endpoint in (connection.source, connection.target):
                component_name, port_name = _split_endpoint(endpoint)
                port = Port(component=component_name, name=port_name, domain=connection.domain)
                ports[port.path] = port
                for variable in self._port_variables(port):
                    variables.setdefault(variable.name, variable)
        for component in self._components.values():
            contract = component_residual_contract(component)
            if contract is None or component.type == "Rotor":
                continue
            model = self._component_model(component)
            for variable in contract.variables(component, model):
                variables.setdefault(variable.name, variable)
            for residual in contract.residuals(component, model):
                residuals.setdefault(residual.name, residual)
        for connection in self.loaded.engine.connections:
            for residual in self._connection_residuals(connection):
                residuals.setdefault(residual.name, residual)
        for residual in self._component_coupling_residuals(variables):
            residuals.setdefault(residual.name, residual)
        for variable in list(variables.values()):
            if self._has_boundary(variable.name):
                residuals.setdefault(
                    f"{variable.name}_boundary_residual",
                    NetworkResidual(
                        f"{variable.name}_boundary_residual",
                        units=variable.units,
                        scale=variable.scale,
                        owner=variable.owner,
                        description="Boundary condition anchor",
                    ),
                )
        self._propagate_initial_values(variables)
        return PortNetworkCatalog(ports=ports, variables=list(variables.values()), residuals=list(residuals.values()))

    def build_problem(self, *, require_square: bool = False) -> NetworkProblem:
        catalog = self.catalog()
        components_with_contracts = [
            (component, component_residual_contract(component), self._component_model(component))
            for component in self._components.values()
            if component_residual_contract(component) is not None and component.type != "Rotor"
        ]

        def evaluate(_t: float, z: Mapping[str, float], inputs: Mapping[str, Any]) -> dict[str, float]:
            values: dict[str, float] = dict(inputs.get("inputs", {})) if isinstance(inputs.get("inputs", {}), dict) else {}
            values.update({key: float(value) for key, value in z.items()})
            for key, value in self._boundary_values(_t).items():
                values.setdefault(f"boundaries.{key}", value)
            result: dict[str, float] = {}
            for component, contract, model in components_with_contracts:
                if contract is None:
                    continue
                evaluated = contract.evaluate(component, ResidualEvaluationContext(z=z, inputs=values, model=model))
                values.update({key: value for key, value in evaluated.items() if key not in catalog.residual_names})
                result.update({key: value for key, value in evaluated.items() if key in catalog.residual_names})
            for connection in self.loaded.engine.connections:
                result.update(_connection_residual_values(connection, values))
            result.update(_component_coupling_residual_values(self._components, self.loaded.engine.connections, values))
            for variable in catalog.variables:
                boundary = _boundary_value(variable.name, values)
                if boundary is not None:
                    result[f"{variable.name}_boundary_residual"] = values[variable.name] - boundary
            return result

        return NetworkProblem(
            catalog.variables,
            catalog.residuals,
            evaluate,
            name=f"{self.loaded.engine.name}_port_network",
            strict_residuals=True,
            require_square=require_square,
        )

    def _port_variables(self, port: Port) -> list[NetworkVariable]:
        if port.domain == "fluid":
            return [
                NetworkVariable(f"{port.path}.P", units="Pa", scale=1.0e6, initial=self._initial_port_value(port, "P", 1.0e5), owner=port.component, lower=1.0),
                NetworkVariable(f"{port.path}.mdot", units="kg/s", scale=1.0, initial=self._initial_port_value(port, "mdot", 0.0), owner=port.component),
                NetworkVariable(f"{port.path}.h", units="J/kg", scale=1.0e6, initial=self._initial_port_value(port, "h", self._initial_port_value(port, "T", 300.0) * 2000.0), owner=port.component),
                NetworkVariable(f"{port.path}.T", units="K", scale=1000.0, initial=self._initial_port_value(port, "T", 300.0), owner=port.component, lower=1.0),
                NetworkVariable(f"{port.path}.rho", units="kg/m^3", scale=1000.0, initial=self._initial_port_value(port, "rho", 1.0), owner=port.component, lower=1.0e-9),
                NetworkVariable(f"{port.path}.gamma", scale=1.0, initial=self._initial_port_value(port, "gamma", 1.2), owner=port.component, lower=1.0001),
            ]
        if port.domain == "shaft":
            return [
                NetworkVariable(f"{port.path}.omega", units="rad/s", scale=1000.0, initial=self._initial_port_value(port, "omega", 0.0), owner=port.component),
                NetworkVariable(f"{port.path}.tau", units="N*m", scale=100.0, owner=port.component),
            ]
        if port.domain == "thermal":
            return [
                NetworkVariable(f"{port.path}.T", units="K", scale=1000.0, initial=self._initial_port_value(port, "T", 300.0), owner=port.component),
                NetworkVariable(f"{port.path}.Q_dot", units="W", scale=1.0e5, owner=port.component),
            ]
        return []

    def _initial_port_value(self, port: Port, prop: str, default: float) -> float:
        boundary = self._boundary_values(0.0).get(f"{port.path}.{prop}")
        if boundary is not None:
            return float(boundary)
        component = self._components.get(port.component)
        if component is None:
            return float(default)
        params = component.parameters
        if prop == "P" and "initial_P" in params:
            return float(params["initial_P"])
        if prop == "T":
            if "initial_T" in params:
                return float(params["initial_T"])
            if "T_adiabatic" in params:
                return float(params["T_adiabatic"])
        if prop == "h":
            if "initial_h" in params:
                return float(params["initial_h"])
            if "initial_T" in params or "T_adiabatic" in params:
                temperature = float(params.get("initial_T", params.get("T_adiabatic", 300.0)))
                cp = float(params.get("cp", params.get("gas_cp", 3500.0)))
                return cp * temperature
        if prop == "rho" and ("initial_rho" in params or "initial_P" in params):
            if "initial_rho" in params:
                return float(params["initial_rho"])
            pressure = float(params.get("initial_P", 101325.0))
            temperature = float(params.get("initial_T", params.get("T_adiabatic", 300.0)))
            gas_r = float(params.get("gas_R", params.get("R", 355.0)))
            return pressure / max(gas_r * temperature, 1.0e-12)
        if prop == "gamma":
            if "gamma" in params:
                return float(params["gamma"])
            if "gas_gamma" in params:
                return float(params["gas_gamma"])
        if prop == "omega":
            if "initial_omega" in params:
                return float(params["initial_omega"])
            if "initial_speed_rpm" in params:
                return float(params["initial_speed_rpm"]) * 3.141592653589793 / 30.0
            pump_map = params.get("pump_map", {})
            if isinstance(pump_map, Mapping) and "speed_design" in pump_map:
                speed = float(pump_map["speed_design"])
                return speed * 3.141592653589793 / 30.0 if speed > 1000.0 else speed
            turbine_map = params.get("turbine_map", {})
            if isinstance(turbine_map, Mapping) and "speed_design" in turbine_map:
                speed = float(turbine_map["speed_design"])
                return speed * 3.141592653589793 / 30.0 if speed > 1000.0 else speed
        if prop == "mdot":
            if "initial_mdot" in params:
                return float(params["initial_mdot"])
            if "mdot_design" in params:
                return float(params["mdot_design"])
            pump_map = params.get("pump_map", {})
            if isinstance(pump_map, Mapping) and "mdot_design" in pump_map:
                return float(pump_map["mdot_design"])
            turbine_map = params.get("turbine_map", {})
            if isinstance(turbine_map, Mapping):
                return float(turbine_map.get("mdot_design", turbine_map.get("mdot_corrected_design", 0.0)))
        if prop == "tau":
            omega = self._initial_port_value(port, "omega", 1000.0)
            if component.type == "Pump":
                pump_map = params.get("pump_map", {})
                if isinstance(pump_map, Mapping):
                    mdot = float(pump_map.get("mdot_design", params.get("mdot_design", 0.0)))
                    d_p = float(pump_map.get("dP_design", params.get("delta_P_design", 0.0)))
                    rho = float(pump_map.get("rho_design", params.get("rho_design", 1000.0)))
                    eta = float(pump_map.get("efficiency_design", params.get("efficiency_design", 0.74)))
                    return mdot * max(d_p, 0.0) / max(rho * eta * max(abs(omega), 1.0), 1.0e-12)
            if component.type == "Turbine":
                turbine_map = params.get("turbine_map", {})
                power = float(turbine_map.get("power_design", params.get("power_design", 0.0))) if isinstance(turbine_map, Mapping) else float(params.get("power_design", 0.0))
                return -power / max(abs(omega), 1.0)
        for key in (prop, f"{port.name}_{prop}"):
            if key in params:
                try:
                    return float(params[key])
                except (TypeError, ValueError):
                    pass
        if prop == "h" and "T" in params:
            cp = float(params.get("cp", params.get("specific_heat", 2000.0)))
            return cp * float(params["T"])
        return float(default)

    def _connection_residuals(self, connection: ConnectionConfig) -> list[NetworkResidual]:
        base = _connection_name(connection)
        if connection.domain == "fluid":
            return [
                NetworkResidual(f"{base}.P_continuity", units="Pa", scale=1.0e6),
                NetworkResidual(f"{base}.h_continuity", units="J/kg", scale=1.0e6),
                NetworkResidual(f"{base}.mdot_continuity", units="kg/s", scale=1.0),
                NetworkResidual(f"{base}.T_continuity", units="K", scale=1000.0),
                NetworkResidual(f"{base}.rho_continuity", units="kg/m^3", scale=1000.0),
                NetworkResidual(f"{base}.gamma_continuity", scale=1.0),
            ]
        if connection.domain == "shaft":
            return [
                NetworkResidual(f"{base}.omega_continuity", units="rad/s", scale=1000.0),
                NetworkResidual(f"{base}.torque_balance", units="N*m", scale=100.0),
            ]
        if connection.domain == "thermal":
            return [
                NetworkResidual(f"{base}.T_continuity", units="K", scale=1000.0),
                NetworkResidual(f"{base}.heat_balance", units="W", scale=1.0e5),
            ]
        return []

    def _component_coupling_residuals(self, variables: Mapping[str, NetworkVariable]) -> list[NetworkResidual]:
        residuals: list[NetworkResidual] = []
        for component in self._components.values():
            variable_names = set(variables)
            if (
                f"{component.name}.delta_P" in variable_names
                and f"{component.name}.outlet.P" in variable_names
            ):
                residuals.append(NetworkResidual(f"{component.name}.delta_P_port_residual", units="Pa", scale=1.0e6, owner=component.name))
            if f"{component.name}.mdot" in variable_names and component.type not in {"CombustionChamber", "Preburner"}:
                for port in _fluid_ports(component):
                    port_mdot = f"{component.name}.{port}.mdot"
                    if port_mdot in variable_names:
                        residuals.append(NetworkResidual(f"{port_mdot}_link_residual", units="kg/s", scale=1.0, owner=component.name))
            if f"{component.name}.omega" in variable_names:
                for port in _shaft_ports(component):
                    port_omega = f"{component.name}.{port}.omega"
                    if port_omega in variable_names:
                        residuals.append(NetworkResidual(f"{port_omega}_link_residual", units="rad/s", scale=1000.0, owner=component.name))
            if component.type == "Rotor":
                for port_omega in sorted(_component_variable_paths(variable_names, component.name, "omega")):
                    residuals.append(NetworkResidual(f"{port_omega}_state_link_residual", units="rad/s", scale=1000.0, owner=component.name))
                if _connected_shaft_components(component.name, self.loaded.engine.connections):
                    residuals.append(NetworkResidual(f"{component.name}.shaft_torque_balance_residual", units="N*m", scale=1000.0, owner=component.name))
                    residuals.append(NetworkResidual(f"{component.name}.shaft_power_balance_residual", units="W", scale=1.0e6, owner=component.name))
            if component.type == "Pump" and f"{component.name}.shaft.tau" in variable_names:
                residuals.append(NetworkResidual(f"{component.name}.shaft.tau_load_residual", units="N*m", scale=100.0, owner=component.name))
            if component.type == "Turbine" and f"{component.name}.shaft.tau" in variable_names:
                residuals.append(NetworkResidual(f"{component.name}.shaft.tau_drive_residual", units="N*m", scale=100.0, owner=component.name))
            if component.type in {"CombustionChamber", "Preburner"} and f"{component.name}.outlet.h" in variable_names and f"{component.name}.h" in variable_names:
                residuals.append(NetworkResidual(f"{component.name}.outlet.h_link_residual", units="J/kg", scale=1.0e6, owner=component.name))
            if component.type in {"CombustionChamber", "Preburner", "GasGenerator"}:
                if f"{component.name}.outlet.mdot" in variable_names and f"{component.name}.mdot" in variable_names:
                    residuals.append(NetworkResidual(f"{component.name}.outlet.mdot_link_residual", units="kg/s", scale=1.0, owner=component.name))
            for residual in _port_property_link_residuals(component, variable_names):
                residuals.append(residual)
        return residuals

    def _has_boundary(self, path: str) -> bool:
        if self.loaded.boundary_conditions is None:
            return False
        return path in evaluate_boundary_conditions(self.loaded.boundary_conditions, 0.0)

    def _component_model(self, component: ComponentConfig) -> dict[str, Any]:
        model = _component_model(component)
        for slot, binding in component.maps.items():
            if binding.ref in self._runtime_maps:
                model[f"{component.name}.map.{slot}"] = self._runtime_maps[binding.ref]
                if binding.output:
                    model[f"{component.name}.map.{slot}.output"] = binding.output
        return model

    def _boundary_values(self, t: float) -> dict[str, float]:
        if self.loaded.boundary_conditions is None:
            return {}
        values: dict[str, float] = {}
        for path, value in evaluate_boundary_conditions(self.loaded.boundary_conditions, t).items():
            if isinstance(value, (int, float)):
                values[path] = float(value)
            elif isinstance(value, dict) and "value" in value and isinstance(value["value"], (int, float)):
                values[path] = float(value["value"])
        return values

    def _propagate_initial_values(self, variables: dict[str, NetworkVariable]) -> None:
        for _ in range(max(len(self.loaded.engine.connections), 1)):
            changed = False
            changed |= self._propagate_component_initial_values(variables)
            for connection in self.loaded.engine.connections:
                if connection.domain == "fluid":
                    properties = FLUID_VARIABLES
                elif connection.domain == "shaft":
                    properties = SHAFT_VARIABLES
                elif connection.domain == "thermal":
                    properties = THERMAL_VARIABLES
                else:
                    continue
                for prop in properties:
                    source = f"{connection.source}.{prop}"
                    target = f"{connection.target}.{prop}"
                    if source in variables and target in variables:
                        changed |= _copy_initial_if_default(variables, source, target)
                        changed |= _copy_initial_if_default(variables, target, source)
            if not changed:
                break

    def _propagate_component_initial_values(self, variables: dict[str, NetworkVariable]) -> bool:
        changed = False
        for component in self._components.values():
            name = component.name
            params = component.parameters
            inputs = _input_fluid_ports(component)
            outputs = _output_fluid_ports(component)
            if component.type == "Pump" and f"{name}.inlet.P" in variables and f"{name}.outlet.P" in variables:
                pump_map = params.get("pump_map", {})
                d_p = float(pump_map.get("dP_design", params.get("delta_P_design", 0.0))) if isinstance(pump_map, Mapping) else float(params.get("delta_P_design", 0.0))
                changed |= _set_initial(variables, f"{name}.outlet.P", max(variables[f"{name}.inlet.P"].initial + d_p, 1.0))
            elif component.type == "MassFlowInjector" and f"{name}.inlet.P" in variables and f"{name}.outlet.P" in variables:
                d_p = float(params.get("delta_P_nominal", 0.0))
                changed |= _set_initial(variables, f"{name}.outlet.P", max(variables[f"{name}.inlet.P"].initial - d_p, 1.0))
            elif component.type in {"Pipe", "Valve", "OrificeCompressible", "Turbine"} and inputs and outputs:
                for output in outputs:
                    for prop in ("P", "h", "T", "rho", "gamma", "mdot"):
                        source = f"{name}.{inputs[0]}.{prop}"
                        target = f"{name}.{output}.{prop}"
                        if source in variables and target in variables:
                            changed |= _copy_initial_if_default(variables, source, target)
            elif component.type == "FlowSplitter" and inputs:
                for output in outputs:
                    for prop in ("P", "h", "T", "rho", "gamma"):
                        source = f"{name}.{inputs[0]}.{prop}"
                        target = f"{name}.{output}.{prop}"
                        if source in variables and target in variables:
                            changed |= _copy_initial_if_default(variables, source, target)
            elif component.type in {"CombustionChamber", "Preburner", "GasGenerator"}:
                for output in outputs:
                    for prop in ("P", "h", "T", "rho", "gamma", "mdot"):
                        source = f"{name}.{prop}"
                        target = f"{name}.{output}.{prop}"
                        if source in variables and target in variables:
                            changed |= _copy_initial_if_default(variables, source, target)
            if f"{name}.mdot" in variables:
                for port in inputs + outputs:
                    target = f"{name}.{port}.mdot"
                    if target in variables:
                        changed |= _copy_initial_if_default(variables, f"{name}.mdot", target)
            if f"{name}.omega" in variables:
                for port in _shaft_ports(component):
                    target = f"{name}.{port}.omega"
                    if target in variables:
                        changed |= _copy_initial_if_default(variables, f"{name}.omega", target)
        return changed


def _connection_residual_values(connection: ConnectionConfig, values: Mapping[str, float]) -> dict[str, float]:
    base = _connection_name(connection)
    source = connection.source
    target = connection.target
    if connection.domain == "fluid":
        return {
            f"{base}.P_continuity": _value(values, f"{source}.P") - _value(values, f"{target}.P"),
            f"{base}.h_continuity": _value(values, f"{source}.h") - _value(values, f"{target}.h"),
            f"{base}.mdot_continuity": _value(values, f"{source}.mdot") - _value(values, f"{target}.mdot"),
            f"{base}.T_continuity": _value(values, f"{source}.T") - _value(values, f"{target}.T"),
            f"{base}.rho_continuity": _value(values, f"{source}.rho") - _value(values, f"{target}.rho"),
            f"{base}.gamma_continuity": _value(values, f"{source}.gamma") - _value(values, f"{target}.gamma"),
        }
    if connection.domain == "shaft":
        return {
            f"{base}.omega_continuity": _value(values, f"{source}.omega") - _value(values, f"{target}.omega"),
            f"{base}.torque_balance": _value(values, f"{source}.tau") + _value(values, f"{target}.tau"),
        }
    if connection.domain == "thermal":
        return {
            f"{base}.T_continuity": _value(values, f"{source}.T") - _value(values, f"{target}.T"),
            f"{base}.heat_balance": _value(values, f"{source}.Q_dot") + _value(values, f"{target}.Q_dot"),
        }
    return {}


def _component_coupling_residual_values(
    components: Mapping[str, ComponentConfig],
    connections: list[ConnectionConfig],
    values: Mapping[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for component in components.values():
        if f"{component.name}.delta_P" in values:
            inlet = _first_existing(values, [f"{component.name}.inlet.P", f"{component.name}.fuel_inlet.P", f"{component.name}.ox_inlet.P"])
            outlet = _first_existing(values, [f"{component.name}.outlet.P"])
            if outlet is not None:
                result[f"{component.name}.delta_P_port_residual"] = outlet - float(inlet or 0.0) - _value(values, f"{component.name}.delta_P")
        if f"{component.name}.mdot" in values and component.type not in {"CombustionChamber", "Preburner"}:
            for port in _fluid_ports(component):
                port_mdot = f"{component.name}.{port}.mdot"
                if port_mdot in values:
                    result[f"{port_mdot}_link_residual"] = _value(values, port_mdot) - _value(values, f"{component.name}.mdot")
        if component.type != "Rotor" and f"{component.name}.omega" in values:
            for port in _shaft_ports(component):
                port_omega = f"{component.name}.{port}.omega"
                if port_omega in values:
                    result[f"{port_omega}_link_residual"] = _value(values, port_omega) - _value(values, f"{component.name}.omega")
        if component.type == "Rotor":
            for port_omega in sorted(_component_value_paths(values, component.name, "omega")):
                result[f"{port_omega}_state_link_residual"] = _value(values, port_omega) - _value(values, f"{component.name}.omega")
            connected = _connected_shaft_components(component.name, connections)
            if connected:
                omega = max(abs(_value(values, f"{component.name}.omega", _value(values, f"{component.name}.shaft.omega", 0.0))), 1.0)
                drive_torque = sum(_value(values, f"{name}.tau_drive", _value(values, f"{name}.power", 0.0) / omega) for name in connected["turbines"])
                load_torque = sum(_value(values, f"{name}.tau_load", _value(values, f"{name}.power", 0.0) / omega) for name in connected["pumps"])
                drive_power = sum(_value(values, f"{name}.power", 0.0) for name in connected["turbines"])
                load_power = sum(_value(values, f"{name}.power", 0.0) for name in connected["pumps"])
                friction = float(component.parameters.get("friction_coeff", 0.0))
                result[f"{component.name}.shaft_torque_balance_residual"] = drive_torque - load_torque - friction * omega
                result[f"{component.name}.shaft_power_balance_residual"] = drive_power - load_power - friction * omega * omega
        if component.type == "Pump" and f"{component.name}.shaft.tau" in values:
            omega = _value(values, f"{component.name}.shaft.omega", _value(values, f"{component.name}.omega", 0.0))
            result[f"{component.name}.shaft.tau_load_residual"] = _value(values, f"{component.name}.shaft.tau") - _pump_torque(component, values, omega)
        if component.type == "Turbine" and f"{component.name}.shaft.tau" in values:
            omega = _value(values, f"{component.name}.shaft.omega", _value(values, f"{component.name}.omega", 0.0))
            result[f"{component.name}.shaft.tau_drive_residual"] = _value(values, f"{component.name}.shaft.tau") + _turbine_torque(component, values, omega)
        if component.type in {"CombustionChamber", "Preburner"} and f"{component.name}.outlet.h" in values and f"{component.name}.h" in values:
            result[f"{component.name}.outlet.h_link_residual"] = _value(values, f"{component.name}.outlet.h") - _value(values, f"{component.name}.h")
        if component.type in {"CombustionChamber", "Preburner", "GasGenerator"}:
            if f"{component.name}.outlet.mdot" in values and f"{component.name}.mdot" in values:
                result[f"{component.name}.outlet.mdot_link_residual"] = _value(values, f"{component.name}.outlet.mdot") - _value(values, f"{component.name}.mdot")
        result.update(_port_property_link_residual_values(component, values))
    return result


def _component_model(component: ComponentConfig) -> dict[str, Any]:
    params = component.parameters
    model: dict[str, Any] = {}
    if component.type == "Valve":
        if "CdA" in params:
            model[f"{component.name}_CdA"] = float(params["CdA"])
        elif "max_area" in params:
            model[f"{component.name}_CdA"] = float(params["max_area"]) * float(params.get("discharge_coeff", 1.0))
    if component.type == "Nozzle" and "conductance" in params:
        model[f"{component.name}_conductance"] = float(params["conductance"])
        model["nozzle_conductance"] = float(params["conductance"])
    if component.type == "MassFlowInjector" and "delta_P_nominal" in params:
        model[f"{component.name}_delta_P"] = float(params["delta_P_nominal"])
    return {**params, **model}


def _fluid_ports(component: ComponentConfig) -> list[str]:
    return [name for name, domain in component_spec(component.type).ports.items() if domain in {"fluid_in", "fluid_out"}]


def _shaft_ports(component: ComponentConfig) -> list[str]:
    return [name for name, domain in component_spec(component.type).ports.items() if domain.startswith("shaft")]


def _connection_name(connection: ConnectionConfig) -> str:
    return f"connection.{connection.source.replace('.', '_')}__{connection.target.replace('.', '_')}"


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    component, port = endpoint.split(".", 1)
    return component, port


def _value(values: Mapping[str, float], path: str, default: float = 0.0) -> float:
    return float(values.get(path, default))


def _copy_initial_if_default(variables: dict[str, NetworkVariable], source: str, target: str) -> bool:
    source_var = variables[source]
    target_var = variables[target]
    if _is_default_initial(target_var) and not _is_default_initial(source_var):
        variables[target] = NetworkVariable(
            target_var.name,
            units=target_var.units,
            scale=target_var.scale,
            initial=source_var.initial,
            owner=target_var.owner,
            description=target_var.description,
            lower=target_var.lower,
            upper=target_var.upper,
        )
        return True
    return False


def _set_initial(variables: dict[str, NetworkVariable], name: str, value: float) -> bool:
    variable = variables[name]
    if abs(float(variable.initial) - float(value)) <= max(abs(float(value)), 1.0) * 1.0e-12:
        return False
    variables[name] = NetworkVariable(
        variable.name,
        units=variable.units,
        scale=variable.scale,
        initial=float(value),
        owner=variable.owner,
        description=variable.description,
        lower=variable.lower,
        upper=variable.upper,
    )
    return True


def _is_default_initial(variable: NetworkVariable) -> bool:
    defaults = {
        "Pa": 1.0e5,
        "kg/s": 0.0,
        "J/kg": 6.0e5,
        "K": 300.0,
        "kg/m^3": 1.0,
        "rad/s": 0.0,
        "N*m": 0.0,
        "W": 0.0,
        "": 1.2 if variable.name.endswith(".gamma") else 0.0,
    }
    default = defaults.get(variable.units, 0.0)
    return abs(float(variable.initial) - float(default)) <= max(abs(default), 1.0) * 1.0e-12


def _first_existing(values: Mapping[str, float], paths: list[str]) -> float | None:
    for path in paths:
        if path in values:
            return float(values[path])
    return None


def _first_existing_name(names: set[str], paths: list[str]) -> str | None:
    for path in paths:
        if path in names:
            return path
    return None

    def _component_model(self, component: ComponentConfig) -> dict[str, Any]:
        model = _component_model(component)
        for slot, binding in component.maps.items():
            if binding.ref in self._runtime_maps:
                model[f"{component.name}.map.{slot}"] = self._runtime_maps[binding.ref]
                if binding.output:
                    model[f"{component.name}.map.{slot}.output"] = binding.output
        return model


def _boundary_value(path: str, values: Mapping[str, Any]) -> float | None:
    boundary_path = f"boundaries.{path}"
    if boundary_path in values:
        value = values[boundary_path]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _component_variable_paths(variable_names: set[str], component: str, leaf: str) -> set[str]:
    prefix = f"{component}."
    suffix = f".{leaf}"
    return {name for name in variable_names if name.startswith(prefix) and name.endswith(suffix) and name != f"{component}.{leaf}"}


def _component_value_paths(values: Mapping[str, float], component: str, leaf: str) -> set[str]:
    prefix = f"{component}."
    suffix = f".{leaf}"
    return {name for name in values if name.startswith(prefix) and name.endswith(suffix) and name != f"{component}.{leaf}"}


def _pump_torque(component: ComponentConfig, values: Mapping[str, float], omega: float) -> float:
    name = component.name
    if f"{name}.tau_load" in values:
        return max(_value(values, f"{name}.tau_load"), 0.0)
    mdot = abs(_first_value(values, (f"{name}.mdot", f"{name}.inlet.mdot", f"{name}.outlet.mdot"), _pump_design(component, "mdot_design", 0.0)))
    delta_p = max(_first_value(values, (f"{name}.delta_P", f"{name}.pressure_rise"), _pump_design(component, "dP_design", 0.0)), 0.0)
    eta = max(min(_first_value(values, (f"{name}.efficiency",), _pump_design(component, "efficiency_design", 0.7)), 1.0), 1.0e-6)
    return mdot * delta_p / max(eta * max(abs(omega), 1.0), 1.0e-12)


def _turbine_torque(component: ComponentConfig, values: Mapping[str, float], omega: float) -> float:
    name = component.name
    if f"{name}.tau_drive" in values:
        return max(_value(values, f"{name}.tau_drive"), 0.0)
    return max(_value(values, f"{name}.power"), 0.0) / max(abs(omega), 1.0)


def _first_value(values: Mapping[str, float], paths: tuple[str, ...], default: float) -> float:
    for path in paths:
        if path in values:
            return float(values[path])
    return float(default)


def _pump_design(component: ComponentConfig, key: str, default: float) -> float:
    pump_map = component.parameters.get("pump_map", {})
    if isinstance(pump_map, Mapping) and key in pump_map:
        return float(pump_map[key])
    return float(component.parameters.get(key, default))


def _connected_shaft_components(rotor: str, connections: list[ConnectionConfig]) -> dict[str, list[str]]:
    connected = {"pumps": [], "turbines": []}
    for connection in connections:
        if connection.domain != "shaft":
            continue
        endpoints = (connection.source, connection.target)
        rotor_endpoints = [endpoint for endpoint in endpoints if endpoint.startswith(f"{rotor}.")]
        if not rotor_endpoints:
            continue
        other = endpoints[1] if endpoints[0] == rotor_endpoints[0] else endpoints[0]
        component_name, _port = _split_endpoint(other)
        if component_name.endswith("pump") or "pump" in component_name:
            connected["pumps"].append(component_name)
        elif component_name.endswith("turbine") or "turbine" in component_name:
            connected["turbines"].append(component_name)
    return connected if connected["pumps"] or connected["turbines"] else {}


def _port_property_link_residuals(component: ComponentConfig, variable_names: set[str]) -> list[NetworkResidual]:
    residuals: list[NetworkResidual] = []
    inputs = _input_fluid_ports(component)
    outputs = _output_fluid_ports(component)
    if not outputs:
        return residuals
    finite_volume = component.type in {"CombustionChamber", "Preburner", "GasGenerator"}
    passthrough_h = component.type in {"Pipe", "Valve", "MassFlowInjector", "OrificeCompressible"}
    passthrough_props = component.type in {
        "Pipe",
        "Valve",
        "MassFlowInjector",
        "Pump",
        "Turbine",
        "OrificeCompressible",
    }
    splitter_props = component.type == "FlowSplitter"
    for output in outputs:
        output_prefix = f"{component.name}.{output}"
        if finite_volume:
            for prop, units, scale in [("P", "Pa", 1.0e6), *_fluid_property_specs(include_h=False)]:
                if f"{output_prefix}.{prop}" in variable_names and f"{component.name}.{prop}" in variable_names:
                    residuals.append(NetworkResidual(f"{output_prefix}.{prop}_state_link_residual", units=units, scale=scale, owner=component.name))
            continue
        if (passthrough_props or splitter_props) and inputs:
            property_specs = _fluid_property_specs(include_h=passthrough_h)
            if splitter_props:
                property_specs = [("P", "Pa", 1.0e6), *property_specs]
            for prop, units, scale in property_specs:
                if f"{output_prefix}.{prop}" not in variable_names:
                    continue
                source = _first_existing_name(variable_names, [f"{component.name}.{port}.{prop}" for port in inputs])
                if source is not None:
                    residuals.append(NetworkResidual(f"{output_prefix}.{prop}_passthrough_residual", units=units, scale=scale, owner=component.name))
    return residuals


def _port_property_link_residual_values(component: ComponentConfig, values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    inputs = _input_fluid_ports(component)
    outputs = _output_fluid_ports(component)
    if not outputs:
        return result
    finite_volume = component.type in {"CombustionChamber", "Preburner", "GasGenerator"}
    passthrough_h = component.type in {"Pipe", "Valve", "MassFlowInjector", "OrificeCompressible"}
    passthrough_props = component.type in {
        "Pipe",
        "Valve",
        "MassFlowInjector",
        "Pump",
        "Turbine",
        "OrificeCompressible",
    }
    splitter_props = component.type == "FlowSplitter"
    value_names = set(values)
    for output in outputs:
        output_prefix = f"{component.name}.{output}"
        if finite_volume:
            for prop, _units, _scale in [("P", "Pa", 1.0e6), *_fluid_property_specs(include_h=False)]:
                output_path = f"{output_prefix}.{prop}"
                state_path = f"{component.name}.{prop}"
                if output_path in values and state_path in values:
                    result[f"{output_path}_state_link_residual"] = _value(values, output_path) - _value(values, state_path)
            continue
        if (passthrough_props or splitter_props) and inputs:
            property_specs = _fluid_property_specs(include_h=passthrough_h)
            if splitter_props:
                property_specs = [("P", "Pa", 1.0e6), *property_specs]
            for prop, _units, _scale in property_specs:
                output_path = f"{output_prefix}.{prop}"
                if output_path not in values:
                    continue
                source = _first_existing_name(value_names, [f"{component.name}.{port}.{prop}" for port in inputs])
                if source is not None:
                    result[f"{output_path}_passthrough_residual"] = _value(values, output_path) - _value(values, source)
    return result


def _input_fluid_ports(component: ComponentConfig) -> list[str]:
    return [name for name, domain in component_spec(component.type).ports.items() if domain == "fluid_in"]


def _output_fluid_ports(component: ComponentConfig) -> list[str]:
    return [name for name, domain in component_spec(component.type).ports.items() if domain == "fluid_out"]


def _fluid_property_specs(*, include_h: bool) -> list[tuple[str, str, float]]:
    specs = [
        ("T", "K", 1000.0),
        ("rho", "kg/m^3", 1000.0),
        ("gamma", "", 1.0),
    ]
    if include_h:
        specs.insert(0, ("h", "J/kg", 1.0e6))
    return specs
