from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from atha.components.registry import component_residual_contract, component_spec
from atha.components.residuals import ResidualEvaluationContext
from atha.config import build_performance_maps
from atha.config.loader import LoadedAnalysisConfig
from atha.config.schema import ComponentConfig, ConnectionConfig
from atha.network.problem import NetworkProblem, NetworkResidual, NetworkVariable


FLUID_VARIABLES = ("P", "mdot", "h")
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
            for key, value in self._static_boundary_values().items():
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
            result.update(_component_coupling_residual_values(self._components, values))
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
                NetworkVariable(f"{port.path}.P", units="Pa", scale=1.0e6, owner=port.component),
                NetworkVariable(f"{port.path}.mdot", units="kg/s", scale=1.0, owner=port.component),
                NetworkVariable(f"{port.path}.h", units="J/kg", scale=1.0e6, owner=port.component),
            ]
        if port.domain == "shaft":
            return [
                NetworkVariable(f"{port.path}.omega", units="rad/s", scale=1000.0, owner=port.component),
                NetworkVariable(f"{port.path}.tau", units="N*m", scale=100.0, owner=port.component),
            ]
        if port.domain == "thermal":
            return [
                NetworkVariable(f"{port.path}.T", units="K", scale=1000.0, owner=port.component),
                NetworkVariable(f"{port.path}.Q_dot", units="W", scale=1.0e5, owner=port.component),
            ]
        return []

    def _connection_residuals(self, connection: ConnectionConfig) -> list[NetworkResidual]:
        base = _connection_name(connection)
        if connection.domain == "fluid":
            return [
                NetworkResidual(f"{base}.P_continuity", units="Pa", scale=1.0e6),
                NetworkResidual(f"{base}.h_continuity", units="J/kg", scale=1.0e6),
                NetworkResidual(f"{base}.mdot_continuity", units="kg/s", scale=1.0),
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
            if f"{component.name}.delta_P" in variable_names:
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
            if component.type == "Pump" and f"{component.name}.shaft.tau" in variable_names:
                residuals.append(NetworkResidual(f"{component.name}.shaft.tau_load_residual", units="N*m", scale=100.0, owner=component.name))
            if component.type == "Turbine" and f"{component.name}.shaft.tau" in variable_names:
                residuals.append(NetworkResidual(f"{component.name}.shaft.tau_drive_residual", units="N*m", scale=100.0, owner=component.name))
        return residuals

    def _has_boundary(self, path: str) -> bool:
        if self.loaded.boundary_conditions is None:
            return False
        return path in self.loaded.boundary_conditions.conditions

    def _component_model(self, component: ComponentConfig) -> dict[str, Any]:
        model = _component_model(component)
        for slot, binding in component.maps.items():
            if binding.ref in self._runtime_maps:
                model[f"{component.name}.map.{slot}"] = self._runtime_maps[binding.ref]
                if binding.output:
                    model[f"{component.name}.map.{slot}.output"] = binding.output
        return model

    def _static_boundary_values(self) -> dict[str, float]:
        if self.loaded.boundary_conditions is None:
            return {}
        values: dict[str, float] = {}
        for path, value in self.loaded.boundary_conditions.conditions.items():
            if isinstance(value, (int, float)):
                values[path] = float(value)
            elif isinstance(value, dict) and "value" in value and isinstance(value["value"], (int, float)):
                values[path] = float(value["value"])
        return values


def _connection_residual_values(connection: ConnectionConfig, values: Mapping[str, float]) -> dict[str, float]:
    base = _connection_name(connection)
    source = connection.source
    target = connection.target
    if connection.domain == "fluid":
        return {
            f"{base}.P_continuity": _value(values, f"{source}.P") - _value(values, f"{target}.P"),
            f"{base}.h_continuity": _value(values, f"{source}.h") - _value(values, f"{target}.h"),
            f"{base}.mdot_continuity": _value(values, f"{source}.mdot") - _value(values, f"{target}.mdot"),
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
    values: Mapping[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for component in components.values():
        if f"{component.name}.delta_P" in values:
            inlet = _first_existing(values, [f"{component.name}.inlet.P", f"{component.name}.fuel_inlet.P", f"{component.name}.ox_inlet.P"])
            outlet = _first_existing(values, [f"{component.name}.outlet.P"])
            if inlet is not None and outlet is not None:
                result[f"{component.name}.delta_P_port_residual"] = outlet - inlet - _value(values, f"{component.name}.delta_P")
        if f"{component.name}.mdot" in values and component.type not in {"CombustionChamber", "Preburner"}:
            for port in _fluid_ports(component):
                port_mdot = f"{component.name}.{port}.mdot"
                if port_mdot in values:
                    result[f"{port_mdot}_link_residual"] = _value(values, port_mdot) - _value(values, f"{component.name}.mdot")
        if f"{component.name}.omega" in values:
            for port in _shaft_ports(component):
                port_omega = f"{component.name}.{port}.omega"
                if port_omega in values:
                    result[f"{port_omega}_link_residual"] = _value(values, port_omega) - _value(values, f"{component.name}.omega")
        if component.type == "Rotor":
            for port_omega in sorted(_component_value_paths(values, component.name, "omega")):
                result[f"{port_omega}_state_link_residual"] = _value(values, port_omega) - _value(values, f"{component.name}.omega")
        if component.type == "Pump" and f"{component.name}.shaft.tau" in values:
            omega = _value(values, f"{component.name}.shaft.omega", _value(values, f"{component.name}.omega", 0.0))
            result[f"{component.name}.shaft.tau_load_residual"] = _value(values, f"{component.name}.shaft.tau") - _pump_torque(component, values, omega)
        if component.type == "Turbine" and f"{component.name}.shaft.tau" in values:
            omega = _value(values, f"{component.name}.shaft.omega", _value(values, f"{component.name}.omega", 0.0))
            result[f"{component.name}.shaft.tau_drive_residual"] = _value(values, f"{component.name}.shaft.tau") + _turbine_torque(component, values, omega)
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


def _first_existing(values: Mapping[str, float], paths: list[str]) -> float | None:
    for path in paths:
        if path in values:
            return float(values[path])
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
    if path in values:
        value = values[path]
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
