from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from atha.components.registry import component_residual_contract, component_spec
from atha.config import build_performance_maps, evaluate_boundary_conditions
from atha.config.balances import balance_configs, wrap_problem_with_balances
from atha.config.loader import LoadedAnalysisConfig
from atha.config.controllers import controller_state_infos
from atha.config.schema import ComponentConfig
from atha.config.transients import TransientSystem
from atha.network import NetworkProblem
from atha.network.ports import PortNetworkBuilder
from atha.components.residuals import ResidualEvaluationContext


@dataclass
class SourceCatalog:
    """Flat source path catalog generated from loaded configuration."""

    sources: set[str] = field(default_factory=set)

    def add(self, *paths: str | None) -> None:
        for path in paths:
            if path:
                self.sources.add(str(path))

    def update(self, paths: Iterable[str]) -> None:
        for path in paths:
            self.add(path)

    def with_sources(self, paths: Iterable[str]) -> "SourceCatalog":
        catalog = SourceCatalog(set(self.sources))
        catalog.update(paths)
        return catalog


@dataclass(frozen=True)
class InitialVectors:
    state_names: list[str]
    X: list[float]
    algebraic_names: list[str]
    Z: list[float]


class EngineAssembler:
    """Phase-9 bridge from loaded YAML to model-independent source metadata."""

    def __init__(self, loaded: LoadedAnalysisConfig) -> None:
        self.loaded = loaded
        self._runtime_maps = build_performance_maps(loaded.maps) if loaded.maps else {}

    def source_catalog(self) -> SourceCatalog:
        catalog = SourceCatalog()
        catalog.add("time", "mdot.total")
        self._add_boundary_sources(catalog)
        self._add_target_sources(catalog)
        self._add_timing_sources(catalog)
        self._add_controller_sources(catalog)
        self._add_transient_sources(catalog)
        self._add_component_sources(catalog)
        self._add_connection_sources(catalog)
        self._add_port_network_sources(catalog)
        return catalog

    def telemetry_sources(self, extra_sources: Iterable[str] = ()) -> set[str]:
        return self.source_catalog().with_sources(extra_sources).sources

    def residual_network_problem(self, *, require_square: bool = True) -> NetworkProblem:
        """Build a square NetworkProblem from registered component residual contracts.

        This is the Phase-11 component-contract assembly bridge. It intentionally
        assembles only component-local residual providers; automatic connection
        port-variable expansion remains Phase 12+ work.
        """

        variables = []
        residuals = []
        contracts = []
        model = self._component_model()
        for component in self.loaded.engine.components.values():
            contract = component_residual_contract(component)
            if contract is None:
                continue
            variables.extend(contract.variables(component, model))
            residuals.extend(contract.residuals(component, model))
            contracts.append((component, contract))

        def evaluate(_t: float, z: dict[str, float], inputs: dict[str, float]) -> dict[str, float]:
            context = ResidualEvaluationContext(z=z, inputs=inputs, model=model)
            result: dict[str, float] = {}
            for component, contract in contracts:
                result.update(contract.evaluate(component, context))
            return result

        return NetworkProblem(variables, residuals, evaluate, name=f"{self.loaded.engine.name}_component_contracts", require_square=require_square)

    def port_network_problem(self, *, require_square: bool = False) -> NetworkProblem:
        """Build the automatic port-variable network problem for this engine."""

        problem = PortNetworkBuilder(self.loaded).build_problem(require_square=require_square)
        balances = balance_configs(self.loaded.analysis_config.analysis.get("balances", {}))
        if not balances:
            return problem
        allowed_sources = self.source_catalog().sources
        allowed_sources.update(problem.variable_names)
        allowed_sources.update(problem.residual_names)
        allowed_sources.update(f"residuals.{name}" for name in problem.residual_names)
        for balance in balances:
            allowed_sources.add(balance.variable)
            allowed_sources.add(f"balances.{balance.name}.residual")
            allowed_sources.add(f"residuals.balances.{balance.name}.residual")
        return wrap_problem_with_balances(problem, balances, allowed_sources=allowed_sources)

    def initial_vectors(self) -> InitialVectors:
        """Generate generic initial state/algebraic vectors from YAML metadata."""

        state_values: dict[str, float] = {}
        for component in self.loaded.engine.components.values():
            spec = component_spec(component.type)
            for state_name in spec.state_names:
                state_values.setdefault(
                    f"{component.name}.{state_name}",
                    _initial_state_value(component, state_name),
                )
            for state_name, value in component.initial_state.items():
                state_values[f"{component.name}.{state_name}"] = float(value)
        transient_system = TransientSystem.from_configs(self.loaded.transients)
        for name, value in zip(transient_system.state_names(), transient_system.initial_state({})):
            state_values.setdefault(name, float(value))
        for state in controller_state_infos(self.loaded.controllers):
            state_values.setdefault(state.name, float(state.initial))
        overrides = self.loaded.analysis_config.analysis.get("initial_state", {})
        if isinstance(overrides, dict):
            for name, value in overrides.items():
                state_values[str(name)] = float(value)

        problem = self.residual_network_problem(require_square=False)
        z_values = {name: float(value) for name, value in zip(problem.variable_names, problem.initial_z)}
        for balance in balance_configs(self.loaded.analysis_config.analysis.get("balances", {})):
            z_values.setdefault(balance.variable, float(balance.initial))
        z_overrides = self.loaded.analysis_config.analysis.get("initial_algebraic", {})
        if isinstance(z_overrides, dict):
            for name, value in z_overrides.items():
                z_values[str(name)] = float(value)
        return InitialVectors(
            state_names=list(state_values),
            X=[state_values[name] for name in state_values],
            algebraic_names=problem.variable_names,
            Z=[z_values[name] for name in problem.variable_names],
        )

    def _component_model(self) -> dict[str, float]:
        model: dict[str, object] = {}
        for component in self.loaded.engine.components.values():
            params = component.parameters
            for slot, binding in component.maps.items():
                if binding.ref in self._runtime_maps:
                    model[f"{component.name}.map.{slot}"] = self._runtime_maps[binding.ref]
                    if binding.output:
                        model[f"{component.name}.map.{slot}.output"] = binding.output
            if component.type == "Valve":
                if "CdA" in params:
                    model[f"{component.name}_CdA"] = float(params["CdA"])
                elif "max_area" in params:
                    model[f"{component.name}_CdA"] = float(params["max_area"]) * float(params.get("discharge_coeff", 1.0))
            elif component.type == "Nozzle" and "conductance" in params:
                model[f"{component.name}_conductance"] = float(params["conductance"])
                model.setdefault("nozzle_conductance", float(params["conductance"]))
            elif component.type == "MassFlowInjector" and "delta_P_nominal" in params:
                model[f"{component.name}_delta_P"] = float(params["delta_P_nominal"])
        return model

    def _add_boundary_sources(self, catalog: SourceCatalog) -> None:
        if self.loaded.boundary_conditions is None:
            return
        catalog.update(self.loaded.boundary_conditions.conditions)
        catalog.update(evaluate_boundary_conditions(self.loaded.boundary_conditions, 0.0))
        for name in self.loaded.boundary_conditions.conditions:
            catalog.add(f"boundaries.{name}")
        for name in evaluate_boundary_conditions(self.loaded.boundary_conditions, 0.0):
            catalog.add(f"boundaries.{name}")

    def _add_target_sources(self, catalog: SourceCatalog) -> None:
        if self.loaded.operating_conditions is None:
            return
        for name, spec in self.loaded.operating_conditions.targets.items():
            catalog.add(f"target.{name}", f"targets.{name}")
            if isinstance(spec, dict):
                schedule = spec.get("schedule", spec)
                outputs = schedule.get("outputs") if isinstance(schedule, dict) else None
                if isinstance(outputs, dict):
                    for output_name in outputs:
                        catalog.add(f"target.{output_name}", f"targets.{name}.{output_name}")
                if name == "pms":
                    for output_name in ("mdot_total", "OF", "mdot_lox", "mdot_fuel", "duration"):
                        catalog.add(f"target.{output_name}", f"targets.{name}.{output_name}")

    def _add_timing_sources(self, catalog: SourceCatalog) -> None:
        if self.loaded.timings is None:
            return
        for event in self.loaded.timings.events:
            target = event.get("target")
            if isinstance(target, str):
                catalog.add(target, f"timings.{target}")

    def _add_controller_sources(self, catalog: SourceCatalog) -> None:
        if self.loaded.controllers is None:
            return
        for name, controller in self.loaded.controllers.controllers.items():
            if not isinstance(controller, dict):
                continue
            if "output" in controller:
                catalog.add(str(controller["output"]))
            outputs = controller.get("outputs")
            if isinstance(outputs, dict):
                catalog.update(str(path) for path in outputs.values())
            if controller.get("type") in {"proportional", "pi", "pid", "limiter", "rate_limiter", "selector", "min", "max"}:
                catalog.add(
                    f"controller.{name}.target",
                    f"controller.{name}.measurement",
                    f"controller.{name}.error",
                    f"controller.{name}.command",
                    f"controller.{name}.raw_command",
                    f"controller.{name}.saturated",
                    f"controller.{name}.integral",
                    f"controller.{name}.derivative",
                    f"controller.{name}.proportional_term",
                    f"controller.{name}.integral_term",
                    f"controller.{name}.derivative_term",
                    f"controller.{name}.rate",
                )
        for state in controller_state_infos(self.loaded.controllers):
            catalog.add(state.name)

    def _add_transient_sources(self, catalog: SourceCatalog) -> None:
        system = TransientSystem.from_configs(self.loaded.transients)
        catalog.update(system.source_catalog())
        for config in self.loaded.transients.values():
            command = config.command.get("path")
            if isinstance(command, str):
                catalog.add(command)

    def _add_component_sources(self, catalog: SourceCatalog) -> None:
        for component in self.loaded.engine.components.values():
            spec = component_spec(component.type)
            catalog.update(spec.source_paths(component.name))
            for residual_name in spec.residual_names:
                catalog.add(f"residuals.{component.name}.{residual_name}", f"{component.name}.{residual_name}")
            contract = component_residual_contract(component)
            if contract is not None:
                model = component.parameters
                catalog.update(variable.name for variable in contract.variables(component, model))
                catalog.update(residual.name for residual in contract.residuals(component, model))
                catalog.update(f"residuals.{residual.name}" for residual in contract.residuals(component, model))
            catalog.update(_component_source_paths(component))

    def _add_connection_sources(self, catalog: SourceCatalog) -> None:
        for connection in self.loaded.engine.connections:
            src_comp, src_port = connection.source.split(".", 1)
            dst_comp, dst_port = connection.target.split(".", 1)
            base = f"connection.{src_comp}.{src_port}__{dst_comp}.{dst_port}"
            if connection.domain == "fluid":
                catalog.add(f"{base}.P", f"{base}.h", f"{base}.mdot")
            elif connection.domain == "shaft":
                catalog.add(f"{base}.omega", f"{base}.tau")
            elif connection.domain == "thermal":
                catalog.add(f"{base}.T_wall", f"{base}.Q_dot")

    def _add_port_network_sources(self, catalog: SourceCatalog) -> None:
        port_catalog = PortNetworkBuilder(self.loaded).catalog()
        catalog.update(port_catalog.source_paths())
        for balance in balance_configs(self.loaded.analysis_config.analysis.get("balances", {})):
            catalog.add(balance.variable, f"balances.{balance.name}.residual", f"residuals.balances.{balance.name}.residual")


def _component_source_paths(component: ComponentConfig) -> set[str]:
    name = component.name
    common = {f"{name}.output"}
    if component.type == "Valve":
        return common | {f"{name}.command", f"{name}.position", f"{name}.mdot", f"{name}.A_frac"}
    if component.type == "Pipe":
        return common | {f"{name}.mdot", f"{name}.mdot_steady", f"{name}.P", f"{name}.dP"}
    if component.type == "MassFlowInjector":
        return common | {f"{name}.mdot", f"{name}.delta_P", f"{name}.P"}
    if component.type in {"CombustionChamber", "Preburner", "GasGenerator"}:
        return common | {f"{name}.P", f"{name}.T", f"{name}.OF", f"{name}.mdot", f"{name}.h"}
    if component.type == "Nozzle":
        return common | {f"{name}.mdot", f"{name}.thrust", f"{name}.Isp_vacuum"}
    if component.type == "Rotor":
        return common | {f"{name}.omega", f"{name}.rpm", f"{name}.tau_drive", f"{name}.tau_load"}
    if component.type == "Pump":
        return common | {
            f"{name}.mdot",
            f"{name}.inlet_mdot",
            f"{name}.inlet.mdot",
            f"{name}.inlet.h",
            f"{name}.outlet.h",
            f"{name}.delta_P",
            f"{name}.pressure_rise",
            f"{name}.power",
            f"{name}.tau_load",
            f"{name}.efficiency",
        }
    if component.type == "Turbine":
        return common | {
            f"{name}.mdot",
            f"{name}.pressure_ratio",
            f"{name}.power",
            f"{name}.tau_drive",
            f"{name}.efficiency",
        }
    if component.type == "FlowSplitter":
        return common | {f"{name}.inlet.mdot", f"{name}.outlet_a.mdot", f"{name}.outlet_b.mdot"}
    if component.type in {"GasVolume", "Volume"}:
        return common | {f"{name}.P", f"{name}.T", f"{name}.mdot"}
    if component.type in {"Outlet", "OutletInertia"}:
        return common | {f"{name}.mdot", f"{name}.mdot_steady"}
    return common


def _initial_state_value(component: ComponentConfig, state_name: str) -> float:
    params = component.parameters
    if f"initial_{state_name}" in params:
        value = params[f"initial_{state_name}"]
    elif state_name == "omega" and "initial_speed_rpm" in params:
        value = float(params["initial_speed_rpm"]) * 3.141592653589793 / 30.0
    elif state_name == "P":
        value = params.get("initial_P", 101325.0)
    elif state_name == "T_wall":
        value = params.get("initial_T_wall", params.get("T_wall", 300.0))
    elif state_name == "h":
        value = params.get("initial_h", 0.0)
    elif state_name == "mdot":
        value = params.get("initial_mdot", params.get("mdot_design", 0.0))
    else:
        value = 0.0
    return float(value)
