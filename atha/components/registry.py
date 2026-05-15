from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from atha.config.schema import ComponentConfig, ConfigError, EngineConfig, TransientConfig
from atha.components.residuals import ComponentResidualContract, residual_contract_for_type


ModelExtractor = Callable[[ComponentConfig, Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ComponentSpec:
    type_name: str
    required_parameters: frozenset[str] = frozenset()
    optional_parameters: frozenset[str] = frozenset()
    transient_capable: bool = False
    model_extractor: ModelExtractor | None = None
    aliases: frozenset[str] = field(default_factory=frozenset)
    allow_extra_parameters: bool = False
    ports: Mapping[str, str] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()
    algebraic_variables: tuple[str, ...] = ()
    residual_names: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = ()
    map_slots: frozenset[str] = frozenset()
    transient_inputs: tuple[str, ...] = ()
    residual_contract: ComponentResidualContract | None = None

    @property
    def accepted_parameters(self) -> frozenset[str]:
        return self.required_parameters | self.optional_parameters

    def source_paths(self, component_name: str) -> set[str]:
        paths = {f"{component_name}.output"}
        paths.update(f"{component_name}.{path}" for path in self.output_paths)
        paths.update(f"{component_name}.{state}" for state in self.state_names)
        paths.update(f"{component_name}.{variable}" for variable in self.algebraic_variables)
        paths.update(f"{component_name}.{path}" for path in self.transient_inputs)
        return paths


class ComponentRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ComponentSpec] = {}

    def register(self, spec: ComponentSpec) -> None:
        names = {spec.type_name, *spec.aliases}
        for name in names:
            self._specs[name] = spec

    def get(self, type_name: str) -> ComponentSpec:
        try:
            return self._specs[type_name]
        except KeyError as exc:
            raise ConfigError(
                f"Unsupported component type '{type_name}'. Known types: {self.known_types()}"
            ) from exc

    def known_types(self) -> list[str]:
        return sorted(self._specs)

    def validate_component(
        self,
        component: ComponentConfig,
        transients: Mapping[str, TransientConfig] | None = None,
    ) -> None:
        spec = self.get(component.type)
        params = set(component.parameters)
        missing = sorted(spec.required_parameters - params)
        if missing:
            raise ConfigError(f"Component '{component.name}' missing required parameter(s): {missing}")
        unknown = sorted(params - spec.accepted_parameters)
        if unknown and not spec.allow_extra_parameters:
            raise ConfigError(
                f"Component '{component.name}' has unsupported parameter(s) {unknown} "
                f"for type '{component.type}'. Accepted: {sorted(spec.accepted_parameters)}"
            )
        if component.transient is not None:
            if not spec.transient_capable:
                raise ConfigError(f"Component '{component.name}' of type '{component.type}' does not support transients")
            if transients is not None and component.transient not in transients:
                raise ConfigError(
                    f"Component '{component.name}' references transient '{component.transient}', "
                    "but Analysis YAML does not bind it"
                )

    def validate_engine(
        self,
        engine: EngineConfig,
        transients: Mapping[str, TransientConfig] | None = None,
    ) -> None:
        for component in engine.components.values():
            self.validate_component(component, transients=transients)

    def extract_engine_model(self, engine: EngineConfig) -> dict[str, Any]:
        model: dict[str, Any] = {}
        for component in engine.components.values():
            spec = self.get(component.type)
            if spec.model_extractor is not None:
                model.update(spec.model_extractor(component, _coerce_numbers(component.parameters)))
        return model

    def residual_contract(self, component: ComponentConfig) -> ComponentResidualContract | None:
        spec = self.get(component.type)
        return spec.residual_contract


def default_component_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register(
        ComponentSpec(
            "Valve",
            required_parameters=frozenset(),
            optional_parameters=frozenset({"max_area", "CdA", "discharge_coeff"}),
            transient_capable=True,
            model_extractor=_extract_valve_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            output_paths=("mdot", "A_frac", "position", "command", "CdA"),
            map_slots=frozenset({"cd_map", "cda_map"}),
            transient_inputs=("command",),
            residual_contract=residual_contract_for_type("Valve"),
        )
    )
    registry.register(
        ComponentSpec(
            "Pipe",
            optional_parameters=frozenset({"length", "diameter", "time_constant", "friction_factor", "conductance", "mdot_design", "fallback_conductance"}),
            model_extractor=_extract_pipe_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            state_names=("mdot",),
            algebraic_variables=("mdot",),
            residual_names=("momentum_residual",),
            output_paths=("mdot", "mdot_steady", "P", "dP"),
            residual_contract=residual_contract_for_type("Pipe"),
        )
    )
    registry.register(
        ComponentSpec(
            "MassFlowInjector",
            optional_parameters=frozenset({"delta_P_nominal"}),
            model_extractor=_extract_injector_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            algebraic_variables=("outlet.P",),
            residual_names=("delta_P_residual",),
            output_paths=("mdot", "delta_P", "P"),
            residual_contract=residual_contract_for_type("MassFlowInjector"),
        )
    )
    registry.register(
        ComponentSpec(
            "CombustionChamber",
            optional_parameters=frozenset({"volume", "gas_T", "gas_R", "initial_P", "initial_T", "efficiency", "design_MR", "T_adiabatic"}),
            model_extractor=_extract_chamber_model,
            ports={"fuel_inlet": "fluid_in", "ox_inlet": "fluid_in", "lox_inlet": "fluid_in", "outlet": "fluid_out"},
            state_names=("P", "h"),
            algebraic_variables=("P", "OF", "T", "mdot"),
            residual_names=("mass_balance_residual", "OF_residual", "pressure_residual", "temperature_residual"),
            output_paths=("P", "T", "OF", "mdot", "h", "rho", "gamma"),
            residual_contract=residual_contract_for_type("CombustionChamber"),
        )
    )
    registry.register(
        ComponentSpec(
            "Nozzle",
            optional_parameters=frozenset({"throat_area", "exit_area", "conductance", "thrust_coefficient", "efficiencies"}),
            model_extractor=_extract_nozzle_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            algebraic_variables=("mdot",),
            residual_names=("mdot_residual",),
            output_paths=("mdot", "thrust", "Isp_vacuum", "Cf", "c_star"),
            residual_contract=residual_contract_for_type("Nozzle"),
        )
    )
    registry.register(
        ComponentSpec(
            "GasVolume",
            required_parameters=frozenset({"volume", "gas_R", "gas_T"}),
            optional_parameters=frozenset({"gamma"}),
            aliases=frozenset({"Volume"}),
            model_extractor=_extract_gas_volume_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            state_names=("P", "h"),
            output_paths=("P", "T", "mdot", "h"),
        )
    )
    registry.register(
        ComponentSpec(
            "OutletInertia",
            optional_parameters=frozenset({"outlet_resistance", "outlet_inertance", "outlet_conductance"}),
            aliases=frozenset({"Outlet"}),
            model_extractor=_extract_passthrough_model,
            ports={"inlet": "fluid_in", "outlet": "fluid_out"},
            state_names=("mdot",),
            output_paths=("mdot", "mdot_steady"),
        )
    )
    registry.register(
        ComponentSpec(
            "Pump",
            optional_parameters=frozenset({"diameter", "pump_map", "delta_P_design", "mdot_design", "omega_design", "efficiency_design"}),
            allow_extra_parameters=True,
            ports={"inlet": "fluid_in", "outlet": "fluid_out", "shaft": "shaft_in"},
            algebraic_variables=("delta_P",),
            residual_names=("delta_P_residual",),
            output_paths=("mdot", "delta_P", "pressure_rise", "power", "tau_load", "efficiency"),
            map_slots=frozenset({"head_map", "efficiency_map"}),
            residual_contract=residual_contract_for_type("Pump"),
        )
    )
    registry.register(
        ComponentSpec(
            "Turbine",
            optional_parameters=frozenset({"efficiency", "pressure_ratio", "power_design", "mdot_design", "diameter", "turbine_map"}),
            allow_extra_parameters=True,
            ports={"inlet": "fluid_in", "outlet": "fluid_out", "shaft": "shaft_out"},
            algebraic_variables=("power",),
            residual_names=("power_residual",),
            output_paths=("mdot", "pressure_ratio", "power", "tau_drive", "efficiency"),
            map_slots=frozenset({"pressure_ratio_map", "efficiency_map"}),
            residual_contract=residual_contract_for_type("Turbine"),
        )
    )
    registry.register(
        ComponentSpec(
            "FlowSplitter",
            optional_parameters=frozenset({"split_fraction"}),
            allow_extra_parameters=True,
            ports={"inlet": "fluid_in", "outlet_a": "fluid_out", "outlet_b": "fluid_out"},
            algebraic_variables=("outlet_a.mdot", "outlet_b.mdot"),
            residual_names=("outlet_a.mdot_residual", "outlet_b.mdot_residual"),
            output_paths=("inlet.mdot", "outlet_a.mdot", "outlet_b.mdot"),
            residual_contract=residual_contract_for_type("FlowSplitter"),
        )
    )
    registry.register(
        ComponentSpec(
            "Rotor",
            allow_extra_parameters=True,
            state_names=("omega",),
            algebraic_variables=("omega",),
            residual_names=("torque_balance",),
            output_paths=("omega", "rpm", "tau_drive", "tau_load"),
            ports={"shaft": "shaft"},
            residual_contract=residual_contract_for_type("Rotor"),
        )
    )
    registry.register(
        ComponentSpec(
            "Preburner",
            allow_extra_parameters=True,
            ports={"fuel_inlet": "fluid_in", "ox_inlet": "fluid_in", "lox_inlet": "fluid_in", "outlet": "fluid_out"},
            state_names=("P", "h"),
            algebraic_variables=("P", "OF", "T", "mdot"),
            residual_names=("mass_balance_residual", "OF_residual", "pressure_residual", "temperature_residual"),
            output_paths=("P", "T", "OF", "mdot", "h", "rho", "gamma"),
            residual_contract=residual_contract_for_type("Preburner"),
        )
    )
    registry.register(
        ComponentSpec(
            "RegenChannel",
            allow_extra_parameters=True,
            ports={"coolant_inlet": "fluid_in", "coolant_outlet": "fluid_out"},
            state_names=("T_wall",),
            algebraic_variables=("T_wall", "Q_dot"),
            residual_names=("heat_balance_residual", "wall_temperature_residual"),
            output_paths=("coolant_outlet.P", "coolant_outlet.h", "coolant_outlet.mdot", "Q_cool", "Q_hot", "T_wall", "Q_dot"),
            residual_contract=residual_contract_for_type("RegenChannel"),
        )
    )
    for type_name in ("GasGenerator", "OrificeCompressible"):
        registry.register(ComponentSpec(type_name, allow_extra_parameters=True))
    return registry


def _extract_valve_model(component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    if "CdA" in params:
        cda = params["CdA"]
    elif "max_area" in params:
        cda = params["max_area"] * params.get("discharge_coeff", 1.0)
    else:
        return {}
    return {f"{component.name}_CdA": cda}


def _extract_pipe_model(component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    if "time_constant" not in params:
        return {}
    return {f"{component.name}_time_constant": params["time_constant"]}


def _extract_injector_model(component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    if "delta_P_nominal" not in params:
        return {}
    return {f"{component.name}_delta_P": params["delta_P_nominal"]}


def _extract_chamber_model(_component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    model = {}
    if "volume" in params:
        model["chamber_volume"] = params["volume"]
    if "gas_T" in params:
        model["chamber_T"] = params["gas_T"]
    if "gas_R" in params:
        model["gas_R"] = params["gas_R"]
    return model


def _extract_nozzle_model(_component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    model = {}
    if "throat_area" in params:
        model["nozzle_throat_area"] = params["throat_area"]
    if "conductance" in params:
        model["nozzle_conductance"] = params["conductance"]
    if "thrust_coefficient" in params:
        model["thrust_coefficient"] = params["thrust_coefficient"]
    return model


def _extract_gas_volume_model(_component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    model = {
        "volume": params["volume"],
        "gas_R": params["gas_R"],
        "gas_T": params["gas_T"],
    }
    if "gamma" in params:
        model["gamma"] = params["gamma"]
    return model


def _extract_passthrough_model(_component: ComponentConfig, params: Mapping[str, Any]) -> dict[str, Any]:
    return dict(params)


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


DEFAULT_COMPONENT_REGISTRY = default_component_registry()


def validate_engine_config(engine: EngineConfig, transients: Mapping[str, TransientConfig] | None = None) -> None:
    DEFAULT_COMPONENT_REGISTRY.validate_engine(engine, transients=transients)


def extract_engine_model(engine: EngineConfig) -> dict[str, Any]:
    return DEFAULT_COMPONENT_REGISTRY.extract_engine_model(engine)


def known_component_types() -> list[str]:
    return DEFAULT_COMPONENT_REGISTRY.known_types()


def component_spec(type_name: str) -> ComponentSpec:
    return DEFAULT_COMPONENT_REGISTRY.get(type_name)


def component_residual_contract(component: ComponentConfig) -> ComponentResidualContract | None:
    return DEFAULT_COMPONENT_REGISTRY.residual_contract(component)
