from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping, Protocol

import numpy as np

from atha.config.loader import LoadedAnalysisConfig
from atha.network import NetworkProblem, NetworkResidual, NetworkSolution, NetworkVariable


class ReducedCycleProvider(Protocol):
    """Bridge a reduced transient plant into the generic DAE execution loop."""

    name: str
    network_problem: NetworkProblem

    def initial_state_overrides(self) -> Mapping[str, float]:
        ...

    def measurements(self, states: Mapping[str, float], algebraics: Mapping[str, float]) -> Mapping[str, float]:
        ...

    def solve_algebraics(self, t: float, inputs: Mapping[str, Any]) -> NetworkSolution:
        ...

    def derivatives(
        self,
        t: float,
        states: Mapping[str, float],
        algebraics: Mapping[str, float],
        commands: Mapping[str, float],
        timings: Mapping[str, Any],
    ) -> Mapping[str, float]:
        ...


def build_reduced_cycle_provider(loaded: LoadedAnalysisConfig) -> ReducedCycleProvider | None:
    cfg = loaded.analysis_config.analysis.get("reduced_cycle")
    if not isinstance(cfg, Mapping) or not cfg.get("enabled", True):
        return None
    provider_name = str(cfg.get("provider", cfg.get("type", "")))
    if provider_name in {"gg_single_shaft", "example20_gg_single_shaft"}:
        return GGSingleShaftReducedCycleProvider(loaded)
    if provider_name in {"two_shaft_gg", "generic_two_shaft_gg"}:
        return TwoShaftGGReducedCycleProvider(loaded)
    if provider_name in {"ffsc_reduced", "ffsc_dae", "example19_ffsc"}:
        return FFSCReducedCycleProvider(loaded)
    raise ValueError(f"unsupported reduced_cycle provider: {provider_name!r}")


@dataclass
class GGSingleShaftReducedCycleProvider:
    loaded: LoadedAnalysisConfig

    def __post_init__(self) -> None:
        self.name = "gg_single_shaft"
        self.impl = importlib.import_module("atha.analysis.gg_single_shaft")
        self.model = self.impl._design_model(self.loaded)
        self.network_problem = _target_problem(
            name="gg_single_shaft_reduced_cycle",
            initial_states=self.initial_state_overrides(),
            initial_transients={
                "main_lox_valve.position": 1.0,
                "main_methane_valve.position": 1.0,
                "lox_generator_valve.position": 0.5,
                "methane_generator_valve.position": 0.5,
            },
            model=self.model,
            target_fn=self._targets,
        )

    def initial_state_overrides(self) -> Mapping[str, float]:
        run = self.loaded.analysis_config.analysis
        initial = run.get("initial_conditions", {}) if isinstance(run.get("initial_conditions", {}), Mapping) else {}
        design = run.get("design", {}) if isinstance(run.get("design", {}), Mapping) else {}
        return {
            "shaft.omega": self.impl._rpm_to_rad_s(float(initial.get("shaft.rpm", 32000.0))),
            "chamber.P": float(initial.get("chamber.P", design.get("chamber_pressure", 5.0e6))),
            "generator.P": float(initial.get("generator.P", 8.0e6)),
        }

    def measurements(self, states: Mapping[str, float], algebraics: Mapping[str, float]) -> Mapping[str, float]:
        values = {**states, **algebraics}
        omega = float(states.get("shaft.omega", 0.0))
        return {
            **values,
            "mdot_total": float(algebraics.get("mdot.total", 0.0)),
            "OF": float(algebraics.get("chamber.OF", 0.0)),
            "shaft.rpm": self.impl._rad_s_to_rpm(omega),
        }

    def solve_algebraics(self, t: float, inputs: Mapping[str, Any]) -> NetworkSolution:
        return _direct_solution(self.network_problem, t, inputs, self._targets, self.initial_state_overrides(), self.model)

    def _targets(
        self,
        states: Mapping[str, float],
        transients: Mapping[str, float],
        model: Mapping[str, float],
    ) -> dict[str, float]:
        values = dict(self.impl._plant_values(states, transients, model))
        values.setdefault("lox_pump.inlet.mdot", float(values.get("lox.mdot_total", 0.0)))
        values.setdefault("methane_pump.inlet.mdot", float(values.get("methane.mdot_total", 0.0)))
        values.setdefault("lox_pump.inlet_mdot", float(values.get("lox.mdot_total", 0.0)))
        values.setdefault("methane_pump.inlet_mdot", float(values.get("methane.mdot_total", 0.0)))
        return values

    def derivatives(
        self,
        t: float,
        states: Mapping[str, float],
        algebraics: Mapping[str, float],
        commands: Mapping[str, float],
        timings: Mapping[str, Any],
    ) -> Mapping[str, float]:
        del t, commands, timings
        pump_power = float(algebraics["pump.power"])
        turbine_power = float(algebraics["turbine.power"])
        omega = max(float(states["shaft.omega"]), 100.0)
        domega = (turbine_power - pump_power) / max(self.model["shaft_inertia"] * omega, 1.0)
        domega -= self.model["shaft_friction"] * (omega - self.model["shaft_omega_design"])
        d_pc = self.model["chamber_pressure_gain"] * (float(algebraics["mdot.total"]) - float(algebraics["nozzle.mdot"]))
        d_pg = self.model["generator_pressure_gain"] * (
            float(algebraics["generator.mdot_in"]) - float(algebraics["generator.mdot_out"])
        )
        return {"shaft.omega": domega, "chamber.P": d_pc, "generator.P": d_pg}


@dataclass
class FFSCReducedCycleProvider:
    loaded: LoadedAnalysisConfig

    def __post_init__(self) -> None:
        self.name = "ffsc_reduced"
        self.impl = importlib.import_module("atha.analysis.ffsc_acceptance")
        registry = importlib.import_module("atha.components.registry")
        self.model = {
            **registry.extract_engine_model(self.loaded.engine),
            **self.impl._ffsc_design_model(self.loaded),
        }
        self.network_problem = _target_problem(
            name="ffsc_reduced_cycle",
            initial_states=self.initial_state_overrides(),
            initial_transients={
                "main_lox_valve.position": 1.0,
                "main_methane_valve.position": 1.0,
                "methane_crossover_valve.position": 0.5,
                "lox_crossover_valve.position": 0.5,
            },
            model=self.model,
            target_fn=self._targets,
        )

    def initial_state_overrides(self) -> Mapping[str, float]:
        run = self.loaded.analysis_config.analysis
        initial = run.get("initial_conditions", {}) if isinstance(run.get("initial_conditions", {}), Mapping) else {}
        design = run.get("design", {}) if isinstance(run.get("design", {}), Mapping) else {}
        return {
            "lox_shaft.omega": float(initial.get("lox_shaft.omega", self.impl._rpm_to_rad_s(32000.0))),
            "methane_shaft.omega": float(initial.get("methane_shaft.omega", self.impl._rpm_to_rad_s(27000.0))),
            "chamber.P": float(initial.get("chamber.P", design.get("chamber_pressure", 5.0e6))),
            "ox_preburner.P": float(initial.get("ox_preburner.P", 8.0e6)),
            "fuel_preburner.P": float(initial.get("fuel_preburner.P", 8.0e6)),
        }

    def measurements(self, states: Mapping[str, float], algebraics: Mapping[str, float]) -> Mapping[str, float]:
        values = {**states, **algebraics}
        return {
            **values,
            "mdot_total": float(algebraics.get("mdot.total", 0.0)),
            "OF": float(algebraics.get("chamber.OF", 0.0)),
            "lox_shaft.rpm": self.impl._rad_s_to_rpm(float(states.get("lox_shaft.omega", 0.0))),
            "methane_shaft.rpm": self.impl._rad_s_to_rpm(float(states.get("methane_shaft.omega", 0.0))),
        }

    def solve_algebraics(self, t: float, inputs: Mapping[str, Any]) -> NetworkSolution:
        return _direct_solution(self.network_problem, t, inputs, self._targets, self.initial_state_overrides(), self.model)

    def _targets(
        self,
        states: Mapping[str, float],
        transients: Mapping[str, float],
        model: Mapping[str, float],
    ) -> dict[str, float]:
        values = dict(self.impl._ffsc_targets(states, transients, model))
        values.setdefault("lox_pump.inlet.mdot", float(values.get("lox.mdot_total", 0.0)))
        values.setdefault("methane_pump.inlet.mdot", float(values.get("methane.mdot_total", 0.0)))
        values.setdefault("lox_pump.inlet_mdot", float(values.get("lox.mdot_total", 0.0)))
        values.setdefault("methane_pump.inlet_mdot", float(values.get("methane.mdot_total", 0.0)))
        return values

    def derivatives(
        self,
        t: float,
        states: Mapping[str, float],
        algebraics: Mapping[str, float],
        commands: Mapping[str, float],
        timings: Mapping[str, Any],
    ) -> Mapping[str, float]:
        del t, commands, timings
        lox_omega = max(float(states["lox_shaft.omega"]), 100.0)
        methane_omega = max(float(states["methane_shaft.omega"]), 100.0)
        lox_power_balance = float(algebraics["lox_turbine.power"]) - float(algebraics["lox_pump.power"])
        methane_power_balance = float(algebraics["methane_turbine.power"]) - float(algebraics["methane_pump.power"])
        domega_lox = lox_power_balance / max(self.model["lox_shaft_inertia"] * lox_omega, 1.0)
        domega_methane = methane_power_balance / max(self.model["methane_shaft_inertia"] * methane_omega, 1.0)
        domega_lox -= self.model["lox_shaft_friction"] * (lox_omega - self.model["lox_omega_design"])
        domega_methane -= self.model["methane_shaft_friction"] * (methane_omega - self.model["methane_omega_design"])
        d_ox_pb = self.model["ox_preburner_pressure_gain"] * (
            float(algebraics["ox_preburner.mdot_in"]) - float(algebraics["ox_preburner.mdot_out"])
        )
        d_fuel_pb = self.model["fuel_preburner_pressure_gain"] * (
            float(algebraics["fuel_preburner.mdot_in"]) - float(algebraics["fuel_preburner.mdot_out"])
        )
        d_pc = self.model["chamber_pressure_gain"] * (float(algebraics["mdot.total"]) - float(algebraics["nozzle.mdot"]))
        return {
            "lox_shaft.omega": domega_lox,
            "methane_shaft.omega": domega_methane,
            "chamber.P": d_pc,
            "ox_preburner.P": d_ox_pb,
            "fuel_preburner.P": d_fuel_pb,
        }


@dataclass
class TwoShaftGGReducedCycleProvider:
    loaded: LoadedAnalysisConfig

    def __post_init__(self) -> None:
        self.name = "two_shaft_gg"
        self.model = self._model()
        self.network_problem = _target_problem(
            name="two_shaft_gg_reduced_cycle",
            initial_states=self.initial_state_overrides(),
            initial_transients={
                "main_lox_valve.position": 1.0,
                "main_fuel_valve.position": 1.0,
                "lox_generator_valve.position": 0.45,
                "fuel_generator_valve.position": 0.45,
            },
            model=self.model,
            target_fn=self._targets,
        )

    def _model(self) -> dict[str, float]:
        analysis = self.loaded.analysis_config.analysis
        design = analysis.get("design", {}) if isinstance(analysis.get("design", {}), Mapping) else {}
        reduced = analysis.get("reduced_cycle", {}) if isinstance(analysis.get("reduced_cycle", {}), Mapping) else {}
        model_cfg = reduced.get("model", {}) if isinstance(reduced.get("model", {}), Mapping) else {}
        thrust = float(design.get("thrust", model_cfg.get("thrust", 5000.0)))
        isp = float(design.get("isp_s", model_cfg.get("isp_s", 245.0)))
        mdot_total = float(design.get("mdot_total", thrust / (max(isp, 1.0) * 9.80665)))
        of_ratio = float(design.get("OF", model_cfg.get("OF", 1.35)))
        mdot_lox = float(design.get("mdot_lox", mdot_total * of_ratio / (1.0 + of_ratio)))
        mdot_fuel = float(design.get("mdot_fuel", mdot_total / (1.0 + of_ratio)))
        chamber_pressure = float(design.get("chamber_pressure", model_cfg.get("chamber_pressure", 1.4e6)))
        ambient_pressure = float(model_cfg.get("ambient_pressure", 101325.0))
        throat_area = thrust / max(float(model_cfg.get("thrust_coefficient", 1.45)) * (chamber_pressure - ambient_pressure), 1.0)
        return {
            "thrust": thrust,
            "isp_s": isp,
            "mdot_total": mdot_total,
            "OF": of_ratio,
            "mdot_lox": mdot_lox,
            "mdot_fuel": mdot_fuel,
            "chamber_pressure": chamber_pressure,
            "ambient_pressure": ambient_pressure,
            "generator_pressure": float(design.get("generator_pressure", model_cfg.get("generator_pressure", 2.0e6))),
            "lox_omega_design": _rpm_to_rad_s(float(design.get("lox_shaft_rpm", model_cfg.get("lox_shaft_rpm", 26000.0)))),
            "fuel_omega_design": _rpm_to_rad_s(float(design.get("fuel_shaft_rpm", model_cfg.get("fuel_shaft_rpm", 30000.0)))),
            "lox_generator_fraction": float(model_cfg.get("lox_generator_fraction", 0.08)),
            "fuel_generator_fraction": float(model_cfg.get("fuel_generator_fraction", 0.10)),
            "lox_inertia": float(model_cfg.get("lox_inertia", 0.020)),
            "fuel_inertia": float(model_cfg.get("fuel_inertia", 0.012)),
            "lox_friction": float(model_cfg.get("lox_friction", 0.10)),
            "fuel_friction": float(model_cfg.get("fuel_friction", 0.12)),
            "lox_pump_dp": float(model_cfg.get("lox_pump_dp", 1.9e6)),
            "fuel_pump_dp": float(model_cfg.get("fuel_pump_dp", 1.5e6)),
            "lox_density": float(model_cfg.get("lox_density", 1140.0)),
            "fuel_density": float(model_cfg.get("fuel_density", 790.0)),
            "lox_pump_efficiency": float(model_cfg.get("lox_pump_efficiency", 0.62)),
            "fuel_pump_efficiency": float(model_cfg.get("fuel_pump_efficiency", 0.58)),
            "lox_turbine_gain": float(model_cfg.get("lox_turbine_gain", 4.2e5)),
            "fuel_turbine_gain": float(model_cfg.get("fuel_turbine_gain", 2.9e5)),
            "chamber_pressure_gain": float(model_cfg.get("chamber_pressure_gain", 4.0e5)),
            "generator_pressure_gain": float(model_cfg.get("generator_pressure_gain", 6.0e5)),
            "nozzle_conductance": mdot_total / max(chamber_pressure - ambient_pressure, 1.0),
            "generator_conductance": float(model_cfg.get("generator_conductance", 3.5e-7)),
            "thrust_coefficient": float(model_cfg.get("thrust_coefficient", 1.45)),
            "throat_area": float(model_cfg.get("throat_area", throat_area)),
        }

    def initial_state_overrides(self) -> Mapping[str, float]:
        run = self.loaded.analysis_config.analysis
        initial = run.get("initial_conditions", {}) if isinstance(run.get("initial_conditions", {}), Mapping) else {}
        return {
            "lox_shaft.omega": _rpm_to_rad_s(float(initial.get("lox_shaft.rpm", initial.get("lox_shaft_rpm", 21000.0)))),
            "fuel_shaft.omega": _rpm_to_rad_s(float(initial.get("fuel_shaft.rpm", initial.get("fuel_shaft_rpm", 24000.0)))),
            "chamber.P": float(initial.get("chamber.P", 0.45 * self.model["chamber_pressure"])),
            "generator.P": float(initial.get("generator.P", 0.70 * self.model["generator_pressure"])),
        }

    def measurements(self, states: Mapping[str, float], algebraics: Mapping[str, float]) -> Mapping[str, float]:
        values = {**states, **algebraics}
        return {
            **values,
            "mdot_total": float(algebraics.get("mdot.total", 0.0)),
            "OF": float(algebraics.get("chamber.OF", 0.0)),
            "lox_shaft.rpm": _rad_s_to_rpm(float(states.get("lox_shaft.omega", 0.0))),
            "fuel_shaft.rpm": _rad_s_to_rpm(float(states.get("fuel_shaft.omega", 0.0))),
        }

    def solve_algebraics(self, t: float, inputs: Mapping[str, Any]) -> NetworkSolution:
        return _direct_solution(self.network_problem, t, inputs, self._targets, self.initial_state_overrides(), self.model)

    def _targets(
        self,
        states: Mapping[str, float],
        transients: Mapping[str, float],
        model: Mapping[str, float],
    ) -> dict[str, float]:
        lox_speed = max(float(states.get("lox_shaft.omega", model["lox_omega_design"])) / model["lox_omega_design"], 0.05)
        fuel_speed = max(float(states.get("fuel_shaft.omega", model["fuel_omega_design"])) / model["fuel_omega_design"], 0.05)
        main_lox = _clamp01(float(transients.get("main_lox_valve.position", transients.get("main_lox_valve.command", 1.0))))
        main_fuel = _clamp01(float(transients.get("main_fuel_valve.position", transients.get("main_fuel_valve.command", 1.0))))
        gen_lox = _clamp01(float(transients.get("lox_generator_valve.position", transients.get("lox_generator_valve.command", 0.45))))
        gen_fuel = _clamp01(float(transients.get("fuel_generator_valve.position", transients.get("fuel_generator_valve.command", 0.45))))

        lox_capacity = model["mdot_lox"] * lox_speed * (0.15 + 0.85 * main_lox)
        fuel_capacity = model["mdot_fuel"] * fuel_speed * (0.15 + 0.85 * main_fuel)
        lox_generator = model["mdot_lox"] * model["lox_generator_fraction"] * lox_speed * gen_lox
        fuel_generator = model["mdot_fuel"] * model["fuel_generator_fraction"] * fuel_speed * gen_fuel
        lox_main = max(lox_capacity - lox_generator, 0.0)
        fuel_main = max(fuel_capacity - fuel_generator, 0.0)
        mdot_total = lox_main + fuel_main
        of_ratio = lox_main / max(fuel_main, 1.0e-9)

        chamber_p = max(float(states.get("chamber.P", model["chamber_pressure"])), model["ambient_pressure"])
        generator_p = max(float(states.get("generator.P", model["generator_pressure"])), model["ambient_pressure"])
        nozzle_mdot = model["nozzle_conductance"] * max(chamber_p - model["ambient_pressure"], 0.0)
        generator_mdot_in = lox_generator + fuel_generator
        generator_mdot_out = model["generator_conductance"] * max(generator_p - model["ambient_pressure"], 0.0)

        lox_dp = model["lox_pump_dp"] * lox_speed * lox_speed
        fuel_dp = model["fuel_pump_dp"] * fuel_speed * fuel_speed
        lox_pump_power = (lox_capacity * lox_dp / max(model["lox_density"] * model["lox_pump_efficiency"], 1.0))
        fuel_pump_power = (fuel_capacity * fuel_dp / max(model["fuel_density"] * model["fuel_pump_efficiency"], 1.0))
        generator_power_factor = generator_mdot_out * max(generator_p - chamber_p, 0.0) / 1.0e6
        lox_turbine_power = model["lox_turbine_gain"] * generator_power_factor * (0.30 + 0.70 * gen_lox)
        fuel_turbine_power = model["fuel_turbine_gain"] * generator_power_factor * (0.30 + 0.70 * gen_fuel)
        thrust = model["thrust_coefficient"] * model["throat_area"] * max(chamber_p - model["ambient_pressure"], 0.0)

        return {
            "lox.mdot_total": lox_capacity,
            "fuel.mdot_total": fuel_capacity,
            "mdot.total": mdot_total,
            "chamber.OF": of_ratio,
            "target.mdot_total": model["mdot_total"],
            "target.OF": model["OF"],
            "lox_pump.inlet.mdot": lox_capacity,
            "fuel_pump.inlet.mdot": fuel_capacity,
            "lox_pump.inlet_mdot": lox_capacity,
            "fuel_pump.inlet_mdot": fuel_capacity,
            "lox_pump.delta_P": lox_dp,
            "fuel_pump.delta_P": fuel_dp,
            "lox_pump.phi": lox_capacity / max(model["mdot_lox"], 1.0e-9),
            "fuel_pump.phi": fuel_capacity / max(model["mdot_fuel"], 1.0e-9),
            "lox_pump.psi": lox_dp / max(model["lox_pump_dp"], 1.0e-9),
            "fuel_pump.psi": fuel_dp / max(model["fuel_pump_dp"], 1.0e-9),
            "lox_pump.efficiency": model["lox_pump_efficiency"],
            "fuel_pump.efficiency": model["fuel_pump_efficiency"],
            "lox_pump.power": lox_pump_power,
            "fuel_pump.power": fuel_pump_power,
            "lox_pump.tau_load": lox_pump_power / max(abs(float(states.get("lox_shaft.omega", 0.0))), 1.0),
            "fuel_pump.tau_load": fuel_pump_power / max(abs(float(states.get("fuel_shaft.omega", 0.0))), 1.0),
            "lox_generator.mdot": lox_generator,
            "fuel_generator.mdot": fuel_generator,
            "generator.mdot": generator_mdot_in,
            "generator.mdot_in": generator_mdot_in,
            "generator.mdot_out": generator_mdot_out,
            "generator.OF": lox_generator / max(fuel_generator, 1.0e-9),
            "lox_turbine.power": lox_turbine_power,
            "fuel_turbine.power": fuel_turbine_power,
            "lox_turbine.tau_drive": lox_turbine_power / max(abs(float(states.get("lox_shaft.omega", 0.0))), 1.0),
            "fuel_turbine.tau_drive": fuel_turbine_power / max(abs(float(states.get("fuel_shaft.omega", 0.0))), 1.0),
            "nozzle.mdot": nozzle_mdot,
            "nozzle.thrust": thrust,
            "lox_shaft.rpm": _rad_s_to_rpm(float(states.get("lox_shaft.omega", 0.0))),
            "fuel_shaft.rpm": _rad_s_to_rpm(float(states.get("fuel_shaft.omega", 0.0))),
        }

    def derivatives(
        self,
        t: float,
        states: Mapping[str, float],
        algebraics: Mapping[str, float],
        commands: Mapping[str, float],
        timings: Mapping[str, Any],
    ) -> Mapping[str, float]:
        del t, commands, timings
        lox_omega = max(float(states["lox_shaft.omega"]), 100.0)
        fuel_omega = max(float(states["fuel_shaft.omega"]), 100.0)
        lox_balance = float(algebraics["lox_turbine.power"]) - float(algebraics["lox_pump.power"])
        fuel_balance = float(algebraics["fuel_turbine.power"]) - float(algebraics["fuel_pump.power"])
        domega_lox = lox_balance / max(self.model["lox_inertia"] * lox_omega, 1.0)
        domega_fuel = fuel_balance / max(self.model["fuel_inertia"] * fuel_omega, 1.0)
        domega_lox -= self.model["lox_friction"] * (lox_omega - self.model["lox_omega_design"])
        domega_fuel -= self.model["fuel_friction"] * (fuel_omega - self.model["fuel_omega_design"])
        d_pc = self.model["chamber_pressure_gain"] * (float(algebraics["mdot.total"]) - float(algebraics["nozzle.mdot"]))
        d_pg = self.model["generator_pressure_gain"] * (
            float(algebraics["generator.mdot_in"]) - float(algebraics["generator.mdot_out"])
        )
        return {
            "lox_shaft.omega": domega_lox,
            "fuel_shaft.omega": domega_fuel,
            "chamber.P": d_pc,
            "generator.P": d_pg,
        }


def _target_problem(
    *,
    name: str,
    initial_states: Mapping[str, float],
    initial_transients: Mapping[str, float],
    model: Mapping[str, float],
    target_fn: Any,
) -> NetworkProblem:
    initial_targets = target_fn(dict(initial_states), dict(initial_transients), dict(model))
    variables = [
        NetworkVariable(path, scale=_scale_for(path, value), initial=float(value))
        for path, value in initial_targets.items()
        if _is_number(value)
    ]
    residuals = [NetworkResidual(f"{variable.name}_residual", scale=variable.scale) for variable in variables]

    def evaluate(_t: float, z: dict[str, float], inputs: dict[str, Any]) -> dict[str, float]:
        payload = inputs.get("inputs", inputs)
        states = {key: float(value) for key, value in payload.items() if key in initial_states and _is_number(value)}
        transients = {key: float(value) for key, value in payload.items() if key in initial_transients and _is_number(value)}
        targets = target_fn({**dict(initial_states), **states}, {**dict(initial_transients), **transients}, dict(model))
        return {f"{path}_residual": float(z[path]) - float(targets[path]) for path in z}

    return NetworkProblem(variables, residuals, evaluate, name=name)


def _direct_solution(
    problem: NetworkProblem,
    t: float,
    inputs: Mapping[str, Any],
    target_fn: Any,
    initial_states: Mapping[str, float],
    model: Mapping[str, float],
) -> NetworkSolution:
    del t
    payload = inputs.get("inputs", inputs)
    states = {key: float(value) for key, value in payload.items() if key in initial_states and _is_number(value)}
    transient_keys = {
        key
        for key in payload
        if key.endswith(".position") or key.endswith(".command")
    }
    transients = {key: float(payload[key]) for key in transient_keys if _is_number(payload[key])}
    values = {
        key: float(value)
        for key, value in target_fn({**dict(initial_states), **states}, transients, dict(model)).items()
        if key in problem.variable_names and _is_number(value)
    }
    z = np.asarray([values[name] for name in problem.variable_names], dtype=float)
    residuals = {name: 0.0 for name in problem.residual_names}
    normalized = {name: 0.0 for name in problem.residual_names}
    return NetworkSolution(
        z=z,
        values=values,
        residuals=residuals,
        normalized_residuals=normalized,
        success=True,
        message="direct reduced-cycle algebraic evaluation",
    )


def _scale_for(path: str, value: float) -> float:
    magnitude = max(abs(float(value)), 1.0)
    if path.endswith(".P") or "delta_P" in path:
        return max(magnitude, 1.0e5)
    if "power" in path:
        return max(magnitude, 1.0e5)
    if "thrust" in path:
        return max(magnitude, 1.0e4)
    if "mdot" in path:
        return max(magnitude, 1.0)
    return magnitude


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating))


def _rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * 2.0 * np.pi / 60.0


def _rad_s_to_rpm(omega: float) -> float:
    return float(omega) * 60.0 / (2.0 * np.pi)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
