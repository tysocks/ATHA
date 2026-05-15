from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from atha.assembly import EngineAssembler
from atha.config import (
    TransientSystem,
    build_performance_maps,
    evaluate_boundary_conditions,
    evaluate_dynamic_controllers,
    evaluate_operating_targets,
    evaluate_timing_events,
    load_analysis_config,
)
from atha.config.controllers import controller_evaluation_period, controller_sample_index
from atha.examples.common import coerce_numbers
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources


@dataclass
class GGSingleShaftSummary:
    config_path: Path
    component_count: int
    connection_count: int
    solver_status: str
    csv: Path | None = None
    plot: Path | None = None
    hdf5: Path | None = None
    manifest: Path | None = None
    residuals_csv: Path | None = None
    residuals_json: Path | None = None
    time: np.ndarray | None = None
    mdot_total: np.ndarray | None = None
    of_ratio: np.ndarray | None = None
    chamber_pressure: np.ndarray | None = None
    thrust: np.ndarray | None = None
    shaft_rpm: np.ndarray | None = None


def run_gg_single_shaft_transient(config_path: str | Path, output_dir: str | Path = "outputs") -> GGSingleShaftSummary:
    path = Path(config_path)
    loaded = load_analysis_config(path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = _design_model(loaded)
    extra_sources = {
        "target.mdot_total",
        "target.OF",
        "mdot.total",
        "chamber.OF",
        "chamber.P",
        "nozzle.thrust",
        "shaft.rpm",
        "shaft.omega",
        "lox_pump.inlet.mdot",
        "methane_pump.inlet.mdot",
        "generator.mdot",
        "generator.P",
        "lox_pump.efficiency",
        "methane_pump.efficiency",
        "lox_pump.delta_P",
        "methane_pump.delta_P",
        "lox_pump.phi",
        "methane_pump.phi",
        "lox_pump.psi",
        "methane_pump.psi",
        "lox_pump.inlet.h",
        "lox_pump.outlet.h",
        "methane_pump.inlet.h",
        "methane_pump.outlet.h",
    }
    extra_sources.update(f"{name}.command" for name in loaded.transients)
    extra_sources.update(f"{name}.position" for name in loaded.transients)
    extra_sources.update(
        {
            f"controller.{controller}.{field}"
            for controller in (loaded.controllers.controllers if loaded.controllers is not None else {})
            for field in ("error", "integral", "derivative", "raw_command", "proportional_term", "integral_term", "derivative_term")
        }
    )
    validate_telemetry_sources(loaded.telemetry, EngineAssembler(loaded).telemetry_sources(extra_sources))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = float(run["time"]["start_s"])
    tf = float(run["time"]["end_s"])
    transient_system = TransientSystem.from_configs(loaded.transients)
    commands0 = _commands(loaded, t0, _zero_measurements(run))
    x0 = np.concatenate(
        (
            np.asarray(
                [
                    _rpm_to_rad_s(float(run.get("initial_conditions", {}).get("shaft.rpm", 32000.0))),
                    float(run.get("initial_conditions", {}).get("chamber.P", run["design"]["chamber_pressure"])),
                    float(run.get("initial_conditions", {}).get("generator.P", 8.0e6)),
                ],
                dtype=float,
            ),
            transient_system.initial_state(commands0),
        )
    )

    controller_period_s = controller_evaluation_period(loaded.controllers)
    controller_cache: dict[int, dict[str, float]] = {}

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        states = _state_dict(y)
        x_transient = np.asarray(y[3:], dtype=float)
        timings = coerce_numbers(evaluate_timing_events(loaded.timings, t))
        transient_sources = transient_system.sample_sources(t, x_transient, timings)
        values = _plant_values(states, transient_sources, model)
        measurements = _measurements(values)
        commands = _commands(
            loaded,
            t,
            measurements,
            controller_period_s=controller_period_s,
            controller_cache=controller_cache,
        )
        transient_sources = transient_system.sample_sources(t, x_transient, {**timings, **commands})
        values = _plant_values(states, transient_sources, model)
        dtransients = transient_system.derivatives(t, x_transient, {**timings, **commands})
        pump_power = values["pump.power"]
        turbine_power = values["turbine.power"]
        domega = (turbine_power - pump_power) / max(model["shaft_inertia"] * max(states["shaft.omega"], 100.0), 1.0)
        domega -= model["shaft_friction"] * (states["shaft.omega"] - model["shaft_omega_design"])
        d_pc = model["chamber_pressure_gain"] * (values["mdot.total"] - values["nozzle.mdot"])
        d_pg = model["generator_pressure_gain"] * (values["generator.mdot_in"] - values["generator.mdot_out"])
        return np.concatenate((np.asarray([domega, d_pc, d_pg], dtype=float), dtransients))

    solver_cfg = loaded.analysis_config.solver.get("dae", loaded.analysis_config.solver.get("transient", {}))
    sol = solve_ivp(
        rhs,
        (t0, tf),
        x0,
        method=str(solver_cfg.get("method", "Radau")),
        rtol=float(solver_cfg.get("rtol", 1.0e-7)),
        atol=float(solver_cfg.get("atol", 1.0e-6)),
        max_step=float(solver_cfg.get("max_step", 0.005)),
    )
    if not sol.success:
        raise RuntimeError(f"GG single-shaft transient failed: {sol.message}")

    t = telemetry_times(loaded.telemetry, sol.t[0], sol.t[-1])
    y = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
    samples: list[dict[str, float]] = []
    mdot_total = np.zeros_like(t)
    of_ratio = np.zeros_like(t)
    chamber_pressure = y[:, 1]
    thrust = np.zeros_like(t)
    shaft_rpm = np.zeros_like(t)

    controller_cache = {}
    for i, ti in enumerate(t):
        states = _state_dict(y[i])
        x_transient = np.asarray(y[i, 3:], dtype=float)
        timings = coerce_numbers(evaluate_timing_events(loaded.timings, float(ti)))
        transient_sources = transient_system.sample_sources(float(ti), x_transient, timings)
        values = _plant_values(states, transient_sources, model)
        measurements = _measurements(values)
        commands = _commands(
            loaded,
            float(ti),
            measurements,
            controller_period_s=controller_period_s,
            controller_cache=controller_cache,
        )
        transient_sources = transient_system.sample_sources(float(ti), x_transient, {**timings, **commands})
        values = _plant_values(states, transient_sources, model)
        targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, float(ti)))
        mdot_total[i] = values["mdot.total"]
        of_ratio[i] = values["chamber.OF"]
        thrust[i] = values["nozzle.thrust"]
        shaft_rpm[i] = _rad_s_to_rpm(states["shaft.omega"])
        sample = {
            "time": float(ti),
            "target.mdot_total": float(targets["mdot_total"]),
            "target.OF": float(targets["OF"]),
            "mdot.total": float(values["mdot.total"]),
            "lox_pump.inlet.mdot": float(values["lox.mdot_total"]),
            "methane_pump.inlet.mdot": float(values["methane.mdot_total"]),
            "chamber.OF": float(values["chamber.OF"]),
            "chamber.P": float(states["chamber.P"]),
            "generator.P": float(states["generator.P"]),
            "generator.mdot": float(values["generator.mdot"]),
            "nozzle.thrust": float(values["nozzle.thrust"]),
            "shaft.omega": float(states["shaft.omega"]),
            "shaft.rpm": float(shaft_rpm[i]),
            **{key: float(value) for key, value in transient_sources.items()},
            **{key: float(value) for key, value in commands.items() if isinstance(value, (int, float, np.floating))},
        }
        sample.update({key: float(value) for key, value in values.items() if isinstance(value, (int, float, np.floating))})
        samples.append(sample)

    artifacts, _headers, _columns = OutputProcessor(
        output_dir=output_dir,
        telemetry_config=loaded.telemetry,
        run_output=run["output"],
        metadata={"analysis": loaded.analysis_config.name, "analysis_type": run.get("type", "")},
    ).write(samples, residuals={"reduced_model_residual": 0.0})
    return GGSingleShaftSummary(
        config_path=path,
        component_count=len(loaded.engine.components),
        connection_count=len(loaded.engine.connections),
        solver_status="solved reduced-order GG single-shaft transient",
        csv=artifacts.csv,
        plot=artifacts.plot,
        hdf5=artifacts.hdf5,
        manifest=artifacts.manifest,
        residuals_csv=artifacts.residuals_csv,
        residuals_json=artifacts.residuals_json,
        time=t,
        mdot_total=mdot_total,
        of_ratio=of_ratio,
        chamber_pressure=chamber_pressure,
        thrust=thrust,
        shaft_rpm=shaft_rpm,
    )


def _plant_values(states: dict[str, float], transients: dict[str, float], model: dict[str, float]) -> dict[str, float]:
    omega = max(states["shaft.omega"], 1.0)
    flow_scale = max(omega / model["shaft_omega_design"], 0.05)
    main_lox = _transient(transients, "main_lox_valve.position", 1.0)
    main_methane = _transient(transients, "main_methane_valve.position", 1.0)
    gen_lox = _transient(transients, "lox_generator_valve.position", 0.5)
    gen_methane = _transient(transients, "methane_generator_valve.position", 0.5)
    lox_capacity = model["mdot_lox_design"] * flow_scale * (0.10 + 0.85 * main_lox + 0.25 * gen_lox)
    methane_capacity = model["mdot_methane_design"] * flow_scale * (0.12 + 0.82 * main_methane + 0.30 * gen_methane)
    lox_main, lox_gen = _split_flow(lox_capacity, model["lox_split_fraction"], main_lox, gen_lox)
    methane_main, methane_gen = _split_flow(methane_capacity, model["methane_split_fraction"], main_methane, gen_methane)
    mdot_total = max(lox_main + methane_main, 0.0)
    of_ratio = lox_main / max(methane_main, 1.0e-9)
    generator_mdot = max(lox_gen + methane_gen, 0.0)
    generator_out = model["generator_conductance"] * max(states["generator.P"] - model["ambient_pressure"], 0.0)
    nozzle_mdot = model["nozzle_conductance"] * max(states["chamber.P"] - model["ambient_pressure"], 0.0)
    thrust = model["thrust_coefficient"] * model["nozzle_throat_area"] * max(states["chamber.P"] - model["ambient_pressure"], 0.0)
    lox_pump_dp, lox_pump_efficiency, lox_phi, lox_psi = _pump_map_point(model, "lox", omega, lox_capacity)
    methane_pump_dp, methane_pump_efficiency, methane_phi, methane_psi = _pump_map_point(model, "methane", omega, methane_capacity)
    pump_power = (
        lox_capacity * lox_pump_dp / max(lox_pump_efficiency, 1.0e-12)
        + methane_capacity * methane_pump_dp / max(methane_pump_efficiency, 1.0e-12)
    )
    lox_pump_outlet_h = model["lox_pump_inlet_h"] + lox_pump_dp / max(model["lox_pump_rho_design"] * lox_pump_efficiency, 1.0e-12)
    methane_pump_outlet_h = model["methane_pump_inlet_h"] + methane_pump_dp / max(model["methane_pump_rho_design"] * methane_pump_efficiency, 1.0e-12)
    turbine_power = model["turbine_power_gain"] * generator_out * (1.0 + 0.05 * generator_mdot)
    return {
        "lox.mdot_total": lox_main + lox_gen,
        "methane.mdot_total": methane_main + methane_gen,
        "mdot.total": mdot_total,
        "chamber.OF": of_ratio,
        "generator.mdot": generator_mdot,
        "generator.mdot_in": generator_mdot,
        "generator.mdot_out": generator_out,
        "nozzle.mdot": nozzle_mdot,
        "nozzle.thrust": thrust,
        "pump.power": pump_power,
        "lox_pump.delta_P": lox_pump_dp,
        "methane_pump.delta_P": methane_pump_dp,
        "lox_pump.phi": lox_phi,
        "methane_pump.phi": methane_phi,
        "lox_pump.psi": lox_psi,
        "methane_pump.psi": methane_psi,
        "lox_pump.efficiency": lox_pump_efficiency,
        "methane_pump.efficiency": methane_pump_efficiency,
        "lox_pump.inlet.h": model["lox_pump_inlet_h"],
        "methane_pump.inlet.h": model["methane_pump_inlet_h"],
        "lox_pump.outlet.h": lox_pump_outlet_h,
        "methane_pump.outlet.h": methane_pump_outlet_h,
        "turbine.power": turbine_power,
    }


def _design_model(loaded) -> dict[str, float]:
    run = coerce_numbers(loaded.analysis_config.analysis)
    components = loaded.engine.components
    design = run["design"]
    boundaries = coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, 0.0))
    mdot_total = float(design.get("mdot_total", 40.0))
    of_ratio = float(design.get("OF", 3.2))
    mdot_methane = mdot_total / (1.0 + of_ratio)
    mdot_lox = mdot_total - mdot_methane
    chamber_pressure = float(design.get("chamber_pressure", 5.0e6))
    generator_pressure = float(run.get("initial_conditions", {}).get("generator.P", 8.0e6))
    lox_pump = components["lox_pump"].parameters["pump_map"]
    methane_pump = components["methane_pump"].parameters["pump_map"]
    runtime_maps = build_performance_maps(loaded.maps)
    lox_speed_design = float(lox_pump.get("speed_design", 32000.0))
    methane_speed_design = float(methane_pump.get("speed_design", lox_speed_design))
    shaft_speed_design = float(design.get("shaft_speed_design_rpm", min(lox_speed_design, methane_speed_design)))
    generator_design_mdot = mdot_total * 0.08
    pump_power_design = (
        mdot_lox * float(lox_pump.get("dP_design", 12.0e6))
        + mdot_methane * float(methane_pump.get("dP_design", 12.5e6))
    ) / 0.72
    return {
        "mdot_total_design": mdot_total,
        "mdot_lox_design": mdot_lox,
        "mdot_methane_design": mdot_methane,
        "shaft_omega_design": _rpm_to_rad_s(shaft_speed_design),
        "shaft_inertia": float(components["shaft"].parameters.get("moment_of_inertia", 0.18)),
        "shaft_friction": float(components["shaft"].parameters.get("friction_coeff", 0.018)),
        "lox_pump_dp_design": float(lox_pump.get("dP_design", 12.0e6)),
        "methane_pump_dp_design": float(methane_pump.get("dP_design", 12.5e6)),
        "lox_pump_diameter": float(components["lox_pump"].parameters.get("diameter", 0.0)),
        "methane_pump_diameter": float(components["methane_pump"].parameters.get("diameter", 0.0)),
        "lox_pump_efficiency": float(lox_pump.get("efficiency_design", 0.74)),
        "methane_pump_efficiency": float(methane_pump.get("efficiency_design", 0.69)),
        "lox_pump_rho_design": float(lox_pump.get("rho_design", boundaries.get("lox_tank.outlet.rho", 1140.0))),
        "methane_pump_rho_design": float(methane_pump.get("rho_design", boundaries.get("methane_tank.outlet.rho", 422.0))),
        "lox_pump_head_map": runtime_maps.get(components["lox_pump"].maps.get("head_map").ref)
        if "head_map" in components["lox_pump"].maps
        else None,
        "methane_pump_head_map": runtime_maps.get(components["methane_pump"].maps.get("head_map").ref)
        if "head_map" in components["methane_pump"].maps
        else None,
        "lox_pump_efficiency_map": runtime_maps.get(components["lox_pump"].maps.get("efficiency_map").ref)
        if "efficiency_map" in components["lox_pump"].maps
        else None,
        "methane_pump_efficiency_map": runtime_maps.get(components["methane_pump"].maps.get("efficiency_map").ref)
        if "efficiency_map" in components["methane_pump"].maps
        else None,
        "lox_pump_inlet_h": float(boundaries.get("lox_tank.outlet.h", 0.0)),
        "methane_pump_inlet_h": float(boundaries.get("methane_tank.outlet.h", 0.0)),
        "pump_efficiency": 0.72,
        "lox_split_fraction": float(components["lox_splitter"].parameters.get("split_fraction", 0.92)),
        "methane_split_fraction": float(components["methane_splitter"].parameters.get("split_fraction", 0.92)),
        "ambient_pressure": float(boundaries.get("nozzle.ambient.P", 101325.0)),
        "nozzle_conductance": mdot_total / max(chamber_pressure - float(boundaries.get("nozzle.ambient.P", 101325.0)), 1.0),
        "nozzle_throat_area": float(components["nozzle"].parameters.get("throat_area", 2.11e-2)),
        "thrust_coefficient": float(components["nozzle"].parameters.get("thrust_coefficient", 1.45)),
        "chamber_pressure_gain": 420.0 * 3500.0 / float(components["chamber"].parameters.get("volume", 0.08)),
        "generator_pressure_gain": 420.0 * 1100.0 / float(components["generator"].parameters.get("volume", 0.004)),
        "generator_conductance": generator_design_mdot / max(generator_pressure - float(boundaries.get("nozzle.ambient.P", 101325.0)), 1.0),
        "turbine_power_gain": pump_power_design / max(generator_design_mdot * (1.0 + 0.05 * generator_design_mdot), 1.0),
    }


def _pump_map_point(model: dict[str, float], propellant: str, omega: float, mdot: float) -> tuple[float, float, float, float]:
    rho = max(float(model[f"{propellant}_pump_rho_design"]), 1.0e-12)
    diameter = max(float(model[f"{propellant}_pump_diameter"]), 1.0e-12)
    omega = max(float(omega), 1.0e-12)
    mdot = max(float(mdot), 0.0)
    phi = mdot / max(rho * omega * diameter**3, 1.0e-30)
    head_map = model.get(f"{propellant}_pump_head_map")
    if head_map is not None:
        head_values = head_map.evaluate({"phi": phi})
        psi = float(head_values.get("psi", head_values.get("head_coefficient", 0.0)))
    else:
        psi = float(model[f"{propellant}_pump_dp_design"]) / max(rho * model["shaft_omega_design"] ** 2 * diameter**2, 1.0e-30)
    delta_p = rho * psi * omega**2 * diameter**2
    efficiency_map = model.get(f"{propellant}_pump_efficiency_map")
    if efficiency_map is not None:
        efficiency_values = efficiency_map.evaluate({"phi": phi})
        efficiency = float(efficiency_values.get("eta", efficiency_values.get("efficiency", model[f"{propellant}_pump_efficiency"])))
    else:
        efficiency = float(model[f"{propellant}_pump_efficiency"])
    return delta_p, min(max(efficiency, 1.0e-6), 1.0), phi, psi


def _commands(
    loaded,
    t: float,
    measurements: dict[str, float],
    *,
    controller_period_s: float | None = None,
    controller_cache: dict[int, dict[str, float]] | None = None,
) -> dict[str, float]:
    targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, t))
    timings = coerce_numbers(evaluate_timing_events(loaded.timings, t))
    sample_index = controller_sample_index(t, controller_period_s)
    if sample_index is not None and controller_cache is not None and sample_index in controller_cache:
        controller_outputs = dict(controller_cache[sample_index])
    else:
        previous_outputs = (
            controller_cache.get(sample_index - 1, {})
            if sample_index is not None and controller_cache is not None
            else {}
        )
        controller_outputs = coerce_numbers(
            evaluate_dynamic_controllers(
                loaded.controllers,
                targets,
                timings,
                measurements,
                dt=controller_period_s or 1.0,
                current_phase=_current_phase(loaded, t),
                previous_outputs=previous_outputs,
            )
        )
        if sample_index is not None and controller_cache is not None:
            controller_cache[sample_index] = dict(controller_outputs)
    return _with_transient_command_defaults(loaded, {**timings, **controller_outputs})


def _with_transient_command_defaults(loaded, commands: dict[str, float]) -> dict[str, float]:
    completed = dict(commands)
    for transient in loaded.transients.values():
        path = transient.command.get("path")
        if not isinstance(path, str) or path in completed:
            continue
        initial = transient.state.get("initial", transient.parameters.get("initial", 0.0))
        completed[path] = float(initial)
    return completed


def _measurements(values: dict[str, float]) -> dict[str, float]:
    return {"mdot_total": values["mdot.total"], "OF": values["chamber.OF"]}


def _zero_measurements(run: dict[str, object]) -> dict[str, float]:
    design = run.get("design", {})
    return {
        "mdot_total": float(design.get("mdot_total", 0.0)) if isinstance(design, dict) else 0.0,
        "OF": float(design.get("OF", 0.0)) if isinstance(design, dict) else 0.0,
    }


def _current_phase(loaded, t: float) -> str | None:
    phases = loaded.analysis_config.analysis.get("time", {}).get("phases", [])
    if not isinstance(phases, list):
        return None
    time_end = float(loaded.analysis_config.analysis.get("time", {}).get("end_s", t))
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = phase.get("name")
        start = float(phase.get("start_s", phase.get("start", 0.0)))
        end = float(phase.get("end_s", phase.get("end", time_end)))
        if start <= float(t) < end or (float(t) == end and end == time_end):
            return str(name) if name is not None else None
    return None


def _state_dict(y: np.ndarray) -> dict[str, float]:
    return {
        "shaft.omega": max(float(y[0]), 100.0),
        "chamber.P": max(float(y[1]), 101325.0),
        "generator.P": max(float(y[2]), 101325.0),
    }


def _split_flow(total: float, split: float, main_position: float, generator_position: float) -> tuple[float, float]:
    main_weight = max(split, 1.0e-6) * max(main_position, 0.02)
    generator_weight = max(1.0 - split, 1.0e-6) * max(generator_position, 0.02)
    denom = main_weight + generator_weight
    return total * main_weight / denom, total * generator_weight / denom


def _transient(values: dict[str, float], path: str, default: float) -> float:
    return float(np.clip(values.get(path, default), 0.0, 1.0))


def _rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * np.pi / 30.0


def _rad_s_to_rpm(rad_s: float) -> float:
    return float(rad_s) * 30.0 / np.pi
