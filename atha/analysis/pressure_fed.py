from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from atha.assembly import EngineAssembler
from atha.components.registry import extract_engine_model
from atha.config import (
    TransientSystem,
    evaluate_boundary_conditions,
    evaluate_dynamic_controllers,
    evaluate_operating_targets,
    evaluate_timing_events,
    load_analysis_config,
)
from atha.config.loader import LoadedAnalysisConfig
from atha.config.schema import ComponentConfig, ConfigError
from atha.examples.common import coerce_numbers
from atha.analysis.linearization import (
    PerturbationConfig,
    StateSpaceLinearization,
    finite_difference_state_space,
    write_linearization_json,
)
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources
from atha.network import NetworkProblem, NetworkResidual, NetworkSolution, NetworkVariable


@dataclass(frozen=True)
class FeedLeg:
    prefix: str
    valve: str
    pipe: str
    injector: str
    chamber_port: str


class PressureFedTCASummary(SimpleNamespace):
    """Dynamic summary for pressure-fed chamber/nozzle examples."""


def run_pressure_fed_tca(config_path: str | Path, output_dir: str | Path = "outputs") -> PressureFedTCASummary:
    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**extract_engine_model(loaded.engine), **run.get("model", {})}
    time_cfg = run["time"]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    t0 = float(time_cfg["start_s"])
    tf = float(time_cfg["end_s"])
    legs = discover_pressure_fed_legs(loaded)
    transient_system = TransientSystem.from_configs(loaded.transients)
    commands0 = _commands(loaded, t0, {})
    x_transient0 = transient_system.initial_state(commands0)
    validate_telemetry_sources(
        loaded.telemetry,
        EngineAssembler(loaded).telemetry_sources(
            _telemetry_source_catalog(transient_system, legs, loaded.operating_conditions is not None)
        ),
    )
    network_problem = build_pressure_fed_algebraic_problem(legs, model)

    initial_mdots = [float(model.get(f"initial_mdot_{leg.prefix}", 0.0)) for leg in legs]
    x0 = np.concatenate(([float(model["initial_Pc"])], np.asarray(initial_mdots, dtype=float), x_transient0))
    z_guess = _initial_algebraic_guess(network_problem, legs, model)
    residual_report: dict[str, float] = {}

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal z_guess, residual_report
        pc = max(float(y[0]), 1.0)
        mdots = {leg.prefix: max(float(y[1 + i]), 0.0) for i, leg in enumerate(legs)}
        x_transient = np.asarray(y[1 + len(legs):], dtype=float)
        commands = _commands(loaded, t, mdots)
        values = _values(loaded, t)
        transient_sources = transient_system.sample_sources(t, x_transient, commands)
        algebraic = _solve_algebraic(network_problem, z_guess, t, legs, pc, values, transient_sources)
        z_guess = algebraic.z
        residual_report = algebraic.normalized_residuals
        mdot_nozzle = algebraic.values["nozzle.mdot"]
        dpc = (float(model["gas_R"]) * float(model["chamber_T"]) / float(model["chamber_volume"])) * (
            sum(mdots.values()) - mdot_nozzle
        )
        dmdots = [
            (algebraic.values[f"{leg.pipe}.mdot_steady"] - mdots[leg.prefix])
            / max(float(model[f"{leg.pipe}_time_constant"]), 1.0e-12)
            for leg in legs
        ]
        dtransients = transient_system.derivatives(t, x_transient, commands)
        return np.concatenate(([dpc], dmdots, dtransients))

    solver_cfg = loaded.analysis_config.solver["transient"]
    sol = solve_ivp(
        rhs,
        (t0, tf),
        x0,
        method=solver_cfg.get("method", "Radau"),
        rtol=float(solver_cfg.get("rtol", 1.0e-7)),
        atol=float(solver_cfg.get("atol", 1.0e-6)),
        max_step=float(solver_cfg.get("max_step", 0.005)),
    )
    if not sol.success:
        raise RuntimeError(f"Pressure-fed TCA transient failed: {sol.message}")

    t = telemetry_times(loaded.telemetry, sol.t[0], sol.t[-1])
    y = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
    pc = y[:, 0]
    mdot_by_prefix = {
        leg.prefix: np.maximum(y[:, 1 + i], 0.0)
        for i, leg in enumerate(legs)
    }
    x_transients = y[:, 1 + len(legs):]
    thrust = np.zeros_like(t)
    valve_positions = {leg.prefix: np.zeros_like(t) for leg in legs}
    of_ratio = _mixture_ratio_array(mdot_by_prefix)
    samples: list[dict[str, float]] = []

    for i, ti in enumerate(t):
        mdot_measurements = {prefix: float(values[i]) for prefix, values in mdot_by_prefix.items()}
        commands = _commands(loaded, float(ti), mdot_measurements)
        targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, float(ti))) if loaded.operating_conditions else {}
        values = _values(loaded, float(ti))
        transient_sources = transient_system.sample_sources(float(ti), x_transients[i], commands)
        algebraic = _solve_algebraic(network_problem, z_guess, float(ti), legs, float(pc[i]), values, transient_sources)
        z_guess = algebraic.z
        thrust[i] = _thrust(pc[i], values["nozzle.ambient.P"], model)
        sample = {
            "time": float(ti),
            "chamber.P": float(pc[i]),
            "chamber.OF": float(of_ratio[i]),
            "mdot.total": float(sum(mdot_measurements.values())),
            "target.mdot_total": float(targets.get("mdot_total", np.nan)),
            "nozzle.mdot": float(algebraic.values["nozzle.mdot"]),
            "nozzle.thrust": float(thrust[i]),
            **transient_sources,
            **{key: float(value) for key, value in commands.items() if isinstance(value, (int, float, np.floating))},
        }
        for leg in legs:
            position = float(transient_sources[f"{leg.valve}.position"])
            valve_positions[leg.prefix][i] = position
            mdot = mdot_measurements[leg.prefix]
            sample[f"{leg.valve}.command"] = float(commands[f"{leg.valve}.command"])
            sample[f"{leg.pipe}.mdot"] = float(mdot)
            sample[f"{leg.injector}.mdot"] = float(mdot)
        samples.append(sample)

    artifacts, _headers, _columns = OutputProcessor(
        output_dir=output_dir,
        telemetry_config=loaded.telemetry,
        run_output=run["output"],
        metadata={"analysis": loaded.analysis_config.name, "analysis_type": run.get("type", "")},
    ).write(
        samples,
        residuals=residual_report,
        state_history={
            "chamber.P": pc,
            **{f"{leg.pipe}.mdot": values for leg, values in zip(legs, mdot_by_prefix.values())},
        },
    )

    summary = _build_summary(artifacts.csv, artifacts.plot, t, pc, thrust, legs, valve_positions, mdot_by_prefix, of_ratio)
    summary.hdf5 = artifacts.hdf5
    summary.manifest = artifacts.manifest
    summary.residuals_csv = artifacts.residuals_csv
    summary.residuals_json = artifacts.residuals_json
    summary.algebraic_names = network_problem.variable_names
    summary.residual_names = network_problem.residual_names
    summary.max_normalized_residual = max((abs(value) for value in residual_report.values()), default=0.0)
    return summary


def linearize_pressure_fed_tca(
    config_path: str | Path,
    output_dir: str | Path = "outputs",
) -> StateSpaceLinearization:
    """Finite-difference linearization around a pressure-fed steady trim."""

    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**extract_engine_model(loaded.engine), **run.get("model", {})}
    legs = discover_pressure_fed_legs(loaded)
    network_problem = build_pressure_fed_algebraic_problem(legs, model)
    t_lin = float(run.get("linearization", {}).get("time_s", run.get("time", {}).get("end_s", 0.0)))
    values = _values(loaded, t_lin)
    commands = _commands(loaded, t_lin, {})
    transient_system = TransientSystem.from_configs(loaded.transients)
    transient_state = transient_system.initial_state(commands)
    transient_sources = transient_system.sample_sources(t_lin, transient_state, commands)
    pc_guess = float(model.get("initial_Pc", run.get("design", {}).get("chamber_pressure", 1.0e6)))
    z_guess = _initial_algebraic_guess(network_problem, legs, model)
    trim = solve_pressure_fed_steady_state(network_problem, legs, values, transient_sources, pc_guess, z_guess=z_guess)

    state_labels = ["chamber.P", *[f"{leg.pipe}.mdot" for leg in legs]]
    input_labels = [f"{leg.valve}.position" for leg in legs]
    output_labels = ["chamber.P", "mdot.total", "nozzle.thrust", *[f"{leg.pipe}.mdot" for leg in legs]]
    x0 = np.array([trim["chamber.P"], *[trim[f"{leg.pipe}.mdot_steady"] for leg in legs]], dtype=float)
    u0 = np.array([float(transient_sources[f"{leg.valve}.position"]) for leg in legs], dtype=float)

    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pc = max(float(x[0]), 1.0)
        mdots = {leg.prefix: max(float(x[1 + i]), 0.0) for i, leg in enumerate(legs)}
        local_sources = dict(transient_sources)
        for leg, position in zip(legs, u):
            local_sources[f"{leg.valve}.position"] = float(np.clip(position, 0.0, 1.0))
        algebraic = _solve_algebraic(network_problem, z_guess, t_lin, legs, pc, values, local_sources)
        dpc = (float(model["gas_R"]) * float(model["chamber_T"]) / float(model["chamber_volume"])) * (
            sum(mdots.values()) - algebraic.values["nozzle.mdot"]
        )
        dmdots = [
            (algebraic.values[f"{leg.pipe}.mdot_steady"] - mdots[leg.prefix])
            / max(float(model[f"{leg.pipe}_time_constant"]), 1.0e-12)
            for leg in legs
        ]
        return np.array([dpc, *dmdots], dtype=float)

    def outputs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pc = max(float(x[0]), 1.0)
        mdots = [max(float(x[1 + i]), 0.0) for i in range(len(legs))]
        return np.array([pc, sum(mdots), _thrust(pc, values["nozzle.ambient.P"], model), *mdots], dtype=float)

    linearization_cfg = run.get("linearization", {}) if isinstance(run.get("linearization", {}), dict) else {}
    perturbation_cfg = _perturbation_config(linearization_cfg.get("perturbations", {}))
    result = finite_difference_state_space(
        dynamics,
        outputs,
        x0,
        u0,
        state_labels=state_labels,
        input_labels=input_labels,
        output_labels=output_labels,
        perturbations=perturbation_cfg,
    )
    path = Path(output_dir) / str(linearization_cfg.get("output", Path(str(run["output"]["csv"])).with_suffix(".linearization.json").name))
    write_linearization_json(path, result)
    return result


def build_pressure_fed_algebraic_problem(legs: list[FeedLeg], model: dict[str, Any]) -> NetworkProblem:
    variables = [
        NetworkVariable(f"{leg.pipe}.mdot_steady", units="kg/s", scale=max(float(model.get(f"initial_mdot_{leg.prefix}", 1.0)), 1.0))
        for leg in legs
    ]
    variables.append(NetworkVariable("nozzle.mdot", units="kg/s", scale=max(sum(variable.scale for variable in variables), 1.0)))
    residuals = [
        NetworkResidual(f"{leg.pipe}.mdot_steady_residual", units="kg/s", scale=max(float(model.get(f"initial_mdot_{leg.prefix}", 1.0)), 1.0))
        for leg in legs
    ]
    residuals.append(NetworkResidual("nozzle.mdot_residual", units="kg/s", scale=max(sum(residual.scale for residual in residuals), 1.0)))

    def evaluate(_t: float, z: Mapping[str, float], inputs: Mapping[str, float]) -> dict[str, float]:
        pc = float(inputs["chamber.P"])
        values = dict(inputs["boundaries"])
        transient_sources = dict(inputs["transients"])
        result = {}
        for leg in legs:
            steady = _leg_mdot_steady(leg, transient_sources[f"{leg.valve}.position"], pc, values, model)
            result[f"{leg.pipe}.mdot_steady_residual"] = float(z[f"{leg.pipe}.mdot_steady"] - steady)
        result["nozzle.mdot_residual"] = float(z["nozzle.mdot"] - _nozzle_mdot(pc, values["nozzle.ambient.P"], model))
        return result

    return NetworkProblem(variables, residuals, evaluate, name="pressure_fed_tca")


def solve_pressure_fed_steady_state(
    network_problem: NetworkProblem,
    legs: list[FeedLeg],
    values: dict[str, Any],
    transient_sources: dict[str, float],
    pc_guess: float,
    z_guess: np.ndarray | None = None,
) -> dict[str, float]:
    """Trim chamber pressure with the same algebraic residual assembly used in transient RHS."""

    z0 = network_problem.initial_z if z_guess is None else np.asarray(z_guess, dtype=float)
    unknown0 = np.concatenate(([float(pc_guess)], z0))
    mdot_scale = max(float(sum(abs(value) for value in z0)), 1.0)
    pc_scale = max(abs(float(pc_guess)), 1.0)

    def residual(unknown: np.ndarray) -> np.ndarray:
        pc = max(float(unknown[0]), 1.0)
        z = np.asarray(unknown[1:], dtype=float)
        algebraic = network_problem.residual_vector(
            0.0,
            z,
            {
                "chamber.P": pc,
                "boundaries": values,
                "transients": transient_sources,
            },
        )
        z_values = network_problem.values_from_z(z)
        mass_balance = sum(z_values[f"{leg.pipe}.mdot_steady"] for leg in legs) - z_values["nozzle.mdot"]
        return np.concatenate((algebraic / network_problem.residual_scales, [mass_balance / mdot_scale]))

    result = root(residual, unknown0, method="hybr")
    if not result.success:
        raise RuntimeError(f"pressure-fed steady trim failed: {result.message}")

    pc = max(float(result.x[0]), 1.0)
    z = np.asarray(result.x[1:], dtype=float)
    solution = network_problem.solve(
        0.0,
        z,
        {
            "chamber.P": pc,
            "boundaries": values,
            "transients": transient_sources,
        },
    )
    return {
        "chamber.P": pc,
        "normalized_pc": pc / pc_scale,
        **solution.values,
    }


def discover_pressure_fed_legs(loaded: LoadedAnalysisConfig) -> list[FeedLeg]:
    components = loaded.engine.components
    outgoing = {name: [] for name in components}
    for conn in loaded.engine.connections:
        source_component, _ = conn.source.split(".", 1)
        target_component, target_port = conn.target.split(".", 1)
        outgoing.setdefault(source_component, []).append((target_component, target_port))

    legs: list[FeedLeg] = []
    for valve in _components_of_type(components, "Valve"):
        pipe = _single_downstream_of_type(valve.name, "Pipe", components, outgoing)
        injector = _single_downstream_of_type(pipe.name, "MassFlowInjector", components, outgoing)
        chamber_name, chamber_port = _single_downstream(injector.name, components, outgoing)
        chamber = components[chamber_name]
        if chamber.type != "CombustionChamber":
            raise ConfigError(f"injector '{injector.name}' must feed a CombustionChamber, got '{chamber.type}'")
        legs.append(
            FeedLeg(
                prefix=_leg_prefix(valve.name),
                valve=valve.name,
                pipe=pipe.name,
                injector=injector.name,
                chamber_port=chamber_port,
            )
        )
    if not legs:
        raise ConfigError("pressure-fed TCA analysis requires at least one Valve -> Pipe -> MassFlowInjector leg")
    return legs


def _initial_algebraic_guess(
    network_problem: NetworkProblem,
    legs: list[FeedLeg],
    model: dict[str, Any],
) -> np.ndarray:
    values = {}
    for leg in legs:
        values[f"{leg.pipe}.mdot_steady"] = float(model.get(f"initial_mdot_{leg.prefix}", 0.0))
    values["nozzle.mdot"] = sum(values.values())
    return np.array([values[name] for name in network_problem.variable_names], dtype=float)


def _solve_algebraic(
    network_problem: NetworkProblem,
    z_guess: np.ndarray,
    t: float,
    legs: list[FeedLeg],
    pc: float,
    values: dict[str, Any],
    transient_sources: dict[str, float],
) -> NetworkSolution:
    _ = legs
    solution = network_problem.solve(
        t,
        z_guess,
        {
            "chamber.P": float(pc),
            "boundaries": values,
            "transients": transient_sources,
        },
    )
    name, residual = solution.max_normalized_residual
    if not solution.success or abs(residual) > 1.0e-6:
        raise RuntimeError(
            f"pressure-fed algebraic solve failed at t={t:.6g}s: "
            f"{solution.message}; max residual {name}={residual:.3e}"
        )
    return solution


def _components_of_type(components: dict[str, ComponentConfig], component_type: str) -> list[ComponentConfig]:
    return [component for component in components.values() if component.type == component_type]


def _single_downstream_of_type(
    name: str,
    component_type: str,
    components: dict[str, ComponentConfig],
    outgoing: dict[str, list[tuple[str, str]]],
) -> ComponentConfig:
    candidates = [components[target] for target, _ in outgoing.get(name, []) if components[target].type == component_type]
    if len(candidates) != 1:
        raise ConfigError(f"component '{name}' must feed exactly one {component_type}; found {len(candidates)}")
    return candidates[0]


def _single_downstream(
    name: str,
    components: dict[str, ComponentConfig],
    outgoing: dict[str, list[tuple[str, str]]],
) -> tuple[str, str]:
    targets = outgoing.get(name, [])
    if len(targets) != 1:
        raise ConfigError(f"component '{name}' must have exactly one downstream connection; found {len(targets)}")
    target, port = targets[0]
    if target not in components:
        raise ConfigError(f"component '{name}' connects to unknown component '{target}'")
    return target, port


def _leg_prefix(valve_name: str) -> str:
    if valve_name.endswith("_valve"):
        return valve_name[: -len("_valve")]
    if valve_name.startswith("valve_"):
        return valve_name[len("valve_"):]
    return valve_name


def _values(loaded: LoadedAnalysisConfig, t: float) -> dict[str, Any]:
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _commands(loaded: LoadedAnalysisConfig, t: float, mdots: dict[str, float]) -> dict[str, float]:
    timings = coerce_numbers(evaluate_timing_events(loaded.timings, t))
    targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, t)) if loaded.operating_conditions else {}
    measurements = dict(mdots)
    measurements["mdot_total"] = sum(mdots.values())
    controller_outputs = evaluate_dynamic_controllers(loaded.controllers, targets, timings, measurements)
    commands = dict(timings)
    commands.update(coerce_numbers(controller_outputs))
    return commands


def _telemetry_source_catalog(transient_system: TransientSystem, legs: list[FeedLeg], has_operating_targets: bool) -> set[str]:
    sources = {
        "time",
        "chamber.P",
        "chamber.OF",
        "mdot.total",
        "nozzle.mdot",
        "nozzle.thrust",
    } | transient_system.source_catalog()
    if has_operating_targets:
        sources.add("target.mdot_total")
    for leg in legs:
        sources.update(
            {
                f"{leg.valve}.command",
                f"{leg.pipe}.mdot",
                f"{leg.injector}.mdot",
            }
        )
    return sources


def _leg_mdot_steady(leg: FeedLeg, valve_position: float, pc: float, values: dict[str, Any], model: dict[str, Any]) -> float:
    p_supply = float(values[f"{leg.prefix}_supply.P"])
    rho = float(values[f"{leg.prefix}_supply.rho"])
    d_p = max(p_supply - pc - float(model[f"{leg.injector}_delta_P"]), 0.0)
    cda = float(model[f"{leg.valve}_CdA"]) * max(min(float(valve_position), 1.0), 0.0)
    return cda * (2.0 * rho * d_p) ** 0.5


def _nozzle_mdot(pc: float, p_ambient: float, model: dict[str, Any]) -> float:
    if pc <= p_ambient:
        return 0.0
    return float(model["nozzle_conductance"]) * (pc - p_ambient)


def _thrust(pc: float, p_ambient: float, model: dict[str, Any]) -> float:
    return float(model["thrust_coefficient"]) * float(model["nozzle_throat_area"]) * max(pc - p_ambient, 0.0)


def _mixture_ratio_array(mdot_by_prefix: dict[str, np.ndarray]) -> np.ndarray:
    if "lox" in mdot_by_prefix and "methane" in mdot_by_prefix:
        return np.divide(
            mdot_by_prefix["lox"],
            mdot_by_prefix["methane"],
            out=np.full_like(mdot_by_prefix["lox"], np.nan),
            where=mdot_by_prefix["methane"] > 1.0e-12,
        )
    return np.full_like(next(iter(mdot_by_prefix.values())), np.nan)


def _build_summary(
    csv: Path,
    plot: Path,
    time: np.ndarray,
    chamber_pressure: np.ndarray,
    thrust: np.ndarray,
    legs: list[FeedLeg],
    valve_positions: dict[str, np.ndarray],
    mdots: dict[str, np.ndarray],
    of_ratio: np.ndarray,
) -> PressureFedTCASummary:
    summary = PressureFedTCASummary(
        csv=csv,
        plot=plot,
        time=time,
        chamber_pressure=chamber_pressure,
        thrust=thrust,
        of_ratio=of_ratio,
        valve_positions=valve_positions,
        mdots=mdots,
    )
    for leg in legs:
        setattr(summary, f"{leg.valve}_position", valve_positions[leg.prefix])
        setattr(summary, f"mdot_{leg.prefix}", mdots[leg.prefix])
    return summary


def _perturbation_config(raw: Any) -> PerturbationConfig:
    if not isinstance(raw, dict):
        raw = {}
    return PerturbationConfig(
        state_default=float(raw.get("state_default", 1.0e-6)),
        input_default=float(raw.get("input_default", 1.0e-6)),
        minimum_absolute=float(raw.get("minimum_absolute", 1.0e-9)),
        per_state={str(k): float(v) for k, v in raw.get("per_state", {}).items()}
        if isinstance(raw.get("per_state", {}), dict)
        else None,
        per_input={str(k): float(v) for k, v in raw.get("per_input", {}).items()}
        if isinstance(raw.get("per_input", {}), dict)
        else None,
    )
