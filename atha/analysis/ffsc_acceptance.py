from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

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
from atha.config.controllers import controller_evaluation_period, controller_sample_index
from atha.examples.common import coerce_numbers
from atha.analysis.linearization import (
    PerturbationConfig,
    StateSpaceLinearization,
    finite_difference_state_space,
    write_linearization_json,
)
from atha.network import NetworkProblem, NetworkResidual, NetworkVariable
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources
from atha.analysis.transient_reporting import transient_run_provenance
from atha.validation.acceptance import build_ffsc_reduced_acceptance_report, write_acceptance_report_json


@dataclass
class FFSCDAEAcceptanceSummary:
    config_path: Path
    component_count: int
    connection_count: int
    map_count: int
    transient_count: int
    target_samples: dict[float, dict[str, float]]
    solver_status: str = "solved"
    csv: Path | None = None
    plot: Path | None = None
    hdf5: Path | None = None
    manifest: Path | None = None
    residuals_csv: Path | None = None
    residuals_json: Path | None = None
    linearization: Path | None = None
    acceptance_report: Path | None = None
    acceptance_passed: bool | None = None
    time: np.ndarray | None = None
    mdot_total: np.ndarray | None = None
    of_ratio: np.ndarray | None = None
    chamber_pressure: np.ndarray | None = None
    thrust: np.ndarray | None = None
    lox_shaft_rpm: np.ndarray | None = None
    methane_shaft_rpm: np.ndarray | None = None


def validate_ffsc_dae_acceptance(config_path: str | Path, output_dir: str | Path = "outputs") -> FFSCDAEAcceptanceSummary:
    path = Path(config_path)
    loaded = load_analysis_config(path)
    validate_telemetry_sources(loaded.telemetry, EngineAssembler(loaded).telemetry_sources())
    samples = {
        t: evaluate_operating_targets(loaded.operating_conditions, t)
        for t in (0.0, 10.0, 25.0)
    }
    return run_ffsc_dae_transient(path, output_dir=output_dir, target_samples=samples)


def run_ffsc_dae_transient(
    config_path: str | Path,
    output_dir: str | Path = "outputs",
    target_samples: dict[float, dict[str, float]] | None = None,
) -> FFSCDAEAcceptanceSummary:
    """Run the first YAML-driven FFSC DAE acceptance transient.

    This runner is intentionally conservative: it solves a reduced-order DAE
    network for the configured FFSC topology, with differential states for
    shafts, chamber/preburner pressures, and transient valve actuators. The
    algebraic block computes pump-capacity branch flows, nozzle flow, thrust,
    OF, and residual diagnostics at every RHS/sample call. It is not yet the
    final ROCETS port-property solver, but it executes example 19 from YAML
    rather than using the previous configuration-only validator.
    """

    path = Path(config_path)
    loaded = load_analysis_config(path)
    validate_telemetry_sources(loaded.telemetry, EngineAssembler(loaded).telemetry_sources())
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**extract_engine_model(loaded.engine), **_ffsc_design_model(loaded)}
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    t0 = float(run["time"]["start_s"])
    tf = float(run["time"]["end_s"])
    transient_system = TransientSystem.from_configs(loaded.transients)
    commands0 = _commands(loaded, t0, _zero_measurements(run))
    x_transient0 = transient_system.initial_state(commands0)
    initial = _initial_conditions(run)
    x0 = np.concatenate(
        (
            np.array(
                [
                    initial.get("lox_shaft.omega", _rpm_to_rad_s(32000.0)),
                    initial.get("methane_shaft.omega", _rpm_to_rad_s(27000.0)),
                    initial.get("chamber.P", float(run["design"]["chamber_pressure"])),
                    initial.get("ox_preburner.P", 8.0e6),
                    initial.get("fuel_preburner.P", 8.0e6),
                ],
                dtype=float,
            ),
            x_transient0,
        )
    )

    algebraic_problem = build_ffsc_reduced_algebraic_problem(model)
    z_guess = algebraic_problem.initial_z.copy()
    residual_report: dict[str, float] = {}
    controller_period_s = controller_evaluation_period(loaded.controllers)
    controller_cache: dict[int, dict[str, float]] = {}

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal z_guess, residual_report
        states = _state_dict(y)
        x_transient = np.asarray(y[5:], dtype=float)
        timings = coerce_numbers(evaluate_timing_events(loaded.timings, t))
        transient_sources = transient_system.sample_sources(t, x_transient, timings)
        algebraic = _solve_ffsc_algebraic(algebraic_problem, z_guess, t, states, transient_sources, model)
        z_guess = algebraic.z
        residual_report = algebraic.normalized_residuals
        measurements = _measurements(algebraic.values)
        commands = _commands(loaded, t, measurements, controller_period_s=controller_period_s, controller_cache=controller_cache)
        transient_sources = transient_system.sample_sources(t, x_transient, commands)
        algebraic = _solve_ffsc_algebraic(algebraic_problem, z_guess, t, states, transient_sources, model)
        z_guess = algebraic.z
        residual_report = algebraic.normalized_residuals
        values = algebraic.values

        dtransients = transient_system.derivatives(t, x_transient, commands)
        lox_power_balance = values["lox_turbine.power"] - values["lox_pump.power"]
        methane_power_balance = values["methane_turbine.power"] - values["methane_pump.power"]
        domega_lox = lox_power_balance / max(model["lox_shaft_inertia"] * max(states["lox_shaft.omega"], 100.0), 1.0)
        domega_methane = methane_power_balance / max(
            model["methane_shaft_inertia"] * max(states["methane_shaft.omega"], 100.0),
            1.0,
        )
        domega_lox -= model["lox_shaft_friction"] * (states["lox_shaft.omega"] - model["lox_omega_design"])
        domega_methane -= model["methane_shaft_friction"] * (
            states["methane_shaft.omega"] - model["methane_omega_design"]
        )

        d_ox_pb = model["ox_preburner_pressure_gain"] * (
            values["ox_preburner.mdot_in"] - values["ox_preburner.mdot_out"]
        )
        d_fuel_pb = model["fuel_preburner_pressure_gain"] * (
            values["fuel_preburner.mdot_in"] - values["fuel_preburner.mdot_out"]
        )
        d_pc = model["chamber_pressure_gain"] * (values["mdot.total"] - values["nozzle.mdot"])
        derivatives = np.array([domega_lox, domega_methane, d_pc, d_ox_pb, d_fuel_pb], dtype=float)
        return np.concatenate((derivatives, dtransients))

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
        raise RuntimeError(f"FFSC DAE transient failed: {sol.message}")

    t = telemetry_times(loaded.telemetry, sol.t[0], sol.t[-1])
    y = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
    samples: list[dict[str, float]] = []
    state_history: dict[str, list[float]] = {
        "lox_shaft.omega": [],
        "methane_shaft.omega": [],
        "chamber.P": [],
        "ox_preburner.P": [],
        "fuel_preburner.P": [],
    }
    algebraic_history: dict[str, list[float]] = {name: [] for name in algebraic_problem.variable_names}
    residual_history: dict[str, list[float]] = {name: [] for name in algebraic_problem.residual_names}
    boundary_history: dict[str, list[float]] = {}

    mdot_total = np.zeros_like(t)
    of_ratio = np.zeros_like(t)
    chamber_pressure = y[:, 2]
    thrust = np.zeros_like(t)
    lox_rpm = np.zeros_like(t)
    methane_rpm = np.zeros_like(t)
    target_mdot_total = np.zeros_like(t)
    target_of = np.zeros_like(t)

    for i, ti in enumerate(t):
        states = _state_dict(y[i])
        x_transient = np.asarray(y[i, 5:], dtype=float)
        timings = coerce_numbers(evaluate_timing_events(loaded.timings, float(ti)))
        transient_sources = transient_system.sample_sources(float(ti), x_transient, timings)
        algebraic = _solve_ffsc_algebraic(algebraic_problem, z_guess, float(ti), states, transient_sources, model)
        measurements = _measurements(algebraic.values)
        commands = _commands(loaded, float(ti), measurements, controller_period_s=controller_period_s, controller_cache=controller_cache)
        transient_sources = transient_system.sample_sources(float(ti), x_transient, commands)
        algebraic = _solve_ffsc_algebraic(algebraic_problem, z_guess, float(ti), states, transient_sources, model)
        z_guess = algebraic.z
        targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, float(ti)))
        boundaries = coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, float(ti)))
        for key, value in boundaries.items():
            if isinstance(value, (int, float, np.floating)):
                boundary_history.setdefault(key, []).append(float(value))
        values = algebraic.values
        mdot_total[i] = values["mdot.total"]
        of_ratio[i] = values["chamber.OF"]
        target_mdot_total[i] = float(targets["mdot_total"])
        target_of[i] = float(targets["OF"])
        thrust[i] = values["nozzle.thrust"]
        lox_rpm[i] = _rad_s_to_rpm(states["lox_shaft.omega"])
        methane_rpm[i] = _rad_s_to_rpm(states["methane_shaft.omega"])
        for key in state_history:
            state_history[key].append(float(states[key]))
        for key in algebraic_history:
            algebraic_history[key].append(float(values[key]))
        for key in residual_history:
            residual_history[key].append(float(algebraic.normalized_residuals[key]))

        sample = {
            "time": float(ti),
            "target.mdot_total": float(targets["mdot_total"]),
            "target.OF": float(targets["OF"]),
            "mdot.total": float(values["mdot.total"]),
            "lox_pump.inlet_mdot": float(values["lox.mdot_total"]),
            "methane_pump.inlet_mdot": float(values["methane.mdot_total"]),
            "lox_pump.inlet.mdot": float(values["lox.mdot_total"]),
            "methane_pump.inlet.mdot": float(values["methane.mdot_total"]),
            "chamber.OF": float(values["chamber.OF"]),
            "chamber.P": float(states["chamber.P"]),
            "nozzle.thrust": float(values["nozzle.thrust"]),
            "lox_shaft.omega": float(states["lox_shaft.omega"]),
            "lox_shaft.rpm": float(lox_rpm[i]),
            "methane_shaft.omega": float(states["methane_shaft.omega"]),
            "methane_shaft.rpm": float(methane_rpm[i]),
            **{key: float(value) for key, value in transient_sources.items()},
            **{key: float(value) for key, value in commands.items() if isinstance(value, (int, float, np.floating))},
            **{f"target.{key}": float(value) for key, value in targets.items() if isinstance(value, (int, float, np.floating))},
            **{f"timings.{key}": float(value) for key, value in timings.items() if isinstance(value, (int, float, np.floating))},
        }
        sample.update({key: float(value) for key, value in values.items() if isinstance(value, (int, float, np.floating))})
        samples.append(sample)

    linearization_path = None
    linearization_cfg = run.get("linearization", {}) if isinstance(run.get("linearization", {}), dict) else {}
    if bool(linearization_cfg.get("enabled", True)):
        linearization = linearize_ffsc_reduced_operating_point(
            x0=y[-1, :5],
            u0=np.array(
                [
                    samples[-1]["main_lox_valve.position"],
                    samples[-1]["main_methane_valve.position"],
                    samples[-1]["methane_crossover_valve.position"],
                    samples[-1]["lox_crossover_valve.position"],
                ],
                dtype=float,
            ),
            model=model,
            perturbations=_perturbation_config(linearization_cfg.get("perturbations", {})),
        )
        linearization_path = write_linearization_json(
            output_dir / str(linearization_cfg.get("output", Path(str(run["output"]["csv"])).with_suffix(".linearization.json").name)),
            linearization,
        )
    acceptance_cfg = run.get("acceptance", {}) if isinstance(run.get("acceptance", {}), dict) else {}
    acceptance_mask = _acceptance_mask(t, acceptance_cfg)
    acceptance_report = build_ffsc_reduced_acceptance_report(
        time=t[acceptance_mask],
        mdot_total=mdot_total[acceptance_mask],
        target_mdot_total=target_mdot_total[acceptance_mask],
        of_ratio=of_ratio[acceptance_mask],
        target_of=target_of[acceptance_mask],
        thrust=thrust[acceptance_mask],
        lox_shaft_rpm=lox_rpm[acceptance_mask],
        methane_shaft_rpm=methane_rpm[acceptance_mask],
        residuals=residual_report,
        linearization_path=linearization_path,
        tolerances=acceptance_cfg.get("tolerances", {}),
    )
    acceptance_report_path = write_acceptance_report_json(
        output_dir / str(run.get("acceptance", {}).get("report", Path(str(run["output"]["csv"])).with_suffix(".acceptance.json").name))
        if isinstance(run.get("acceptance", {}), dict)
        else output_dir / Path(str(run["output"]["csv"])).with_suffix(".acceptance.json").name,
        acceptance_report,
    )
    provenance = transient_run_provenance(
        loaded=loaded,
        config_path=path,
        run=run,
        residuals=residual_report,
        residual_history={key: np.asarray(value, dtype=float) for key, value in residual_history.items()},
        acceptance_report=acceptance_report_path,
        acceptance_passed=acceptance_report.passed,
        extra={
            "runner": "ffsc_reduced_dae_transient",
            "linearization": None if linearization_path is None else str(linearization_path),
        },
    )
    artifacts, _headers, _columns = OutputProcessor(
        output_dir=output_dir,
        telemetry_config=loaded.telemetry,
        run_output=run["output"],
        metadata={
            "analysis": loaded.analysis_config.name,
            "analysis_type": run.get("type", ""),
            "provenance": provenance,
        },
    ).write(
        samples,
        residuals=residual_report,
        state_history={key: np.asarray(value, dtype=float) for key, value in state_history.items()},
        algebraic_history={key: np.asarray(value, dtype=float) for key, value in algebraic_history.items()},
        residual_history={key: np.asarray(value, dtype=float) for key, value in residual_history.items()},
        boundary_history={key: np.asarray(value, dtype=float) for key, value in boundary_history.items()},
    )

    if target_samples is None:
        target_samples = {ts: evaluate_operating_targets(loaded.operating_conditions, ts) for ts in (0.0, 10.0, 25.0)}
    return FFSCDAEAcceptanceSummary(
        config_path=path,
        component_count=len(loaded.engine.components),
        connection_count=len(loaded.engine.connections),
        map_count=len(loaded.maps),
        transient_count=len(loaded.transients),
        target_samples=target_samples,
        solver_status="solved reduced-order FFSC DAE network",
        csv=artifacts.csv,
        plot=artifacts.plot,
        hdf5=artifacts.hdf5,
        manifest=artifacts.manifest,
        residuals_csv=artifacts.residuals_csv,
        residuals_json=artifacts.residuals_json,
        linearization=linearization_path,
        acceptance_report=acceptance_report_path,
        acceptance_passed=acceptance_report.passed,
        time=t,
        mdot_total=mdot_total,
        of_ratio=of_ratio,
        chamber_pressure=chamber_pressure,
        thrust=thrust,
        lox_shaft_rpm=lox_rpm,
        methane_shaft_rpm=methane_rpm,
    )


def linearize_ffsc_reduced_operating_point(
    *,
    x0: np.ndarray,
    u0: np.ndarray,
    model: dict[str, float],
    perturbations: PerturbationConfig | None = None,
) -> StateSpaceLinearization:
    """Linearize the reduced FFSC DAE plant around one operating point."""

    state_labels = [
        "lox_shaft.omega",
        "methane_shaft.omega",
        "chamber.P",
        "ox_preburner.P",
        "fuel_preburner.P",
    ]
    input_labels = [
        "main_lox_valve.position",
        "main_methane_valve.position",
        "methane_crossover_valve.position",
        "lox_crossover_valve.position",
    ]
    output_labels = [
        "mdot.total",
        "chamber.OF",
        "nozzle.thrust",
        "lox_shaft.rpm",
        "methane_shaft.rpm",
    ]

    def transients_from_u(u: np.ndarray) -> dict[str, float]:
        return {label: float(np.clip(value, 0.0, 1.0)) for label, value in zip(input_labels, u)}

    def values(x: np.ndarray, u: np.ndarray) -> dict[str, float]:
        return _ffsc_targets(_state_dict(x), transients_from_u(u), model)

    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        states = _state_dict(x)
        algebraic = values(x, u)
        lox_power_balance = algebraic["lox_turbine.power"] - algebraic["lox_pump.power"]
        methane_power_balance = algebraic["methane_turbine.power"] - algebraic["methane_pump.power"]
        domega_lox = lox_power_balance / max(model["lox_shaft_inertia"] * max(states["lox_shaft.omega"], 100.0), 1.0)
        domega_methane = methane_power_balance / max(
            model["methane_shaft_inertia"] * max(states["methane_shaft.omega"], 100.0),
            1.0,
        )
        domega_lox -= model["lox_shaft_friction"] * (states["lox_shaft.omega"] - model["lox_omega_design"])
        domega_methane -= model["methane_shaft_friction"] * (
            states["methane_shaft.omega"] - model["methane_omega_design"]
        )
        d_pc = model["chamber_pressure_gain"] * (algebraic["mdot.total"] - algebraic["nozzle.mdot"])
        d_ox_pb = model["ox_preburner_pressure_gain"] * (
            algebraic["ox_preburner.mdot_in"] - algebraic["ox_preburner.mdot_out"]
        )
        d_fuel_pb = model["fuel_preburner_pressure_gain"] * (
            algebraic["fuel_preburner.mdot_in"] - algebraic["fuel_preburner.mdot_out"]
        )
        return np.array([domega_lox, domega_methane, d_pc, d_ox_pb, d_fuel_pb], dtype=float)

    def outputs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        states = _state_dict(x)
        algebraic = values(x, u)
        return np.array(
            [
                algebraic["mdot.total"],
                algebraic["chamber.OF"],
                algebraic["nozzle.thrust"],
                _rad_s_to_rpm(states["lox_shaft.omega"]),
                _rad_s_to_rpm(states["methane_shaft.omega"]),
            ],
            dtype=float,
        )

    return finite_difference_state_space(
        dynamics,
        outputs,
        np.asarray(x0, dtype=float),
        np.asarray(u0, dtype=float),
        state_labels=state_labels,
        input_labels=input_labels,
        output_labels=output_labels,
        perturbations=perturbations,
    )


def build_ffsc_reduced_algebraic_problem(model: dict[str, float]) -> NetworkProblem:
    variables = [
        NetworkVariable("lox.mdot_total", units="kg/s", scale=40.0, initial=model["mdot_lox_design"]),
        NetworkVariable("methane.mdot_total", units="kg/s", scale=15.0, initial=model["mdot_methane_design"]),
        NetworkVariable("mdot.total", units="kg/s", scale=50.0, initial=model["mdot_total_design"]),
        NetworkVariable("chamber.OF", scale=4.0, initial=model["of_design"]),
        NetworkVariable("nozzle.mdot", units="kg/s", scale=50.0, initial=model["mdot_total_design"]),
        NetworkVariable("nozzle.thrust", units="N", scale=150000.0, initial=150000.0),
        NetworkVariable("ox_preburner.mdot_in", units="kg/s", scale=40.0, initial=30.0),
        NetworkVariable("fuel_preburner.mdot_in", units="kg/s", scale=20.0, initial=10.0),
        NetworkVariable("ox_preburner.mdot_out", units="kg/s", scale=40.0, initial=30.0),
        NetworkVariable("fuel_preburner.mdot_out", units="kg/s", scale=20.0, initial=10.0),
        NetworkVariable("lox_pump.power", units="W", scale=1.0e6, initial=5.0e5),
        NetworkVariable("methane_pump.power", units="W", scale=1.0e6, initial=2.0e5),
        NetworkVariable("lox_turbine.power", units="W", scale=1.0e6, initial=5.0e5),
        NetworkVariable("methane_turbine.power", units="W", scale=1.0e6, initial=2.0e5),
    ]
    residuals = [NetworkResidual(f"{variable.name}_residual", scale=variable.scale) for variable in variables]

    def evaluate(_t: float, z: dict[str, float], inputs: dict[str, float]) -> dict[str, float]:
        states = inputs["states"]
        transients = inputs.get("transients", {})
        targets = _ffsc_targets(states, transients, model)
        return {
            f"{name}_residual": float(z[name] - targets[name])
            for name in z
        }

    return NetworkProblem(variables, residuals, evaluate, name="ffsc_reduced_dae")


def _solve_ffsc_algebraic(
    problem: NetworkProblem,
    z_guess: np.ndarray,
    t: float,
    states: dict[str, float],
    transients: dict[str, float],
    model: dict[str, float],
):
    solution = problem.solve(t, z_guess, {"states": states, "transients": transients, "model": model})
    solution.values.update(_ffsc_targets(states, transients, model))
    name, residual = solution.max_normalized_residual
    if not solution.success or abs(residual) > 1.0e-6:
        raise RuntimeError(
            f"FFSC reduced algebraic solve failed at t={t:.6g}s: "
            f"{solution.message}; max residual {name}={residual:.3e}"
        )
    return solution


def _ffsc_targets(states: dict[str, float], transients: dict[str, float], model: dict[str, float]) -> dict[str, float]:
    lox_speed_ratio = max(states["lox_shaft.omega"] / model["lox_omega_design"], 0.05)
    methane_speed_ratio = max(states["methane_shaft.omega"] / model["methane_omega_design"], 0.05)
    main_lox = _transient(transients, "main_lox_valve.position", 1.0)
    main_methane = _transient(transients, "main_methane_valve.position", 1.0)
    lox_cross = _transient(transients, "lox_crossover_valve.position", 0.5)
    methane_cross = _transient(transients, "methane_crossover_valve.position", 0.5)

    lox_total_capacity = model["mdot_lox_design"] * lox_speed_ratio * (
        0.40 + 0.10 * main_lox + 0.70 * methane_cross + 0.30 * lox_cross
    )
    methane_total_capacity = model["mdot_methane_design"] * methane_speed_ratio * (
        0.85 + 0.10 * main_methane + 0.10 * lox_cross
    )
    lox_main, lox_cross_flow = _split_flow(lox_total_capacity, model["lox_split_fraction"], main_lox, lox_cross)
    methane_main, methane_cross_flow = _split_flow(
        methane_total_capacity,
        model["methane_split_fraction"],
        main_methane,
        methane_cross,
    )

    lox_total = max(lox_main + lox_cross_flow, 0.0)
    methane_total = max(methane_main + methane_cross_flow, 1.0e-9)
    mdot_total = lox_total + methane_total
    of_ratio = lox_total / methane_total
    nozzle_mdot = _nozzle_mdot(states["chamber.P"], model)
    thrust = model["thrust_coefficient"] * model["nozzle_throat_area"] * max(
        states["chamber.P"] - model["ambient_pressure"],
        0.0,
    )
    ox_pb_in = lox_main + methane_cross_flow
    fuel_pb_in = methane_main + lox_cross_flow
    ox_pb_out = model["ox_preburner_conductance"] * max(states["ox_preburner.P"] - states["chamber.P"], 0.0)
    fuel_pb_out = model["fuel_preburner_conductance"] * max(states["fuel_preburner.P"] - states["chamber.P"], 0.0)
    lox_pump_power = lox_total * model["lox_pump_dp_design"] * lox_speed_ratio**2 / model["lox_pump_efficiency"]
    methane_pump_power = (
        methane_total * model["methane_pump_dp_design"] * methane_speed_ratio**2 / model["methane_pump_efficiency"]
    )
    lox_pump_dp = model["lox_pump_dp_design"] * lox_speed_ratio**2
    methane_pump_dp = model["methane_pump_dp_design"] * methane_speed_ratio**2
    lox_pump_outlet_h = model["lox_pump_inlet_h"] + lox_pump_dp / max(model["lox_pump_rho_design"] * model["lox_pump_efficiency"], 1.0e-12)
    methane_pump_outlet_h = model["methane_pump_inlet_h"] + methane_pump_dp / max(model["methane_pump_rho_design"] * model["methane_pump_efficiency"], 1.0e-12)
    lox_turbine_power = model["lox_turbine_power_gain"] * ox_pb_out * (1.0 + 0.8 * methane_cross_flow)
    methane_turbine_power = model["methane_turbine_power_gain"] * fuel_pb_out * (1.0 + 0.6 * lox_cross_flow)
    return {
        "lox.mdot_total": lox_total,
        "methane.mdot_total": methane_total,
        "mdot.total": mdot_total,
        "chamber.OF": of_ratio,
        "nozzle.mdot": nozzle_mdot,
        "nozzle.thrust": thrust,
        "ox_preburner.mdot_in": ox_pb_in,
        "fuel_preburner.mdot_in": fuel_pb_in,
        "ox_preburner.mdot_out": ox_pb_out,
        "fuel_preburner.mdot_out": fuel_pb_out,
        "lox_pump.power": lox_pump_power,
        "methane_pump.power": methane_pump_power,
        "lox_pump.efficiency": model["lox_pump_efficiency"],
        "methane_pump.efficiency": model["methane_pump_efficiency"],
        "lox_pump.inlet.h": model["lox_pump_inlet_h"],
        "methane_pump.inlet.h": model["methane_pump_inlet_h"],
        "lox_pump.outlet.h": lox_pump_outlet_h,
        "methane_pump.outlet.h": methane_pump_outlet_h,
        "lox_turbine.power": lox_turbine_power,
        "methane_turbine.power": methane_turbine_power,
    }


def _ffsc_design_model(loaded) -> dict[str, float]:
    components = loaded.engine.components
    run = coerce_numbers(loaded.analysis_config.analysis)
    design = run.get("design", {})
    boundaries = coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, 0.0))
    lox_pump = components["lox_pump"].parameters["pump_map"]
    methane_pump = components["methane_pump"].parameters["pump_map"]
    lox_shaft = components["lox_shaft"].parameters
    methane_shaft = components["methane_shaft"].parameters
    chamber = components["chamber"].parameters
    ox_pb = components["ox_preburner"].parameters
    fuel_pb = components["fuel_preburner"].parameters
    nozzle = components["nozzle"].parameters
    mdot_total = float(design.get("mdot_total", 40.0))
    mdot_lox = float(design.get("mdot_lox", mdot_total * 0.76))
    mdot_methane = float(design.get("mdot_methane", mdot_total - mdot_lox))
    chamber_pressure = float(design.get("chamber_pressure", chamber.get("initial_P", 5.0e6)))
    ox_pb_pressure = float(ox_pb.get("initial_P", 8.0e6))
    fuel_pb_pressure = float(fuel_pb.get("initial_P", 8.0e6))
    lox_speed_design = _rpm_to_rad_s(float(lox_pump.get("speed_design", 32000.0)))
    methane_speed_design = _rpm_to_rad_s(float(methane_pump.get("speed_design", 27000.0)))
    lox_split = float(components["lox_splitter"].parameters.get("split_fraction", 0.92))
    methane_split = float(components["methane_splitter"].parameters.get("split_fraction", 0.90))
    initial_lox_capacity = mdot_lox * (0.40 + 0.10 + 0.70 * 0.5 + 0.30 * 0.5)
    initial_methane_capacity = mdot_methane * (0.85 + 0.10 + 0.10 * 0.5)
    lox_main_design, lox_cross_design = _split_flow(initial_lox_capacity, lox_split, 1.0, 0.5)
    methane_main_design, methane_cross_design = _split_flow(initial_methane_capacity, methane_split, 1.0, 0.5)
    ox_pb_out_design = mdot_lox * 0.78
    fuel_pb_out_design = mdot_methane * 1.25
    lox_pump_power_design = initial_lox_capacity * float(lox_pump.get("dP_design", 13.0e6)) / float(
        lox_pump.get("efficiency_design", 0.74)
    )
    methane_pump_power_design = initial_methane_capacity * float(methane_pump.get("dP_design", 13.5e6)) / float(
        methane_pump.get("efficiency_design", 0.69)
    )
    return {
        "mdot_total_design": mdot_total,
        "mdot_lox_design": mdot_lox,
        "mdot_methane_design": mdot_methane,
        "of_design": mdot_lox / max(mdot_methane, 1.0e-9),
        "lox_omega_design": lox_speed_design,
        "methane_omega_design": methane_speed_design,
        "lox_pump_dp_design": float(lox_pump.get("dP_design", 13.0e6)),
        "methane_pump_dp_design": float(methane_pump.get("dP_design", 13.5e6)),
        "lox_pump_efficiency": float(lox_pump.get("efficiency_design", 0.74)),
        "methane_pump_efficiency": float(methane_pump.get("efficiency_design", 0.69)),
        "lox_pump_rho_design": float(lox_pump.get("rho_design", boundaries.get("lox_tank.outlet.rho", 1140.0))),
        "methane_pump_rho_design": float(methane_pump.get("rho_design", boundaries.get("methane_tank.outlet.rho", 422.0))),
        "lox_pump_inlet_h": float(boundaries.get("lox_tank.outlet.h", 0.0)),
        "methane_pump_inlet_h": float(boundaries.get("methane_tank.outlet.h", 0.0)),
        "lox_shaft_inertia": float(lox_shaft.get("moment_of_inertia", 0.12)),
        "methane_shaft_inertia": float(methane_shaft.get("moment_of_inertia", 0.08)),
        "lox_shaft_friction": float(lox_shaft.get("friction_coeff", 0.02)),
        "methane_shaft_friction": float(methane_shaft.get("friction_coeff", 0.02)),
        "lox_split_fraction": lox_split,
        "methane_split_fraction": methane_split,
        "nozzle_conductance": mdot_total / max(chamber_pressure - float(boundaries.get("nozzle.ambient.P", 101325.0)), 1.0),
        "nozzle_throat_area": float(nozzle.get("throat_area", 2.11e-2)),
        "thrust_coefficient": float(nozzle.get("thrust_coefficient", 1.45)),
        "ambient_pressure": float(boundaries.get("nozzle.ambient.P", 101325.0)),
        "chamber_pressure_gain": float(chamber.get("gas_R", 420.0)) * float(chamber.get("T_adiabatic", 3600.0))
        / float(chamber.get("volume", 0.08)),
        "ox_preburner_pressure_gain": float(ox_pb.get("gas_R", 420.0)) * float(ox_pb.get("T_adiabatic", 950.0))
        / float(ox_pb.get("volume", 7.5e-4)),
        "fuel_preburner_pressure_gain": float(fuel_pb.get("gas_R", 420.0)) * float(fuel_pb.get("T_adiabatic", 900.0))
        / float(fuel_pb.get("volume", 7.5e-4)),
        "ox_preburner_conductance": ox_pb_out_design / max(ox_pb_pressure - chamber_pressure, 1.0),
        "fuel_preburner_conductance": fuel_pb_out_design / max(fuel_pb_pressure - chamber_pressure, 1.0),
        "lox_turbine_power_gain": lox_pump_power_design
        / max(ox_pb_out_design * (1.0 + 0.8 * methane_cross_design), 1.0),
        "methane_turbine_power_gain": methane_pump_power_design
        / max(fuel_pb_out_design * (1.0 + 0.6 * lox_cross_design), 1.0),
    }


def _split_flow(total: float, split: float, main_position: float, crossover_position: float) -> tuple[float, float]:
    main_weight = max(split, 1.0e-6) * max(main_position, 0.02)
    cross_weight = max(1.0 - split, 1.0e-6) * max(crossover_position, 0.02)
    denom = main_weight + cross_weight
    return total * main_weight / denom, total * cross_weight / denom


def _nozzle_mdot(pc: float, model: dict[str, float]) -> float:
    return float(model["nozzle_conductance"]) * max(float(pc) - float(model["ambient_pressure"]), 0.0)


def _state_dict(y: np.ndarray) -> dict[str, float]:
    return {
        "lox_shaft.omega": max(float(y[0]), 100.0),
        "methane_shaft.omega": max(float(y[1]), 100.0),
        "chamber.P": max(float(y[2]), 101325.0),
        "ox_preburner.P": max(float(y[3]), 101325.0),
        "fuel_preburner.P": max(float(y[4]), 101325.0),
    }


def _measurements(values: dict[str, float]) -> dict[str, float]:
    return {
        "mdot_total": float(values.get("mdot.total", 0.0)),
        "OF": float(values.get("chamber.OF", 0.0)),
    }


def _zero_measurements(run: dict[str, object]) -> dict[str, float]:
    design = run.get("design", {})
    return {
        "mdot_total": float(design.get("mdot_total", 0.0)) if isinstance(design, dict) else 0.0,
        "OF": float(design.get("OF", 0.0)) if isinstance(design, dict) else 0.0,
    }


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
    commands = dict(timings)
    commands.update(controller_outputs)
    return _with_transient_command_defaults(loaded, commands)


def _with_transient_command_defaults(loaded, commands: dict[str, float]) -> dict[str, float]:
    completed = dict(commands)
    for transient in loaded.transients.values():
        path = transient.command.get("path")
        if not isinstance(path, str) or path in completed:
            continue
        initial = transient.state.get("initial", transient.parameters.get("initial", 0.0))
        completed[path] = float(initial)
    return completed


def _acceptance_mask(time: np.ndarray, acceptance_cfg: dict[str, object]) -> np.ndarray:
    if not acceptance_cfg:
        return np.ones_like(time, dtype=bool)
    end = acceptance_cfg.get("evaluation_end_s", acceptance_cfg.get("evaluation_time_s"))
    if end is None:
        return np.ones_like(time, dtype=bool)
    mask = time <= float(end)
    if not np.any(mask):
        mask[0] = True
    return mask


def _initial_conditions(run: dict[str, object]) -> dict[str, float]:
    values = run.get("initial_conditions", {})
    if not isinstance(values, dict):
        return {}
    return {str(key): float(value) for key, value in values.items()}


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


def _transient(values: dict[str, float], path: str, default: float) -> float:
    return float(values.get(path, default))


def _rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * 2.0 * np.pi / 60.0


def _rad_s_to_rpm(omega: float) -> float:
    return float(omega) * 60.0 / (2.0 * np.pi)


def _perturbation_config(raw: object) -> PerturbationConfig:
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
