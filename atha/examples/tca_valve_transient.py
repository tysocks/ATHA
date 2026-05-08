from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from atha.config import (
    TransientSystem,
    evaluate_boundary_conditions,
    evaluate_timing_events,
    load_analysis_config,
)
from atha.examples.common import coerce_numbers
from atha.output.telemetry import build_telemetry_rows, write_telemetry_csv


@dataclass
class TCAValveTransientSummary:
    csv: Path
    plot: Path
    time: np.ndarray
    chamber_pressure: np.ndarray
    methane_valve_position: np.ndarray
    lox_valve_position: np.ndarray
    mdot_methane: np.ndarray
    mdot_lox: np.ndarray
    of_ratio: np.ndarray
    thrust: np.ndarray


def run_tca_valve_transient(config_path: str | Path, output_dir: str | Path = "outputs") -> TCAValveTransientSummary:
    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**_model_from_engine(loaded), **run.get("model", {})}
    time_cfg = run["time"]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    t0 = float(time_cfg["start_s"])
    tf = float(time_cfg["end_s"])
    transient_system = TransientSystem.from_configs(loaded.transients)
    commands0 = _commands(loaded, t0)
    X_transient0 = transient_system.initial_state(commands0)
    X0 = np.concatenate(
        (
            X_transient0,
            [
                float(model["initial_Pc"]),
                float(model.get("initial_mdot_methane", 0.0)),
                float(model.get("initial_mdot_lox", 0.0)),
            ],
        )
    )

    def rhs(t: float, y):
        x_transient = np.asarray(y[: transient_system.n_states], dtype=float)
        pc = max(float(y[transient_system.n_states]), 1.0)
        mdot_methane = max(float(y[transient_system.n_states + 1]), 0.0)
        mdot_lox = max(float(y[transient_system.n_states + 2]), 0.0)

        commands = _commands(loaded, t)
        positions = transient_system.evaluate(t, x_transient, commands)
        values = _values(loaded, t)
        mdot_methane_ss = _leg_mdot_steady("methane", positions["methane_valve.position"], pc, values, model)
        mdot_lox_ss = _leg_mdot_steady("lox", positions["lox_valve.position"], pc, values, model)
        mdot_nozzle = _nozzle_mdot(pc, values["nozzle.ambient.P"], model)

        dtransients = transient_system.derivatives(t, x_transient, commands)
        dpc = (float(model["gas_R"]) * float(model["chamber_T"]) / float(model["chamber_volume"])) * (
            mdot_methane + mdot_lox - mdot_nozzle
        )
        dmdot_methane = (mdot_methane_ss - mdot_methane) / max(float(model["methane_pipe_time_constant"]), 1.0e-12)
        dmdot_lox = (mdot_lox_ss - mdot_lox) / max(float(model["lox_pipe_time_constant"]), 1.0e-12)
        return np.concatenate((dtransients, [dpc, dmdot_methane, dmdot_lox]))

    solver_cfg = loaded.analysis_config.solver["transient"]
    sol = solve_ivp(
        rhs,
        (t0, tf),
        X0,
        method=solver_cfg.get("method", "Radau"),
        rtol=float(solver_cfg.get("rtol", 1.0e-7)),
        atol=float(solver_cfg.get("atol", 1.0e-6)),
        max_step=float(solver_cfg.get("max_step", 0.005)),
    )
    if not sol.success:
        raise RuntimeError(f"TCA valve transient failed: {sol.message}")

    t = _telemetry_times(loaded, sol.t[0], sol.t[-1])
    y = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
    pc_index = transient_system.n_states
    pc = y[:, pc_index]
    mdot_methane = np.maximum(y[:, pc_index + 1], 0.0)
    mdot_lox = np.maximum(y[:, pc_index + 2], 0.0)
    methane_valve = np.zeros_like(t)
    lox_valve = np.zeros_like(t)
    thrust = np.zeros_like(t)
    of_ratio = np.divide(mdot_lox, mdot_methane, out=np.full_like(mdot_lox, np.nan), where=mdot_methane > 1.0e-12)
    samples = []

    for i, ti in enumerate(t):
        x_transient = y[i, : transient_system.n_states]
        commands = _commands(loaded, float(ti))
        positions = transient_system.evaluate(float(ti), x_transient, commands)
        values = _values(loaded, float(ti))
        methane_valve[i] = positions["methane_valve.position"]
        lox_valve[i] = positions["lox_valve.position"]
        thrust[i] = _thrust(pc[i], values["nozzle.ambient.P"], model)
        samples.append(
            {
                "time": float(ti),
                "methane_valve.command": float(commands["methane_valve.command"]),
                "lox_valve.command": float(commands["lox_valve.command"]),
                "methane_valve.position": float(methane_valve[i]),
                "lox_valve.position": float(lox_valve[i]),
                "methane_pipe.mdot": float(mdot_methane[i]),
                "lox_pipe.mdot": float(mdot_lox[i]),
                "methane_injector.mdot": float(mdot_methane[i]),
                "lox_injector.mdot": float(mdot_lox[i]),
                "chamber.P": float(pc[i]),
                "chamber.OF": float(of_ratio[i]),
                "nozzle.mdot": float(_nozzle_mdot(pc[i], values["nozzle.ambient.P"], model)),
                "nozzle.thrust": float(thrust[i]),
            }
        )

    headers, columns = build_telemetry_rows(loaded.telemetry, samples)
    out_csv = write_telemetry_csv(output_dir / run["output"]["csv"], headers, columns)
    out_plot = output_dir / run["output"]["plot"]
    _plot(out_plot, t, methane_valve, lox_valve, mdot_methane, mdot_lox, of_ratio, pc, thrust)

    return TCAValveTransientSummary(
        csv=out_csv,
        plot=out_plot,
        time=t,
        chamber_pressure=pc,
        methane_valve_position=methane_valve,
        lox_valve_position=lox_valve,
        mdot_methane=mdot_methane,
        mdot_lox=mdot_lox,
        of_ratio=of_ratio,
        thrust=thrust,
    )


def _model_from_engine(loaded) -> dict:
    model = {}
    for component in loaded.engine.components.values():
        params = coerce_numbers(component.parameters)
        if component.name == "methane_valve":
            model["methane_valve_CdA"] = params.get("CdA", params["max_area"] * params.get("discharge_coeff", 1.0))
        elif component.name == "lox_valve":
            model["lox_valve_CdA"] = params.get("CdA", params["max_area"] * params.get("discharge_coeff", 1.0))
        elif component.name == "methane_pipe":
            model["methane_pipe_time_constant"] = params["time_constant"]
        elif component.name == "lox_pipe":
            model["lox_pipe_time_constant"] = params["time_constant"]
        elif component.name == "methane_injector":
            model["methane_injector_delta_P"] = params["delta_P_nominal"]
        elif component.name == "lox_injector":
            model["lox_injector_delta_P"] = params["delta_P_nominal"]
        elif component.name == "chamber":
            model["chamber_volume"] = params["volume"]
            model["chamber_T"] = params["gas_T"]
            model["gas_R"] = params["gas_R"]
        elif component.name == "nozzle":
            model["nozzle_throat_area"] = params["throat_area"]
            model["nozzle_conductance"] = params["conductance"]
            model["thrust_coefficient"] = params["thrust_coefficient"]
    return model


def _values(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _commands(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_timing_events(loaded.timings, t))


def _leg_mdot_steady(propellant: str, valve_position: float, pc: float, values: dict, model: dict) -> float:
    p_supply = float(values[f"{propellant}_supply.P"])
    rho = float(values[f"{propellant}_supply.rho"])
    dP = max(p_supply - pc - float(model[f"{propellant}_injector_delta_P"]), 0.0)
    cda = float(model[f"{propellant}_valve_CdA"]) * max(min(valve_position, 1.0), 0.0)
    return cda * (2.0 * rho * dP) ** 0.5


def _nozzle_mdot(pc: float, p_ambient: float, model: dict) -> float:
    if pc <= p_ambient:
        return 0.0
    return float(model["nozzle_conductance"]) * (pc - p_ambient)


def _thrust(pc: float, p_ambient: float, model: dict) -> float:
    return float(model["thrust_coefficient"]) * float(model["nozzle_throat_area"]) * max(pc - p_ambient, 0.0)


def _telemetry_times(loaded, t0: float, tf: float) -> np.ndarray:
    rate = float(loaded.telemetry.sample_rate_hz or 100.0)
    dt = 1.0 / rate
    t = np.arange(t0, tf + 0.5 * dt, dt)
    t[-1] = min(t[-1], tf)
    if t[-1] < tf:
        t = np.append(t, tf)
    return t


def _plot(path: Path, t, methane_valve, lox_valve, mdot_methane, mdot_lox, of_ratio, pc, thrust) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(t, methane_valve, label="methane linear")
    axes[0, 0].plot(t, lox_valve, label="LOX first order")
    axes[0, 0].set(ylabel="Position [-]", title="Valve actual response")
    axes[0, 0].legend()
    axes[0, 1].plot(t, mdot_methane, label="methane")
    axes[0, 1].plot(t, mdot_lox, label="LOX")
    axes[0, 1].set(ylabel="Mass flow [kg/s]", title="Injector flow")
    axes[0, 1].legend()
    axes[1, 0].plot(t, pc / 1e5, label="Pc")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Pc [bar]", title="Chamber pressure")
    ax_of = axes[1, 0].twinx()
    ax_of.plot(t, of_ratio, color="tab:green", alpha=0.7, label="OF")
    ax_of.set_ylabel("OF [-]")
    axes[1, 1].plot(t, thrust)
    axes[1, 1].set(xlabel="Time [s]", ylabel="Thrust [N]", title="Nozzle thrust")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
