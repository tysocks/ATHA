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
class TwoValveChainSummary:
    csv: Path
    plot: Path
    time: np.ndarray
    chamber_pressure: np.ndarray
    valve_a_position: np.ndarray
    valve_b_position: np.ndarray
    mdot_a: np.ndarray
    mdot_b: np.ndarray
    thrust: np.ndarray


def run_two_valve_chain(config_path: str | Path, output_dir: str | Path = "outputs") -> TwoValveChainSummary:
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
                float(model.get("initial_mdot_a", 0.0)),
                float(model.get("initial_mdot_b", 0.0)),
            ],
        )
    )

    def rhs(t: float, y):
        x_transient = np.asarray(y[: transient_system.n_states], dtype=float)
        Pc = max(float(y[transient_system.n_states]), 1.0)
        mdot_a = max(float(y[transient_system.n_states + 1]), 0.0)
        mdot_b = max(float(y[transient_system.n_states + 2]), 0.0)

        commands = _commands(loaded, t)
        positions = transient_system.evaluate(t, x_transient, commands)
        values = _values(loaded, t)
        mdot_a_ss = _leg_mdot_steady("a", positions["valve_a.position"], Pc, values, model)
        mdot_b_ss = _leg_mdot_steady("b", positions["valve_b.position"], Pc, values, model)
        mdot_nozzle = _nozzle_mdot(Pc, values["nozzle.ambient.P"], model)

        dtransients = transient_system.derivatives(t, x_transient, commands)
        dPc = (float(model["gas_R"]) * float(model["chamber_T"]) / float(model["chamber_volume"])) * (
            mdot_a + mdot_b - mdot_nozzle
        )
        dmdot_a = (mdot_a_ss - mdot_a) / max(float(model["pipe_a_time_constant"]), 1.0e-12)
        dmdot_b = (mdot_b_ss - mdot_b) / max(float(model["pipe_b_time_constant"]), 1.0e-12)
        return np.concatenate((dtransients, [dPc, dmdot_a, dmdot_b]))

    solver_cfg = loaded.analysis_config.solver["transient"]
    sol = solve_ivp(
        rhs,
        (t0, tf),
        X0,
        method=solver_cfg.get("method", "Radau"),
        rtol=float(solver_cfg.get("rtol", 1.0e-7)),
        atol=float(solver_cfg.get("atol", 1.0e-5)),
        max_step=float(solver_cfg.get("max_step", 0.01)),
    )
    if not sol.success:
        raise RuntimeError(f"Two-valve transient failed: {sol.message}")

    t = _telemetry_times(loaded, sol.t[0], sol.t[-1])
    Y = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
    pc_index = transient_system.n_states
    pc = Y[:, pc_index]
    mdot_a = np.maximum(Y[:, pc_index + 1], 0.0)
    mdot_b = np.maximum(Y[:, pc_index + 2], 0.0)
    valve_a = np.zeros_like(t)
    valve_b = np.zeros_like(t)
    thrust = np.zeros_like(t)
    samples = []

    for i, ti in enumerate(t):
        x_transient = Y[i, : transient_system.n_states]
        commands = _commands(loaded, float(ti))
        positions = transient_system.evaluate(float(ti), x_transient, commands)
        values = _values(loaded, float(ti))
        valve_a[i] = positions["valve_a.position"]
        valve_b[i] = positions["valve_b.position"]
        thrust[i] = _thrust(pc[i], values["nozzle.ambient.P"], model)
        samples.append(
            {
                "time": float(ti),
                "valve_a.command": float(commands["valve_a.command"]),
                "valve_b.command": float(commands["valve_b.command"]),
                "valve_a.position": float(valve_a[i]),
                "valve_b.position": float(valve_b[i]),
                "pipe_a.mdot": float(mdot_a[i]),
                "pipe_b.mdot": float(mdot_b[i]),
                "injector_a.mdot": float(mdot_a[i]),
                "injector_b.mdot": float(mdot_b[i]),
                "chamber.P": float(pc[i]),
                "nozzle.mdot": float(_nozzle_mdot(pc[i], values["nozzle.ambient.P"], model)),
                "nozzle.thrust": float(thrust[i]),
            }
        )

    headers, columns = build_telemetry_rows(loaded.telemetry, samples)
    out_csv = write_telemetry_csv(output_dir / run["output"]["csv"], headers, columns)
    out_plot = output_dir / run["output"]["plot"]
    _plot(out_plot, t, valve_a, valve_b, mdot_a, mdot_b, pc, thrust)

    return TwoValveChainSummary(
        csv=out_csv,
        plot=out_plot,
        time=t,
        chamber_pressure=pc,
        valve_a_position=valve_a,
        valve_b_position=valve_b,
        mdot_a=mdot_a,
        mdot_b=mdot_b,
        thrust=thrust,
    )


def _values(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _model_from_engine(loaded) -> dict:
    model = {}
    for component in loaded.engine.components.values():
        params = coerce_numbers(component.parameters)
        if component.name == "valve_a":
            model["valve_a_CdA"] = params.get("CdA", params["max_area"] * params.get("discharge_coeff", 1.0))
        elif component.name == "valve_b":
            model["valve_b_CdA"] = params.get("CdA", params["max_area"] * params.get("discharge_coeff", 1.0))
        elif component.name == "pipe_a":
            model["pipe_a_time_constant"] = params["time_constant"]
        elif component.name == "pipe_b":
            model["pipe_b_time_constant"] = params["time_constant"]
        elif component.name == "injector_a":
            model["injector_a_delta_P"] = params["delta_P_nominal"]
        elif component.name == "injector_b":
            model["injector_b_delta_P"] = params["delta_P_nominal"]
        elif component.name == "chamber":
            model["chamber_volume"] = params["volume"]
            model["chamber_T"] = params["gas_T"]
            model["gas_R"] = params["gas_R"]
        elif component.name == "nozzle":
            model["nozzle_throat_area"] = params["throat_area"]
            model["nozzle_conductance"] = params["conductance"]
            model["thrust_coefficient"] = params["thrust_coefficient"]
    return model


def _commands(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_timing_events(loaded.timings, t))


def _leg_mdot_steady(prefix: str, valve_position: float, Pc: float, values: dict, model: dict) -> float:
    P_supply = float(values[f"{prefix}_supply.P"])
    rho = float(values[f"{prefix}_supply.rho"])
    dP = max(P_supply - Pc - float(model[f"injector_{prefix}_delta_P"]), 0.0)
    cda = float(model[f"valve_{prefix}_CdA"]) * max(min(valve_position, 1.0), 0.0)
    return cda * (2.0 * rho * dP) ** 0.5


def _nozzle_mdot(Pc: float, P_ambient: float, model: dict) -> float:
    if Pc <= P_ambient:
        return 0.0
    return float(model["nozzle_conductance"]) * (Pc - P_ambient)


def _thrust(Pc: float, P_ambient: float, model: dict) -> float:
    pressure_term = max(Pc - P_ambient, 0.0)
    return float(model["thrust_coefficient"]) * float(model["nozzle_throat_area"]) * pressure_term


def _telemetry_times(loaded, t0: float, tf: float) -> np.ndarray:
    rate = float(loaded.telemetry.sample_rate_hz or 100.0)
    dt = 1.0 / rate
    t = np.arange(t0, tf + 0.5 * dt, dt)
    t[-1] = min(t[-1], tf)
    if t[-1] < tf:
        t = np.append(t, tf)
    return t


def _plot(path: Path, t, valve_a, valve_b, mdot_a, mdot_b, pc, thrust) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(t, valve_a, label="valve A linear")
    axes[0, 0].plot(t, valve_b, label="valve B first order")
    axes[0, 0].set(ylabel="Position [-]", title="Valve actual response")
    axes[0, 0].legend()
    axes[0, 1].plot(t, mdot_a, label="pipe A")
    axes[0, 1].plot(t, mdot_b, label="pipe B")
    axes[0, 1].set(ylabel="Mass flow [kg/s]", title="Pipe/injector flow")
    axes[0, 1].legend()
    axes[1, 0].plot(t, pc / 1e5)
    axes[1, 0].set(xlabel="Time [s]", ylabel="Pc [bar]", title="Chamber pressure")
    axes[1, 1].plot(t, thrust)
    axes[1, 1].set(xlabel="Time [s]", ylabel="Thrust [N]", title="Nozzle thrust")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
