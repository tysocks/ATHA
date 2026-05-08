from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from atha.config import evaluate_boundary_conditions, evaluate_timing_events, load_analysis_config
from atha.config.transients import TransientSystem
from atha.examples.common import coerce_numbers
from atha.output.telemetry import build_telemetry_rows, write_telemetry_csv


@dataclass
class ValveVolumeSummary:
    csv: Path
    plot: Path
    time: np.ndarray
    pressure: np.ndarray
    mdot_in: np.ndarray
    mdot_out: np.ndarray
    valve_command: np.ndarray
    valve_position: np.ndarray


def run_valve_volume_profile(config_path: str | Path, output_dir: str | Path = "outputs") -> ValveVolumeSummary:
    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**_model_from_engine(loaded), **run.get("model", {})}
    time_cfg = run["time"]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    t0 = float(time_cfg["start_s"])
    tf = float(time_cfg["end_s"])
    values0 = _values(loaded, t0)
    P0 = float(model.get("initial_P", values0["ambient.P"]))
    mdot_out0 = float(model.get("initial_mdot_out", 0.0))
    transient_system = TransientSystem.from_configs(loaded.transients)
    commands0 = _commands(loaded, t0)
    X_transient0 = transient_system.initial_state(commands0)
    if X_transient0.size == 0:
        valve_position0 = float(model.get("initial_valve_position", _valve_command(loaded, t0)))
        X_transient0 = np.array([valve_position0], dtype=float)

    def rhs(t: float, y):
        P = max(float(y[0]), 1.0)
        X_transient = np.asarray(y[1:-1], dtype=float)
        mdot_out = float(y[-1])
        values = _values(loaded, t)
        commands = _commands(loaded, t)
        transient_outputs = transient_system.evaluate(t, X_transient, commands) if transient_system.n_states else {}
        valve_position = min(max(float(transient_outputs.get("valve.position", X_transient[0])), 0.0), 1.0)
        mdot_in = _valve_mdot(values["supply.P"], P, values["supply.T"], valve_position, model)
        mdot_out_used = max(mdot_out, 0.0)
        dPdt = (float(model["gas_R"]) * float(model["gas_T"]) / float(model["volume"])) * (mdot_in - mdot_out_used)
        dtransient_dt = transient_system.derivatives(t, X_transient, commands) if transient_system.n_states else np.array([0.0])
        dmdot_out_dt = _outlet_mdot_derivative(P, values["ambient.P"], mdot_out_used, model)
        return np.concatenate(([dPdt], dtransient_dt, [dmdot_out_dt]))

    solver_cfg = loaded.analysis_config.solver["transient"]
    sol = solve_ivp(
        rhs,
        (t0, tf),
        np.concatenate(([P0], X_transient0, [mdot_out0])),
        method=solver_cfg.get("method", "Radau"),
        rtol=float(solver_cfg.get("rtol", 1e-7)),
        atol=float(solver_cfg.get("atol", 1e-3)),
        max_step=float(solver_cfg.get("max_step", 0.01)),
    )
    if not sol.success:
        raise RuntimeError(f"Valve-volume transient failed: {sol.message}")

    t = _telemetry_times(loaded, sol.t[0], sol.t[-1])
    pressure = np.interp(t, sol.t, sol.y[0])
    transient_state = np.vstack([np.interp(t, sol.t, sol.y[i]) for i in range(1, 1 + X_transient0.size)]).T
    mdot_out = np.maximum(np.interp(t, sol.t, sol.y[-1]), 0.0)
    valve_position = np.zeros_like(t)
    valve_command = np.zeros_like(t)
    mdot_in = np.zeros_like(t)
    samples = []
    for i, ti in enumerate(t):
        values = _values(loaded, float(ti))
        commands = _commands(loaded, float(ti))
        valve_command[i] = float(commands.get("valve.command", _valve_command(loaded, float(ti))))
        transient_outputs = transient_system.evaluate(float(ti), transient_state[i], commands) if transient_system.n_states else {}
        valve_position[i] = min(max(float(transient_outputs.get("valve.position", transient_state[i, 0])), 0.0), 1.0)
        mdot_in[i] = _valve_mdot(values["supply.P"], pressure[i], values["supply.T"], valve_position[i], model)
        samples.append(
            {
                "time": float(ti),
                "valve.command": float(valve_command[i]),
                "valve.position": float(valve_position[i]),
                "downstream.P": float(pressure[i]),
                "valve.mdot": float(mdot_in[i]),
                "outlet.mdot": float(mdot_out[i]),
                "outlet.mdot_steady": float(_outlet_mdot_steady(pressure[i], values["ambient.P"], model)),
            }
        )

    headers, columns = build_telemetry_rows(loaded.telemetry, samples)
    out_csv = write_telemetry_csv(output_dir / run["output"]["csv"], headers, columns)
    out_plot = output_dir / run["output"]["plot"]
    _plot(out_plot, t, pressure, valve_command, valve_position, mdot_in, mdot_out)

    return ValveVolumeSummary(
        csv=out_csv,
        plot=out_plot,
        time=t,
        pressure=pressure,
        mdot_in=mdot_in,
        mdot_out=mdot_out,
        valve_command=valve_command,
        valve_position=valve_position,
    )


def _values(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _model_from_engine(loaded) -> dict:
    model = {}
    for component in loaded.engine.components.values():
        prefix = component.name
        params = coerce_numbers(component.parameters)
        if component.type == "Valve":
            if "CdA" in params:
                model["valve_CdA"] = params["CdA"]
            elif "max_area" in params:
                model["valve_CdA"] = params["max_area"] * params.get("discharge_coeff", 1.0)
        elif component.type in {"GasVolume", "Volume"}:
            model["volume"] = params["volume"]
            model["gas_R"] = params["gas_R"]
            model["gas_T"] = params["gas_T"]
            if "gamma" in params:
                model["gamma"] = params["gamma"]
        elif component.type in {"Outlet", "OutletInertia"}:
            for key in ("outlet_resistance", "outlet_inertance", "outlet_conductance"):
                if key in params:
                    model[key] = params[key]
        else:
            for key, value in params.items():
                model[f"{prefix}_{key}"] = value
    return model


def _commands(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_timing_events(loaded.timings, t))


def _valve_command(loaded, t: float) -> float:
    timings = _commands(loaded, t)
    value = timings.get("valve.command", timings.get("valve.position", 0.0))
    return min(max(float(value), 0.0), 1.0)


def _valve_mdot(P_up: float, P_down: float, T_up: float, valve_position: float, model: dict) -> float:
    if P_up <= P_down or valve_position <= 0.0:
        return 0.0
    gamma = float(model["gamma"])
    R = float(model["gas_R"])
    cda = float(model["valve_CdA"]) * valve_position
    pr = max(P_down / P_up, 1.0e-9)
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    common = cda * P_up * (gamma / (R * T_up)) ** 0.5
    if pr <= pr_crit:
        coeff = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return common * coeff
    term = pr ** (2.0 / gamma) - pr ** ((gamma + 1.0) / gamma)
    return common * (2.0 / (gamma - 1.0) * max(term, 0.0)) ** 0.5


def _outlet_mdot_steady(P: float, P_ambient: float, model: dict) -> float:
    if P <= P_ambient:
        return 0.0
    if "outlet_resistance" in model:
        return (P - P_ambient) / float(model["outlet_resistance"])
    return float(model["outlet_conductance"]) * (P - P_ambient)


def _outlet_mdot_derivative(P: float, P_ambient: float, mdot_out: float, model: dict) -> float:
    if "outlet_resistance" in model:
        resistance = float(model["outlet_resistance"])
    else:
        resistance = 1.0 / float(model["outlet_conductance"])
    if "outlet_inertance" in model:
        pressure_force = max(P - P_ambient, 0.0)
        pressure_loss = resistance * mdot_out
        return (pressure_force - pressure_loss) / max(float(model["outlet_inertance"]), 1.0e-12)

    tau = max(float(model.get("outlet_flow_time_constant", 0.25)), 1.0e-9)
    return (_outlet_mdot_steady(P, P_ambient, model) - mdot_out) / tau


def _telemetry_times(loaded, t0: float, tf: float) -> np.ndarray:
    rate = float(loaded.telemetry.sample_rate_hz or 100.0)
    dt = 1.0 / rate
    t = np.arange(t0, tf + 0.5 * dt, dt)
    t[-1] = min(t[-1], tf)
    if t[-1] < tf:
        t = np.append(t, tf)
    return t


def _plot(path: Path, t, pressure, valve_command, valve_position, mdot_in, mdot_out) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, valve_command, "--", label="command")
    axes[0].plot(t, valve_position, label="actual")
    axes[0].set(ylabel="Valve [-]", title="Timed valve command and actuator response")
    axes[0].legend()
    axes[1].plot(t, pressure / 1e5)
    axes[1].set(ylabel="Downstream P [bar]", title="Downstream volume response")
    axes[2].plot(t, mdot_in, label="inlet")
    axes[2].plot(t, mdot_out, label="outlet")
    axes[2].set(xlabel="Time [s]", ylabel="Mass flow [kg/s]", title="Flow response")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
