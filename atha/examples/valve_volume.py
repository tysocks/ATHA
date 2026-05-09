from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from atha.assembly import EngineAssembler
from atha.components.registry import extract_engine_model
from atha.config import evaluate_boundary_conditions, evaluate_timing_events, load_analysis_config
from atha.config.transients import TransientSystem
from atha.examples.common import coerce_numbers
from atha.output.processor import OutputProcessor
from atha.output.sampling import telemetry_times
from atha.output.telemetry import validate_telemetry_sources


@dataclass
class ValveVolumeSummary:
    csv: Path
    plot: Path
    hdf5: Path
    manifest: Path
    time: np.ndarray
    pressure: np.ndarray
    mdot_in: np.ndarray
    mdot_out: np.ndarray
    valve_command: np.ndarray
    valve_position: np.ndarray


def run_valve_volume_profile(config_path: str | Path, output_dir: str | Path = "outputs") -> ValveVolumeSummary:
    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    model = {**extract_engine_model(loaded.engine), **run.get("model", {})}
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
    validate_telemetry_sources(loaded.telemetry, EngineAssembler(loaded).telemetry_sources(_telemetry_source_catalog(transient_system)))
    if X_transient0.size == 0:
        valve_position0 = float(model.get("initial_valve_position", _valve_command(loaded, t0)))
        X_transient0 = np.array([valve_position0], dtype=float)

    def rhs(t: float, y):
        P = max(float(y[0]), 1.0)
        X_transient = np.asarray(y[1:-1], dtype=float)
        mdot_out = float(y[-1])
        values = _values(loaded, t)
        commands = _commands(loaded, t)
        transient_outputs = transient_system.sample_sources(t, X_transient, commands)
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

    t = telemetry_times(loaded.telemetry, sol.t[0], sol.t[-1])
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
        transient_outputs = transient_system.sample_sources(float(ti), transient_state[i], commands)
        valve_position[i] = min(max(float(transient_outputs.get("valve.position", transient_state[i, 0])), 0.0), 1.0)
        mdot_in[i] = _valve_mdot(values["supply.P"], pressure[i], values["supply.T"], valve_position[i], model)
        samples.append(
            {
                "time": float(ti),
                "valve.command": float(valve_command[i]),
                **transient_outputs,
                "downstream.P": float(pressure[i]),
                "valve.mdot": float(mdot_in[i]),
                "outlet.mdot": float(mdot_out[i]),
                "outlet.mdot_steady": float(_outlet_mdot_steady(pressure[i], values["ambient.P"], model)),
            }
        )

    artifacts, _headers, _columns = OutputProcessor(
        output_dir=output_dir,
        telemetry_config=loaded.telemetry,
        run_output=run["output"],
        metadata={"analysis": loaded.analysis_config.name, "analysis_type": run.get("type", "")},
    ).write(
        samples,
        state_history={
            "downstream.P": pressure,
            "outlet.mdot": mdot_out,
            "valve.position": valve_position,
        },
    )

    return ValveVolumeSummary(
        csv=artifacts.csv,
        plot=artifacts.plot,
        hdf5=artifacts.hdf5,
        manifest=artifacts.manifest,
        time=t,
        pressure=pressure,
        mdot_in=mdot_in,
        mdot_out=mdot_out,
        valve_command=valve_command,
        valve_position=valve_position,
    )


def _values(loaded, t: float) -> dict:
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _telemetry_source_catalog(transient_system: TransientSystem) -> set[str]:
    return {
        "time",
        "valve.command",
        "downstream.P",
        "valve.mdot",
        "outlet.mdot",
        "outlet.mdot_steady",
    } | transient_system.source_catalog()


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
