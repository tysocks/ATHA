from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from atha.components.factory import build_component_from_config
from atha.config import (
    evaluate_controllers,
    evaluate_operating_targets,
    evaluate_timing_events,
    load_analysis_config,
)
from atha.core.engine import Engine
from atha.examples.common import (
    boundary_values,
    coerce_numbers,
    connect_from_config,
    evaluate_outputs,
)
from atha.output.telemetry import build_telemetry_rows, write_telemetry_csv
from atha.solver.steady_state import SteadyStateSolver
from atha.solver.transient import TransientSolver
from atha.thermo.cantera_backend import CanteraBackend
from atha.thermo.coolprop_backend import CoolPropBackend


@dataclass
class TCAResultSummary:
    csv: Path
    plot: Path
    time: np.ndarray
    pc: np.ndarray
    thrust: np.ndarray
    mdot_total: np.ndarray
    of_ratio: np.ndarray
    valve_position: np.ndarray | None = None


def build_tca_engine(loaded):
    run = coerce_numbers(loaded.analysis_config.analysis)
    combustion = run["combustion"]
    fluids = run["fluids"]

    engine = Engine(loaded.engine.name)
    lox = CoolPropBackend(fluids["oxidizer"])
    methane = CoolPropBackend(fluids["fuel"])

    context = {"combustion": combustion}
    for comp_cfg in loaded.engine.components.values():
        engine.add_component(build_component_from_config(comp_cfg, context))
    connect_from_config(engine, loaded.engine.connections)
    return engine.compile(), engine, lox, methane


def make_tca_bcs_fn(loaded, lox, methane):
    values0 = boundary_values(loaded, 0.0)
    has_valves = "lox_valve" in loaded.engine.components and "fuel_valve" in loaded.engine.components

    def bcs(t: float) -> dict:
        values = boundary_values(loaded, t)
        targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, t))
        timings = coerce_numbers(evaluate_timing_events(loaded.timings, t))
        commands = evaluate_controllers(loaded.controllers, targets, timings)

        lox_state = lox.state_from_PT(values["lox_manifold.P"], values["lox_manifold.T"])
        fuel_state = methane.state_from_PT(values["fuel_manifold.P"], values["fuel_manifold.T"])
        values_out = {
            "lox_inj.inlet.P": values["lox_manifold.P"],
            "lox_inj.inlet.T": values["lox_manifold.T"],
            "lox_inj.inlet.h": lox_state.h,
            "lox_inj.inlet.rho": lox_state.rho,
            "lox_inj.inlet.mdot": commands["lox_inj.inlet.mdot"],
            "fuel_inj.inlet.P": values["fuel_manifold.P"],
            "fuel_inj.inlet.T": values["fuel_manifold.T"],
            "fuel_inj.inlet.h": fuel_state.h,
            "fuel_inj.inlet.rho": fuel_state.rho,
            "fuel_inj.inlet.mdot": commands["fuel_inj.inlet.mdot"],
            "nozzle.P_ambient": values.get("nozzle.ambient.P", values0["nozzle.ambient.P"]),
            "pms.mdot_total": commands["commands.pms.mdot_total"],
            "pms.OF": commands["commands.pms.OF"],
        }
        if has_valves:
            values_out.update(
                {
                    "lox_valve.inlet.P": values["lox_manifold.P"],
                    "lox_valve.inlet.T": values["lox_manifold.T"],
                    "lox_valve.inlet.h": lox_state.h,
                    "lox_valve.inlet.rho": lox_state.rho,
                    "lox_valve.valve.A_frac": commands["lox_valve.valve.A_frac"],
                    "fuel_valve.inlet.P": values["fuel_manifold.P"],
                    "fuel_valve.inlet.T": values["fuel_manifold.T"],
                    "fuel_valve.inlet.h": fuel_state.h,
                    "fuel_valve.inlet.rho": fuel_state.rho,
                    "fuel_valve.valve.A_frac": commands["fuel_valve.valve.A_frac"],
                    "lox_valve.position": commands["lox_valve.valve.A_frac"],
                    "fuel_valve.position": commands["fuel_valve.valve.A_frac"],
                }
            )
            if values_out["lox_valve.position"] < 0.01 and values_out["fuel_valve.position"] < 0.01:
                values_out["chamber.outlet.mdot"] = 0.0
                values_out["nozzle.inlet.mdot"] = 0.0
        return values_out

    return bcs


def run_tca_profile(config_path: str | Path, output_dir: str | Path = "outputs") -> TCAResultSummary:
    loaded = load_analysis_config(config_path)
    run = coerce_numbers(loaded.analysis_config.analysis)
    layout, _, lox, methane = build_tca_engine(loaded)

    initial_targets = coerce_numbers(evaluate_operating_targets(loaded.operating_conditions, 0.0))
    start_time = float(run.get("time", {}).get("start_s", 0.0))
    duration = float(initial_targets["pms"]["duration"])
    bcs_fn = make_tca_bcs_fn(loaded, lox, methane)

    X0 = layout.assemble_state_vector()
    bcs0 = bcs_fn(start_time)
    if abs(bcs0["lox_inj.inlet.mdot"]) > 1e-12 or abs(bcs0["fuel_inj.inlet.mdot"]) > 1e-12:
        X0 = SteadyStateSolver(layout, **loaded.analysis_config.solver.get("steady_trim", {})).solve(X0, bcs0)

    solver = TransientSolver(layout, **loaded.analysis_config.solver["transient"])
    result = solver.integrate((start_time, duration), X0, bcs_fn)
    return _export_tca_result(loaded, run, layout, result, bcs_fn, Path(output_dir))


def _export_tca_result(loaded, run: dict, layout, result, bcs_fn, output_dir: Path) -> TCAResultSummary:
    t, X = _sample_result_for_output(result, loaded.telemetry)
    state_names = layout.all_state_names()
    pc = X[:, state_names.index("chamber.P")]
    mdot_total = np.zeros_like(t)
    of_ratio = np.zeros_like(t)
    mdot_lox = np.zeros_like(t)
    mdot_fuel = np.zeros_like(t)
    thrust = np.zeros_like(t)
    isp = np.zeros_like(t)
    has_valves = "lox_valve" in loaded.engine.components
    valve_position = np.zeros_like(t) if has_valves else None
    fuel_valve_position = np.zeros_like(t) if has_valves else None

    for i, ti in enumerate(t):
        bcs = bcs_fn(float(ti))
        mdot_total[i] = bcs["pms.mdot_total"]
        of_ratio[i] = bcs["pms.OF"]
        mdot_lox[i] = bcs["lox_inj.inlet.mdot"]
        mdot_fuel[i] = bcs["fuel_inj.inlet.mdot"]
        if has_valves:
            valve_position[i] = bcs["lox_valve.position"]
            fuel_valve_position[i] = bcs["fuel_valve.position"]
        outputs = evaluate_outputs(layout, X[i], bcs)
        thrust[i] = outputs["nozzle"].get("thrust", np.nan)
        isp[i] = outputs["nozzle"].get("Isp_vacuum", np.nan)

    output_dir.mkdir(exist_ok=True)
    out_csv = output_dir / run["output"]["csv"]
    columns = {
        "time_s": t,
        "mdot_total_kg_s": mdot_total,
        "OF": of_ratio,
        "mdot_lox_kg_s": mdot_lox,
        "mdot_fuel_kg_s": mdot_fuel,
        "Pc_Pa": pc,
        "thrust_N": thrust,
        "Isp_vac_s": isp,
    }
    header = ["time_s", "mdot_total_kg_s", "OF", "mdot_lox_kg_s", "mdot_fuel_kg_s"]
    if has_valves:
        columns["lox_valve_position"] = valve_position
        columns["fuel_valve_position"] = fuel_valve_position
        header.extend(["lox_valve_position", "fuel_valve_position"])
    header.extend(["Pc_Pa", "thrust_N", "Isp_vac_s"])
    if loaded.telemetry is not None:
        samples = _build_samples(t, X, columns, layout, bcs_fn)
        header, telemetry_columns = build_telemetry_rows(loaded.telemetry, samples)
        write_telemetry_csv(out_csv, header, telemetry_columns)
    else:
        write_telemetry_csv(out_csv, header, columns)

    out_plot = output_dir / run["output"]["plot"]
    _plot_tca(out_plot, t, mdot_total, of_ratio, mdot_lox, mdot_fuel, pc, thrust, valve_position)
    return TCAResultSummary(
        csv=out_csv,
        plot=out_plot,
        time=t,
        pc=pc,
        thrust=thrust,
        mdot_total=mdot_total,
        of_ratio=of_ratio,
        valve_position=valve_position,
    )


def _build_samples(t, X, columns, layout, bcs_fn) -> list[dict]:
    samples = []
    for i, ti in enumerate(t):
        bcs = bcs_fn(float(ti))
        outputs = evaluate_outputs(layout, X[i], bcs)
        sample = {
            "time": float(ti),
            "time_s": float(ti),
            "pms.mdot_total": float(bcs["pms.mdot_total"]),
            "pms.OF": float(bcs["pms.OF"]),
            "chamber.P": float(columns["Pc_Pa"][i]),
            "nozzle.thrust": float(outputs["nozzle"].get("thrust", np.nan)),
            "mdot.lox": float(columns["mdot_lox_kg_s"][i]),
            "mdot.fuel": float(columns["mdot_fuel_kg_s"][i]),
        }
        if "lox_valve_position" in columns:
            sample["lox_valve.position"] = float(columns["lox_valve_position"][i])
            sample["fuel_valve.position"] = float(columns["fuel_valve_position"][i])
        samples.append(sample)
    return samples


def _sample_result_for_output(result, telemetry_config):
    if telemetry_config is None or telemetry_config.sample_rate_hz is None:
        return result.t, result.X
    rate = float(telemetry_config.sample_rate_hz)
    if rate <= 0.0:
        return result.t, result.X
    dt = 1.0 / rate
    t0 = float(result.t[0])
    tf = float(result.t[-1])
    t = np.arange(t0, tf + 0.5 * dt, dt)
    t[-1] = min(t[-1], tf)
    if t[-1] < tf:
        t = np.append(t, tf)
    X = np.column_stack([np.interp(t, result.t, result.X[:, i]) for i in range(result.X.shape[1])])
    return t, X


def _plot_tca(path, t, mdot_total, of_ratio, mdot_lox, mdot_fuel, pc, thrust, valve_position):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(mdot_total, of_ratio, marker="o", markersize=2)
    axes[0, 0].set(xlabel="Total mass flow [kg/s]", ylabel="OF [-]", title="PMS runbox perimeter")
    axes[0, 1].plot(t, pc / 1e6)
    axes[0, 1].set(xlabel="Time [s]", ylabel="Pc [MPa]", title="Chamber pressure")
    axes[1, 0].plot(t, mdot_total, label="total")
    axes[1, 0].plot(t, mdot_lox, label="LOX")
    axes[1, 0].plot(t, mdot_fuel, label="CH4")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Mass flow [kg/s]", title="Commanded flows")
    axes[1, 0].legend()
    axes[1, 1].plot(t, thrust / 1000, label="thrust")
    if valve_position is not None:
        axes[1, 1].plot(t, valve_position * max(np.nanmax(thrust) / 1000, 1.0), label="valve cmd")
        axes[1, 1].set(ylabel="Thrust [kN] / valve [-]", title="Valve timing and thrust")
    else:
        axes[1, 1].set(ylabel="Thrust [kN]", title="Nozzle thrust")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
