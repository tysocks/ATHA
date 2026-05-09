from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from atha.components.combustion_chamber import CombustionChamber
from atha.components.gas_generator import GasGenerator
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
from atha.components.pump import Pump, PumpMap
from atha.components.regen_channel import RegenChannel
from atha.components.rotor import Rotor
from atha.components.turbine import Turbine, TurbineMap
from atha.config import build_performance_maps, evaluate_boundary_conditions, load_analysis_config
from atha.core.engine import Engine
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.monte_carlo import MonteCarloResult, MonteCarloRunner, ParameterType, UncertainParameter
from atha.solver.steady_state import SteadyStateSolver
from atha.thermo.cantera_backend import CanteraBackend
from atha.thermo.coolprop_backend import CoolPropBackend


@dataclass
class NominalMCSweepSummary:
    config_path: Path
    csv: Path | None
    plot: Path | None
    nominal: dict[str, float]
    monte_carlo: MonteCarloResult | None
    monte_carlo_file: Path | None
    histogram: Path | None
    sweep: dict[str, np.ndarray] | None
    sweep_plot: Path | None


def run_nominal_mc_sweep(config_path: str | Path, output_dir: str | Path = "outputs") -> NominalMCSweepSummary:
    loaded = load_analysis_config(config_path)
    maps = build_performance_maps(loaded.maps)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    nominal = solve_nominal(loaded, maps)
    mc_result, mc_file, histogram = run_monte_carlo(loaded, maps, output_dir)
    sweep_result, sweep_plot = run_speed_sweep(loaded, maps, output_dir)
    return NominalMCSweepSummary(
        config_path=Path(config_path),
        csv=None,
        plot=sweep_plot,
        nominal=nominal,
        monte_carlo=mc_result,
        monte_carlo_file=mc_file,
        histogram=histogram,
        sweep=sweep_result,
        sweep_plot=sweep_plot,
    )


def solve_nominal(loaded, maps) -> dict[str, float]:
    layout, engine, lox, fuel = build_engine(loaded, maps)
    bcs = make_bcs(loaded, lox, fuel)
    solver_cfg = loaded.analysis_config.solver["steady_trim"]
    X = SteadyStateSolver(layout, **solver_cfg).solve(layout.assemble_state_vector(), bcs)
    layout.scatter_state_vector(X)
    nominal = _outputs(engine)
    print(f"\n{loaded.analysis_config.name} nominal")
    print(f"  Pc      : {nominal['Pc'] / 1e6:.3f} MPa")
    print(f"  GG T    : {nominal['gg_T']:.0f} K")
    print(f"  Shaft   : {nominal['shaft_rpm']:.0f} rpm")
    print(f"  Thrust  : {nominal['thrust']:.0f} N")
    print(f"  Isp vac : {nominal['Isp_vacuum']:.1f} s")
    if "regen_T_wall" in nominal:
        print(f"  Regen Tw: {nominal['regen_T_wall']:.0f} K")
    return nominal


def run_monte_carlo(loaded, maps, output_dir: Path) -> tuple[MonteCarloResult | None, Path | None, Path | None]:
    run = _coerce_numbers(loaded.analysis_config.analysis)
    mc = run.get("monte_carlo")
    if not isinstance(mc, dict):
        return None, None, None
    params = [
        UncertainParameter(p["name"], getattr(ParameterType, p["distribution"].upper()), **p["settings"])
        for p in mc["parameters"]
    ]
    names = [p.name for p in params]

    def evaluate(row: np.ndarray) -> float:
        sample = {names[i]: float(row[i]) for i in range(len(names))}
        layout, engine, lox, fuel = build_engine(
            loaded,
            maps,
            lox_eta_scale=sample.get("lox_eta_scale", 1.0),
            fuel_eta_scale=sample.get("fuel_eta_scale", 1.0),
            gg_efficiency=sample.get("gg_efficiency"),
        )
        bcs = make_bcs(
            loaded,
            lox,
            fuel,
            T_lox=sample.get("T_lox_inlet"),
            T_fuel=sample.get("T_fuel_inlet"),
        )
        X = SteadyStateSolver(layout).solve(layout.assemble_state_vector(), bcs)
        layout.scatter_state_vector(X)
        return float(engine["nozzle"].last_outputs["thrust"])

    result = MonteCarloRunner(
        params=params,
        evaluate_fn=evaluate,
        n_samples=int(mc["samples"]),
        sampler=mc["sampler"],
        n_jobs=int(mc["n_jobs"]),
        seed=int(mc["seed"]),
    ).run()
    result.print_summary()
    result_path = output_dir / str(mc["result_file"])
    result.save(str(result_path))
    fig = result.plot_histogram(bins=25, title=f"{loaded.analysis_config.name} thrust distribution", show=False)
    histogram_path = output_dir / str(mc["histogram_file"])
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    return result, result_path, histogram_path


def run_speed_sweep(loaded, maps, output_dir: Path) -> tuple[dict[str, np.ndarray] | None, Path | None]:
    run = _coerce_numbers(loaded.analysis_config.analysis)
    sweep = run.get("speed_sweep")
    if not isinstance(sweep, dict):
        return None, None
    speeds_rpm = np.linspace(float(sweep["rpm_min"]), float(sweep["rpm_max"]), int(sweep["points"]))
    thrust = np.zeros_like(speeds_rpm)
    pc = np.zeros_like(speeds_rpm)
    isp = np.zeros_like(speeds_rpm)

    for i, rpm in enumerate(speeds_rpm):
        layout, engine, lox, fuel = build_engine(loaded, maps)
        bcs = make_bcs(loaded, lox, fuel, omega=rpm * np.pi / 30.0)
        X = SteadyStateSolver(layout).solve(layout.assemble_state_vector(), bcs)
        layout.scatter_state_vector(X)
        values = _outputs(engine)
        thrust[i] = values["thrust"]
        pc[i] = values["Pc"]
        isp[i] = values["Isp_vacuum"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    speed_scale = 1000.0 if np.max(speeds_rpm) > 10000.0 else 1.0
    speed_label = "Shaft speed [krpm]" if speed_scale == 1000.0 else "Shaft speed [rpm]"
    axes[0].plot(speeds_rpm / speed_scale, thrust / 1000.0)
    axes[0].set(xlabel=speed_label, ylabel="Thrust [kN]")
    axes[1].plot(speeds_rpm / speed_scale, pc / 1e6)
    axes[1].set(xlabel=speed_label, ylabel="Pc [MPa]")
    axes[2].plot(speeds_rpm / speed_scale, isp)
    axes[2].set(xlabel=speed_label, ylabel="Isp vac [s]")
    fig.suptitle(f"{loaded.analysis_config.name} speed sweep")
    fig.tight_layout()
    plot_path = output_dir / str(sweep["plot_file"])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSpeed sweep: {thrust.min() / 1000:.2f}-{thrust.max() / 1000:.2f} kN")
    return {"speed_rpm": speeds_rpm, "thrust": thrust, "Pc": pc, "Isp_vacuum": isp}, plot_path


def build_engine(loaded, maps, lox_eta_scale: float = 1.0, fuel_eta_scale: float = 1.0, gg_efficiency=None):
    engine = Engine(loaded.engine.name)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    fluids = run["fluids"]
    combustion = run["combustion"]
    lox = CoolPropBackend(fluids["oxidizer"])
    fuel = CoolPropBackend(fluids["fuel"])

    shaft = Rotor("shaft", **_params(loaded, "shaft"))
    lox_cfg = _params(loaded, "lox_pump")
    fuel_cfg = _params(loaded, "fuel_pump")
    lox_pump = Pump("lox_pump", pump_map=PumpMap.from_design_point(**lox_cfg.pop("pump_map")), fluid=lox, **lox_cfg)
    fuel_pump = Pump("fuel_pump", pump_map=PumpMap.from_design_point(**fuel_cfg.pop("pump_map")), fluid=fuel, **fuel_cfg)
    lox_pump._efficiency_map = maps["lox_pump_efficiency"]
    fuel_pump._efficiency_map = maps["fuel_pump_efficiency"]
    lox_pump.map_efficiency_scale = lox_eta_scale
    fuel_pump.map_efficiency_scale = fuel_eta_scale

    gg_cfg = _params(loaded, "gg")
    gg_eff_nominal = gg_cfg.pop("efficiency")
    gg = GasGenerator(
        "gg",
        thermo=CanteraBackend(combustion["mechanism"]),
        fuel=combustion["fuel"],
        oxidizer=combustion["oxidizer"],
        efficiency=gg_efficiency if gg_efficiency is not None else gg_eff_nominal,
        **gg_cfg,
    )
    turbine_cfg = _params(loaded, "turbine")
    turbine = Turbine("turbine", turbine_map=TurbineMap.from_design_point(**turbine_cfg.pop("turbine_map")), **turbine_cfg)
    chamber = CombustionChamber(
        "chamber",
        thermo=CanteraBackend(combustion["mechanism"], initial_X=combustion.get("chamber_initial_X")),
        fuel=combustion["fuel"],
        oxidizer=combustion["oxidizer"],
        **_params(loaded, "chamber"),
    )
    nozzle_cfg = _params(loaded, "nozzle")
    nozzle = Nozzle("nozzle", efficiencies=JANNAFEfficiencies(**nozzle_cfg.pop("efficiencies")), **nozzle_cfg)
    lox_inj = OrificeCompressible("lox_inj", **_params(loaded, "lox_inj"))
    fuel_inj = OrificeCompressible("fuel_inj", **_params(loaded, "fuel_inj"))
    components = [shaft, lox_pump, fuel_pump, gg, turbine]
    if "regen" in loaded.engine.components:
        components.append(RegenChannel("regen", fluid=fuel, **_params(loaded, "regen")))
    components.extend([chamber, nozzle, lox_inj, fuel_inj])
    for component in components:
        engine.add_component(component)
    for conn in loaded.engine.connections:
        src_comp, src_port = conn.source.split(".", 1)
        dst_comp, dst_port = conn.target.split(".", 1)
        engine.connect(engine[src_comp].port(src_port), engine[dst_comp].port(dst_port))
    return engine.compile(), engine, lox, fuel


def make_bcs(loaded, lox, fuel, t: float = 0.0, T_lox=None, T_fuel=None, omega=None) -> dict[str, float]:
    values = _boundary_values(loaded, t)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    T_lox = values["lox_tank.outlet.T"] if T_lox is None else T_lox
    T_fuel = values["fuel_tank.outlet.T"] if T_fuel is None else T_fuel
    omega = values["shaft.omega_override"] if omega is None else omega
    bcs = {
        "lox_pump.inlet.P": values["lox_tank.outlet.P"],
        "lox_pump.inlet.h": lox.state_from_PT(values["lox_tank.outlet.P"], T_lox).h,
        "fuel_pump.inlet.P": values["fuel_tank.outlet.P"],
        "fuel_pump.inlet.h": fuel.state_from_PT(values["fuel_tank.outlet.P"], T_fuel).h,
        "nozzle.P_ambient": values["nozzle.ambient.P"],
        "shaft.omega_override": omega,
    }
    if "gas_T" in run.get("combustion", {}):
        bcs["gas.T"] = run["combustion"]["gas_T"]
    if "Pc" in run.get("design", {}):
        bcs["gas.P"] = run["design"]["Pc"]
    return bcs


def _params(loaded, name: str) -> dict[str, Any]:
    return _coerce_numbers(dict(loaded.engine.components[name].parameters))


def _boundary_values(loaded, t: float = 0.0) -> dict[str, Any]:
    return _coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def _outputs(engine) -> dict[str, float]:
    values = {
        "Pc": float(engine["chamber"]._state_values["P"]),
        "gg_T": float(engine["gg"]._state_values["T"]),
        "shaft_rpm": float(engine["shaft"]._state_values["omega"] * 30.0 / np.pi),
        "thrust": float(engine["nozzle"].last_outputs["thrust"]),
        "Isp_vacuum": float(engine["nozzle"].last_outputs["Isp_vacuum"]),
    }
    if "regen" in engine:
        values["regen_T_wall"] = float(engine["regen"]._state_values["T_wall"])
    return values


def _coerce_numbers(value):
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
