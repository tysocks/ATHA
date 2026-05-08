"""YAML-driven gas-generator LOX/methane cycle with regen and sweeps."""

from __future__ import annotations

from pathlib import Path

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
from atha.monte_carlo import MonteCarloRunner, ParameterType, UncertainParameter
from atha.solver.steady_state import SteadyStateSolver
from atha.thermo.cantera_backend import CanteraBackend
from atha.thermo.coolprop_backend import CoolPropBackend


CONFIG_PATH = Path(__file__).parent / "configs" / "analysis.yaml"
OUTPUT_DIR = Path("outputs")


def _params(loaded, name: str) -> dict:
    return _coerce_numbers(dict(loaded.engine.components[name].parameters))


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


def _boundary_values(loaded, t: float = 0.0) -> dict:
    return _coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def build_engine(loaded, maps, lox_eta_scale=1.0, fuel_eta_scale=1.0, gg_efficiency=None):
    engine = Engine(loaded.engine.name)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    fluids = run["fluids"]
    combustion = run["combustion"]
    lox = CoolPropBackend(fluids["oxidizer"])
    fuel = CoolPropBackend(fluids["fuel"])

    shaft = Rotor("shaft", **_params(loaded, "shaft"))

    lox_cfg = _params(loaded, "lox_pump")
    fuel_cfg = _params(loaded, "fuel_pump")
    lox_pump = Pump(
        "lox_pump",
        pump_map=PumpMap.from_design_point(**lox_cfg.pop("pump_map")),
        fluid=lox,
        **lox_cfg,
    )
    fuel_pump = Pump(
        "fuel_pump",
        pump_map=PumpMap.from_design_point(**fuel_cfg.pop("pump_map")),
        fluid=fuel,
        **fuel_cfg,
    )
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
    turbine = Turbine(
        "turbine",
        turbine_map=TurbineMap.from_design_point(**turbine_cfg.pop("turbine_map")),
        **turbine_cfg,
    )
    regen = RegenChannel("regen", fluid=fuel, **_params(loaded, "regen"))
    chamber = CombustionChamber(
        "chamber",
        thermo=CanteraBackend(combustion["mechanism"], initial_X=combustion["chamber_initial_X"]),
        fuel=combustion["fuel"],
        oxidizer=combustion["oxidizer"],
        **_params(loaded, "chamber"),
    )
    nozzle_cfg = _params(loaded, "nozzle")
    nozzle = Nozzle(
        "nozzle",
        efficiencies=JANNAFEfficiencies(**nozzle_cfg.pop("efficiencies")),
        **nozzle_cfg,
    )
    lox_inj = OrificeCompressible("lox_inj", **_params(loaded, "lox_inj"))
    fuel_inj = OrificeCompressible("fuel_inj", **_params(loaded, "fuel_inj"))

    for comp in [shaft, lox_pump, fuel_pump, gg, turbine, regen, chamber, nozzle, lox_inj, fuel_inj]:
        engine.add_component(comp)
    for conn in loaded.engine.connections:
        src_comp, src_port = conn.source.split(".", 1)
        dst_comp, dst_port = conn.target.split(".", 1)
        engine.connect(engine[src_comp].port(src_port), engine[dst_comp].port(dst_port))

    return engine.compile(), engine, lox, fuel


def make_bcs(loaded, lox, fuel, t=0.0, T_lox=None, T_fuel=None, omega=None):
    values = _boundary_values(loaded, t)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    T_lox = values["lox_tank.outlet.T"] if T_lox is None else T_lox
    T_fuel = values["fuel_tank.outlet.T"] if T_fuel is None else T_fuel
    omega = values["shaft.omega_override"] if omega is None else omega
    return {
        "lox_pump.inlet.P": values["lox_tank.outlet.P"],
        "lox_pump.inlet.h": lox.state_from_PT(values["lox_tank.outlet.P"], T_lox).h,
        "fuel_pump.inlet.P": values["fuel_tank.outlet.P"],
        "fuel_pump.inlet.h": fuel.state_from_PT(values["fuel_tank.outlet.P"], T_fuel).h,
        "nozzle.P_ambient": values["nozzle.ambient.P"],
        "gas.T": run["combustion"]["gas_T"],
        "gas.P": run["design"]["Pc"],
        "shaft.omega_override": omega,
    }


def solve_nominal(loaded, maps):
    layout, engine, lox, fuel = build_engine(loaded, maps)
    bcs = make_bcs(loaded, lox, fuel)
    X = SteadyStateSolver(layout, **loaded.analysis_config.solver["steady_trim"]).solve(
        layout.assemble_state_vector(), bcs
    )
    layout.scatter_state_vector(X)
    print("\nGG LOX/methane nominal")
    print(f"  Pc        : {engine['chamber']._state_values['P'] / 1e6:.3f} MPa")
    print(f"  GG T      : {engine['gg']._state_values['T']:.0f} K")
    print(f"  Shaft     : {engine['shaft']._state_values['omega'] * 30 / np.pi:.0f} rpm")
    print(f"  Thrust    : {engine['nozzle'].last_outputs['thrust']:.0f} N")
    print(f"  Isp vac   : {engine['nozzle'].last_outputs['Isp_vacuum']:.1f} s")
    print(f"  Regen Tw  : {engine['regen']._state_values['T_wall']:.0f} K")


def run_monte_carlo(loaded, maps):
    mc = _coerce_numbers(loaded.analysis_config.analysis["monte_carlo"])
    params = [
        UncertainParameter(p["name"], getattr(ParameterType, p["distribution"].upper()), **p["settings"])
        for p in mc["parameters"]
    ]
    names = [p.name for p in params]

    def evaluate(row):
        sample = {names[i]: float(row[i]) for i in range(len(names))}
        layout, engine, lox, fuel = build_engine(
            loaded,
            maps,
            lox_eta_scale=sample["lox_eta_scale"],
            fuel_eta_scale=sample["fuel_eta_scale"],
            gg_efficiency=sample["gg_efficiency"],
        )
        bcs = make_bcs(loaded, lox, fuel, T_lox=sample["T_lox_inlet"], T_fuel=sample["T_fuel_inlet"])
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
    OUTPUT_DIR.mkdir(exist_ok=True)
    result.save(str(OUTPUT_DIR / mc["result_file"]))
    result.plot_histogram(bins=25, title="GG LOX/methane thrust distribution")
    plt.savefig(OUTPUT_DIR / mc["histogram_file"], dpi=150)
    plt.close()


def run_speed_sweep(loaded, maps):
    sweep = _coerce_numbers(loaded.analysis_config.analysis["speed_sweep"])
    speeds = np.linspace(float(sweep["rpm_min"]), float(sweep["rpm_max"]), int(sweep["points"]))
    thrust = np.zeros_like(speeds)
    pc = np.zeros_like(speeds)
    isp = np.zeros_like(speeds)

    for i, rpm in enumerate(speeds):
        layout, engine, lox, fuel = build_engine(loaded, maps)
        bcs = make_bcs(loaded, lox, fuel, omega=rpm * np.pi / 30)
        X = SteadyStateSolver(layout).solve(layout.assemble_state_vector(), bcs)
        layout.scatter_state_vector(X)
        thrust[i] = engine["nozzle"].last_outputs["thrust"]
        pc[i] = engine["chamber"]._state_values["P"]
        isp[i] = engine["nozzle"].last_outputs["Isp_vacuum"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(speeds / 1000, thrust / 1000)
    axes[0].set(xlabel="Shaft speed [krpm]", ylabel="Thrust [kN]")
    axes[1].plot(speeds / 1000, pc / 1e6)
    axes[1].set(xlabel="Shaft speed [krpm]", ylabel="Pc [MPa]")
    axes[2].plot(speeds / 1000, isp)
    axes[2].set(xlabel="Shaft speed [krpm]", ylabel="Isp vac [s]")
    fig.suptitle("GG LOX/methane speed sweep")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / sweep["plot_file"], dpi=150)
    plt.close(fig)
    print(f"\nSpeed sweep: {thrust.min() / 1000:.2f}-{thrust.max() / 1000:.2f} kN")


def main():
    loaded = load_analysis_config(CONFIG_PATH)
    maps = build_performance_maps(loaded.maps)
    solve_nominal(loaded, maps)
    run_monte_carlo(loaded, maps)
    run_speed_sweep(loaded, maps)


if __name__ == "__main__":
    main()
