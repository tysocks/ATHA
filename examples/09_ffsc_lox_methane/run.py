"""YAML-driven full-flow staged-combustion LOX/methane example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.preburner import Preburner
from atha.components.pump import Pump, PumpMap
from atha.components.regen_channel import RegenChannel
from atha.components.rotor import Rotor
from atha.components.turbine import Turbine, TurbineMap
from atha.config import build_performance_maps, evaluate_boundary_conditions, load_analysis_config
from atha.core.engine import Engine
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.profiles import ControlCommand, PhaseDefinition, PhaseMode, SafetyLimit, TestProfile
from atha.solver.steady_state import SteadyStateSolver
from atha.thermo.cantera_backend import CanteraBackend
from atha.thermo.coolprop_backend import CoolPropBackend


CONFIG_PATH = Path(__file__).parent / "configs" / "analysis.yaml"


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


def build_engine(loaded, maps):
    engine = Engine(loaded.engine.name)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    fluids = run["fluids"]
    combustion = run["combustion"]
    lox = CoolPropBackend(fluids["oxidizer"])
    fuel = CoolPropBackend(fluids["fuel"])

    lox_shaft = Rotor("lox_shaft", **_params(loaded, "lox_shaft"))
    fuel_shaft = Rotor("fuel_shaft", **_params(loaded, "fuel_shaft"))

    lox_cfg = _params(loaded, "lox_pump")
    fuel_cfg = _params(loaded, "fuel_pump")
    lox_pump = Pump("lox_pump", pump_map=PumpMap.from_design_point(**lox_cfg.pop("pump_map")), fluid=lox, **lox_cfg)
    fuel_pump = Pump("fuel_pump", pump_map=PumpMap.from_design_point(**fuel_cfg.pop("pump_map")), fluid=fuel, **fuel_cfg)
    lox_pump._efficiency_map = maps["lox_pump_efficiency"]
    fuel_pump._efficiency_map = maps["fuel_pump_efficiency"]

    ox_preburner = Preburner(
        "ox_preburner",
        thermo=CanteraBackend(combustion["mechanism"]),
        fuel=combustion["fuel"],
        oxidizer=combustion["oxidizer"],
        **_params(loaded, "ox_preburner"),
    )
    fuel_preburner = Preburner(
        "fuel_preburner",
        thermo=CanteraBackend(combustion["mechanism"]),
        fuel=combustion["fuel"],
        oxidizer=combustion["oxidizer"],
        **_params(loaded, "fuel_preburner"),
    )
    ox_turb_cfg = _params(loaded, "lox_turbine")
    fuel_turb_cfg = _params(loaded, "fuel_turbine")
    lox_turbine = Turbine(
        "lox_turbine",
        turbine_map=TurbineMap.from_design_point(**ox_turb_cfg.pop("turbine_map")),
        **ox_turb_cfg,
    )
    fuel_turbine = Turbine(
        "fuel_turbine",
        turbine_map=TurbineMap.from_design_point(**fuel_turb_cfg.pop("turbine_map")),
        **fuel_turb_cfg,
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

    for comp in [
        lox_shaft,
        fuel_shaft,
        lox_pump,
        fuel_pump,
        ox_preburner,
        fuel_preburner,
        lox_turbine,
        fuel_turbine,
        regen,
        chamber,
        nozzle,
    ]:
        engine.add_component(comp)
    for conn in loaded.engine.connections:
        src_comp, src_port = conn.source.split(".", 1)
        dst_comp, dst_port = conn.target.split(".", 1)
        engine.connect(engine[src_comp].port(src_port), engine[dst_comp].port(dst_port))
    return engine.compile(), engine, lox, fuel


def make_bcs(loaded, lox, fuel, t=0.0):
    values = _boundary_values(loaded, t)
    run = _coerce_numbers(loaded.analysis_config.analysis)
    return {
        "lox_pump.inlet.P": values["lox_tank.outlet.P"],
        "lox_pump.inlet.h": lox.state_from_PT(values["lox_tank.outlet.P"], values["lox_tank.outlet.T"]).h,
        "fuel_pump.inlet.P": values["fuel_tank.outlet.P"],
        "fuel_pump.inlet.h": fuel.state_from_PT(values["fuel_tank.outlet.P"], values["fuel_tank.outlet.T"]).h,
        "nozzle.P_ambient": values["nozzle.ambient.P"],
        "gas.T": run["combustion"]["gas_T"],
        "gas.P": run["design"]["Pc"],
        "chamber.lox_inlet.mdot": run["design"]["mdot_lox"],
        "chamber.fuel_inlet.mdot": run["design"]["mdot_fuel"],
        "lox_shaft.omega_override": values["lox_shaft.omega_override"],
        "fuel_shaft.omega_override": values["fuel_shaft.omega_override"],
    }


def solve_nominal(loaded, maps):
    layout, engine, lox, fuel = build_engine(loaded, maps)
    bcs = make_bcs(loaded, lox, fuel)
    X = SteadyStateSolver(layout, **loaded.analysis_config.solver["steady_trim"]).solve(
        layout.assemble_state_vector(), bcs
    )
    layout.scatter_state_vector(X)
    print("\nFFSC LOX/methane nominal")
    print(f"  Pc              : {engine['chamber']._state_values['P'] / 1e6:.3f} MPa")
    print(f"  Ox preburner T  : {engine['ox_preburner']._state_values['T']:.0f} K")
    print(f"  Fuel preburner T: {engine['fuel_preburner']._state_values['T']:.0f} K")
    print(f"  LOX shaft       : {engine['lox_shaft']._state_values['omega'] * 30 / np.pi:.0f} rpm")
    print(f"  Fuel shaft      : {engine['fuel_shaft']._state_values['omega'] * 30 / np.pi:.0f} rpm")
    print(f"  Thrust          : {engine['nozzle'].last_outputs['thrust']:.0f} N")
    print(f"  Isp vac         : {engine['nozzle'].last_outputs['Isp_vacuum']:.1f} s")
    return layout, engine, lox, fuel


def run_profile(loaded, layout, lox, fuel):
    profile_cfg = _coerce_numbers(loaded.analysis_config.analysis["profile"])
    design = _coerce_numbers(loaded.analysis_config.analysis["design"])
    values = _boundary_values(loaded, 0.0)
    h_lox = lox.state_from_PT(values["lox_tank.outlet.P"], values["lox_tank.outlet.T"]).h
    h_fuel = fuel.state_from_PT(values["fuel_tank.outlet.P"], values["fuel_tank.outlet.T"]).h

    def ramp(t):
        return min(1.0, max(0.0, t / profile_cfg["ramp_duration"]))

    profile = TestProfile(
        name="ffsc_yaml_start",
        phases=[
            PhaseDefinition(
                name="startup",
                mode=PhaseMode.TRANSIENT,
                duration=float(profile_cfg["duration"]),
                control_commands=[
                    ControlCommand("lox_pump.inlet.mdot", fn=lambda t: design["mdot_lox"] * ramp(t)),
                    ControlCommand("fuel_pump.inlet.mdot", fn=lambda t: design["mdot_fuel"] * ramp(t)),
                    ControlCommand("lox_pump.inlet.h", fn=lambda t: h_lox),
                    ControlCommand("fuel_pump.inlet.h", fn=lambda t: h_fuel),
                ],
                recording_rate_hz=float(profile_cfg["recording_rate_hz"]),
            )
        ],
        global_limits=[
            SafetyLimit("Pc_max", "chamber", "P", upper_limit=float(profile_cfg["Pc_max"]), is_hard=True),
            SafetyLimit("Twall_max", "regen", "T_wall", upper_limit=float(profile_cfg["T_wall_max"]), is_hard=True),
        ],
    )
    result = profile.execute(layout, layout.assemble_state_vector(), bcs_fn=lambda t: make_bcs(loaded, lox, fuel, t))
    if result.success:
        startup = result.get_phase("startup")
        print("\nStartup profile")
        print(f"  LOX shaft peak : {startup.get('lox_shaft', 'omega').max() * 30 / np.pi:.0f} rpm")
        print(f"  Fuel shaft peak: {startup.get('fuel_shaft', 'omega').max() * 30 / np.pi:.0f} rpm")
        print(f"  Regen Tw end   : {startup.get('regen', 'T_wall')[-1]:.0f} K")
    else:
        print(f"\nStartup abort: {result.abort_reason} at t={result.abort_time:.3f} s")


def main():
    loaded = load_analysis_config(CONFIG_PATH)
    maps = build_performance_maps(loaded.maps)
    layout, engine, lox, fuel = solve_nominal(loaded, maps)
    run_profile(loaded, layout, lox, fuel)


if __name__ == "__main__":
    main()
