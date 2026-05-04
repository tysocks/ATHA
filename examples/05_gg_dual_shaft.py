"""
Gas Generator Cycle — Separate LOX and Fuel Turbopumps
=======================================================
Engine topology:

  LOX tank ──► LOX pump ──┬──► Main LOX injector ──► Combustion chamber ──► Nozzle
                           │                                ▲
  Fuel tank ──► Fuel pump ─┼──► Main fuel injector ────────┘
                           │
                           ├──► Gas generator (small bleed, fuel-rich)
                           │         │
                           │         ▼
                           │    LOX turbine (drives LOX pump shaft only)
                           │         │
                           └─────────┴──► Fuel turbine (drives fuel pump shaft only)
                                         │
                                    (both exhausts dump overboard)

Key difference from single-shaft GG: the LOX and fuel turbopumps run on
independent shafts. The GG exhaust can be split between the two turbines
(or run in series). Independent shafts allow each pump to find its own
equilibrium speed, which improves off-design performance and avoids
coupled instability modes.

Architecture shown here: GG exhaust split by a flow divider, each half
driving one turbine in parallel, then both exhausts dump overboard.

NOTE: Requires planned components — Pump, Turbine, Rotor, GasGenerator,
      CombustionChamber, Nozzle, OrificeCompressible, FlowSplitter.
"""

import numpy as np
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.gas_generator import GasGenerator
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
from atha.components.flow_splitter import FlowSplitter
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.solver.steady_state import SteadyStateSolver
from atha.profiles import (
    TestProfile, PhaseDefinition, PhaseMode, ControlCommand, SafetyLimit,
)

# ---------------------------------------------------------------------------
# Propellants
# ---------------------------------------------------------------------------
lox     = CoolPropBackend("Oxygen")
ethanol = CoolPropBackend("Ethanol")

P_lox_tank  = 4e5;  T_lox  = 91.0
P_fuel_tank = 4e5;  T_fuel = 293.0

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = ethanol.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design  = 2.0e6
MR_main    = 2.8
At         = 1.963e-3
Ae_At      = 8.0

mdot_total = 5500.0 / (290.0 * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

GG_BLEED   = 0.035
MR_gg      = 0.40
mdot_gg_fuel = GG_BLEED * mdot_fuel
mdot_gg_lox  = mdot_gg_fuel * MR_gg

P_pump_out = Pc_design * 1.55

# GG exhaust split fraction: fraction to LOX turbine (rest to fuel turbine)
# Tune so both shafts are power-balanced at nominal operating point
LOX_TURB_SPLIT = 0.55   # 55% of GG flow to LOX turbine

# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  20000, 0.68)
fuel_map = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 24000, 0.65)

# Turbines sized for their respective shaft power requirements
turb_lox_map  = TurbineMap.from_design_point(PR_design=2.5, eta_design=0.70, mdot_corrected_design=0.028)
turb_fuel_map = TurbineMap.from_design_point(PR_design=2.5, eta_design=0.70, mdot_corrected_design=0.022)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("gg_dual")

lox_shaft  = Rotor("lox_shaft",  moment_of_inertia=0.06, initial_speed_rpm=20000)
fuel_shaft = Rotor("fuel_shaft", moment_of_inertia=0.05, initial_speed_rpm=24000)

lox_pump  = Pump("lox_pump",  diameter=0.09, pump_map=lox_map,  fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.07, pump_map=fuel_map, fluid=ethanol)

gg_thermo = CanteraBackend("gri30.yaml")
gg = GasGenerator(
    "gg", volume=1e-4,
    thermo=gg_thermo,
    fuel="C2H5OH", oxidizer="O2",
    design_MR=MR_gg, efficiency=0.93,
    initial_P=P_pump_out, initial_T=1050.0,
)

# Flow splitter: divides GG exhaust between two turbines
gg_splitter = FlowSplitter("gg_split", split_fraction=LOX_TURB_SPLIT)

lox_turbine  = Turbine("lox_turbine",  diameter=0.09, turbine_map=turb_lox_map)
fuel_turbine = Turbine("fuel_turbine", diameter=0.08, turbine_map=turb_fuel_map)

cc_thermo = CanteraBackend("gri30.yaml")
chamber = CombustionChamber(
    "chamber", volume=5e-4,
    thermo=cc_thermo,
    fuel="C2H5OH", oxidizer="O2",
    efficiency=0.96,
    initial_P=Pc_design, initial_T=3400.0,
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.983)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

lox_inj  = OrificeCompressible("lox_inj",  Cd=0.72, area=1.5e-4)
fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=1.2e-4)

for comp in [lox_shaft, fuel_shaft, lox_pump, fuel_pump, gg, gg_splitter,
             lox_turbine, fuel_turbine, chamber, nozzle, lox_inj, fuel_inj]:
    engine.add_component(comp)

# GG propellant bleed
engine.connect(lox_pump.port("bleed"),  gg.port("lox_inlet"))
engine.connect(fuel_pump.port("bleed"), gg.port("fuel_inlet"))
engine.connect(gg.port("outlet"),       gg_splitter.port("inlet"))

# Parallel turbines from GG splitter
engine.connect(gg_splitter.port("outlet_a"), lox_turbine.port("inlet"))
engine.connect(gg_splitter.port("outlet_b"), fuel_turbine.port("inlet"))
# Both turbine exhausts dump overboard (terminal — no downstream connection)

# Shaft connections — independent shafts
engine.connect(lox_turbine.port("shaft"),  lox_shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"),     lox_shaft.port("pump"))
engine.connect(fuel_turbine.port("shaft"), fuel_shaft.port("turbine_in"))
engine.connect(fuel_pump.port("shaft"),    fuel_shaft.port("pump"))

# Main propellant flow
engine.connect(lox_pump.port("outlet"),  lox_inj.port("inlet"))
engine.connect(fuel_pump.port("outlet"), fuel_inj.port("inlet"))
engine.connect(lox_inj.port("outlet"),   chamber.port("lox_inlet"))
engine.connect(fuel_inj.port("outlet"),  chamber.port("fuel_inlet"))
engine.connect(chamber.port("outlet"),   nozzle.port("inlet"))

layout = engine.compile()
X0 = layout.assemble_state_vector()

# ---------------------------------------------------------------------------
# Steady-state trim balance
# ---------------------------------------------------------------------------
bcs = {
    "lox_pump.inlet.P":  P_lox_tank,  "lox_pump.inlet.h":  h_lox_inlet,
    "fuel_pump.inlet.P": P_fuel_tank, "fuel_pump.inlet.h": h_fuel_inlet,
    "nozzle.P_ambient":  0.0,
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs)
layout.scatter_state_vector(X_ss)

print(f"\nGG Dual-Shaft Steady-State Balance:")
print(f"  Chamber pressure   : {chamber._state_values['P']/1e6:.3f} MPa")
print(f"  GG temperature     : {gg._state_values['T']:.0f} K")
print(f"  LOX shaft speed    : {lox_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Fuel shaft speed   : {fuel_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust      : {nozzle.last_outputs['thrust']:.0f} N")
print(f"  Vacuum Isp         : {nozzle.last_outputs['Isp_vacuum']:.1f} s")

# ---------------------------------------------------------------------------
# Transient: LOX turbopump shaft spin-up from cold (engine start)
# ---------------------------------------------------------------------------
# Simulate the first 3 seconds of engine start:
# t=0:   ignition, GG comes online — turbines begin spooling
# t=0.5: LOX and fuel pumps begin to provide flow as shafts accelerate
# t=2.0: main valves open to design flow
# The dual-shaft configuration lets each turbopump find its own equilibrium.

def gg_flow_schedule(t):
    """GG flow ramps up from ignition."""
    return min(1.0, t / 0.3) * GG_BLEED

def main_valve_schedule(t):
    """Main propellant valves open after shaft speed is sufficient."""
    if t < 1.5:
        return 0.0
    return min(1.0, (t - 1.5) / 0.5)

profile = TestProfile(
    name="gg_dual_start",
    phases=[
        PhaseDefinition(
            name="spinup",
            mode=PhaseMode.TRANSIENT,
            duration=4.0,
            control_commands=[
                ControlCommand("lox_pump.inlet.mdot",
                               fn=lambda t: mdot_lox  * (gg_flow_schedule(t) + main_valve_schedule(t) * (1 - GG_BLEED))),
                ControlCommand("fuel_pump.inlet.mdot",
                               fn=lambda t: mdot_fuel * (gg_flow_schedule(t) + main_valve_schedule(t) * (1 - GG_BLEED))),
                ControlCommand("lox_pump.inlet.h",  fn=lambda t: h_lox_inlet),
                ControlCommand("fuel_pump.inlet.h", fn=lambda t: h_fuel_inlet),
            ],
            recording_rate_hz=200.0,
        ),
        PhaseDefinition(
            name="mainstage",
            mode=PhaseMode.STEADY_TRIM,
            duration=10.0,
        ),
    ],
    global_limits=[
        SafetyLimit("Pc_max",      "chamber",    "P",     upper_limit=2.6e6, is_hard=True),
        SafetyLimit("lox_rpm_max", "lox_shaft",  "omega", upper_limit=26000 * np.pi/30, is_hard=True),
        SafetyLimit("fuel_rpm_max","fuel_shaft", "omega", upper_limit=30000 * np.pi/30, is_hard=True),
        SafetyLimit("T_gg_max",    "gg",         "T",     upper_limit=1200.0, is_hard=True),
    ],
)

result = profile.execute(layout, X0)

if result.success:
    spinup = result.get_phase("spinup")
    print(f"\nStart transient: LOX shaft peak = "
          f"{spinup.get('lox_shaft', 'omega').max() * 30/np.pi:.0f} rpm  "
          f"Fuel shaft peak = "
          f"{spinup.get('fuel_shaft', 'omega').max() * 30/np.pi:.0f} rpm")
    result.plot_timeline()
else:
    print(f"\nAbort: {result.abort_reason} at t={result.abort_time:.3f} s")
