"""
Full-Flow Staged Combustion (FFSC) — LOX / Liquid Methane at 10 MPa
====================================================================
Engine topology:

  LOX tank ──► LOX pump ──► Ox-rich preburner (MR ~50) ──► LOX turbine ──► Main chamber
                                     ▲                                           ▲
  LCH4 tank ──► CH4 pump ──► RegenChannel ──► Fuel-rich preburner (MR ~0.25) ──► CH4 turbine ──┘
                                     ▲
                          (small LOX bleed to fuel-rich PB)

All propellant passes through a preburner before reaching the main chamber:
  - LOX side: ox-rich preburner (MR=50) produces ~730 K hot gas → LOX turbine
  - CH4 side: regen channel cools chamber wall, then fuel-rich preburner (MR=0.25)
              produces ~680 K hot gas → CH4 turbine

The methane path is: pump → RegenChannel → fuel-rich preburner → CH4 turbine → chamber.
The regen channel pre-heats the methane before combustion while cooling the
chamber/nozzle wall — the expander-adjacent benefit of FFSC architecture.

Both turbine exhausts combine as the reactants in the main combustion chamber.
Two independent turbopump shafts — each side runs at its own equilibrium speed.

Engine parameters (development-scale, 20 kN vacuum):
  Pc = 10.0 MPa, MR_main = 3.5 (LOX/LCH4), At = 1.03e-3 m², Ae/At = 50

Compare to:
  03_ffsc_cycle.py   — LOX/ethanol at 2 MPa
  08_orsc_lox_methane.py — same propellants, ORSC topology
"""

import numpy as np
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.preburner import Preburner
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.regen_channel import RegenChannel
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
methane = CoolPropBackend("Methane")

P_lox_tank  = 4e5;   T_lox  = 91.0    # K  subcooled LOX
P_fuel_tank = 4e5;   T_fuel = 108.0   # K  subcooled LCH4

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = methane.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design = 10.0e6   # Pa
MR_main   = 3.5
F_design  = 20000.0  # N  vacuum
Isp_est   = 375.0    # s  FFSC is ~5% higher Isp than ORSC
At        = 1.03e-3  # m²
Ae_At     = 50.0

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

# Preburner mixture ratios
MR_ox_pb   = 50.0    # ox-rich:  all LOX + trace CH4 → ~730 K
MR_fuel_pb = 0.25    # fuel-rich: all CH4 + trace LOX → ~680 K

# Each preburner uses all of its primary propellant + small bleed of the other
mdot_ox_pb_fuel  = mdot_lox / MR_ox_pb        # small CH4 bleed into ox PB
mdot_fuel_pb_lox = mdot_fuel * MR_fuel_pb      # small LOX bleed into fuel PB

# Pump outlet pressure: Pc × 1.7
P_pump_out = Pc_design * 1.7

# ---------------------------------------------------------------------------
# Performance maps (design-point based)
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  32000, 0.74)
fuel_map = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 27000, 0.69)

turb_ox_map   = TurbineMap.from_design_point(PR_design=2.2, eta_design=0.74,
                                              mdot_corrected_design=0.07)
turb_fuel_map = TurbineMap.from_design_point(PR_design=2.0, eta_design=0.72,
                                              mdot_corrected_design=0.05)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("ffsc_lox_ch4")

# Two independent shafts — LOX side and CH4 side
lox_shaft  = Rotor("lox_shaft",  moment_of_inertia=0.08, initial_speed_rpm=32000)
fuel_shaft = Rotor("fuel_shaft", moment_of_inertia=0.05, initial_speed_rpm=27000)

lox_pump  = Pump("lox_pump",  diameter=0.085, pump_map=lox_map,  fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.060, pump_map=fuel_map, fluid=methane)

ox_thermo   = CanteraBackend("gri30.yaml")
fuel_thermo = CanteraBackend("gri30.yaml")
cc_thermo   = CanteraBackend("gri30.yaml")

# Ox-rich preburner: all LOX + small CH4 bleed
ox_preburner = Preburner(
    "ox_preburner",
    volume=1.5e-4,
    thermo=ox_thermo,
    fuel="CH4",
    oxidizer="O2",
    design_MR=MR_ox_pb,
    efficiency=0.94,
    initial_P=P_pump_out,
    initial_T=730.0,         # K  ox-rich LOX/CH4 at MR=50
)

# Fuel-rich preburner: pre-heated CH4 (after regen) + small LOX bleed
fuel_preburner = Preburner(
    "fuel_preburner",
    volume=1.5e-4,
    thermo=fuel_thermo,
    fuel="CH4",
    oxidizer="O2",
    design_MR=MR_fuel_pb,
    efficiency=0.94,
    initial_P=P_pump_out,
    initial_T=680.0,         # K  fuel-rich LOX/CH4 at MR=0.25
)

lox_turbine  = Turbine("lox_turbine",  diameter=0.10, turbine_map=turb_ox_map)
fuel_turbine = Turbine("fuel_turbine", diameter=0.09, turbine_map=turb_fuel_map)

# Regenerative cooling channel — CH4 cools chamber/nozzle before the preburner
regen = RegenChannel(
    "regen",
    fluid=methane,
    channel_area=5e-5,        # m²   total cross-section
    hydraulic_diam=3e-3,      # m
    channel_length=0.8,       # m    chamber + nozzle contour
    hot_area=0.15,            # m²   gas-side area
    cool_area=0.18,           # m²   coolant-side area
    wall_mass=2.5,            # kg   CuCrZr alloy jacket
    wall_cp=390.0,            # J/(kg·K)
    h_hot_design=8000.0,      # W/(m²·K)  area-averaged Bartz over 0.15 m² contour
    Pc_design=Pc_design,
    recovery_factor=0.90,
    initial_T_wall=300.0,     # K   cold start
)

chamber = CombustionChamber(
    "chamber",
    volume=3e-4,
    thermo=cc_thermo,
    fuel="CH4",
    oxidizer="O2",
    efficiency=0.97,
    initial_P=Pc_design,
    initial_T=3500.0,
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

for comp in [lox_shaft, fuel_shaft, lox_pump, fuel_pump,
             ox_preburner, fuel_preburner,
             lox_turbine, fuel_turbine,
             regen, chamber, nozzle]:
    engine.add_component(comp)

# LOX side: pump → ox-rich preburner → LOX turbine → chamber lox inlet
engine.connect(lox_pump.port("outlet"),        ox_preburner.port("lox_inlet"))
engine.connect(fuel_pump.port("bleed_ox"),     ox_preburner.port("fuel_inlet"))  # CH4 bleed to ox PB
engine.connect(ox_preburner.port("outlet"),    lox_turbine.port("inlet"))
engine.connect(lox_turbine.port("outlet"),     chamber.port("lox_inlet"))

# CH4 side: pump → regen → fuel-rich preburner → CH4 turbine → chamber fuel inlet
engine.connect(fuel_pump.port("outlet"),       regen.port("coolant_inlet"))
engine.connect(regen.port("coolant_outlet"),   fuel_preburner.port("fuel_inlet"))
engine.connect(lox_pump.port("bleed_fuel"),    fuel_preburner.port("lox_inlet"))  # LOX bleed to fuel PB
engine.connect(fuel_preburner.port("outlet"),  fuel_turbine.port("inlet"))
engine.connect(fuel_turbine.port("outlet"),    chamber.port("fuel_inlet"))

# Chamber → nozzle
engine.connect(chamber.port("outlet"),         nozzle.port("inlet"))

# Independent shaft connections
engine.connect(lox_turbine.port("shaft"),  lox_shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"),     lox_shaft.port("pump"))
engine.connect(fuel_turbine.port("shaft"), fuel_shaft.port("turbine_in"))
engine.connect(fuel_pump.port("shaft"),    fuel_shaft.port("pump"))

layout = engine.compile()
X0 = layout.assemble_state_vector()

# ---------------------------------------------------------------------------
# Steady-state trim balance
# ---------------------------------------------------------------------------
bcs_ss = {
    "lox_pump.inlet.P":  P_lox_tank,   "lox_pump.inlet.h":  h_lox_inlet,
    "fuel_pump.inlet.P": P_fuel_tank,  "fuel_pump.inlet.h": h_fuel_inlet,
    "nozzle.P_ambient":  0.0,
    "gas.T": 3500.0,    # K   main chamber temperature estimate
    "gas.P": Pc_design, # Pa
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs_ss)
layout.scatter_state_vector(X_ss)

T_wall_ss = regen._state_values["T_wall"]
ch4_T_out = regen.last_outputs.get("T_bulk_out", float("nan"))
Q_cool    = regen.last_outputs.get("Q_cool",     float("nan"))
delta_P_r = regen.last_outputs.get("delta_P",    float("nan"))

print(f"\nFFSC LOX/CH4 at 10 MPa — Steady-State Balance:")
print(f"  Chamber pressure      : {chamber._state_values['P']/1e6:.3f} MPa  (design {Pc_design/1e6:.1f})")
print(f"  Ox preburner temp     : {ox_preburner._state_values['T']:.0f} K")
print(f"  Fuel preburner temp   : {fuel_preburner._state_values['T']:.0f} K")
print(f"  LOX shaft speed       : {lox_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  CH4 shaft speed       : {fuel_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust         : {nozzle.last_outputs['thrust']:.0f} N   (design {F_design:.0f})")
print(f"  Vacuum Isp            : {nozzle.last_outputs['Isp_vacuum']:.1f} s")
print(f"\n  Regen channel (steady state):")
print(f"    Wall temperature    : {T_wall_ss:.0f} K")
print(f"    CH4 outlet temp     : {ch4_T_out:.1f} K  (inlet {T_fuel:.0f} K)")
print(f"    Heat absorbed       : {Q_cool/1000:.2f} kW")
print(f"    Channel dP          : {delta_P_r/1e5:.3f} bar")

# ---------------------------------------------------------------------------
# Start transient: engine ignition and spinup
#   t = 0.0 s  ignition — both preburners come online at low flow
#   t = 0.3 s  shafts accelerate, main valves begin to open
#   t = 2.0 s  full-flow mainstage
# Both shafts spool independently, demonstrating dual-shaft FFSC dynamics.
# The regen wall heats from cold (300 K) to steady state during spinup.
# ---------------------------------------------------------------------------
def gg_ramp(t):
    return min(1.0, t / 0.3)

def main_valve(t):
    if t < 1.2:
        return 0.0
    return min(1.0, (t - 1.2) / 0.8)

def make_bcs(t):
    return {
        "lox_pump.inlet.P":  P_lox_tank,   "lox_pump.inlet.h":  h_lox_inlet,
        "fuel_pump.inlet.P": P_fuel_tank,  "fuel_pump.inlet.h": h_fuel_inlet,
        "nozzle.P_ambient":  0.0,
        "gas.T": min(chamber._state_values.get("T", 3500.0), 4000.0),
        "gas.P": min(chamber._state_values.get("P", Pc_design), 14e6),
    }

profile = TestProfile(
    name="ffsc_ch4_start",
    phases=[
        PhaseDefinition(
            name="spinup",
            mode=PhaseMode.TRANSIENT,
            duration=4.0,
            control_commands=[
                ControlCommand("lox_pump.inlet.mdot",
                               fn=lambda t: mdot_lox  * (0.04 * gg_ramp(t) + 0.96 * main_valve(t))),
                ControlCommand("fuel_pump.inlet.mdot",
                               fn=lambda t: mdot_fuel * (0.05 * gg_ramp(t) + 0.95 * main_valve(t))),
                ControlCommand("lox_pump.inlet.h",   fn=lambda t: h_lox_inlet),
                ControlCommand("fuel_pump.inlet.h",  fn=lambda t: h_fuel_inlet),
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
        SafetyLimit("Pc_max",       "chamber",       "P",      upper_limit=13.0e6,         is_hard=True),
        SafetyLimit("T_ox_pb_max",  "ox_preburner",  "T",      upper_limit=900.0,          is_hard=True),
        SafetyLimit("T_fl_pb_max",  "fuel_preburner","T",      upper_limit=900.0,          is_hard=True),
        SafetyLimit("lox_rpm_max",  "lox_shaft",     "omega",  upper_limit=40000*np.pi/30, is_hard=True),
        SafetyLimit("fuel_rpm_max", "fuel_shaft",    "omega",  upper_limit=34000*np.pi/30, is_hard=True),
        SafetyLimit("T_wall_max",   "regen",         "T_wall", upper_limit=1200.0,         is_hard=True),
    ],
)

result = profile.execute(layout, X0, bcs_fn=make_bcs)

if result.success:
    spinup   = result.get_phase("spinup")
    lox_rpm  = spinup.get("lox_shaft",  "omega").max() * 30 / np.pi
    fuel_rpm = spinup.get("fuel_shaft", "omega").max() * 30 / np.pi
    T_wall   = spinup.get("regen",      "T_wall")
    print(f"\nStart transient:")
    print(f"  LOX shaft peak    : {lox_rpm:.0f} rpm")
    print(f"  CH4 shaft peak    : {fuel_rpm:.0f} rpm")
    print(f"  T_wall at end     : {T_wall[-1]:.0f} K  (started at {T_wall[0]:.0f} K)")
    result.plot_timeline()
else:
    print(f"\nAbort: {result.abort_reason} at t={result.abort_time:.3f} s")
