"""
Oxidizer-Rich Staged Combustion (ORSC) — LOX / Liquid Methane at 10 MPa
========================================================================
Engine topology:

  LOX tank ──► LOX pump ──► Ox-rich preburner (MR ~50) ──► Turbine ──► main chamber
                                      ▲                                      ▲
  LCH4 tank ──► CH4 pump ──(small CH4 bleed)                                │
                            │                                                 │
                            └──► RegenChannel ──► (main CH4 flow) ──────────┘

All LOX passes through the ox-rich preburner. A small methane bleed burns at
MR ~50 producing cool (~730 K) ox-rich working fluid that drives the turbine.
Turbine exhaust mixes with the main methane stream in the combustion chamber.
Both pumps share a single turbopump shaft.

The main CH4 flow passes through a regenerative cooling channel around the
chamber/nozzle before reaching the fuel injector, absorbing heat from the wall
and pre-heating the propellant.

Engine parameters (development-scale, 20 kN vacuum):
  Pc = 10.0 MPa, MR_main = 3.5 (LOX/LCH4), At = 1.03e-3 m², Ae/At = 50

The high chamber pressure relative to the ethanol examples (02_orsc_cycle.py)
requires ~1.8× pump outlet pressure and a higher-energy preburner circuit.
"""

import matplotlib; matplotlib.use("Agg")
import numpy as np
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.preburner import Preburner
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
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
P_fuel_tank = 4e5;   T_fuel = 108.0   # K  subcooled LCH4 (NBP at 1 bar = 111.7 K)

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = methane.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design = 10.0e6   # Pa
MR_main   = 3.5      # LOX / LCH4  (stoichiometric ≈ 4.0)
F_design  = 20000.0  # N  vacuum
Isp_est   = 357.0    # s  estimated vacuum Isp for ORSC LOX/LCH4
At        = 1.03e-3  # m²  ~36 mm throat
Ae_At     = 50.0     # vacuum nozzle

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

# Preburner: all LOX + small CH4 bleed at MR ~50 → T_pb ~730 K
MR_pb         = 50.0
mdot_pb_fuel  = mdot_lox / MR_pb        # small CH4 fraction to preburner
mdot_main_fuel = mdot_fuel - mdot_pb_fuel  # remainder to main chamber

# Pump outlet pressure: Pc × 1.8 covers injector ΔP, preburner ΔP, line losses
P_pump_out = Pc_design * 1.8

# ---------------------------------------------------------------------------
# Performance maps (design-point based)
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_lox,        P_pump_out - P_lox_tank,  28000, 0.72)
fuel_map = PumpMap.from_design_point(mdot_fuel,        P_pump_out - P_fuel_tank, 28000, 0.68)
turb_map = TurbineMap.from_design_point(PR_design=1.5, eta_design=0.73,
                                        mdot_corrected_design=0.06)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("orsc_lox_ch4")

shaft = Rotor("shaft", moment_of_inertia=0.10, initial_speed_rpm=28000)

lox_pump  = Pump("lox_pump",  diameter=0.085, pump_map=lox_map,  fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.060, pump_map=fuel_map, fluid=methane)

pb_thermo = CanteraBackend("gri30.yaml", initial_X="O2:0.885,H2O:0.077,CO2:0.038")
preburner = Preburner(
    "preburner",
    volume=1.5e-4,
    thermo=pb_thermo,
    fuel="CH4",
    oxidizer="O2",
    design_MR=MR_pb,
    efficiency=0.94,
    initial_P=P_pump_out,
    initial_T=730.0,         # K  ox-rich LOX/CH4 at MR=50
)

turbine = Turbine("turbine", diameter=0.10, turbine_map=turb_map)

# Regenerative cooling channel — main CH4 flow cools chamber/nozzle wall
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

cc_thermo = CanteraBackend("gri30.yaml",
                           initial_X="H2O:0.60,CO2:0.25,CO:0.08,H2:0.07")
chamber = CombustionChamber(
    "chamber",
    volume=3e-4,
    thermo=cc_thermo,
    fuel="CH4",
    oxidizer="O2",
    efficiency=0.97,
    initial_P=Pc_design,
    initial_T=3500.0,         # K  LOX/CH4 adiabatic flame at MR=3.5
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

# Injector orifices
# lox_inj handles turbine exhaust (ox-rich gas ~730 K, mostly O2): R=265 J/kg·K, gamma=1.4
# Area sized for ~4.4 kg/s at 12 MPa turbine-out, 10 MPa chamber
lox_inj  = OrificeCompressible("lox_inj",  Cd=0.72, area=3.0e-4, gamma=1.4, R_gas=265.0)
# fuel_inj handles supercritical methane after regen: R=518 J/kg·K, gamma=1.3
fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=0.9e-4, gamma=1.3, R_gas=518.3)

for comp in [shaft, lox_pump, fuel_pump, preburner, turbine,
             regen, chamber, nozzle, lox_inj, fuel_inj]:
    engine.add_component(comp)

# All LOX → preburner; small CH4 bleed → preburner
engine.connect(lox_pump.port("outlet"),     preburner.port("lox_inlet"))
engine.connect(fuel_pump.port("bleed"),     preburner.port("fuel_inlet"))
engine.connect(preburner.port("outlet"),    turbine.port("inlet"))
# Turbine exhaust (ox-rich hot gas) → chamber LOX port
engine.connect(turbine.port("outlet"),      lox_inj.port("inlet"))
engine.connect(lox_inj.port("outlet"),      chamber.port("lox_inlet"))
# Main CH4 flow: pump → regen → injector → chamber
engine.connect(fuel_pump.port("outlet"),    regen.port("coolant_inlet"))
engine.connect(regen.port("coolant_outlet"), fuel_inj.port("inlet"))
engine.connect(fuel_inj.port("outlet"),     chamber.port("fuel_inlet"))
engine.connect(chamber.port("outlet"),      nozzle.port("inlet"))

# Shaft
engine.connect(turbine.port("shaft"),   shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"),  shaft.port("pump_lox"))
engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

layout = engine.compile()
X0 = layout.assemble_state_vector()

# ---------------------------------------------------------------------------
# Steady-state trim balance
# Hot gas conditions fed to regen via BCS
# ---------------------------------------------------------------------------
OMEGA_DESIGN = 28000 * np.pi / 30.0  # rad/s

bcs_ss = {
    "lox_pump.inlet.P":    P_lox_tank,    "lox_pump.inlet.h":    h_lox_inlet,
    "lox_pump.inlet.mdot": mdot_lox,
    "fuel_pump.inlet.P":   P_fuel_tank,   "fuel_pump.inlet.h":   h_fuel_inlet,
    "fuel_pump.inlet.mdot": mdot_fuel,
    # Pin bleed flow so pump's bare-mdot broadcast doesn't flood the preburner.
    # Without this the pump propagates mdot_fuel (total) to BOTH regen and preburner.
    "preburner.fuel_inlet.mdot": mdot_pb_fuel,
    "nozzle.P_ambient":    0.0,
    "gas.T": 3500.0,    # K   chamber flame temperature estimate
    "gas.P": Pc_design, # Pa
    "shaft.omega_override": OMEGA_DESIGN,
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs_ss)
layout.scatter_state_vector(X_ss)

T_wall_ss = regen._state_values["T_wall"]
ch4_T_out = regen.last_outputs.get("T_bulk_out", float("nan"))
Q_cool    = regen.last_outputs.get("Q_cool",     float("nan"))
delta_P_r = regen.last_outputs.get("delta_P",    float("nan"))

print(f"\nORSC LOX/CH4 at 10 MPa — Steady-State Balance:")
print(f"  Chamber pressure   : {chamber._state_values['P']/1e6:.3f} MPa  (design {Pc_design/1e6:.1f})")
print(f"  Preburner temp     : {preburner._state_values['T']:.0f} K")
print(f"  Shaft speed        : {shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust      : {nozzle.last_outputs['thrust']:.0f} N   (design {F_design:.0f})")
print(f"  Vacuum Isp         : {nozzle.last_outputs['Isp_vacuum']:.1f} s")
print(f"\n  Regen channel (steady state):")
print(f"    Wall temperature : {T_wall_ss:.0f} K")
print(f"    CH4 outlet temp  : {ch4_T_out:.1f} K  (inlet {T_fuel:.0f} K)")
print(f"    Heat absorbed    : {Q_cool/1000:.2f} kW")
print(f"    Channel dP       : {delta_P_r/1e5:.3f} bar")

# ---------------------------------------------------------------------------
# Throttle transient: 100% → 65% → 100% RPL
# BCS function supplies live chamber conditions to the regen each step
# ---------------------------------------------------------------------------
def throttle(t):
    if   t < 2.0:  return 1.0
    elif t < 3.0:  return 1.0 - 0.35 * (t - 2.0)
    elif t < 8.0:  return 0.65
    elif t < 9.0:  return 0.65 + 0.35 * (t - 8.0)
    return 1.0

def make_bcs(t):
    # Provides defaults — ControlCommands in the transient phase override pump mdots.
    return {
        "lox_pump.inlet.P":    P_lox_tank,    "lox_pump.inlet.h":    h_lox_inlet,
        "lox_pump.inlet.mdot": mdot_lox,      # default (overridden by throttle command)
        "fuel_pump.inlet.P":   P_fuel_tank,   "fuel_pump.inlet.h":   h_fuel_inlet,
        "fuel_pump.inlet.mdot": mdot_fuel,    # default (overridden by throttle command)
        # Override bare-mdot broadcast from fuel_pump so preburner only sees its bleed fraction.
        "preburner.fuel_inlet.mdot": mdot_pb_fuel,
        "nozzle.P_ambient":    0.0,
        "gas.T": min(chamber._state_values.get("T", 3500.0), 4000.0),
        "gas.P": min(chamber._state_values.get("P", Pc_design), 14e6),
    }

_diag_prev_t = [-1.0]
def make_bcs_diag(t):
    bcs = make_bcs(t)
    if t - _diag_prev_t[0] >= 0.05:
        T_w = regen._state_values.get("T_wall", float("nan"))
        T_ch = chamber._state_values.get("T", float("nan"))
        Pc   = chamber._state_values.get("P", float("nan"))
        print(f"  [diag t={t:.4f}] T_wall={T_w:.1f}K  T_ch={T_ch:.1f}K  Pc={Pc/1e6:.3f}MPa  gas.T={bcs['gas.T']:.1f}K")
        _diag_prev_t[0] = t
    return bcs

profile = TestProfile(
    name="orsc_ch4_throttle",
    phases=[
        PhaseDefinition(
            name="trim",
            mode=PhaseMode.STEADY_TRIM,
            duration=5.0,
            trim_targets={"shaft.omega_override": OMEGA_DESIGN},
        ),
        PhaseDefinition(
            name="throttle",
            mode=PhaseMode.TRANSIENT,
            duration=12.0,
            control_commands=[
                ControlCommand("lox_pump.inlet.mdot",  fn=lambda t: mdot_lox   * throttle(t)),
                ControlCommand("fuel_pump.inlet.mdot", fn=lambda t: mdot_fuel  * throttle(t)),
                ControlCommand("preburner.fuel_inlet.mdot", fn=lambda t: mdot_pb_fuel * throttle(t)),
                ControlCommand("lox_pump.inlet.h",     fn=lambda t: h_lox_inlet),
                ControlCommand("fuel_pump.inlet.h",    fn=lambda t: h_fuel_inlet),
                ControlCommand("shaft.omega_override", fn=lambda t: OMEGA_DESIGN),
            ],
            recording_rate_hz=200.0,
        ),
    ],
    global_limits=[
        SafetyLimit("Pc_max",      "chamber",   "P",      upper_limit=13.0e6,         is_hard=True),
        SafetyLimit("T_pb_max",    "preburner", "T",      upper_limit=900.0,          is_hard=True),
        SafetyLimit("rpm_max",     "shaft",     "omega",  upper_limit=36000*np.pi/30, is_hard=True),
        SafetyLimit("T_wall_max",  "regen",     "T_wall", upper_limit=2000.0,         is_hard=True),
    ],
)

result = profile.execute(layout, X_ss, bcs_fn=make_bcs_diag)

if result.success:
    trans  = result.get_phase("throttle")
    Pc     = trans.get("chamber", "P")
    rpm    = trans.get("shaft",   "omega") * 30 / np.pi
    T_wall = trans.get("regen",   "T_wall")
    print(f"\nThrottle transient:")
    print(f"  Min Pc     = {Pc.min()/1e6:.3f} MPa  at 65% throttle")
    print(f"  Min RPM    = {rpm.min():.0f}")
    print(f"  T_wall min = {T_wall.min():.0f} K  (throttled)  max = {T_wall.max():.0f} K")
    result.plot_timeline()
else:
    print(f"\nAbort: {result.abort_reason} at t={result.abort_time:.3f} s")
