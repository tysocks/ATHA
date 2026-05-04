"""
Oxidizer-Rich Staged Combustion (ORSC) Cycle
=============================================
Engine topology:
  LOX tank ──► LOX pump ──┬──► Preburner (ox-rich, MR ~50) ──► Turbine ──► Main chamber
                           │                                                     ▲
  Fuel tank ──► Fuel pump ─┴─(small fuel bleed to preburner)───────────────────┘
                           │
                           └──► (remaining fuel to main injector)

All LOX passes through the preburner. A small fuel fraction drives combustion
at high MR (~50) producing cool (~700 K) oxidizer-rich hot gas that drives the
turbine. Turbine exhaust combines with main fuel supply in the combustion
chamber. Both pumps share a single turbine shaft (common ORSC architecture).

Reference engine parameters (small development scale, ~5 kN thrust):
  Pc = 2.0 MPa, MR_main = 2.8 (LOX/ethanol), At = 1.96e-3 m²

NOTE: Requires planned components — Pump, Turbine, Rotor, Preburner,
      CombustionChamber, Nozzle, OrificeCompressible, CoolPropBackend.
"""

import numpy as np
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.preburner import Preburner
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
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

# Tank inlet conditions
P_lox_tank  = 4e5      # Pa — tank ullage + head
T_lox       = 91.0     # K  — subcooled LOX
P_fuel_tank = 4e5      # Pa
T_fuel      = 293.0    # K  — ambient ethanol

h_lox_inlet  = lox.state_from_PT(P_lox_tank, T_lox).h
h_fuel_inlet = ethanol.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design     = 2.0e6   # Pa    — main chamber pressure
MR_main       = 2.8     # —     — LOX/ethanol mass mixture ratio
F_design      = 5500.0  # N     — design thrust (vacuum)
At            = 1.963e-3 # m²  — throat area (50 mm diameter)
Ae_At         = 8.0     # —     — nozzle expansion ratio

# Propellant split
mdot_total    = F_design / (300.0 * 9.80665)   # kg/s — Isp_vac ~300s estimated
mdot_lox      = mdot_total * MR_main / (1 + MR_main)
mdot_fuel     = mdot_total / (1 + MR_main)

# Preburner bleed: 3.5% of fuel bleed feeds preburner at MR_pb ~50
MR_preburner  = 50.0
mdot_pb_fuel  = 0.035 * mdot_fuel
mdot_pb_lox   = mdot_pb_fuel * MR_preburner    # all LOX goes through PB

# Pump outlet pressure: Pc + injector ΔP + preburner ΔP + lines
P_pump_out    = Pc_design * 1.6   # ×1.6 for injector drop + margin

# ---------------------------------------------------------------------------
# Performance maps (design-point based — replace with measured maps)
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_design=mdot_lox,  dP_design=P_pump_out - P_lox_tank,  speed_design=25000, efficiency_design=0.70)
fuel_map = PumpMap.from_design_point(mdot_design=mdot_fuel, dP_design=P_pump_out - P_fuel_tank, speed_design=25000, efficiency_design=0.65)
turb_map = TurbineMap.from_design_point(PR_design=2.2, eta_design=0.72, mdot_corrected_design=0.08)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("orsc_dev1")

# Single turbopump shaft (LOX pump + fuel pump + turbine)
shaft = Rotor("shaft", moment_of_inertia=0.12, initial_speed_rpm=25000)

lox_pump  = Pump("lox_pump",  diameter=0.09, pump_map=lox_map,  fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.07, pump_map=fuel_map, fluid=ethanol)

# Ox-rich preburner: all LOX + small fuel bleed
# CanteraBackend provides equilibrium products as working fluid for turbine
pb_thermo = CanteraBackend("gri30.yaml")
preburner = Preburner(
    "preburner",
    volume=2e-4,             # m³  — small volume, fast dynamics
    thermo=pb_thermo,
    fuel="C2H5OH",           # ethanol
    oxidizer="O2",
    design_MR=MR_preburner,
    efficiency=0.94,
    initial_P=P_pump_out,
    initial_T=720.0,         # K  — ox-rich flame temp ~700–750 K
)

turbine = Turbine("turbine", diameter=0.12, turbine_map=turb_map)

# Main combustion chamber
cc_thermo = CanteraBackend("gri30.yaml")
chamber = CombustionChamber(
    "chamber",
    volume=5e-4,             # m³
    thermo=cc_thermo,
    fuel="C2H5OH",
    oxidizer="O2",
    efficiency=0.96,
    initial_P=Pc_design,
    initial_T=3400.0,
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.983)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

# Main injector orifices (combine turbine exhaust LOX + main fuel supply)
lox_injector  = OrificeCompressible("lox_inj",  Cd=0.72, area=1.5e-4)
fuel_injector = OrificeCompressible("fuel_inj", Cd=0.72, area=1.2e-4)

# Register all components
for comp in [shaft, lox_pump, fuel_pump, preburner, turbine,
             chamber, nozzle, lox_injector, fuel_injector]:
    engine.add_component(comp)

# Wire up ports
engine.connect(lox_pump.port("outlet"),   preburner.port("lox_inlet"))
engine.connect(fuel_pump.port("outlet"),  preburner.port("fuel_inlet"))   # bleed only
engine.connect(preburner.port("outlet"),  turbine.port("inlet"))
engine.connect(turbine.port("outlet"),    lox_injector.port("inlet"))     # hot ox-rich gas → chamber
engine.connect(fuel_pump.port("outlet2"), fuel_injector.port("inlet"))    # main fuel flow
engine.connect(lox_injector.port("outlet"),  chamber.port("lox_inlet"))
engine.connect(fuel_injector.port("outlet"), chamber.port("fuel_inlet"))
engine.connect(chamber.port("outlet"),    nozzle.port("inlet"))

# Shaft connections
engine.connect(turbine.port("shaft"),  shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"), shaft.port("pump_lox"))
engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

layout = engine.compile()
X0 = layout.assemble_state_vector()

# ---------------------------------------------------------------------------
# Steady-state trim balance
# ---------------------------------------------------------------------------
bcs = {
    "lox_pump.inlet.P":  P_lox_tank,
    "lox_pump.inlet.h":  h_lox_inlet,
    "fuel_pump.inlet.P": P_fuel_tank,
    "fuel_pump.inlet.h": h_fuel_inlet,
    "nozzle.P_ambient":  0.0,            # vacuum
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs)
layout.scatter_state_vector(X_ss)

Pc_ss    = chamber._state_values["P"]
T_pb_ss  = preburner._state_values["T"]
omega_ss = shaft._state_values["omega"]
thrust   = nozzle.last_outputs["thrust"]
Isp      = nozzle.last_outputs["Isp_vacuum"]

print(f"\nORSC Steady-State Balance:")
print(f"  Chamber pressure   : {Pc_ss/1e6:.3f} MPa   (design {Pc_design/1e6:.3f})")
print(f"  Preburner temp     : {T_pb_ss:.0f} K")
print(f"  Shaft speed        : {omega_ss * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust      : {thrust:.0f} N    (design {F_design:.0f})")
print(f"  Vacuum Isp         : {Isp:.1f} s")

# ---------------------------------------------------------------------------
# Throttle transient: 100% → 65% → 100% RPL
# ---------------------------------------------------------------------------
h_ref  = ethanol.state_from_PT(P_fuel_tank, T_fuel).h

def throttle_schedule(t):
    """Return RPL fraction 0–1 as a function of time."""
    if   t < 2.0:  return 1.0
    elif t < 3.0:  return 1.0 - 0.35 * (t - 2.0)   # ramp down to 65% over 1 s
    elif t < 8.0:  return 0.65
    elif t < 9.0:  return 0.65 + 0.35 * (t - 8.0)  # ramp back to 100%
    else:           return 1.0

profile = TestProfile(
    name="orsc_throttle_sweep",
    phases=[
        PhaseDefinition(
            name="mainstage_trim",
            mode=PhaseMode.STEADY_TRIM,
            duration=5.0,
        ),
        PhaseDefinition(
            name="throttle_transient",
            mode=PhaseMode.TRANSIENT,
            duration=12.0,
            control_commands=[
                ControlCommand("lox_pump.inlet.mdot",
                               fn=lambda t: mdot_lox  * throttle_schedule(t)),
                ControlCommand("fuel_pump.inlet.mdot",
                               fn=lambda t: mdot_fuel * throttle_schedule(t)),
            ],
            recording_rate_hz=200.0,
        ),
    ],
    global_limits=[
        SafetyLimit("Pc_max",   "chamber",   "P",      upper_limit=2.6e6, is_hard=True),
        SafetyLimit("T_pb_max", "preburner", "T",      upper_limit=900.0, is_hard=True),
        SafetyLimit("omega_max","shaft",     "omega",  upper_limit=32000 * np.pi/30, is_hard=True),
    ],
)

result = profile.execute(layout, X_ss)

if result.success:
    trans = result.get_phase("throttle_transient")
    Pc    = trans.get("chamber", "P")
    rpm   = trans.get("shaft",   "omega") * 30 / np.pi
    print(f"\nThrottle transient: min Pc = {Pc.min()/1e6:.3f} MPa  "
          f"min RPM = {rpm.min():.0f}  settled = {result.success}")
    result.plot_timeline()
else:
    print(f"\nAbort: {result.abort_reason} at t={result.abort_time:.3f} s")
