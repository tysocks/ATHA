"""
Full-Flow Staged Combustion (FFSC) Cycle
=========================================
Engine topology:

  LOX tank ──► LOX pump ──► Ox-rich preburner (MR ~50) ──► LOX turbine ──► Main chamber
                                        ▲                                        ▲
  Fuel tank ──► Fuel pump ──► Fuel-rich preburner (MR ~0.3) ──► Fuel turbine ──┘
                                        │
                                (all propellant goes through preburners — no bypass)

Both preburners run at extreme mixture ratios to limit temperature while using
all of the propellant as turbine working fluid. The preburner exhausts both
enter the main combustion chamber where final combustion occurs.

This is the most thermodynamically efficient liquid cycle (Raptor-like).
Two separate turbopump shafts: one for LOX side, one for fuel side.

Reference engine parameters:
  Pc = 2.0 MPa, MR_main = 2.8 (LOX/ethanol), At = 1.96e-3 m²

NOTE: Requires planned components — Pump, Turbine, Rotor, Preburner,
      CombustionChamber, Nozzle, CoolPropBackend, CanteraBackend.
"""

import numpy as np
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.preburner import Preburner
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.solver.steady_state import SteadyStateSolver

# ---------------------------------------------------------------------------
# Propellants
# ---------------------------------------------------------------------------
lox     = CoolPropBackend("Oxygen")
ethanol = CoolPropBackend("Ethanol")

P_lox_tank  = 4e5;   T_lox  = 91.0    # K  subcooled
P_fuel_tank = 4e5;   T_fuel = 293.0   # K  ambient

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = ethanol.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design  = 2.0e6    # Pa
MR_main    = 2.8
At         = 1.963e-3  # m²  50 mm throat
Ae_At      = 8.0
F_design   = 5500.0   # N  vacuum
Isp_est    = 310.0    # s  estimated vacuum Isp (slightly higher than ORSC)

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

# Preburner mixture ratios
MR_ox_pb   = 50.0    # ox-rich:  nearly pure LOX with trace fuel, T ~700 K
MR_fuel_pb = 0.30    # fuel-rich: nearly pure fuel with trace LOX, T ~850 K

# Pump outlet pressure (FFSC needs higher pump pressures)
P_pump_out = Pc_design * 1.7   # Pa — extra head for dual-preburner routing

# ---------------------------------------------------------------------------
# Performance maps
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  28000, 0.72)
fuel_map = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 28000, 0.67)

turb_ox_map   = TurbineMap.from_design_point(PR_design=2.0, eta_design=0.73, mdot_corrected_design=0.06)
turb_fuel_map = TurbineMap.from_design_point(PR_design=1.9, eta_design=0.71, mdot_corrected_design=0.04)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("ffsc_dev1")

# Two independent shafts — LOX side and fuel side
lox_shaft  = Rotor("lox_shaft",  moment_of_inertia=0.08, initial_speed_rpm=28000)
fuel_shaft = Rotor("fuel_shaft", moment_of_inertia=0.06, initial_speed_rpm=28000)

lox_pump  = Pump("lox_pump",  diameter=0.09, pump_map=lox_map,  fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.07, pump_map=fuel_map, fluid=ethanol)

ox_thermo   = CanteraBackend("gri30.yaml")
fuel_thermo = CanteraBackend("gri30.yaml")
cc_thermo   = CanteraBackend("gri30.yaml")

# Ox-rich preburner: all LOX + small fuel bleed
ox_preburner = Preburner(
    "ox_preburner",
    volume=2e-4,
    thermo=ox_thermo,
    fuel="C2H5OH", oxidizer="O2",
    design_MR=MR_ox_pb,
    efficiency=0.93,
    initial_P=P_pump_out,
    initial_T=720.0,
)

# Fuel-rich preburner: all fuel + small LOX bleed
fuel_preburner = Preburner(
    "fuel_preburner",
    volume=2e-4,
    thermo=fuel_thermo,
    fuel="C2H5OH", oxidizer="O2",
    design_MR=MR_fuel_pb,
    efficiency=0.93,
    initial_P=P_pump_out,
    initial_T=850.0,
)

lox_turbine  = Turbine("lox_turbine",  diameter=0.11, turbine_map=turb_ox_map)
fuel_turbine = Turbine("fuel_turbine", diameter=0.10, turbine_map=turb_fuel_map)

chamber = CombustionChamber(
    "chamber", volume=5e-4,
    thermo=cc_thermo,
    fuel="C2H5OH", oxidizer="O2",
    efficiency=0.97,
    initial_P=Pc_design, initial_T=3500.0,
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.983)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

for comp in [lox_shaft, fuel_shaft, lox_pump, fuel_pump,
             ox_preburner, fuel_preburner,
             lox_turbine, fuel_turbine,
             chamber, nozzle]:
    engine.add_component(comp)

# LOX side: pump → ox-rich preburner → LOX turbine → chamber lox inlet
engine.connect(lox_pump.port("outlet"),        ox_preburner.port("lox_inlet"))
engine.connect(fuel_pump.port("bleed_ox"),     ox_preburner.port("fuel_inlet"))   # small fuel bleed
engine.connect(ox_preburner.port("outlet"),    lox_turbine.port("inlet"))
engine.connect(lox_turbine.port("outlet"),     chamber.port("lox_inlet"))

# Fuel side: pump → fuel-rich preburner → fuel turbine → chamber fuel inlet
engine.connect(fuel_pump.port("outlet"),       fuel_preburner.port("fuel_inlet"))
engine.connect(lox_pump.port("bleed_fuel"),    fuel_preburner.port("lox_inlet"))  # small LOX bleed
engine.connect(fuel_preburner.port("outlet"),  fuel_turbine.port("inlet"))
engine.connect(fuel_turbine.port("outlet"),    chamber.port("fuel_inlet"))

# Chamber → nozzle
engine.connect(chamber.port("outlet"), nozzle.port("inlet"))

# Shaft connections
engine.connect(lox_turbine.port("shaft"),  lox_shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"),     lox_shaft.port("pump"))
engine.connect(fuel_turbine.port("shaft"), fuel_shaft.port("turbine_in"))
engine.connect(fuel_pump.port("shaft"),    fuel_shaft.port("pump"))

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
    "nozzle.P_ambient":  0.0,
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs)
layout.scatter_state_vector(X_ss)

print(f"\nFFSC Steady-State Balance:")
print(f"  Chamber pressure   : {chamber._state_values['P']/1e6:.3f} MPa")
print(f"  Ox preburner temp  : {ox_preburner._state_values['T']:.0f} K")
print(f"  Fuel preburner temp: {fuel_preburner._state_values['T']:.0f} K")
print(f"  LOX shaft speed    : {lox_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Fuel shaft speed   : {fuel_shaft._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust      : {nozzle.last_outputs['thrust']:.0f} N")
print(f"  Vacuum Isp         : {nozzle.last_outputs['Isp_vacuum']:.1f} s")
