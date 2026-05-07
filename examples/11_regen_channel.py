"""
Regenerative Cooling Channel — LOX / Liquid Methane GG Engine
=============================================================
Extends the GG LOX/CH4 engine from example 10 by routing the methane
propellant through a regenerative cooling jacket around the combustion
chamber and nozzle before it reaches the injector.

Regen channel roles in this engine:
  - Cools the chamber/nozzle wall  (thermal management)
  - Pre-heats the methane          (increases coolant enthalpy → higher Isp)

Engine flow path with regen:

  LCH4 tank ──► CH4 pump ──► RegenChannel ──► CH4 injector ──► Chamber ──► Nozzle
                                   ▲
                              (absorbs heat from
                               chamber/nozzle wall)

The RegenChannel has one dynamic state: T_wall [K].  In steady state
dT_wall/dt = 0 → Q_hot (Bartz) = Q_cool (Dittus-Boelter).

This example shows:
  1. Steady-state balance — wall temperature and coolant temperature rise
  2. Transient heat soak — engine starts cold, wall heats to equilibrium
  3. Sensitivity to coolant flow rate (throttle effect on wall temperature)

Regen channel geometry (representative chamber + nozzle jacket):
  Channel cross-section : 5.0 cm²  (total, all channels combined)
  Hydraulic diameter    : 3 mm
  Channel length        : 0.8 m    (chamber + nozzle contour)
  Hot-area              : 0.15 m²  (combustion-gas-side surface)
  Cool-area             : 0.18 m²  (coolant-side surface, slightly larger)
  Wall mass             : 2.5 kg   (CuCrZr alloy jacket)
  Wall Cp               : 390 J/(kg·K)  (CuCrZr at operating temperature)
  h_hot (area-avg)      :  8 000 W/(m²·K)
"""

import numpy as np
import matplotlib.pyplot as plt

from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.gas_generator import GasGenerator
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
from atha.components.regen_channel import RegenChannel
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.solver.steady_state import SteadyStateSolver
from atha.solver.transient import TransientSolver
from atha.profiles import (
    TestProfile, PhaseDefinition, PhaseMode, ControlCommand, SafetyLimit,
)

# ---------------------------------------------------------------------------
# Propellants
# ---------------------------------------------------------------------------
lox     = CoolPropBackend("Oxygen")
methane = CoolPropBackend("Methane")

P_lox_tank  = 4e5;   T_lox  = 91.0
P_fuel_tank = 4e5;   T_fuel = 108.0

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = methane.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters (same as example 10)
# ---------------------------------------------------------------------------
Pc_design = 10.0e6
MR_main   = 3.5
F_design  = 20000.0
At        = 1.03e-3
Ae_At     = 50.0
Isp_est   = 348.0

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

GG_BLEED   = 0.07
MR_gg      = 0.35
P_pump_out = Pc_design * 1.55

lox_map_base  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  30000, 0.70)
fuel_map_base = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 25000, 0.67)
turb_map      = TurbineMap.from_design_point(PR_design=5.5, eta_design=0.72,
                                              mdot_corrected_design=0.025)

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------
engine = Engine("gg_lox_ch4_regen")

shaft = Rotor("shaft", moment_of_inertia=0.10, initial_speed_rpm=28000)

lox_pump  = Pump("lox_pump",  diameter=0.085, pump_map=lox_map_base, fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.055, pump_map=fuel_map_base, fluid=methane)

gg_thermo = CanteraBackend("gri30.yaml")
gg = GasGenerator(
    "gg",
    volume=1e-4,
    thermo=gg_thermo,
    fuel="CH4", oxidizer="O2",
    design_MR=MR_gg,
    efficiency=0.93,
    initial_P=P_pump_out,
    initial_T=1050.0,
)

turbine = Turbine("turbine", diameter=0.10, turbine_map=turb_map)

# Regenerative cooling channel around chamber + nozzle
# Methane flows through this before reaching the injector
regen = RegenChannel(
    "regen",
    fluid=methane,
    channel_area=5e-5,         # m²   total channel cross-section
    hydraulic_diam=3e-3,       # m    hydraulic diameter
    channel_length=0.8,        # m    total coolant path
    hot_area=0.15,             # m²   hot-gas-side area
    cool_area=0.18,            # m²   coolant-side area
    wall_mass=2.5,             # kg   CuCrZr alloy jacket
    wall_cp=390.0,             # J/(kg·K)
    h_hot_design=8000.0,       # W/(m²·K)  area-averaged Bartz over 0.15 m² contour
    Pc_design=Pc_design,
    recovery_factor=0.90,
    initial_T_wall=300.0,      # K    engine starts cold
)

cc_thermo = CanteraBackend("gri30.yaml")
chamber = CombustionChamber(
    "chamber",
    volume=3e-4,
    thermo=cc_thermo,
    fuel="CH4", oxidizer="O2",
    efficiency=0.97,
    initial_P=Pc_design,
    initial_T=3500.0,
)

nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

lox_inj  = OrificeCompressible("lox_inj",  Cd=0.72, area=1.2e-4)
fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=0.9e-4)

for comp in [shaft, lox_pump, fuel_pump, gg, turbine,
             regen, chamber, nozzle, lox_inj, fuel_inj]:
    engine.add_component(comp)

# GG bleed
engine.connect(lox_pump.port("bleed"),  gg.port("lox_inlet"))
engine.connect(fuel_pump.port("bleed"), gg.port("fuel_inlet"))
engine.connect(gg.port("outlet"),       turbine.port("inlet"))

# Main propellant — CH4 passes through regen before injector
engine.connect(lox_pump.port("outlet"),        lox_inj.port("inlet"))
engine.connect(fuel_pump.port("outlet"),       regen.port("coolant_inlet"))
engine.connect(regen.port("coolant_outlet"),   fuel_inj.port("inlet"))
engine.connect(lox_inj.port("outlet"),         chamber.port("lox_inlet"))
engine.connect(fuel_inj.port("outlet"),        chamber.port("fuel_inlet"))
engine.connect(chamber.port("outlet"),         nozzle.port("inlet"))

# Shaft
engine.connect(turbine.port("shaft"),   shaft.port("turbine_in"))
engine.connect(lox_pump.port("shaft"),  shaft.port("pump_lox"))
engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

layout = engine.compile()
X0 = layout.assemble_state_vector()

# ---------------------------------------------------------------------------
# Steady-state trim balance
# Hot gas conditions fed from chamber state via BCS
# ---------------------------------------------------------------------------
bcs_ss = {
    "lox_pump.inlet.P":  P_lox_tank,   "lox_pump.inlet.h":  h_lox_inlet,
    "fuel_pump.inlet.P": P_fuel_tank,  "fuel_pump.inlet.h": h_fuel_inlet,
    "nozzle.P_ambient":  0.0,
    # RegenChannel reads hot gas conditions from these BCS keys
    "gas.T": 3500.0,    # K   initial estimate; solver will update
    "gas.P": Pc_design, # Pa
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs_ss)
layout.scatter_state_vector(X_ss)

T_wall_ss = regen._state_values["T_wall"]
ch4_T_out = regen.last_outputs.get("T_bulk_out", float("nan"))
Q_cool    = regen.last_outputs.get("Q_cool",     float("nan"))
delta_P_r = regen.last_outputs.get("delta_P",    float("nan"))

print(f"\nGG LOX/CH4 10 MPa — Steady-State with Regen:")
print(f"  Chamber pressure      : {chamber._state_values['P']/1e6:.3f} MPa")
print(f"  Shaft speed           : {shaft._state_values['omega']*30/np.pi:.0f} rpm")
print(f"  Vacuum thrust         : {nozzle.last_outputs['thrust']:.0f} N")
print(f"  Vacuum Isp            : {nozzle.last_outputs['Isp_vacuum']:.1f} s")
print(f"\n  Regen channel (steady state):")
print(f"    Wall temperature    : {T_wall_ss:.0f} K")
print(f"    CH4 outlet temp     : {ch4_T_out:.1f} K  (inlet {T_fuel:.0f} K)")
print(f"    Heat absorbed       : {Q_cool/1000:.2f} kW")
print(f"    Channel dP          : {delta_P_r/1e5:.3f} bar")

# ---------------------------------------------------------------------------
# Transient: heat soak from cold start
# Wall heats from 300 K to steady-state temperature over ~10–30 s
# ---------------------------------------------------------------------------
def make_bcs(t):
    return {
        "lox_pump.inlet.P":  P_lox_tank,   "lox_pump.inlet.h":  h_lox_inlet,
        "fuel_pump.inlet.P": P_fuel_tank,  "fuel_pump.inlet.h": h_fuel_inlet,
        "nozzle.P_ambient":  0.0,
        # Use current chamber state for hot-gas inputs to regen (clamped to physical range)
        "gas.T": min(chamber._state_values.get("T", 3500.0), 4000.0),
        "gas.P": min(chamber._state_values.get("P", Pc_design), 14e6),
    }

profile = TestProfile(
    name="regen_heatsoak",
    phases=[
        PhaseDefinition(
            name="mainstage",
            mode=PhaseMode.TRANSIENT,
            duration=30.0,
            control_commands=[
                ControlCommand("lox_pump.inlet.mdot",  fn=lambda t: mdot_lox),
                ControlCommand("fuel_pump.inlet.mdot", fn=lambda t: mdot_fuel),
                ControlCommand("lox_pump.inlet.h",     fn=lambda t: h_lox_inlet),
                ControlCommand("fuel_pump.inlet.h",    fn=lambda t: h_fuel_inlet),
            ],
            recording_rate_hz=50.0,
        ),
    ],
    global_limits=[
        SafetyLimit("T_wall_max", "regen",   "T_wall", upper_limit=1200.0,  is_hard=True),
        SafetyLimit("Pc_max",     "chamber", "P",      upper_limit=13.0e6,  is_hard=True),
    ],
)

# Start from X0 (cold wall) so we see the heat soak transient
result = profile.execute(layout, X0, bcs_fn=make_bcs)

if result.success:
    phase    = result.get_phase("mainstage")
    t        = phase.t
    T_wall   = phase.get("regen",   "T_wall")
    Pc_trace = phase.get("chamber", "P")

    print(f"\nHeat soak transient (cold start → {t[-1]:.0f} s):")
    print(f"  T_wall:  {T_wall[0]:.0f} K → {T_wall[-1]:.0f} K  (SS target ≈ {T_wall_ss:.0f} K)")
    print(f"  Pc:      {Pc_trace.mean()/1e6:.3f} MPa  (mean over transient)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(t, T_wall, "r-", label="T_wall")
    ax1.axhline(T_wall_ss, color="r", linestyle="--", alpha=0.5, label="SS target")
    ax1.set(ylabel="Wall temperature [K]", title="Regen Channel — Cold Start Heat Soak")
    ax1.legend()

    ax2.plot(t, Pc_trace / 1e6, "b-")
    ax2.set(xlabel="Time [s]", ylabel="Chamber pressure [MPa]")

    plt.tight_layout()
    plt.savefig("outputs/11_regen_heatsoak.png", dpi=150)
    plt.show()
else:
    print(f"\nAbort: {result.abort_reason} at t={result.abort_time:.3f} s")

# ---------------------------------------------------------------------------
# Sensitivity: wall temperature vs coolant flow (throttle effect)
# As thrust is throttled down, less coolant flows → wall runs hotter
# ---------------------------------------------------------------------------
print("\nWall temperature vs throttle level (steady state):")
print(f"  {'Throttle':>10}  {'ṁ_CH4 kg/s':>12}  {'T_wall K':>10}  {'Q_cool kW':>10}")

throttle_levels = [1.0, 0.85, 0.70, 0.55, 0.40]
for thr in throttle_levels:
    bcs_thr = dict(bcs_ss)
    bcs_thr["lox_pump.inlet.mdot"]  = mdot_lox  * thr
    bcs_thr["fuel_pump.inlet.mdot"] = mdot_fuel * thr
    bcs_thr["gas.P"] = Pc_design * thr          # rough Pc scaling
    try:
        X_thr = solver.solve(X0, bcs_thr)
        layout.scatter_state_vector(X_thr)
        T_w = regen._state_values["T_wall"]
        Qc  = regen.last_outputs.get("Q_cool", float("nan"))
        print(f"  {thr*100:>9.0f}%  {mdot_fuel*thr:>12.3f}  {T_w:>10.0f}  {Qc/1000:>10.2f}")
    except Exception as e:
        print(f"  {thr*100:>9.0f}%  solver did not converge: {e}")
