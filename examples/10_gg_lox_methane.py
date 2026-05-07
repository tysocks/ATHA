"""
Gas Generator Cycle — LOX / Liquid Methane at 10 MPa
Single Turbopump Shaft · Monte Carlo + Speed Sweep
====================================================
Engine topology:

  LOX tank ──► LOX pump ──┬──► Main LOX injector ──► Combustion chamber ──► Nozzle
                           │                                  ▲
  LCH4 tank ──► CH4 pump ─┼──► RegenChannel ──► Main CH4 injector ──────────┘
                           │
                           └──► Gas generator (fuel-rich bleed, ~7% total flow)
                                      │
                                      ▼
                                 Turbine (single shaft)
                                      │
                               (exhaust dump overboard)

The main CH4 flow passes through a regenerative cooling channel around the
chamber/nozzle before reaching the fuel injector, pre-heating the propellant.

Both pumps driven by a single turbine shaft. The GG burns a fuel-rich LCH4/LOX
mixture (~MR 0.35) producing moderate-temperature (~1050 K) working fluid.

High chamber pressure (10 MPa) demands ~7% GG bleed vs ~3.5% for low-pressure
ethanol cycles — the pump work grows with ΔP² and the turbine work must keep up.

Engine parameters:
  Pc = 10.0 MPa, MR_main = 3.5 (LOX/LCH4), At = 1.03e-3 m², Ae/At = 50
  F_design ≈ 20 kN vacuum, Isp_vac ≈ 348 s (GG cycle penalty vs staged combustion)

Three analyses:
  1. Nominal steady-state balance
  2. Monte Carlo — pump efficiency, GG efficiency, propellant temperatures
  3. Speed sweep — thrust/Pc/Isp vs turbopump speed (pump ``shaft.omega`` BCs)

Compare to:
  04_gg_single_shaft_mc_sweep.py — LOX/ethanol at 2 MPa
  08_orsc_lox_methane.py         — same propellants, ORSC topology (+25 s Isp)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
from atha.monte_carlo import MonteCarloRunner, UncertainParameter, ParameterType

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
Isp_est   = 348.0    # s  GG cycle is ~10 s lower than ORSC due to bleed loss
At        = 1.03e-3  # m²
Ae_At     = 50.0

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

# GG bleed: 7% of total flow at fuel-rich MR=0.35
GG_BLEED   = 0.07
MR_gg      = 0.35
mdot_gg    = GG_BLEED * mdot_total
mdot_gg_fuel = mdot_gg / (1 + MR_gg)
mdot_gg_lox  = mdot_gg - mdot_gg_fuel

# Pump outlet pressure
P_pump_out = Pc_design * 1.55

# ---------------------------------------------------------------------------
# Engine factory (used for MC parameter injection)
# ---------------------------------------------------------------------------
lox_map_base  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  28000, 0.70)
fuel_map_base = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 28000, 0.67)
turb_map      = TurbineMap.from_design_point(PR_design=5.5, eta_design=0.72,
                                              mdot_corrected_design=0.025)


def build_engine(
    lox_eta_scale:  float = 1.0,
    fuel_eta_scale: float = 1.0,
    gg_efficiency:  float = 0.93,
    T_lox_inlet:    float = 91.0,
    T_fuel_inlet:   float = 108.0,
):
    engine = Engine("gg_lox_ch4")

    shaft = Rotor("shaft", moment_of_inertia=0.10, initial_speed_rpm=28000)

    lox_pump  = Pump("lox_pump",  diameter=0.085,
                     pump_map=lox_map_base.scale_efficiency(lox_eta_scale),  fluid=lox)
    fuel_pump = Pump("fuel_pump", diameter=0.055,
                     pump_map=fuel_map_base.scale_efficiency(fuel_eta_scale), fluid=methane)

    gg_thermo = CanteraBackend("gri30.yaml")
    gg = GasGenerator(
        "gg",
        volume=1e-4,
        thermo=gg_thermo,
        fuel="CH4",
        oxidizer="O2",
        design_MR=MR_gg,
        efficiency=gg_efficiency,
        initial_P=P_pump_out,
        initial_T=1050.0,        # K  fuel-rich LOX/CH4 at MR=0.35
    )

    turbine = Turbine("turbine", diameter=0.10, turbine_map=turb_map)

    # Approximate CH4/O2 combustion products at MR=3.5 (slightly fuel-rich).
    # Setting initial composition avoids Cantera's H2 default which gives
    # unrealistically low density and breaks the nozzle mass-flow calculation.
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
        initial_T=3500.0,
    )

    nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
    nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At,
                    efficiencies=nozzle_eff)

    # Regenerative cooling channel — main CH4 flow cools chamber/nozzle wall
    regen = RegenChannel(
        "regen",
        fluid=methane,
        channel_area=5e-5,       # m²   total coolant cross-section
        hydraulic_diam=3e-3,     # m
        channel_length=0.7,      # m    chamber + nozzle contour (GG shorter than ORSC)
        hot_area=0.13,           # m²   gas-side area
        cool_area=0.16,          # m²   coolant-side area
        wall_mass=2.0,           # kg   CuCrZr alloy jacket
        wall_cp=390.0,           # J/(kg·K)
        h_hot_design=8000.0,     # W/(m²·K)  area-averaged Bartz over 0.13 m² contour
        Pc_design=Pc_design,
        recovery_factor=0.90,
        initial_T_wall=300.0,
    )

    lox_inj  = OrificeCompressible("lox_inj",  Cd=0.72, area=1.2e-4)
    fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=0.9e-4)

    for comp in [shaft, lox_pump, fuel_pump, gg, turbine,
                 regen, chamber, nozzle, lox_inj, fuel_inj]:
        engine.add_component(comp)

    # GG bleed from pump outlet
    engine.connect(lox_pump.port("bleed"),  gg.port("lox_inlet"))
    engine.connect(fuel_pump.port("bleed"), gg.port("fuel_inlet"))
    engine.connect(gg.port("outlet"),       turbine.port("inlet"))

    # Main propellant to chamber: CH4 passes through regen first
    engine.connect(lox_pump.port("outlet"),       lox_inj.port("inlet"))
    engine.connect(fuel_pump.port("outlet"),      regen.port("coolant_inlet"))
    engine.connect(regen.port("coolant_outlet"),  fuel_inj.port("inlet"))
    engine.connect(lox_inj.port("outlet"),        chamber.port("lox_inlet"))
    engine.connect(fuel_inj.port("outlet"),       chamber.port("fuel_inlet"))
    engine.connect(chamber.port("outlet"),        nozzle.port("inlet"))

    # Shaft
    engine.connect(turbine.port("shaft"),   shaft.port("turbine_in"))
    engine.connect(lox_pump.port("shaft"),  shaft.port("pump_lox"))
    engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

    layout = engine.compile()
    return layout, engine


# ---------------------------------------------------------------------------
# 1. Nominal steady-state
# ---------------------------------------------------------------------------
layout, eng = build_engine()
X0 = layout.assemble_state_vector()

OMEGA_DESIGN = 28000 * np.pi / 30.0  # rad/s

bcs = {
    "lox_pump.inlet.P":  P_lox_tank,   "lox_pump.inlet.h":  h_lox_inlet,
    "fuel_pump.inlet.P": P_fuel_tank,  "fuel_pump.inlet.h": h_fuel_inlet,
    "nozzle.P_ambient":  0.0,
    "gas.T": 3500.0,    # K  chamber flame temperature for regen
    "gas.P": Pc_design, # Pa
    # Pin shaft to design speed; full torque balance needs turbine choked-flow model
    "shaft.omega_override": OMEGA_DESIGN,
}

solver = SteadyStateSolver(layout, tol=1e-8)
X_ss   = solver.solve(X0, bcs)
layout.scatter_state_vector(X_ss)

chamber_comp = eng["chamber"]
shaft_comp   = eng["shaft"]
nozzle_comp  = eng["nozzle"]
gg_comp      = eng["gg"]
regen_comp   = eng["regen"]

print(f"\nGG LOX/CH4 at 10 MPa — Steady-State Balance:")
print(f"  Chamber pressure : {chamber_comp._state_values['P']/1e6:.3f} MPa  (design {Pc_design/1e6:.1f})")
print(f"  GG temperature   : {gg_comp._state_values['T']:.0f} K")
print(f"  Shaft speed      : {shaft_comp._state_values['omega'] * 30/np.pi:.0f} rpm")
print(f"  Vacuum thrust    : {nozzle_comp.last_outputs['thrust']:.0f} N   (design {F_design:.0f})")
print(f"  Vacuum Isp       : {nozzle_comp.last_outputs['Isp_vacuum']:.1f} s")
print(f"  GG bleed loss    : ~{GG_BLEED*100:.0f}% of total flow")
print(f"  Regen wall temp  : {regen_comp._state_values['T_wall']:.0f} K")
print(f"  Regen CH4 T_out  : {regen_comp.last_outputs.get('T_bulk_out', float('nan')):.0f} K")
print(f"  Regen heat load  : {regen_comp.last_outputs.get('Q_cool', float('nan'))/1e3:.1f} kW")
print(f"  Regen delta_P    : {regen_comp.last_outputs.get('delta_P', float('nan'))/1e3:.1f} kPa")

# ---------------------------------------------------------------------------
# 2. Monte Carlo — hardware variability
# ---------------------------------------------------------------------------
params = [
    UncertainParameter("lox_eta_scale",  ParameterType.NORMAL, mean=1.0,   std=0.025),
    UncertainParameter("fuel_eta_scale", ParameterType.NORMAL, mean=1.0,   std=0.025),
    UncertainParameter("gg_efficiency",  ParameterType.NORMAL, mean=0.93,  std=0.02),
    UncertainParameter("T_lox_inlet",    ParameterType.NORMAL, mean=91.0,  std=0.8),
    UncertainParameter("T_fuel_inlet",   ParameterType.NORMAL, mean=108.0, std=1.0),
]


def evaluate_engine(X: dict) -> float:
    h_lox  = lox.state_from_PT(P_lox_tank,  X["T_lox_inlet"]).h
    h_fuel = methane.state_from_PT(P_fuel_tank, X["T_fuel_inlet"]).h
    lay, e = build_engine(
        lox_eta_scale  = X["lox_eta_scale"],
        fuel_eta_scale = X["fuel_eta_scale"],
        gg_efficiency  = X["gg_efficiency"],
        T_lox_inlet    = X["T_lox_inlet"],
        T_fuel_inlet   = X["T_fuel_inlet"],
    )
    bcs_mc = {
        "lox_pump.inlet.P":  P_lox_tank,  "lox_pump.inlet.h":  h_lox,
        "fuel_pump.inlet.P": P_fuel_tank, "fuel_pump.inlet.h": h_fuel,
        "nozzle.P_ambient":  0.0,
        "gas.T": 3500.0, "gas.P": Pc_design,
        "shaft.omega_override": OMEGA_DESIGN,
    }
    X_sol = SteadyStateSolver(lay).solve(lay.assemble_state_vector(), bcs_mc)
    lay.scatter_state_vector(X_sol)
    return e["nozzle"].last_outputs["thrust"]


_mc_names10 = [p.name for p in params]

def evaluate_engine_row10(X: np.ndarray) -> float:
    d = {_mc_names10[i]: float(X[i]) for i in range(len(_mc_names10))}
    return evaluate_engine(d)

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate_engine_row10,
    n_samples=300,
    sampler="lhs",
    n_jobs=-1,
    seed=11,
)
mc_result = runner.run()
mc_result.print_summary()
mc_result.plot_histogram(bins=30, title="GG LOX/CH4 — Vacuum Thrust Distribution")
mc_result.save("outputs/gg_ch4_thrust_mc_300.hdf5")

# ---------------------------------------------------------------------------
# 3. Speed sweep — thrust envelope via turbopump shaft speed
#
# ``shaft.omega_override`` drives the rotor state to the target speed; the
# solver propagates omega to pumps and turbine through port connections so
# pump discharge pressure and chamber pressure vary correctly with speed.
# ---------------------------------------------------------------------------
def evaluate_at_speed(X: dict) -> dict:
    lay, e = build_engine()
    omega = float(X["omega"])
    bcs_sw = dict(bcs)
    bcs_sw["shaft.omega_override"] = omega
    X_sol = SteadyStateSolver(lay).solve(lay.assemble_state_vector(), bcs_sw)
    lay.scatter_state_vector(X_sol)
    return {
        "thrust": e["nozzle"].last_outputs["thrust"],
        "Pc":     e["chamber"]._state_values["P"],
        "Isp":    e["nozzle"].last_outputs["Isp_vacuum"],
        "P_lox_pump_out": e["lox_pump"].last_outputs["outlet.P"],
    }

speeds_rpm = np.linspace(15000, 33000, 15)
thrust_arr = np.zeros(len(speeds_rpm))
Pc_arr     = np.zeros(len(speeds_rpm))
Isp_arr    = np.zeros(len(speeds_rpm))
P_lox_out  = np.zeros(len(speeds_rpm))

for i, rpm in enumerate(speeds_rpm):
    out = evaluate_at_speed({"omega": rpm * np.pi / 30})
    thrust_arr[i] = out["thrust"]
    Pc_arr[i]     = out["Pc"]
    Isp_arr[i]    = out["Isp"]
    P_lox_out[i]  = out["P_lox_pump_out"]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(speeds_rpm / 1000, thrust_arr / 1000)
axes[0].set(xlabel="Pump speed [krpm]", ylabel="Thrust [kN]", title="Thrust vs pump speed")
ax1 = axes[1]
ax1.plot(speeds_rpm / 1000, Pc_arr / 1e6, label="Pc (chamber state)")
ax1.set(xlabel="Pump speed [krpm]", ylabel="Pc [MPa]")
ax1.tick_params(axis="y", labelcolor="C0")
ax1b = ax1.twinx()
ax1b.plot(speeds_rpm / 1000, P_lox_out / 1e5, color="C1", label="LOX pump outlet")
ax1b.set(ylabel="LOX pump outlet [bar]")
ax1b.tick_params(axis="y", labelcolor="C1")
ax1.set_title("Pc vs pump speed (outlet P responds)")
axes[2].plot(speeds_rpm / 1000, Isp_arr)
axes[2].set(xlabel="Pump speed [krpm]", ylabel="Isp [s]", title="Isp vs pump speed")
plt.suptitle("GG LOX/CH4 at 10 MPa — pump speed sweep", fontsize=13)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/gg_ch4_speed_sweep.png", dpi=150)
plt.close()

print(f"\nSpeed sweep range  ({speeds_rpm[0]:.0f}–{speeds_rpm[-1]:.0f} rpm):")
print(f"  Thrust : {thrust_arr.min()/1000:.2f} – {thrust_arr.max()/1000:.2f} kN")
print(f"  Pc     : {Pc_arr.min()/1e6:.3f} – {Pc_arr.max()/1e6:.3f} MPa")
print(f"  LOX pump outlet: {P_lox_out.min()/1e5:.2f} – {P_lox_out.max()/1e5:.2f} bar")
print(f"  Isp    : {Isp_arr.min():.1f} – {Isp_arr.max():.1f} s")
