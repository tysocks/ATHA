"""
Minimal Speed Sweep — Pump / Injector / Chamber / Nozzle / Shaft
================================================================
Demonstrates how pump shaft speed (via ``shaft.omega_override`` BC) controls
chamber pressure and thrust.  Uses only IdealGasBackend; no Cantera or CoolProp.

Engine topology (no turbine — omega is externally pinned):

  Tank ──► LOX Pump ──► LOX Injector ──► Combustion Chamber ──► Nozzle
  Tank ──► Fuel Pump ──► Fuel Injector ──┘
              │
          Rotor (shaft) ← omega_override from BCS

Physics of the speed-pressure coupling:
  1. Pump head    ΔP ∝ ω²     → P_pump_out rises with speed
  2. Injector ṁ ∝ √(P_pump - Pc)  → more flow at higher speed
  3. Choked nozzle ṁ ∝ Pc   → Pc rises until ṁ_in = ṁ_out
  → Pc and Thrust increase with pump speed (nonlinearly)

Isp should remain roughly constant because choked nozzle Cf depends only on
γ and area ratio, not on Pc directly.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.rotor import Rotor
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
from atha.thermo.ideal_gas import IdealGasBackend
from atha.solver.steady_state import SteadyStateSolver

# ── Fluid properties ─────────────────────────────────────────────────────────
GAMMA = 1.4
R_GAS = 287.0                           # J/(kg·K)  — air-like, for simplicity
CP    = GAMMA * R_GAS / (GAMMA - 1.0)  # 1004.5 J/(kg·K)

fluid = IdealGasBackend(gamma=GAMMA, R=R_GAS)

P_TANK  = 5e5    # Pa   tank pressure
T_TANK  = 300.0  # K    propellant temperature
h_inlet = fluid.state_from_PT(P_TANK, T_TANK).h  # J/kg

# ── Design-point parameters ───────────────────────────────────────────────────
Pc_design        = 2.0e6    # Pa
T_ADIABATIC      = 2000.0   # K  adiabatic flame temperature
OMEGA_DESIGN_RPM = 10_000   # rpm

P_pump_out  = Pc_design * 1.5           # Pa  pump discharge
dP_pump     = P_pump_out - P_TANK       # Pa  pump head

At    = 6e-4    # m²  nozzle throat
Ae_At = 10.0

ox_map   = PumpMap.from_design_point(0.5, dP_pump, OMEGA_DESIGN_RPM, 0.70)
fuel_map = PumpMap.from_design_point(0.5, dP_pump, OMEGA_DESIGN_RPM, 0.70)


def build_engine():
    eng = Engine("speed_sweep_demo")

    shaft     = Rotor("shaft",     moment_of_inertia=0.05,
                      initial_speed_rpm=OMEGA_DESIGN_RPM)
    ox_pump   = Pump("ox_pump",   diameter=0.08, pump_map=ox_map,   fluid=fluid)
    fuel_pump = Pump("fuel_pump", diameter=0.08, pump_map=fuel_map, fluid=fluid)
    ox_inj    = OrificeCompressible("ox_inj",   Cd=0.72, area=1.2e-4,
                                    gamma=GAMMA, R_gas=R_GAS)
    fuel_inj  = OrificeCompressible("fuel_inj", Cd=0.72, area=1.2e-4,
                                    gamma=GAMMA, R_gas=R_GAS)
    chamber   = CombustionChamber(
        "chamber", volume=3e-4,
        thermo=IdealGasBackend(gamma=GAMMA, R=R_GAS),
        fuel="fuel", oxidizer="ox",
        efficiency=1.0,
        T_adiabatic=T_ADIABATIC,
        initial_P=Pc_design, initial_T=T_ADIABATIC,
    )
    nozzle    = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, gamma=GAMMA)

    for c in [shaft, ox_pump, fuel_pump, ox_inj, fuel_inj, chamber, nozzle]:
        eng.add_component(c)

    # Shaft → pumps (reverse propagation carries omega to pump.shaft.omega)
    eng.connect(ox_pump.port("shaft"),   shaft.port("pump_ox"))
    eng.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

    # Propellant flow path
    eng.connect(ox_pump.port("outlet"),   ox_inj.port("inlet"))
    eng.connect(fuel_pump.port("outlet"), fuel_inj.port("inlet"))
    eng.connect(ox_inj.port("outlet"),   chamber.port("lox_inlet"))
    eng.connect(fuel_inj.port("outlet"), chamber.port("fuel_inlet"))
    eng.connect(chamber.port("outlet"),  nozzle.port("inlet"))

    return eng.compile(), eng


# ── Base BCS (shared between nominal and sweep) ───────────────────────────────
bcs_base = {
    "ox_pump.inlet.P":   P_TANK, "ox_pump.inlet.h":   h_inlet,
    "fuel_pump.inlet.P": P_TANK, "fuel_pump.inlet.h": h_inlet,
    "nozzle.P_ambient":  0.0,
    "shaft.omega_override": OMEGA_DESIGN_RPM * np.pi / 30.0,
}

# ── 1. Nominal steady-state ───────────────────────────────────────────────────
layout, eng = build_engine()
X_ss = SteadyStateSolver(layout, tol=1e-8).solve(
    layout.assemble_state_vector(), bcs_base)
layout.scatter_state_vector(X_ss)

Pc_nom     = eng["chamber"]._state_values["P"]
thrust_nom = eng["nozzle"].last_outputs["thrust"]
Isp_nom    = eng["nozzle"].last_outputs["Isp_vacuum"]
mdot_nom   = eng["nozzle"].last_outputs["mdot"]

print(f"\nNominal Steady-State at {OMEGA_DESIGN_RPM} rpm:")
print(f"  Chamber pressure  : {Pc_nom/1e6:.3f} MPa  (design {Pc_design/1e6:.1f} MPa)")
print(f"  Vacuum thrust     : {thrust_nom:.0f} N")
print(f"  Vacuum Isp        : {Isp_nom:.1f} s")
print(f"  Total mdot        : {mdot_nom:.3f} kg/s")
print(f"  Pump outlet P     : {eng['ox_pump'].last_outputs['outlet.P']/1e6:.3f} MPa"
      f"  (design {P_pump_out/1e6:.1f} MPa)")

# ── 2. Speed sweep: 50% – 130% design speed ──────────────────────────────────
def evaluate_at_speed(rpm: float) -> dict:
    lay, e = build_engine()
    bcs = dict(bcs_base)
    bcs["shaft.omega_override"] = rpm * np.pi / 30.0
    X = SteadyStateSolver(lay, tol=1e-7).solve(lay.assemble_state_vector(), bcs)
    lay.scatter_state_vector(X)
    return {
        "Pc":     e["chamber"]._state_values["P"],
        "thrust": e["nozzle"].last_outputs["thrust"],
        "Isp":    e["nozzle"].last_outputs["Isp_vacuum"],
        "P_pump": e["ox_pump"].last_outputs["outlet.P"],
    }


speeds_rpm = np.linspace(5000, 13000, 11)
results    = [evaluate_at_speed(r) for r in speeds_rpm]

Pc_arr     = np.array([r["Pc"]     for r in results])
thrust_arr = np.array([r["thrust"] for r in results])
Isp_arr    = np.array([r["Isp"]    for r in results])
P_pump_arr = np.array([r["P_pump"] for r in results])

print(f"\nSpeed sweep ({speeds_rpm[0]:.0f} – {speeds_rpm[-1]:.0f} rpm):")
print(f"  {'rpm':>8}  {'Pc [MPa]':>9}  {'F [N]':>8}  {'Pump [MPa]':>11}")
for i, rpm in enumerate(speeds_rpm):
    print(f"  {rpm:8.0f}  {Pc_arr[i]/1e6:9.3f}  {thrust_arr[i]:8.0f}  "
          f"{P_pump_arr[i]/1e6:11.3f}")

print(f"\n  Pump:   {P_pump_arr.min()/1e6:.3f} - {P_pump_arr.max()/1e6:.3f} MPa "
      f"(ratio {P_pump_arr.max()/P_pump_arr.min():.2f}x, expect {(13000/5000)**2:.2f}x from omega^2)")
print(f"  Pc:     {Pc_arr.min()/1e6:.3f} - {Pc_arr.max()/1e6:.3f} MPa")
print(f"  Thrust: {thrust_arr.min():.0f} - {thrust_arr.max():.0f} N")
print(f"  Isp:    {Isp_arr.min():.1f} - {Isp_arr.max():.1f} s  (should be ~constant)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(speeds_rpm / 1000, thrust_arr, "o-", color="C0")
axes[0].axvline(OMEGA_DESIGN_RPM / 1000, ls="--", color="gray", label="design")
axes[0].set(xlabel="Pump speed [krpm]", ylabel="Vacuum Thrust [N]",
            title="Thrust vs pump speed")
axes[0].legend()

ax1 = axes[1]
ax1.plot(speeds_rpm / 1000, Pc_arr / 1e6, "o-", color="C0", label="Pc")
ax1.set(xlabel="Pump speed [krpm]", ylabel="Chamber Pc [MPa]")
ax1.tick_params(axis="y", labelcolor="C0")
ax1b = ax1.twinx()
ax1b.plot(speeds_rpm / 1000, P_pump_arr / 1e6, "s--", color="C1", label="P_pump")
ax1b.set(ylabel="Pump outlet [MPa]")
ax1b.tick_params(axis="y", labelcolor="C1")
ax1.set_title("Pressures vs pump speed")
ax1.axvline(OMEGA_DESIGN_RPM / 1000, ls="--", color="gray")

axes[2].plot(speeds_rpm / 1000, Isp_arr, "o-", color="C2")
axes[2].axvline(OMEGA_DESIGN_RPM / 1000, ls="--", color="gray")
axes[2].set(xlabel="Pump speed [krpm]", ylabel="Isp [s]",
            title="Isp vs pump speed (≈ constant)")

plt.suptitle(
    "Speed Sweep: P_pump ∝ ω² → Pc and Thrust follow through injector coupling",
    fontsize=10)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/00_speed_sweep_minimal.png", dpi=150)
print("\nPlot saved -> outputs/00_speed_sweep_minimal.png")
plt.close()
