"""
Performance Maps — General-Purpose Axis Mapping
================================================
Demonstrates defining PerformanceMaps whose axes can be ANY named quantity:
states, inputs, computed outputs, or cross-component values from the BCS dict.

Use cases shown:
  1. Bellows-style CdA that grows with upstream pressure (1 axis: "inlet.P")
  2. 2-D pump efficiency map (axes: shaft speed and flow rate)
  3. Cross-component Isp map: turbine inlet temperature drives nozzle Isp
     (axes: "turbine.inlet.T", "chamber.P")
  4. Scatter-data discharge coefficient from test measurements
  5. CSV import — auto-detect structured vs scattered

Maps are backend-agnostic: the same PerformanceMap class handles arrays,
scattered data, callables, and constants with a unified API.
"""

import numpy as np
import matplotlib.pyplot as plt

from atha.maps.performance_map import PerformanceMap

# ─────────────────────────────────────────────────────────────────────────────
# 1. Bellows / pressure-expanding orifice  (1 axis: "inlet.P")
#    CdA grows nonlinearly with upstream pressure — typical of a flexible-wall
#    orifice or a bellows-type check-valve seat.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Bellows CdA map  (axis: 'inlet.P')")
print("=" * 60)

P_cal   = np.array([0.5e5, 1e5, 2e5, 3e5, 4e5, 5e5])   # Pa, calibration points
CdA_cal = np.array([8e-5,  1e-4, 1.4e-4, 1.65e-4, 1.8e-4, 1.9e-4])

bellows_cda_map = PerformanceMap.from_arrays(
    axes={"inlet.P": P_cal},
    outputs={"CdA": CdA_cal},
    extrapolation="clamp",
)

from atha.components.orifice import OrificeCompressible

orifice = OrificeCompressible("bellows", area=1e-4, cda_map=bellows_cda_map)

P_range = np.linspace(0.5e5, 5e5, 40)
mdots   = []
for P in P_range:
    out = orifice.compute_outputs(0.0, {}, {
        "inlet.P":  P,
        "outlet.P": 0.3e5,
        "inlet.T":  300.0,
    })
    mdots.append(out["mdot"])

print(f"  mdot at 1 bar -> {mdots[0]*1000:.3f} g/s")
print(f"  mdot at 5 bar -> {mdots[-1]*1000:.3f} g/s")

plt.figure(figsize=(7, 4))
plt.plot(P_range / 1e5, [m * 1000 for m in mdots], "b-")
plt.scatter(P_cal / 1e5, [
    orifice.compute_outputs(0.0, {}, {"inlet.P": p, "outlet.P": 0.3e5, "inlet.T": 300.0})["mdot"] * 1000
    for p in P_cal
], color="r", zorder=5, label="calibration points")
plt.xlabel("Inlet pressure [bar]")
plt.ylabel("Mass flow [g/s]")
plt.title("Bellows orifice: CdA(inlet.P)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/07_bellows_cda.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pump efficiency map  (2 axes: "shaft.omega", "inlet.mdot")
#    Typical centrifugal pump hill chart.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Pump 2-D efficiency map  (axes: 'shaft.omega', 'inlet.mdot')")
print("=" * 60)

omega_design = 22000 * np.pi / 30          # rad/s  (22 000 rpm)
mdot_design  = 1.93                         # kg/s

omega_vals = np.linspace(0.5, 1.2, 8) * omega_design
mdot_vals  = np.linspace(0.3, 1.5, 10) * mdot_design

# Synthetic hill-chart: peak efficiency at (omega_design, mdot_design)
def synthetic_eta(omega, mdot):
    phi  = mdot / (mdot_design * omega / omega_design)   # corrected flow coeff
    dphi = phi - 1.0
    return max(0.2, 0.70 - 0.18 * dphi**2 - 0.05 * ((omega / omega_design) - 1.0)**2)

eta_grid = np.array([[synthetic_eta(om, md) for md in mdot_vals]
                     for om in omega_vals])

pump_eta_map = PerformanceMap.from_arrays(
    axes={"shaft.omega": omega_vals, "inlet.mdot": mdot_vals},
    outputs={"efficiency": eta_grid},
    extrapolation="clamp",
)

from atha.components.pump import Pump, PumpMap
from atha.thermo.coolprop_backend import CoolPropBackend

lox = CoolPropBackend("Oxygen")
pump_base = PumpMap(mdot_design, 2.7e6, omega_design, 0.70)
pump = Pump(
    "lox_pump",
    diameter=0.08,
    pump_map=pump_base,
    fluid=lox,
    efficiency_map=pump_eta_map,
)

# Evaluate at a few operating points
for omega_rpm, mdot in [(18000, 1.5), (22000, 1.93), (24000, 2.2)]:
    fs_in = lox.state_from_PT(3e5, 91.0)
    out = pump.compute_outputs(0.0, {}, {
        "shaft.omega": omega_rpm * np.pi / 30,
        "inlet.mdot":  mdot,
        "inlet.P":     3e5,
        "inlet.h":     fs_in.h,
    })
    print(f"  {omega_rpm} rpm, {mdot:.2f} kg/s -> "
          f"dP={out['delta_P']/1e6:.3f} MPa, eta={out['efficiency']:.3f}")

# Plot efficiency contours
plt.figure(figsize=(8, 5))
OM, MD = np.meshgrid(omega_vals * 30 / np.pi, mdot_vals, indexing="ij")
plt.contourf(OM, MD, eta_grid, levels=20, cmap="RdYlGn")
plt.colorbar(label="η [-]")
plt.xlabel("Shaft speed [rpm]")
plt.ylabel("Mass flow [kg/s]")
plt.title("LOX pump efficiency map")
plt.tight_layout()
plt.savefig("outputs/07_pump_efficiency_map.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-component Isp map  (axes: "turbine.T_exit", "chamber.P")
#    The delivered specific impulse is mapped as a function of turbine-exit
#    temperature (which affects propellant enthalpy) and chamber pressure.
#    This demonstrates that map axes are not restricted to a single component.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Cross-component Isp map  (axes: 'turbine.T_exit', 'chamber.P')")
print("=" * 60)

T_turb = np.linspace(900, 1300, 5)     # K,  turbine exit temperature
Pc     = np.linspace(1e6, 3e6, 5)      # Pa, chamber pressure

# Synthetic Isp surface (realistic trends, not physics-exact)
def synthetic_Isp(T, P):
    return 280.0 + 5e-3 * (T - 1100) + 8.0 * (P / 2e6 - 1.0)

Isp_grid = np.array([[synthetic_Isp(t, p) for p in Pc] for t in T_turb])

isp_map = PerformanceMap.from_arrays(
    axes={"turbine.T_exit": T_turb, "chamber.P": Pc},
    outputs={"Isp": Isp_grid},
    extrapolation="clamp",
)

# The map is evaluated using the BCS context dict (flat namespace)
bcs_context = {
    "turbine.T_exit": 1050.0,
    "chamber.P":      2.1e6,
    "other_keys":     "ignored",
}
result = isp_map.evaluate(bcs_context)
print(f"  Isp at T_turb=1050K, Pc=2.1 MPa -> {result['Isp']:.1f} s")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scattered test data — discharge coefficient from fire tests
#    Each test point has (dP, temperature) and a measured Cd.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Scattered Cd map from test data")
print("=" * 60)

rng = np.random.default_rng(42)
n_tests = 30
dP_meas  = rng.uniform(5e4, 5e5, n_tests)
T_meas   = rng.uniform(250, 350, n_tests)
Cd_meas  = 0.72 + 0.05 * (dP_meas / 3e5 - 1.0) + 0.01 * (T_meas / 300 - 1.0) \
           + 0.005 * rng.standard_normal(n_tests)   # measurement noise

scattered_cd_map = PerformanceMap.from_scattered(
    points={"dP": dP_meas, "inlet.T": T_meas},
    outputs={"Cd": Cd_meas},
)

q = scattered_cd_map.evaluate({"dP": 2e5, "inlet.T": 300.0})
print(f"  Predicted Cd at dP=2 bar, T=300 K -> {q['Cd']:.4f}")
print(f"  (training data centroid Cd = {Cd_meas.mean():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Callable map — analytical model as a map
#    c* efficiency as a function of chamber pressure and MR,
#    based on a simple empirical correlation.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Callable map — η_cstar(chamber.P, MR)")
print("=" * 60)

def eta_cstar_model(**kw):
    Pc = kw["chamber.P"]
    MR = kw["MR"]
    eta = 0.97 * (1.0 - 0.01 * abs(MR - 2.8) - 2e-9 * abs(Pc - 2e6))
    return {"eta_cstar": max(0.8, min(1.0, eta))}

cstar_map = PerformanceMap.from_callable(
    fn=eta_cstar_model,
    axes=["chamber.P", "MR"],
    outputs=["eta_cstar"],
    bounds={"chamber.P": (0.5e6, 5e6), "MR": (1.5, 5.0)},
    extrapolation="clamp",
)

# Evaluate via context dict (dotted axis name would fail as a kwarg)
for Pc, MR in [(2e6, 2.8), (3e6, 2.2), (1e6, 4.0)]:
    res = cstar_map.evaluate({"chamber.P": Pc, "MR": MR})
    print(f"  Pc={Pc/1e6:.1f} MPa, MR={MR:.1f} -> eta_c*={res['eta_cstar']:.4f}")

# Rasterize so we can save it to HDF5
grid = {
    "chamber.P": np.linspace(0.5e6, 5e6, 20),
    "MR":        np.linspace(1.5,   5.0, 20),
}
cstar_map_arr = cstar_map.rasterize(grid)
cstar_map_arr.save("outputs/07_eta_cstar_map.h5")
print(f"\n  Rasterized callable map saved -> outputs/07_eta_cstar_map.h5")

loaded = PerformanceMap.load("outputs/07_eta_cstar_map.h5")
check  = loaded.evaluate({"chamber.P": 2e6, "MR": 2.8})
print(f"  Reloaded map check: η_c* = {check['eta_cstar']:.4f}")

print("\nDone — all example outputs written to outputs/")
