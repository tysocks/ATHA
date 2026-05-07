import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.pump import Pump, PumpMap
from atha.thermo.coolprop_backend import CoolPropBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import finish_plot


lox = CoolPropBackend("Oxygen")
methane = CoolPropBackend("Methane")

P_lox_tank, T_lox = 4e5, 91.0
P_fuel_tank, T_fuel = 4e5, 108.0
Pc_design = 10.0e6
P_pump_out = Pc_design * 1.55
mdot_total = 20000.0 / (348.0 * 9.80665)
mdot_lox = mdot_total * 3.5 / (1 + 3.5)
mdot_fuel = mdot_total / (1 + 3.5)

lox_map = PumpMap.from_design_point(mdot_lox, P_pump_out - P_lox_tank, 30000, 0.70)
fuel_map = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 25000, 0.67)

lox_pump = Pump("lox_pump", diameter=0.085, pump_map=lox_map, fluid=lox)
fuel_pump = Pump("fuel_pump", diameter=0.055, pump_map=fuel_map, fluid=methane)

h_lox = lox.state_from_PT(P_lox_tank, T_lox).h
h_fuel = methane.state_from_PT(P_fuel_tank, T_fuel).h

t = np.linspace(0.0, 2.5, 200)
# Inputs:
# 1) Pump speed ramp (setpoint)
# 2) Outlet pressure ramp (setpoint)
rpm = np.linspace(24000.0, 32000.0, len(t))
omega = rpm * np.pi / 30.0
p_out = np.linspace(10000000, 11000000, len(t))

lox_dp, fuel_dp = np.zeros_like(t), np.zeros_like(t)
lox_tau, fuel_tau = np.zeros_like(t), np.zeros_like(t)
lox_mdot, fuel_mdot = np.zeros_like(t), np.zeros_like(t)

for i in range(len(t)):
    out_l = lox_pump.compute_outputs(
        t[i], {}, {"shaft.omega": omega[i], "inlet.P": P_lox_tank, "inlet.h": h_lox, "outlet.P": p_out[i]}
    )
    out_f = fuel_pump.compute_outputs(
        t[i], {}, {"shaft.omega": omega[i], "inlet.P": P_fuel_tank, "inlet.h": h_fuel, "outlet.P": p_out[i]}
    )
    lox_dp[i], fuel_dp[i] = out_l["delta_P"], out_f["delta_P"]
    lox_tau[i], fuel_tau[i] = out_l["tau_load"], out_f["tau_load"]
    lox_mdot[i], fuel_mdot[i] = out_l["inlet.mdot"], out_f["inlet.mdot"]

print(f"LOX pump dP range: {lox_dp.min()/1e5:.1f} -> {lox_dp.max()/1e5:.1f} bar")
print(f"Fuel pump dP range: {fuel_dp.min()/1e5:.1f} -> {fuel_dp.max()/1e5:.1f} bar")
print(f"LOX pump mdot range: {lox_mdot.min():.2f} -> {lox_mdot.max():.2f} kg/s")
print(f"Fuel pump mdot range: {fuel_mdot.min():.2f} -> {fuel_mdot.max():.2f} kg/s")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(rpm / 1000.0, lox_dp / 1e6, label="LOX pump")
ax[0].plot(rpm / 1000.0, fuel_dp / 1e6, label="Fuel pump")
ax[0].set(xlabel="Speed [krpm]", ylabel="delta_P [MPa]", title="Pump pressure rise")
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].plot(rpm / 1000.0, lox_mdot, label="LOX pump")
ax[1].plot(rpm / 1000.0, fuel_mdot, label="Fuel pump")
ax[1].set(xlabel="Speed [krpm]", ylabel="Mass flow rate [kg/s]", title="Pump mass flow rate")
ax[1].grid(alpha=0.3)
ax[1].legend()

ax[2].plot(rpm / 1000.0, lox_tau, label="LOX pump")
ax[2].plot(rpm / 1000.0, fuel_tau, label="Fuel pump")
ax[2].set(xlabel="Speed [krpm]", ylabel="tau_load [N*m]", title="Pump shaft load")
ax[2].grid(alpha=0.3)
ax[2].legend()

finish_plot("Example 10 Component Check: Pump pressure-speed to mdot", "02_pumps_speed_ramp.png")
