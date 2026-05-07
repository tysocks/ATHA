import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.combustion_chamber import CombustionChamber
from atha.thermo.cantera_backend import CanteraBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import euler_integrate, finish_plot


Pc_design = 10.0e6
cc = CombustionChamber(
    "chamber",
    volume=3e-4,
    thermo=CanteraBackend("gri30.yaml"),
    fuel="CH4",
    oxidizer="O2",
    efficiency=0.97,
    initial_P=Pc_design,
    initial_T=3500.0,
)

def input_fn(t, _state):
    ramp = min(max((t - 0.3) / 1.0, 0.0), 1.0)
    mdot_lox = 10.2 * ramp
    mdot_fuel = 2.9 * ramp
    mdot_out = (mdot_lox + mdot_fuel)
    return {"lox_inlet.mdot": mdot_lox, "fuel_inlet.mdot": mdot_fuel, "outlet.mdot": mdot_out}


# Integrate the chamber's dynamic states (P, h) forward in time.
t, states, _outputs_hist = euler_integrate(
    cc,
    state0=dict(cc._state_values),
    input_fn=input_fn,
    t_final=2.5,
    dt=0.001,
)

Pc = states["P"]
mdot_total = np.zeros_like(t)
of_ratio = np.zeros_like(t)

dP_dt = np.zeros_like(t)
dh_dt = np.zeros_like(t)

for i in range(len(t)):
    s = {k: states[k][i] for k in states}
    inputs = input_fn(t[i], s)
    out = cc.compute_outputs(t[i], s, inputs)
    derivs = cc.get_state_derivatives(t[i], s, inputs, out)
    dP_dt[i] = derivs["P"]
    dh_dt[i] = derivs["h"]
    mdot_lox = float(inputs.get("lox_inlet.mdot", 0.0))
    mdot_fuel = float(inputs.get("fuel_inlet.mdot", 0.0))
    mdot_total[i] = mdot_lox + mdot_fuel
    of_ratio[i] = mdot_lox / max(mdot_fuel, 1e-12)

print(f"Chamber dP/dt range: {dP_dt.min()/1e6:.2f} -> {dP_dt.max()/1e6:.2f} MPa/s")

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

ax[0, 0].plot(t, mdot_total, label="mdot_total")
ax[0, 0].set(ylabel="Total mass flow [kg/s]", title="Total mass flow rate")
ax[0, 0].grid(alpha=0.3)

ax[0, 1].plot(t, of_ratio, label="O/F")
ax[0, 1].set(ylabel="O/F [-]", title="Mixture ratio (O/F)")
ax[0, 1].grid(alpha=0.3)

ax[1, 0].plot(t, Pc / 1e6, label="Pc")
ax[1, 0].set(xlabel="Time [s]", ylabel="Chamber pressure [MPa]", title="Chamber pressure")
ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(t, dP_dt / 1e6, label="dP/dt [MPa/s]")
ax[1, 1].plot(t, dh_dt / 1e6, label="dh/dt [MJ/kg/s]")
ax[1, 1].set(xlabel="Time [s]", ylabel="State derivative (scaled)", title="Dynamic tendency")
ax[1, 1].legend()
ax[1, 1].grid(alpha=0.3)

finish_plot("Example 10 Component Check: Chamber dynamic tendency", "06_chamber_transient.png")
