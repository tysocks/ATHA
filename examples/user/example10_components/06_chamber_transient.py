import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.combustion_chamber import CombustionChamber
from atha.thermo.cantera_backend import CanteraBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import finish_plot


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

state = dict(cc._state_values)


def input_fn(t):
    ramp = min(max((t - 0.3) / 1.0, 0.0), 1.0)
    mdot_lox = 10.2 * ramp
    mdot_fuel = 2.9 * ramp
    mdot_out = (mdot_lox + mdot_fuel) * (0.95 + 0.04 * ramp)
    return {"lox_inlet.mdot": mdot_lox, "fuel_inlet.mdot": mdot_fuel, "outlet.mdot": mdot_out}


t = np.linspace(0.0, 2.5, 220)
dP_dt = np.zeros_like(t)
dh_dt = np.zeros_like(t)

for i in range(len(t)):
    inputs = input_fn(t[i])
    out = cc.compute_outputs(t[i], state, inputs)
    derivs = cc.get_state_derivatives(t[i], state, inputs, out)
    dP_dt[i] = derivs["P"]
    dh_dt[i] = derivs["h"]

print(f"Chamber dP/dt range: {dP_dt.min()/1e6:.2f} -> {dP_dt.max()/1e6:.2f} MPa/s")

plt.figure(figsize=(8, 4))
plt.plot(t, dP_dt / 1e6, label="dP/dt")
plt.plot(t, dh_dt / 1e6, label="dh/dt")
plt.xlabel("Time [s]")
plt.ylabel("State derivative [scaled]")
plt.legend()
plt.grid(alpha=0.3)
finish_plot("Example 10 Component Check: Chamber dynamic tendency", "06_chamber_transient.png")
