import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.gas_generator import GasGenerator
from atha.thermo.cantera_backend import CanteraBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import euler_integrate, finish_plot


Pc_design = 10.0e6
gg_thermo = CanteraBackend("gri30.yaml")
gg = GasGenerator(
    "gg",
    volume=1e-4,
    thermo=gg_thermo,
    fuel="CH4",
    oxidizer="O2",
    design_MR=0.35,
    efficiency=0.93,
    initial_P=Pc_design * 1.55,
    initial_T=1050.0,
)

state0 = dict(gg._state_values)
print(state0)

def input_fn(t, _s):
    ramp = min(max((t - 0.2) / 0.8, 0.0), 1.0)
    mdot_f = 0.07 * ramp
    mdot_o = 0.0245 * ramp
    return {
        "fuel_inlet.mdot": mdot_f,
        "lox_inlet.mdot": mdot_o,
        "outlet.mdot": 0.092 * ramp,
    }


t, states, _ = euler_integrate(gg, state0, input_fn, t_final=2.5, dt=0.002)
P_bar = states["P"] / 1e5
T = gg_thermo.state_from_Ph(states["P"][-1], states["h"][-1]).T
print(f"GG pressure range: {P_bar.min():.1f} -> {P_bar.max():.1f} bar")
print(f"GG final inferred temperature: {T:.0f} K")

plt.figure(figsize=(8, 4))
plt.plot(t, P_bar)
plt.xlabel("Time [s]")
plt.ylabel("GG pressure [bar]")
plt.grid(alpha=0.3)
finish_plot("Example 10 Component Check: Gas generator pressure transient", "03_gas_generator_transient.png")
