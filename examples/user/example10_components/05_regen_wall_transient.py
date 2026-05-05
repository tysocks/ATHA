import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.regen_channel import RegenChannel
from atha.thermo.coolprop_backend import CoolPropBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import euler_integrate, finish_plot


methane = CoolPropBackend("Methane")
Pc_design = 10.0e6
T_in = 108.0
h_in = methane.state_from_PT(4e5, T_in).h

regen = RegenChannel(
    "regen",
    fluid=methane,
    channel_area=5e-5,
    hydraulic_diam=3e-3,
    channel_length=0.7,
    hot_area=0.13,
    cool_area=0.16,
    wall_mass=2.0,
    wall_cp=390.0,
    h_hot_design=50000.0,
    Pc_design=Pc_design,
    recovery_factor=0.90,
    initial_T_wall=300.0,
)


def input_fn(t, _s):
    mdot = 0.4 + 0.8 * min(t / 1.0, 1.0)
    gas_T = 2600.0 + 900.0 * min(t / 1.2, 1.0)
    return {
        "coolant_inlet.mdot": mdot,
        "coolant_inlet.P": Pc_design * 1.55,
        "coolant_inlet.h": h_in,
        "gas.T": gas_T,
        "gas.P": Pc_design,
    }


t, states, outs = euler_integrate(regen, {"T_wall": 300.0}, input_fn, t_final=3.0, dt=0.003)
T_wall = states["T_wall"]
q_hot = np.array([o["Q_hot"] for o in outs])
q_cool = np.array([o["Q_cool"] for o in outs])

print(f"Regen wall T range: {T_wall.min():.1f} -> {T_wall.max():.1f} K")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, T_wall)
ax[0].set(xlabel="Time [s]", ylabel="Wall temperature [K]", title="Regen wall thermal transient")
ax[0].grid(alpha=0.3)
ax[1].plot(t, q_hot / 1e3, label="Q_hot")
ax[1].plot(t, q_cool / 1e3, label="Q_cool")
ax[1].set(xlabel="Time [s]", ylabel="Heat rate [kW]", title="Hot-side vs coolant-side heat flow")
ax[1].legend()
ax[1].grid(alpha=0.3)
finish_plot("Example 10 Component Check: Regen channel", "05_regen_wall_transient.png")
