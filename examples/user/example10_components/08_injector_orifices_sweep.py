import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.orifice import OrificeCompressible
sys.path.append(str(Path(__file__).resolve().parent))
from _common import finish_plot


lox_inj = OrificeCompressible("lox_inj", Cd=0.72, area=1.2e-4)
fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=0.9e-4)

t = np.linspace(0.0, 1.5, 140)
P_in = np.linspace(60e5, 155e5, len(t))
P_out = np.linspace(30e5, 100e5, len(t))
T_in = np.full_like(t, 330.0)

mdot_lox = np.zeros_like(t)
mdot_fuel = np.zeros_like(t)

for i in range(len(t)):
    inp = {"inlet.P": P_in[i], "outlet.P": P_out[i], "inlet.T": T_in[i]}
    mdot_lox[i] = lox_inj.compute_outputs(t[i], {}, inp)["mdot"]
    mdot_fuel[i] = fuel_inj.compute_outputs(t[i], {}, inp)["mdot"]

print(f"LOX injector mdot range: {mdot_lox.min():.3f} -> {mdot_lox.max():.3f} kg/s")
print(f"Fuel injector mdot range: {mdot_fuel.min():.3f} -> {mdot_fuel.max():.3f} kg/s")

plt.figure(figsize=(8, 4))
plt.plot(t, mdot_lox, label="LOX injector")
plt.plot(t, mdot_fuel, label="Fuel injector")
plt.xlabel("Time [s]")
plt.ylabel("Mass flow [kg/s]")
plt.title("Injector orifice flow response to pressure ramp")
plt.grid(alpha=0.3)
plt.legend()
finish_plot("Example 10 Component Check: Injector orifice sweep", "08_injector_orifices_sweep.png")
