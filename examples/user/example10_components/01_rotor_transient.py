import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.rotor import Rotor
sys.path.append(str(Path(__file__).resolve().parent))
from _common import euler_integrate, finish_plot


rotor = Rotor("shaft", moment_of_inertia=0.10, friction_coeff=0.01, initial_speed_rpm=5000)
rotor.port("turbine_in")
rotor.port("pump_lox")
rotor.port("pump_fuel")


def input_fn(t, _s):
    tau_drive = 150.0 if t > 0.4 else 60.0
    return {
        "turbine_in.tau": tau_drive,
        "pump_lox.tau": 25.0,
        "pump_fuel.tau": 20.0,
    }


t, states, _ = euler_integrate(rotor, {"omega": rotor._state_values["omega"]}, input_fn, t_final=3.0, dt=0.002)
rpm = states["omega"] * 30.0 / np.pi

print(f"Rotor rpm range: {rpm.min():.0f} -> {rpm.max():.0f}")

plt.figure(figsize=(8, 4))
plt.plot(t, rpm)
plt.xlabel("Time [s]")
plt.ylabel("Shaft speed [rpm]")
plt.grid(alpha=0.3)
finish_plot("Example 10 Component Check: Rotor transient", "01_rotor_transient.png")
