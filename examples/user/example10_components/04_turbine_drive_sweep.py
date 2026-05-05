import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.turbine import Turbine, TurbineMap
from atha.thermo.ideal_gas import IdealGasBackend
sys.path.append(str(Path(__file__).resolve().parent))
from _common import finish_plot


turb_map = TurbineMap.from_design_point(PR_design=5.5, eta_design=0.72, mdot_corrected_design=0.025)
turbine = Turbine("turbine", diameter=0.10, turbine_map=turb_map)
gas = IdealGasBackend(gamma=1.3, R=350.0)

t = np.linspace(0.0, 2.0, 180)
P_in = np.linspace(45e5, 80e5, len(t))
P_out = np.full_like(t, 15e5)
omega = np.linspace(18000.0, 32000.0, len(t)) * np.pi / 30.0
h_in = np.array([gas.state_from_PT(P_in[i], 1050.0).h for i in range(len(t))])

power = np.zeros_like(t)
tau_drive = np.zeros_like(t)

for i in range(len(t)):
    out = turbine.compute_outputs(
        t[i], {}, {"inlet.P": P_in[i], "inlet.h": h_in[i], "inlet.mdot": 0.09, "outlet.P": P_out[i], "shaft.omega": omega[i]}
    )
    power[i] = out["power"]
    tau_drive[i] = out["tau_drive"]

print(f"Turbine power range: {power.min()/1e3:.1f} -> {power.max()/1e3:.1f} kW")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, power / 1e3)
ax[0].set(xlabel="Time [s]", ylabel="Power [kW]", title="Turbine power transient")
ax[0].grid(alpha=0.3)
ax[1].plot(t, tau_drive)
ax[1].set(xlabel="Time [s]", ylabel="tau_drive [N*m]", title="Turbine shaft torque transient")
ax[1].grid(alpha=0.3)
finish_plot("Example 10 Component Check: Turbine drive sweep", "04_turbine_drive_sweep.png")
