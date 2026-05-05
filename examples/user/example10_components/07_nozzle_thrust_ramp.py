import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from atha.components.nozzle import Nozzle
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
sys.path.append(str(Path(__file__).resolve().parent))
from _common import finish_plot


thermo = CanteraBackend("gri30.yaml")
eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
nozzle = Nozzle(
    "nozzle",
    throat_area=1.03e-3,
    exit_area=1.03e-3 * 50.0,
    thermo=thermo,
    efficiencies=eff,
)

t = np.linspace(0.0, 1.5, 160)
Pc = np.linspace(4.0e6, 11.0e6, len(t))
mdot = np.linspace(3.0, 7.0, len(t))
h = np.array([thermo.state_from_PT(Pc[i], 3500.0).h for i in range(len(t))])
thrust = np.zeros_like(t)
isp = np.zeros_like(t)

for i in range(len(t)):
    out = nozzle.compute_outputs(t[i], {}, {"inlet.P": Pc[i], "inlet.h": h[i], "inlet.mdot": mdot[i], "P_ambient": 0.0})
    thrust[i] = out["thrust"]
    isp[i] = out["Isp_vacuum"]

print(f"Nozzle thrust range: {thrust.min()/1e3:.2f} -> {thrust.max()/1e3:.2f} kN")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, thrust / 1e3)
ax[0].set(xlabel="Time [s]", ylabel="Thrust [kN]", title="Nozzle thrust ramp")
ax[0].grid(alpha=0.3)
ax[1].plot(t, isp)
ax[1].set(xlabel="Time [s]", ylabel="Isp [s]", title="Nozzle Isp ramp")
ax[1].grid(alpha=0.3)
finish_plot("Example 10 Component Check: Nozzle thrust ramp", "07_nozzle_thrust_ramp.png")
