"""
Thrust Chamber Assembly Runbox — Injector + Chamber + Nozzle
===========================================================

Minimal topology for fast iteration:

  LOX mdot ─► LOX injector ─┐
                             ├──► Combustion chamber ─► Nozzle ─► thrust/Isp
  Fuel mdot ─► Fuel injector ┘

Analyses:
  1) Steady-state solve at a nominal operating point
  2) Transient run (mdot ramps)
  3) Sweep analysis (grid over mdot_lox, mdot_fuel)
  4) Monte Carlo (uncertainty on inlet mdots and chamber efficiency)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atha.core.engine import Engine
from atha.components.injector import MassFlowInjector
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.solver.steady_state import SteadyStateSolver
from atha.solver.transient import TransientSolver
from atha.monte_carlo import MonteCarloRunner, UncertainParameter, ParameterType

# ---------------------------------------------------------------------------
# Propellants (for inlet enthalpies)
# ---------------------------------------------------------------------------
lox = CoolPropBackend("Oxygen")
methane = CoolPropBackend("Methane")

P_lox_tank, T_lox = 4e5, 91.0
P_fuel_tank, T_fuel = 4e5, 108.0
h_lox_inlet = lox.state_from_PT(P_lox_tank, T_lox).h
h_fuel_inlet = methane.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Nominal design point
# ---------------------------------------------------------------------------
Pc_design = 10.0e6
MR_main = 3.5
F_design = 20000.0
Isp_est = 348.0
At = 1.03e-3
Ae_At = 50.0

mdot_total = F_design / (Isp_est * 9.80665)
mdot_lox = mdot_total * MR_main / (1 + MR_main)
mdot_fuel = mdot_total / (1 + MR_main)


def build_engine(*, chamber_efficiency: float = 0.97):
    engine = Engine("runbox")

    cc_thermo = CanteraBackend(
        "gri30.yaml",
        initial_X="H2O:0.60,CO2:0.25,CO:0.08,H2:0.07",
    )
    chamber = CombustionChamber(
        "chamber",
        volume=3e-4,
        thermo=cc_thermo,
        fuel="CH4",
        oxidizer="O2",
        efficiency=chamber_efficiency,
        initial_P=Pc_design,
        initial_T=3500.0,
    )

    nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.985)
    nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

    lox_inj = MassFlowInjector("lox_inj")
    fuel_inj = MassFlowInjector("fuel_inj")

    for c in (lox_inj, fuel_inj, chamber, nozzle):
        engine.add_component(c)

    engine.connect(lox_inj.port("outlet"), chamber.port("lox_inlet"))
    engine.connect(fuel_inj.port("outlet"), chamber.port("fuel_inlet"))
    engine.connect(chamber.port("outlet"), nozzle.port("inlet"))

    return engine.compile(), engine


def base_bcs(*, mdot_lox_in: float, mdot_fuel_in: float) -> dict:
    return {
        "lox_inj.inlet.T": T_lox,
        "lox_inj.inlet.mdot": float(mdot_lox_in),

        "fuel_inj.inlet.T": T_fuel,
        "fuel_inj.inlet.mdot": float(mdot_fuel_in),

        "nozzle.P_ambient": 0.0,
    }


# ---------------------------------------------------------------------------
# 1) Steady state
# ---------------------------------------------------------------------------
layout, eng = build_engine()
X0 = layout.assemble_state_vector()

bcs = base_bcs(mdot_lox_in=mdot_lox, mdot_fuel_in=mdot_fuel)
X_ss = SteadyStateSolver(layout, tol=1e-8).solve(X0, bcs)
layout.scatter_state_vector(X_ss)

Pc_ss = eng["chamber"]._state_values["P"]
thrust_ss = eng["nozzle"].last_outputs["thrust"]
Isp_ss = eng["nozzle"].last_outputs["Isp_vacuum"]

print("\nRunbox steady-state:")
print(f"  Pc      : {Pc_ss/1e6:.3f} MPa")
print(f"  Thrust  : {thrust_ss:.1f} N")
print(f"  Isp_vac : {Isp_ss:.2f} s")
print(f"  mdot_lox / mdot_fuel : {mdot_lox:.3f} / {mdot_fuel:.3f} kg/s   (O/F={mdot_lox/max(mdot_fuel,1e-12):.3f})")


# ---------------------------------------------------------------------------
# 2) Transient
# ---------------------------------------------------------------------------
layout_t, eng_t = build_engine()
# Start transient from the steady-state operating point to avoid invalid nozzle
# states during the early ramp.
X0_t = X_ss.copy()

def bcs_fn(t: float) -> dict:
    # Small perturbation around the steady-state operating point.
    # Starting at the same mdot as X0_t avoids stiff start-up behavior.
    ramp = 1.0 + 0.25 * min(max((t - 0.2) / 0.8, 0.0), 1.0)  # 1.00 → 1.25
    return base_bcs(mdot_lox_in=mdot_lox * ramp, mdot_fuel_in=mdot_fuel * ramp)

sol = TransientSolver(layout_t, rtol=1e-3, atol=1e-6, max_step=1e-2).integrate(
    (0.0, 2.0), X0_t, bcs_fn
)
Pc_t = sol.get("chamber", "P") / 1e6

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(9, 4))
plt.plot(sol.t, Pc_t, label="Pc")
plt.xlabel("Time [s]")
plt.ylabel("Chamber pressure [MPa]")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/runbox_transient_pc.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3) Sweep analysis (mdot grid)
# ---------------------------------------------------------------------------
lox_vals = np.linspace(0.6 * mdot_lox, 1.4 * mdot_lox, 11)
fuel_vals = np.linspace(0.6 * mdot_fuel, 1.4 * mdot_fuel, 11)

TH = np.zeros((len(lox_vals), len(fuel_vals)))
PC = np.zeros_like(TH)
ISP = np.zeros_like(TH)

for i, ml in enumerate(lox_vals):
    for j, mf in enumerate(fuel_vals):
        lay, e = build_engine()
        X_sol = SteadyStateSolver(lay, tol=1e-8).solve(lay.assemble_state_vector(), base_bcs(mdot_lox_in=ml, mdot_fuel_in=mf))
        lay.scatter_state_vector(X_sol)
        TH[i, j] = e["nozzle"].last_outputs["thrust"]
        PC[i, j] = e["chamber"]._state_values["P"]
        ISP[i, j] = e["nozzle"].last_outputs["Isp_vacuum"]

L, F = np.meshgrid(fuel_vals, lox_vals)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

pcm0 = axes[0].pcolormesh(F, L, TH / 1000.0, shading="auto")
axes[0].set(xlabel="mdot_lox [kg/s]", ylabel="mdot_fuel [kg/s]", title="Thrust [kN]")
plt.colorbar(pcm0, ax=axes[0])

pcm1 = axes[1].pcolormesh(F, L, PC / 1e6, shading="auto")
axes[1].set(xlabel="mdot_lox [kg/s]", ylabel="mdot_fuel [kg/s]", title="Pc [MPa]")
plt.colorbar(pcm1, ax=axes[1])

pcm2 = axes[2].pcolormesh(F, L, ISP, shading="auto")
axes[2].set(xlabel="mdot_lox [kg/s]", ylabel="mdot_fuel [kg/s]", title="Isp_vac [s]")
plt.colorbar(pcm2, ax=axes[2])

plt.suptitle("Runbox sweep: mdot_lox × mdot_fuel", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/runbox_mdot_sweep.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 4) Monte Carlo
# ---------------------------------------------------------------------------
params = [
    UncertainParameter("mdot_lox", ParameterType.NORMAL, mean=mdot_lox, std=0.03 * mdot_lox),
    UncertainParameter("mdot_fuel", ParameterType.NORMAL, mean=mdot_fuel, std=0.03 * mdot_fuel),
    UncertainParameter("chamber_efficiency", ParameterType.NORMAL, mean=0.97, std=0.01),
]

_mc_names = [p.name for p in params]

def evaluate_engine_row(X: np.ndarray) -> float:
    d = {_mc_names[i]: float(X[i]) for i in range(len(_mc_names))}
    lay, e = build_engine(chamber_efficiency=d["chamber_efficiency"])
    X_sol = SteadyStateSolver(lay, tol=1e-8).solve(
        lay.assemble_state_vector(),
        base_bcs(mdot_lox_in=d["mdot_lox"], mdot_fuel_in=d["mdot_fuel"]),
    )
    lay.scatter_state_vector(X_sol)
    return e["nozzle"].last_outputs["thrust"]

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate_engine_row,
    n_samples=200,
    sampler="lhs",
    n_jobs=-1,
    seed=11,
)
mc_result = runner.run()
mc_result.print_summary()
mc_result.plot_histogram(bins=30, title="Runbox — Vacuum Thrust Distribution")
mc_result.save("outputs/runbox_thrust_mc_200.hdf5")
