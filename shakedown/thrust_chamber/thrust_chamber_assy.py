"""
Thrust Chamber Assembly Runbox — Injector + Chamber + Nozzle
===========================================================

Minimal topology for fast iteration:

  LOX mdot ─► LOX injector ─┐
                             ├──► Combustion chamber ─► Nozzle ─► thrust/Isp
  Fuel mdot ─► Fuel injector ┘

Analyses:
  1) Connectivity diagram (component layout)
  2) Transient run driven by PMS profile (expanded telemetry)
  3) Sweep analysis: O/F (x) vs mdot_total (y)
  4) Monte Carlo + histograms for Isp, Pc, OF, and mdot_total
"""

import os
import csv
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
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.maps.performance_map import PerformanceMap
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
PMS_PROFILE_PATH = os.path.join(
    os.path.dirname(__file__), "setpoints", "runbox.csv"
)
CHAMBER_EFFICIENCY = 0.97
MC_SAMPLES = 200
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ---------------------------------------------------------------------------
# Performance maps
# ---------------------------------------------------------------------------
_of_axis = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)
_mdot_axis = np.array([2.0, 4.5, 7.0], dtype=float)
_mdot_scale_grid = np.array(
    [
        [0.985, 0.995, 1.005],
        [0.990, 1.000, 1.010],
        [0.995, 1.005, 1.015],
        [1.000, 1.010, 1.020],
    ],
    dtype=float,
)
INJECTOR_MDOT_SCALE_MAP = PerformanceMap.from_arrays(
    axes={"of_ratio": _of_axis, "mdot_total": _mdot_axis},
    outputs={"mdot_scale": _mdot_scale_grid},
    extrapolation="clamp",
)

_pc_axis = np.array([6e6, 8e6, 10e6, 12e6], dtype=float)
_mdot_noz_axis = np.array([2.0, 4.5, 7.0], dtype=float)
_cf_grid = np.array(
    [
        [1.46, 1.49, 1.52],
        [1.50, 1.53, 1.56],
        [1.54, 1.57, 1.60],
        [1.56, 1.59, 1.62],
    ],
    dtype=float,
)
NOZZLE_CF_MAP = PerformanceMap.from_arrays(
    axes={"inlet.P": _pc_axis, "inlet.mdot": _mdot_noz_axis},
    outputs={"Cf": _cf_grid},
    extrapolation="clamp",
)


def _load_pms_profile(path: str) -> dict:
    """
    Load PMS profile CSV with columns:
      - time_s (or alias)
      - mdot_total_kg_s (or alias)
      - of_ratio (or alias)
    """
    time_aliases = {"time", "time_s", "t", "seconds", "elapsed_time_s"}
    mdot_aliases = {"mdot_total", "mdot_total_kg_s", "massflow", "massflow_rate", "massflow_kg_s"}
    of_aliases = {"of", "of_ratio", "o_f", "oxidizer_fuel_ratio"}

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"PMS profile has no header: {path}")

        lower_to_orig = {h.strip().lower(): h for h in reader.fieldnames}

        def pick(aliases: set, label: str) -> str:
            for a in aliases:
                if a in lower_to_orig:
                    return lower_to_orig[a]
            raise ValueError(f"Missing PMS '{label}' column in {path}")

        time_key = pick(time_aliases, "time")
        mdot_key = pick(mdot_aliases, "mdot_total")
        of_key = pick(of_aliases, "of_ratio")

        times = []
        mdot_total_vals = []
        of_vals = []
        for row in reader:
            t = float(row[time_key])
            m = float(row[mdot_key])
            of = float(row[of_key])
            times.append(t)
            mdot_total_vals.append(m)
            of_vals.append(of)

    if len(times) < 2:
        raise ValueError("PMS profile requires at least two points")
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("PMS profile time values must be strictly increasing")

    return {
        "time_s": np.asarray(times, dtype=float),
        "mdot_total_kg_s": np.asarray(mdot_total_vals, dtype=float),
        "of_ratio": np.asarray(of_vals, dtype=float),
    }


def _mdot_split_from_pms_profile(t: float, profile: dict) -> tuple[float, float, float, float]:
    """Return (mdot_lox, mdot_fuel, mdot_total, of) at time t from PMS profile."""
    times = profile["time_s"]
    mdot_total_t = float(np.interp(t, times, profile["mdot_total_kg_s"], left=profile["mdot_total_kg_s"][0], right=profile["mdot_total_kg_s"][-1]))
    of_t = max(float(np.interp(t, times, profile["of_ratio"], left=profile["of_ratio"][0], right=profile["of_ratio"][-1])), 1e-12)
    mdot_fuel_t = mdot_total_t / (1.0 + of_t)
    mdot_lox_t = mdot_total_t - mdot_fuel_t
    return mdot_lox_t, mdot_fuel_t, mdot_total_t, of_t


def _apply_injector_maps(
    mdot_lox_in: float, mdot_fuel_in: float, mdot_total_in: float, of_ratio: float
) -> tuple[float, float]:
    """Scale injector-delivered mdot using the runbox performance map."""
    mdot_scale = INJECTOR_MDOT_SCALE_MAP.evaluate(
        {"of_ratio": of_ratio, "mdot_total": mdot_total_in}
    )["mdot_scale"]
    return mdot_lox_in * mdot_scale, mdot_fuel_in * mdot_scale


def build_engine(*, chamber_efficiency: float):
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
    nozzle = Nozzle(
        "nozzle",
        throat_area=At,
        exit_area=At * Ae_At,
        efficiencies=nozzle_eff,
        cf_map=NOZZLE_CF_MAP,
    )

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
# 1) Component connectivity diagram
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig_conn, ax_conn = plt.subplots(figsize=(9, 2.5))
ax_conn.axis("off")
nodes = {
    "lox_inj": (0.12, 0.65),
    "fuel_inj": (0.12, 0.25),
    "chamber": (0.5, 0.45),
    "nozzle": (0.82, 0.45),
}
for _name, (_x, _y) in nodes.items():
    ax_conn.text(_x, _y, _name, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#f3f6ff", ec="#4a6fa5"))
arrow_kw = dict(arrowstyle="->", lw=1.8, color="#444")
ax_conn.annotate("", xy=nodes["chamber"], xytext=nodes["lox_inj"], arrowprops=arrow_kw)
ax_conn.annotate("", xy=nodes["chamber"], xytext=nodes["fuel_inj"], arrowprops=arrow_kw)
ax_conn.annotate("", xy=nodes["nozzle"], xytext=nodes["chamber"], arrowprops=arrow_kw)
ax_conn.set_title("Connected Engine Components")
fig_conn.tight_layout()
fig_conn.savefig(os.path.join(OUTPUT_DIR, "runbox_components.png"), dpi=150)
plt.close(fig_conn)


# ---------------------------------------------------------------------------
# 2) Transient
# ---------------------------------------------------------------------------
layout_t, eng_t = build_engine(chamber_efficiency=CHAMBER_EFFICIENCY)
X0_t = layout_t.assemble_state_vector()
profile = _load_pms_profile(PMS_PROFILE_PATH)

def bcs_fn(t: float) -> dict:
    mdot_lox_t, mdot_fuel_t, mdot_total_t, of_t = _mdot_split_from_pms_profile(t, profile)
    mdot_lox_t, mdot_fuel_t = _apply_injector_maps(
        mdot_lox_t, mdot_fuel_t, mdot_total_t, of_t
    )
    return base_bcs(mdot_lox_in=mdot_lox_t, mdot_fuel_in=mdot_fuel_t)

sol = TransientSolver(layout_t, rtol=1e-3, atol=1e-6, max_step=1e-2).integrate((float(profile["time_s"][0]), float(profile["time_s"][-1])), X0_t, bcs_fn)
telemetry = {
    "time_s": sol.t.copy(),
    "mass_flow_rate_kg_s": np.zeros_like(sol.t),
    "of_ratio": np.zeros_like(sol.t),
    "chamber_pressure_MPa": np.zeros_like(sol.t),
    "lox_mdot_kg_s": np.zeros_like(sol.t),
    "fuel_mdot_kg_s": np.zeros_like(sol.t),
    "isp_vac_s": np.zeros_like(sol.t),
    "exit_pressure_MPa": np.zeros_like(sol.t),
    "fuel_inlet_pressure_MPa": np.full_like(sol.t, P_fuel_tank / 1e6),
    "lox_inlet_pressure_MPa": np.full_like(sol.t, P_lox_tank / 1e6),
    "chamber_temperature_K": np.zeros_like(sol.t),
    "thrust_N": np.zeros_like(sol.t),
}

for _i, _t in enumerate(sol.t):
    layout_t.scatter_state_vector(sol.X[_i])
    mdot_lox_t, mdot_fuel_t, mdot_total_t, of_t = _mdot_split_from_pms_profile(float(_t), profile)
    mdot_lox_t, mdot_fuel_t = _apply_injector_maps(
        mdot_lox_t, mdot_fuel_t, mdot_total_t, of_t
    )
    inputs = base_bcs(mdot_lox_in=mdot_lox_t, mdot_fuel_in=mdot_fuel_t)
    chamber = eng_t["chamber"]
    nozzle = eng_t["nozzle"]
    chamber_states = {k: chamber._state_values[k] for k in chamber.state_names}
    chamber_out = chamber.compute_outputs(float(_t), chamber_states, inputs)
    nozzle_inputs = {
        "inlet.P": chamber_out["outlet.P"],
        "inlet.h": chamber_out["outlet.h"],
        "inlet.mdot": mdot_total_t,
        "inlet.gamma": chamber_out.get("gamma", 1.2),
        "inlet.rho": chamber_out.get("rho", 0.0),
        "P_ambient": 0.0,
    }
    nozzle_out = nozzle.compute_outputs(float(_t), {}, nozzle_inputs)
    gamma = float(max(chamber_out.get("gamma", 1.2), 1.01))
    me = SimplifiedJANNAF._exit_mach(gamma, nozzle._epsilon)
    pc = chamber_out["outlet.P"]
    pe = pc * (1.0 + (gamma - 1.0) / 2.0 * me ** 2) ** (-gamma / (gamma - 1.0))
    telemetry["mass_flow_rate_kg_s"][_i] = mdot_total_t
    telemetry["of_ratio"][_i] = of_t
    telemetry["chamber_pressure_MPa"][_i] = chamber_out["outlet.P"] / 1e6
    telemetry["lox_mdot_kg_s"][_i] = mdot_lox_t
    telemetry["fuel_mdot_kg_s"][_i] = mdot_fuel_t
    telemetry["isp_vac_s"][_i] = nozzle_out.get("Isp_vacuum", float("nan"))
    telemetry["exit_pressure_MPa"][_i] = pe / 1e6
    telemetry["chamber_temperature_K"][_i] = chamber_out.get("T", float("nan"))
    telemetry["thrust_N"][_i] = nozzle_out.get("thrust", float("nan"))

fig_tr, axes_tr = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
plots = [
    ("mass_flow_rate_kg_s", "Mass flow [kg/s]"),
    ("of_ratio", "O/F [-]"),
    ("chamber_pressure_MPa", "Pc [MPa]"),
    ("lox_mdot_kg_s", "LOX mdot [kg/s]"),
    ("fuel_mdot_kg_s", "Fuel mdot [kg/s]"),
    ("isp_vac_s", "Isp [s]"),
    ("exit_pressure_MPa", "Exit pressure [MPa]"),
    ("fuel_inlet_pressure_MPa", "Fuel inlet P [MPa]"),
    ("lox_inlet_pressure_MPa", "LOX inlet P [MPa]"),
    ("chamber_temperature_K", "Chamber T [K]"),
    ("thrust_N", "Thrust [N]"),
]
for _ax, (key, ylabel) in zip(axes_tr.flatten(), plots):
    _ax.plot(telemetry["time_s"], telemetry[key])
    _ax.set_ylabel(ylabel)
    _ax.grid(alpha=0.3)
for _ax in axes_tr[-1]:
    _ax.set_xlabel("Time [s]")
fig_tr.suptitle("Transient Solver Telemetry", fontsize=13)
fig_tr.tight_layout()
fig_tr.savefig(os.path.join(OUTPUT_DIR, "runbox_transient_telemetry.png"), dpi=150)
plt.close(fig_tr)


# ---------------------------------------------------------------------------
# 3) Sweep analysis (x=O/F, y=mdot_total)
# ---------------------------------------------------------------------------
of_vals = np.linspace(1.5, 4.5, 13)
mdot_total_vals = np.linspace(2.0, 7.0, 13)
TH = np.zeros((len(mdot_total_vals), len(of_vals)))
PC = np.zeros_like(TH)

for _i, mdot_t in enumerate(mdot_total_vals):
    for _j, of in enumerate(of_vals):
        mdot_f = mdot_t / (1.0 + of)
        mdot_l = mdot_t - mdot_f
        mdot_l, mdot_f = _apply_injector_maps(mdot_l, mdot_f, mdot_t, of)
        lay, e = build_engine(chamber_efficiency=CHAMBER_EFFICIENCY)
        X_sol = SteadyStateSolver(lay, tol=1e-8).solve(lay.assemble_state_vector(), base_bcs(mdot_lox_in=mdot_l, mdot_fuel_in=mdot_f))
        lay.scatter_state_vector(X_sol)
        TH[_i, _j] = e["nozzle"].last_outputs["thrust"] / 1000.0
        PC[_i, _j] = e["chamber"]._state_values["P"] / 1e6

O, M = np.meshgrid(of_vals, mdot_total_vals)
fig_sw, axes_sw = plt.subplots(1, 2, figsize=(12, 4.5))
c0 = axes_sw[0].pcolormesh(O, M, TH, shading="auto")
axes_sw[0].set(xlabel="O/F ratio [-]", ylabel="mdot_total [kg/s]", title="Thrust [kN]")
fig_sw.colorbar(c0, ax=axes_sw[0])
c1 = axes_sw[1].pcolormesh(O, M, PC, shading="auto")
axes_sw[1].set(xlabel="O/F ratio [-]", ylabel="mdot_total [kg/s]", title="Pc [MPa]")
fig_sw.colorbar(c1, ax=axes_sw[1])
fig_sw.tight_layout()
fig_sw.savefig(os.path.join(OUTPUT_DIR, "runbox_mdot_sweep.png"), dpi=150)
plt.close(fig_sw)


# ---------------------------------------------------------------------------
# 4) Monte Carlo
# ---------------------------------------------------------------------------
params = [
    UncertainParameter("mdot_total", ParameterType.NORMAL, mean=mdot_total, std=0.03 * mdot_total),
    UncertainParameter("of_ratio", ParameterType.NORMAL, mean=3.5, std=0.12),
    UncertainParameter("chamber_efficiency", ParameterType.NORMAL, mean=CHAMBER_EFFICIENCY, std=0.01),
]
_mc_names = [p.name for p in params]
mc_records = {"Isp": [], "Pc": [], "OF": [], "Mdot_t": []}

def evaluate_engine_row(X: np.ndarray) -> float:
    d = {_mc_names[_i]: float(X[_i]) for _i in range(len(_mc_names))}
    of = max(d["of_ratio"], 1e-6)
    mdot_t = max(d["mdot_total"], 1e-6)
    mdot_f = mdot_t / (1.0 + of)
    mdot_l = mdot_t - mdot_f
    mdot_l, mdot_f = _apply_injector_maps(mdot_l, mdot_f, mdot_t, of)
    lay, e = build_engine(chamber_efficiency=d["chamber_efficiency"])
    X_sol = SteadyStateSolver(lay, tol=1e-8).solve(
        lay.assemble_state_vector(),
        base_bcs(mdot_lox_in=mdot_l, mdot_fuel_in=mdot_f),
    )
    lay.scatter_state_vector(X_sol)
    mc_records["Isp"].append(float(e["nozzle"].last_outputs.get("Isp_vacuum", np.nan)))
    mc_records["Pc"].append(float(e["chamber"]._state_values["P"] / 1e6))
    mc_records["OF"].append(float(of))
    mc_records["Mdot_t"].append(float(mdot_t))
    return e["nozzle"].last_outputs["thrust"]

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate_engine_row,
    n_samples=MC_SAMPLES,
    sampler="lhs",
    n_jobs=1,
    seed=11,
)
mc_result = runner.run()
mc_result.print_summary()
fig_mc, axes_mc = plt.subplots(2, 2, figsize=(10, 7))
hist_data = [
    (mc_records["Isp"], "Isp [s]"),
    (mc_records["Pc"], "Pc [MPa]"),
    (mc_records["OF"], "OF [-]"),
    (mc_records["Mdot_t"], "Mdot_total [kg/s]"),
]
for _ax, (_vals, _title) in zip(axes_mc.flatten(), hist_data):
    arr = np.asarray(_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    _ax.hist(arr, bins=20, alpha=0.8, edgecolor="white")
    _ax.set_title(_title)
    _ax.grid(alpha=0.3)
fig_mc.suptitle("Monte Carlo Histograms")
fig_mc.tight_layout()
fig_mc.savefig(os.path.join(OUTPUT_DIR, "runbox_mc_histograms.png"), dpi=150)
plt.close(fig_mc)
mc_result.save(os.path.join(OUTPUT_DIR, "runbox_thrust_mc_200.hdf5"))
