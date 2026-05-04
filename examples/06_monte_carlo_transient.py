"""
Monte Carlo with Transient Analysis
=====================================
Demonstrates using ProfileMonteCarloRunner to propagate hardware uncertainty
through a full transient firing sequence, not just a steady-state point.

Each MC sample:
  1. Perturbs pump efficiency, GG efficiency, and inlet temperature
  2. Runs a full TestProfile: steady trim → 5 s mainstage transient
  3. Extracts a scalar metric (e.g., peak chamber pressure, time to 90% thrust)

This answers questions like:
  "Given our hardware tolerance stack, what is the 95th-percentile peak
   chamber pressure during the throttle ramp?"
  "What fraction of runs will trip the 2.6 MPa abort limit?"

Computational cost: N_samples × t_sim_seconds × (Radau steps/s)
  Example: 200 samples × 5 s transient ≈ 3–8 min on 8 cores
  Use n_jobs=-1 to parallelise across all available cores.

NOTE: Uses planned components. ProfileMonteCarloRunner is implemented.
"""

import importlib.util
from pathlib import Path

import numpy as np
from atha.monte_carlo import ProfileMonteCarloRunner, UncertainParameter, ParameterType
from atha.profiles import (
    TestProfile, PhaseDefinition, PhaseMode, ControlCommand, SafetyLimit,
)
from atha.solver.steady_state import SteadyStateSolver

# Reuse engine builder from 04_gg_single_shaft_mc_sweep.py (numeric prefix → load by path)
_gg04 = Path(__file__).resolve().parent / "04_gg_single_shaft_mc_sweep.py"
_spec = importlib.util.spec_from_file_location("_atha_example_gg_single", _gg04)
_gg_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gg_mod)
build_engine = _gg_mod.build_engine
P_lox_tank = _gg_mod.P_lox_tank
P_fuel_tank = _gg_mod.P_fuel_tank
h_lox_inlet = _gg_mod.h_lox_inlet
h_fuel_inlet = _gg_mod.h_fuel_inlet
mdot_lox = _gg_mod.mdot_lox
mdot_fuel = _gg_mod.mdot_fuel
lox = _gg_mod.lox
ethanol = _gg_mod.ethanol

# ---------------------------------------------------------------------------
# Uncertain parameters
# ---------------------------------------------------------------------------
params = [
    UncertainParameter("lox_eta_scale",  ParameterType.NORMAL, mean=1.0,   std=0.025),
    UncertainParameter("fuel_eta_scale", ParameterType.NORMAL, mean=1.0,   std=0.025),
    UncertainParameter("gg_efficiency",  ParameterType.NORMAL, mean=0.93,  std=0.02),
    UncertainParameter("T_lox",          ParameterType.NORMAL, mean=91.0,  std=0.8),
]

# ---------------------------------------------------------------------------
# Profile template (throttle ramp to 65% and back)
# ---------------------------------------------------------------------------
def make_profile(h_lox):
    def throttle(t):
        if   t < 1.0:  return 1.0
        elif t < 2.0:  return 1.0 - 0.35 * (t - 1.0)
        elif t < 4.0:  return 0.65
        elif t < 5.0:  return 0.65 + 0.35 * (t - 4.0)
        return 1.0

    return TestProfile(
        name="transient_mc",
        phases=[
            PhaseDefinition(
                name="trim",
                mode=PhaseMode.STEADY_TRIM,
                duration=0.5,
            ),
            PhaseDefinition(
                name="throttle",
                mode=PhaseMode.TRANSIENT,
                duration=2.0,
                control_commands=[
                    ControlCommand("lox_pump.inlet.mdot",  fn=lambda t: mdot_lox  * throttle(t)),
                    ControlCommand("fuel_pump.inlet.mdot", fn=lambda t: mdot_fuel * throttle(t)),
                    ControlCommand("lox_pump.inlet.h",     fn=lambda t: h_lox),
                    ControlCommand("fuel_pump.inlet.h",    fn=lambda t: h_fuel_inlet),
                ],
                recording_rate_hz=100.0,
            ),
        ],
        global_limits=[
            SafetyLimit("Pc_max", "chamber", "P", upper_limit=2.6e6, is_hard=True),
        ],
    )

# ---------------------------------------------------------------------------
# apply_params: rebuild the engine with perturbed parameters
# The layout must also be rebuilt since component parameters are baked into it.
# ---------------------------------------------------------------------------
def apply_params(param_values: dict):
    """Return (profile, layout, X0) tuple for this MC sample."""
    h_lox  = lox.state_from_PT(P_lox_tank, param_values["T_lox"]).h
    layout, eng = build_engine(
        lox_eta_scale  = param_values["lox_eta_scale"],
        fuel_eta_scale = param_values["fuel_eta_scale"],
        gg_efficiency  = param_values["gg_efficiency"],
        T_lox_inlet    = param_values["T_lox"],
    )
    bcs = {
        "lox_pump.inlet.P":  P_lox_tank,  "lox_pump.inlet.h":  h_lox,
        "fuel_pump.inlet.P": P_fuel_tank, "fuel_pump.inlet.h": h_fuel_inlet,
        "nozzle.P_ambient":  0.0,
    }
    X0 = layout.assemble_state_vector()
    profile = make_profile(h_lox)
    return profile, layout, X0

# ---------------------------------------------------------------------------
# extract_metric: scalar from a completed TestProfileResult
# ---------------------------------------------------------------------------
def peak_chamber_pressure(profile_result) -> float:
    """Return peak Pc [MPa] during the throttle transient."""
    if not profile_result.success:
        return float("nan")
    trans = profile_result.get_phase("throttle")
    return trans.get("chamber", "P").max() / 1e6   # MPa

def time_to_restabilise(profile_result) -> float:
    """Return time [s] for Pc to return to within 2% of nominal after throttle-up."""
    if not profile_result.success:
        return float("nan")
    trans   = profile_result.get_phase("throttle")
    Pc      = trans.get("chamber", "P")
    t       = trans.t
    Pc_nom  = Pc[0]
    mask    = (t > 0.5 * t[-1]) & (np.abs(Pc - Pc_nom) / Pc_nom < 0.02)
    return float(t[mask][0]) if mask.any() else float("nan")

# ---------------------------------------------------------------------------
# Run MC — peak pressure metric
# ---------------------------------------------------------------------------
runner = ProfileMonteCarloRunner(
    params=params,
    apply_params_fn=apply_params,
    extract_metric=peak_chamber_pressure,
    n_samples=24,
    sampler="lhs",
    n_jobs=-1,              # all cores (threading backend; see MonteCarloRunner)
    seed=99,
)

result_Pc = runner.run()
result_Pc.print_summary()
# MonteCarloResult — 200 samples
#   mean     :  2.01 MPa
#   std      :  0.04 MPa
#   CV       :  2.0 %
#   5th/95th :  1.94 / 2.08 MPa
#   NaN (aborted): X / 200 runs

result_Pc.plot_histogram(bins=30, title="Peak Chamber Pressure Distribution (throttle transient)")
result_Pc.save("outputs/gg_transient_Pc_mc_24.hdf5")

# Abort rate (non-finite metric samples, e.g. failed sims)
n_abort = int(result_Pc.n_failed)
n_tot = len(result_Pc.Y_samples)
print(f"\nAbort rate: {n_abort}/{n_tot} = {100*n_abort/max(n_tot,1):.1f}%")

# ---------------------------------------------------------------------------
# Re-run for restabilisation time metric (same samples via fixed seed)
# ---------------------------------------------------------------------------
runner2 = ProfileMonteCarloRunner(
    params=params,
    apply_params_fn=apply_params,
    extract_metric=time_to_restabilise,
    n_samples=24, sampler="lhs", n_jobs=-1, seed=99,
)
result_t = runner2.run()
result_t.print_summary()
result_t.plot_histogram(bins=30, title="Throttle Restabilisation Time Distribution")
result_t.save("outputs/gg_transient_t_settle_mc_24.hdf5")
