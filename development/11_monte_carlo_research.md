# Monte Carlo Analysis Research: Uncertainty Quantification for ATHA

## Executive Summary

This document synthesizes JANNAF, SAE ARP4900, SALib, and aerospace industry best practices for implementing Monte Carlo uncertainty quantification (MCUQ) in ATHA. The goal is to enable statistical prediction of engine performance variations due to manufacturing tolerances, efficiency uncertainties, and propellant property variations.

**Key finding**: Latin Hypercube Sampling (LHS) + Sobol sensitivity analysis provides the best balance of computational efficiency and insight, requiring 500–2000 samples for rocket engines depending on dimensionality and desired accuracy. For 10 uncertain parameters, LHS converges at ~500 samples vs. ~10,000 for pure random sampling.

---

## Part 1: Uncertainty Sources in Rocket Engine Simulation

### Manufacturing Tolerances (SAE ARP4900)

| Component | Parameter | Typical 3σ Range | Isp Impact |
|-----------|-----------|-----------------|------------|
| Throat | Area (At) | ±1.5% | ±0.3% |
| Nozzle | Expansion ratio (ε) | ±2% | ±0.5% |
| Nozzle | Half-angle divergence | ±0.5° | ±0.2% |
| Injector | Flow distribution uniformity | ±2% radial | ±1% MR spatial |
| Coolant channels | Hydraulic diameter | ±3% | ±2% cooling |

SAE ARP4900 insight: Nozzle manufacturing is the critical path. ~80% of engine components are non-critical (deterministic design). The remaining 10–15% require probabilistic analysis.

### Thermodynamic Efficiency Uncertainties (JANNAF CPIA 246)

| Efficiency Factor | Nominal | 1σ Uncertainty | Notes |
|------------------|---------|----------------|-------|
| η_c* (combustion) | 0.975 | ±0.005 | Injector mixing quality |
| η_Cd (discharge) | 0.985 | ±0.003 | Manifold pressure losses |
| η_velocity | 0.990 | ±0.0025 | 1D viscous nozzle losses |
| η_divergence | 0.983 | ±0.004 | Boundary layer separation |
| η_two_phase | 1.000 | ±0.025 | If condensation occurs |
| η_boundary_layer | 0.990 | ±0.0025 | Skin friction + heat loss |
| **Combined Isp** | — | **±2%** | Multiplicative cascade |

**JANNAF structure**: Efficiency factors are **multiplicative**, not additive.
`Isp_actual = Isp_theoretical × η_c* × η_Cd × η_v × η_div × η_BL`

Degrading η_div from 0.983 to 0.975 multiplies Isp by 0.975, cutting ~3.5 seconds off 450 s Isp.

### Propellant Property Uncertainties

| Property | LOX | LH2 | Variation Source |
|----------|-----|-----|-----------------|
| Density | 1141 kg/m³ | 71 kg/m³ | ±1% from storage temperature |
| Specific enthalpy | -179 kJ/kg | -72 kJ/kg | ±0.5% from phase/composition |
| Mixture ratio | 6.0 | — | ±1% from flow control accuracy |
| T_adiabatic | 3560 K | — | ±1% from MR and composition |

Real-world data: LOX density varies ±0.5% across 80K–120K storage temperatures. LH2 trace N₂ contamination (5–50 ppm) shifts T_ad by ±10 K.

### Turbomachinery Uncertainties

| Source | Typical 1σ | Mechanism |
|--------|-----------|-----------|
| Turbine isentropic efficiency | ±1.5% | Blade clearances, manufacturing |
| Pump volumetric efficiency | ±1.0% | Impeller balance, cavitation |
| Feed-line ΔP | ±3% | Pipe friction, valve Cv scatter |
| Heat transfer coefficient | ±5% | Bartz correlation ±20% point-wise |
| Ignition timing | ±50 ms | Igniter reliability |

---

## Part 2: Sampling Strategy — LHS vs. Pure Monte Carlo

### Pure Random Sampling

```
Convergence: Error ~ O(1/√N)  → Need ~10,000 samples for 1% accuracy
Disadvantage: Clustering and gaps in parameter space
```

For 10 uncertain parameters: 10,000 samples × 30 s/sample = ~83 hours (serial).

### Latin Hypercube Sampling (LHS)

**Concept**: Stratified sampling — divide each parameter's range into N bins of equal probability, randomly pair one bin from each parameter (no replacement).

```python
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm

N = 500
k = 8   # number of uncertain parameters

sampler = LatinHypercube(d=k, seed=42)
unit_samples = sampler.random(N)         # shape (500, 8), values in [0,1]

# Transform [0,1] to physical space for Normal distribution:
# param_samples[:, j] = μ + σ * norm.ppf(unit_samples[:, j])
```

**Advantages over pure MC**:
- Stratified coverage: no clustering, even space-filling
- Convergence: Error ~ O(1/N) for smooth functions (10× faster than MC)
- For k=10 parameters: **500–1000 samples** achieves <1% accuracy in mean/variance

### Saltelli Sampling for Sobol Indices

For variance-based sensitivity analysis (Sobol indices), use Saltelli's scheme:

```python
from SALib.sample import saltelli

problem = {
    'num_vars': 8,
    'names': ['Pc', 'MR', 'eta_cstar', 'eta_div', 'At', 'epsilon',
              'eta_pump', 'eta_turbine'],
    'bounds': [
        [18e6, 23e6],       # Pc [Pa]
        [5.5, 6.5],         # MR
        [0.970, 0.990],     # η_c*
        [0.975, 0.991],     # η_div
        [0.0680, 0.0694],   # At [m²]
        [74, 81],           # ε
        [0.88, 0.92],       # η_pump
        [0.85, 0.91],       # η_turbine
    ]
}

# Saltelli generates N*(k+2) evaluations: for N=500, k=8 → 5000 model runs
param_values = saltelli.sample(problem, N=500, calc_second_order=False)
# Shape: (5000, 8)
```

**Cost tradeoff**: Saltelli costs N*(k+2) runs vs. N runs for basic LHS. For k=8, N=500: 5000 vs. 500 runs. Use Saltelli only when sensitivity indices are needed.

---

## Part 3: Sobol Sensitivity Analysis

### First-Order and Total-Order Indices

```
First-Order Index S_i:
  S_i = Var(E[Y | X_i]) / Var(Y)
  
  Interpretation: Fraction of output variance explained by X_i alone.
  Range: [0, 1]
  Example: S_Pc = 0.35 → 35% of Isp variance from Pc variation alone

Total-Order Index S_Ti:
  S_Ti = 1 - Var(E[Y | X_¬i]) / Var(Y)
  
  Interpretation: Total effect including all interactions with other parameters.
  Always ≥ S_i. If S_Ti >> S_i, strong interactions with other parameters.
  Example: S_T_MR = 0.31 → MR has interactions with Pc worth 0.03 extra variance

Key insight: Parameters with S_Ti ≈ S_i are non-interactive (safe to treat independently).
```

### Computing Indices with SALib

```python
from SALib.analyze import sobol

# Y must be the output array aligned with param_values from saltelli.sample()
Y = np.array([evaluate_engine_isp(X) for X in param_values])

Si = sobol.analyze(problem, Y, print_to_console=True, conf_level=0.95)

# Si['S1']      — first-order indices, shape (k,)
# Si['ST']      — total-order indices, shape (k,)
# Si['S1_conf'] — 95% confidence intervals, shape (k,)
# Si['ST_conf'] — 95% confidence intervals, shape (k,)
```

### Expected Results for LOX/LH2 Staged Combustion

Based on physics and JANNAF guidance, expected Sobol ranking for Isp sensitivity:

| Parameter | S_i (expected) | S_Ti (expected) | Action |
|-----------|---------------|-----------------|--------|
| Pc (chamber pressure) | 0.30–0.40 | 0.32–0.42 | Tighten pressure control |
| MR (mixture ratio) | 0.25–0.35 | 0.27–0.38 | Add closed-loop MR feedback |
| η_c* (combustion eff.) | 0.15–0.25 | 0.17–0.27 | Optimize injector mixing |
| η_div (divergence) | 0.10–0.18 | 0.12–0.20 | Better nozzle contour |
| At (throat area) | 0.03–0.06 | 0.04–0.07 | Limited ROI in tightening |
| ε (expansion ratio) | 0.01–0.03 | 0.01–0.04 | Not a driver |
| η_pump | <0.01 | <0.01 | Negligible at this design |
| η_turbine | <0.01 | <0.01 | Negligible at this design |

**Interpretation rule**: Parameters with S_Ti > 0.05 deserve engineering attention. Below 0.01, they are not worth tightening tolerances on.

---

## Part 4: Parallelization Strategies

### joblib (Single Machine, Recommended Default)

```python
from joblib import Parallel, delayed

def evaluate_one(X, idx, engine_template, bcs_nominal):
    """Worker function — runs in isolated subprocess."""
    import copy
    engine_copy = copy.deepcopy(engine_template)

    # Apply perturbed parameters
    # X is array of parameter values in order of uncertain_params list
    ...

    layout = engine_copy.compile()
    solver = SteadyStateSolver(layout)
    X_sol = solver.solve(X0_nominal, bcs_nominal)
    return extract_performance(layout, X_sol)

results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(evaluate_one)(X, i, engine, bcs)
    for i, X in enumerate(param_values)
)

# n_jobs=-1: use all CPU cores
# backend='loky': multiprocessing-safe for numpy/scipy (avoids GIL issues)
# Expected speedup: ~0.8 × n_cores
```

**When to use joblib**: Laptop/workstation with 2–32 cores. Simple setup, no cluster needed.

### Ray (Cluster-Ready)

```python
import ray

ray.init(num_cpus=64)   # or ray.init(address="auto") for a cluster

@ray.remote
def evaluate_one_remote(X, idx, engine_bytes, bcs):
    import pickle
    engine = pickle.loads(engine_bytes)
    # ... same as above
    return result

engine_bytes = ray.put(pickle.dumps(engine))   # share object efficiently

futures = [
    evaluate_one_remote.remote(X, i, engine_bytes, bcs)
    for i, X in enumerate(param_values)
]
results = ray.get(futures)
ray.shutdown()
```

**When to use Ray**: HPC cluster, 100+ cores, or when you want fault tolerance and a monitoring dashboard.

### Recommended Strategy

| Environment | Tool | n_jobs setting |
|-------------|------|---------------|
| Laptop (4–8 cores) | joblib | n_jobs=-1 |
| Workstation (16–32 cores) | joblib | n_jobs=-1 |
| HPC cluster (100+ cores) | Ray | ray.init() |

---

## Part 5: Convergence Criteria and Sample Size Selection

### Coefficient of Variation (CV) Test

```
CV(μ) = σ_Y / (√N × μ)

Stop when CV < target (typically 1%)

Required N:  N = (σ_Y / (target × μ))²

Example — Isp = 450 s, σ = 2 s, target CV = 1%:
  N = (2 / (0.01 × 450))² = (0.444)² → N ≈ 20 per batch
  But Sobol indices need much more: N_saltelli = 500–1000
```

### Recommended Sample Sizes

| Objective | k params | N (LHS) | N (Saltelli) | Wall Time (8 cores) |
|-----------|----------|---------|--------------|---------------------|
| Mean/variance | 5–10 | 200–500 | — | 10 min – 2 hr |
| Sensitivity (Sobol S_i, S_Ti) | 5–10 | — | 500–1000 × (k+2) | 2–10 hr |
| Sobol + 2nd-order (S_ij) | 5–10 | — | 1000–2000 × (2k+2) | 20–40 hr |

### Adaptive Convergence Check

```python
def check_convergence(Y_samples, target_cv=0.01):
    """Check if MC mean has converged."""
    mu = np.mean(Y_samples)
    sigma = np.std(Y_samples)
    cv = sigma / abs(mu) / np.sqrt(len(Y_samples))
    return cv < target_cv, cv

# Run in batches of 100, check convergence
Y_all = []
for batch in range(50):   # up to 5000 samples
    batch_results = run_batch(100)
    Y_all.extend(batch_results)
    converged, cv = check_convergence(np.array(Y_all))
    print(f"N={len(Y_all)}: CV={cv:.4f}")
    if converged:
        print(f"Converged at N={len(Y_all)}")
        break
```

---

## Part 6: Statistics and Reporting

### Key Metrics to Report

```python
def compute_mc_statistics(Y_valid):
    """Compute standard MC statistics for a scalar output."""
    N = len(Y_valid)
    mu = np.mean(Y_valid)
    sigma = np.std(Y_valid, ddof=1)

    return {
        "N_samples": N,
        "mean": mu,
        "std": sigma,
        "cv_pct": 100 * sigma / mu,
        "min": np.min(Y_valid),
        "max": np.max(Y_valid),
        "median": np.median(Y_valid),
        "p5": np.percentile(Y_valid, 5),
        "p95": np.percentile(Y_valid, 95),
        "p1": np.percentile(Y_valid, 1),
        "p99": np.percentile(Y_valid, 99),
        # 95% CI on the mean estimate:
        "mean_ci_95": 1.96 * sigma / np.sqrt(N),
        # Reliability index (distance from 95th percentile to mean, in σ)
        "beta_index": (np.percentile(Y_valid, 95) - mu) / sigma,
    }
```

### HDF5 Output Format

```python
import h5py

def save_mc_results(filename, param_samples, Y_samples, converged,
                    param_names, stats, sobol_indices=None):
    with h5py.File(filename, 'w') as f:
        # Raw data
        f.create_dataset("param_samples", data=param_samples)
        f.create_dataset("Y_samples", data=Y_samples)
        f.create_dataset("converged", data=converged)
        f.attrs["param_names"] = param_names

        # Statistics
        stats_grp = f.create_group("statistics")
        for key, val in stats.items():
            stats_grp.attrs[key] = val

        # Sobol indices (if computed)
        if sobol_indices is not None:
            sobol_grp = f.create_group("sobol")
            sobol_grp.create_dataset("S1", data=sobol_indices["S1"])
            sobol_grp.create_dataset("ST", data=sobol_indices["ST"])
            sobol_grp.create_dataset("S1_conf", data=sobol_indices["S1_conf"])
            sobol_grp.create_dataset("ST_conf", data=sobol_indices["ST_conf"])
            sobol_grp.attrs["param_names"] = param_names
```

---

## Part 7: Integration with Test Profiles

### Combined Workflow

The highest-value use case: run Monte Carlo over **full test profiles**, not just steady-state points.

```python
class ProfileMonteCarloAnalysis:
    """Run a TestProfile for each MC sample."""

    def evaluate_one(self, X, profile):
        # 1. Apply perturbed parameters to engine copy
        # 2. Execute full test profile (startup → mainstage → shutdown)
        # 3. Extract metrics: mainstage Isp mean, peak Pc, shutdown time

        try:
            engine_copy = self.perturb_engine(X)
            layout = engine_copy.compile()
            result = profile.execute(layout, X0_nominal)

            return {
                "success": result.abort_reason is None,
                "Isp_mainstage": result.mean_phase_metric("mainstage", "Isp"),
                "Pc_peak": result.max_metric("chamber.P"),
                "throttle_settling_time": result.settling_time("mainstage_65pct"),
                "abort_reason": result.abort_reason,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run(self, N_samples):
        param_samples = self.generate_lhs_samples(N_samples)
        results = Parallel(n_jobs=-1)(
            delayed(self.evaluate_one)(X, self.profile)
            for X in param_samples
        )

        success_rate = np.mean([r["success"] for r in results])
        Isp_dist = [r["Isp_mainstage"] for r in results if r["success"]]

        return {
            "success_rate": success_rate,
            "Isp_stats": compute_mc_statistics(Isp_dist),
            "raw": results,
        }
```

**Key insight**: Profile MC captures failure modes that steady-state MC misses:
- Startup abort due to Pc overshoot
- Throttle transient instability at an intermediate power level
- Shutdown incomplete (Pc doesn't decay to safe level within time limit)

---

## Part 8: Recommended ATHA Implementation Architecture

### New Modules Needed

```
atha/
├── monte_carlo/
│   ├── __init__.py
│   ├── parameters.py       # UncertainParameter, ParameterSet
│   ├── sampling.py         # LHSSampler, SaltelliSampler
│   ├── runner.py           # MonteCarloRunner (joblib parallel)
│   ├── sensitivity.py      # SobolAnalysis wrapping SALib
│   ├── statistics.py       # MCStatistics dataclass, compute_statistics()
│   └── results.py          # MonteCarloResult, HDF5 save/load, plotting
```

### API Design Goals

1. **Minimal boilerplate**: Defining 8 uncertain parameters and running MC should take <20 lines
2. **Pluggable engines**: Same analysis code works for JANNAF-only and full transient engines
3. **Progressive complexity**: basic MC in 5 lines, Sobol analysis opt-in with one flag
4. **Reproducibility**: random_seed always exposed and defaulted

```python
# Target API
from atha.monte_carlo import (
    UncertainParameter, ParameterType,
    MonteCarloRunner, MonteCarloConfig
)

params = [
    UncertainParameter("Pc", nominal=20.6e6, dist=ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("MR", nominal=6.0,    dist=ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("eta_cstar", nominal=0.975, dist=ParameterType.NORMAL, sigma=0.005),
]

config = MonteCarloConfig(params, n_samples=500, sensitivity=True, n_jobs=-1, seed=42)
runner = MonteCarloRunner(config)
result = runner.run(evaluate_fn=lambda X: jannaf.compute_with_params(X).Isp)

result.print_summary()
result.plot_histogram("Isp")
result.plot_sobol_indices()
result.save("ttbe_mc.hdf5")
```

---

## References

1. **JANNAF Rocket Engine Performance Manual** (CPIA 246, 1975) — Efficiency factor uncertainty ranges
2. **SAE ARP4900** — Liquid Rocket Engine Reliability Certification; probabilistic tolerance analysis
3. **SALib: Sensitivity Analysis Library** (Herman & Usher, 2017) — https://salib.readthedocs.io/
4. **scipy.stats.qmc** — Latin Hypercube Sampling in SciPy (1.7+)
5. **joblib Documentation** — Parallel computing in Python
6. **Saltelli et al., "Variance Based Sensitivity Analysis"** (Comput. Phys. Commun., 2010)
7. **McKay, Beckman, Conover, "Comparison of Three Methods"** (Technometrics, 1979) — Original LHS paper
8. **NASA/TM-2010-216420** — "Applying Monte Carlo Simulation to Launch Vehicle Design"
9. **ATHA Architecture Decisions**: `05_architecture_decisions.md`
