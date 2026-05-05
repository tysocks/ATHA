# ATHA — Advanced Transient and High-fidelity Analysis

Python library for detailed simulation of liquid rocket engine steady-state and transient performance, implementing the modeling capabilities of the ROCKETS system (NASA/P&W 1991) and JANNAF performance methodology (CPIA 246, 1975).

## Features

- **Component-based engine modeling** — build any engine cycle by connecting components in a graph
- **Three run modes** — steady-state Newton-Raphson, transient Radau integration, and linearization
- **JANNAF performance** — simplified efficiency-factor chain (η_c*, η_Cd, η_v, η_div)
- **Thermodynamics** — ideal gas (testing), CoolProp (LOX/LH2/LCH4), Cantera (combustion)
- **Stiff ODE integration** — SciPy Radau handles acoustic-to-thermal timescale ratios of 10⁵
- **Phase 1 DAE foundation** — ordered variable/residual registries, structured layout evaluation, and scaled nonlinear solve diagnostics
- **Performance maps** — import test data, simulation results, or analytical models as maps on any axis (pressure, speed, temperature, cross-component values) and plug them into any component parameter
- **Test profiles** — multi-phase sequential simulations with safety limits and automatic abort
- **Monte Carlo analysis** — Latin Hypercube and Saltelli sampling, parallel runs, Sobol sensitivity indices
- **Regenerative cooling** — lumped `RegenChannel` with Bartz hot-side HTC, Dittus-Boelter coolant HTC, and wall temperature dynamics
- **Validated** against ROCKETS TTBE reference data (c* within 0.04%, Isp within 3.3%)

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, NumPy, SciPy, CoolProp, Cantera, Pydantic

## Quick Start

### Steady-State Single Volume

```python
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.steady_state import SteadyStateSolver

gas = IdealGasBackend(gamma=1.4, R=287.0)
vol = Volume("chamber", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
vol.add_inlet("inlet")
vol.add_outlet("outlet")

engine = Engine("test")
engine.add_component(vol)
layout = engine.compile()

X0 = layout.assemble_state_vector()
bcs = {"inlet.mdot": 1.0, "inlet.h": gas.state_from_PT(1e5, 300.0).h,
       "outlet.mdot": 1.0}

solver = SteadyStateSolver(layout)
X_ss = solver.solve(X0, bcs)
```

### Transient Pressure Rise

```python
from atha.solver.transient import TransientSolver

solver = TransientSolver(layout, method="Radau", max_step=0.05)

def bcs(t):
    return {"inlet.mdot": 0.1, "inlet.h": gas.state_from_PT(1e5, 300.0).h}

result = solver.integrate((0.0, 5.0), X0, bcs)
P = result.get("chamber", "P")   # numpy array over time
print(f"Final pressure: {P[-1]/1e5:.2f} bar")
```

### Nozzle Performance (JANNAF)

```python
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend

thermo = IdealGasBackend(gamma=1.24, R=711.0)   # LOX/LH2 products at MR=6
eff = JANNAFEfficiencies(eta_cstar=0.975, eta_Cd=0.98,
                          eta_velocity=0.99, eta_divergence=0.9830)
# First argument must be a ThermoBackend (gas), not JANNAFEfficiencies.
jannaf = SimplifiedJANNAF(thermo=thermo, efficiencies=eff,
                           throat_area=0.0687, exit_area=0.0687*77.5,
                           ambient_pressure=0.0)  # vacuum

result = jannaf.compute(P_chamber=20.6e6, T_chamber=3560.0,
                         MR=6.0, mdot_total=468.0)
print(f"Isp: {result.Isp:.1f} s   Thrust: {result.thrust/1000:.1f} kN")
```

### TTBE Validation Run

```python
from atha.validation.ttbe import run_ttbe_100pct_rpl

r = run_ttbe_100pct_rpl()
print(f"c*:    {r.c_star:.1f} m/s  (ref: 2365)")
print(f"Isp:   {r.Isp_vacuum:.1f} s    (ref: ~453)")
print(f"Thrust:{r.thrust_vacuum/1000:.1f} kN")
```

## Project Layout

```
atha/
├── core/          # Engine graph, compile step, port system
├── solver/        # SteadyStateSolver, TransientSolver, nonlinear utilities
├── thermo/        # IdealGasBackend, CoolPropBackend, CanteraBackend
├── components/    # Volume, Rotor, Pipe, Nozzle, Pump, Turbine, ...
├── jannaf/        # Simplified performance calculation
├── maps/          # PerformanceMap — any-axis interpolation from data, scatter, callable
├── profiles/      # Multi-phase test profiles, safety limits, HDF5 I/O
├── monte_carlo/   # Uncertainty quantification, Sobol sensitivity, parallel runs
├── validation/    # TTBE reference model
└── examples/      # Gas generator, staged combustion, pressure-fed

tests/
├── unit/          # Per-component and solver tests
└── integration/   # Multi-component and end-to-end scenarios

development/       # Technical reference documents
```

## Numerical Core Status

ATHA is being migrated toward an index-1 DAE architecture for realistic engine
cycle simulation. The first Phase 1 pieces are now available:

- `VariableRegistry` and `ResidualRegistry` record ordered metadata for states,
  algebraic variables, commands, parameters, outputs, and residuals.
- `EngineLayout.evaluate(t, X, Z, U)` returns an `EvaluationResult` containing
  `dXdt`, algebraic residuals, named outputs, residual names, and residual
  scales.
- Compiled fluid, shaft, and thermal connections register named residuals such
  as `connection.source.outlet__sink.inlet.mdot`, giving solver diagnostics a
  first-class view of graph coupling constraints.
- `TransientSolver` now routes RHS evaluation through `EngineLayout.evaluate`,
  preserving existing boundary-condition dictionary workflows while creating a
  common path for algebraic solves.
- `atha.solver.nonlinear.solve_nonlinear` provides the initial dense
  finite-difference Newton utility with residual scaling and named diagnostics.
- For square component-level algebraic systems, `TransientSolver` solves `Rz=0`
  inside the RHS and warm-starts from the previous successful `Z`.

Current limitation: this is a compatibility foundation, not the completed DAE
engine solver. Connection constraints are now registered and evaluated, but they
are not yet backed by a full port-variable `Z` vector or included in a square
global algebraic solve. That assembled connection solve is the next Phase 1 step
before broad component-fidelity upgrades.

## Running Tests

```bash
pytest tests/ -v                          # all tests
pytest tests/unit/ -v                     # unit only
pytest tests/integration/ -v              # integration only
pytest tests/ --co -q                     # list test names
```

## Test Profiles

Test profiles define multi-phase engine firing sequences. Each phase runs in sequence; the final state vector of one phase becomes the initial condition of the next. A hard safety limit anywhere in the sequence triggers an immediate abort and returns a failed `TestProfileResult`.

### Phase Modes


| Mode          | Description                                               |
| ------------- | --------------------------------------------------------- |
| `STEADY_TRIM` | Newton-Raphson solve to find a stationary operating point |
| `TRANSIENT`   | Time-integration driven by `ControlCommand` schedules     |
| `DWELL`       | Hold the current state vector for a fixed duration        |


### Defining a Multi-Phase Profile

```python
from atha.profiles import (
    PhaseDefinition, PhaseMode, ControlCommand,
    SafetyLimit, TestProfile,
)
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine

# --- build engine ---
gas = IdealGasBackend(gamma=1.4, R=287.0)
vol = Volume("chamber", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
vol.add_inlet("inlet")
engine = Engine("e")
engine.add_component(vol)
layout = engine.compile()
X0 = layout.assemble_state_vector()
h_ref = gas.state_from_PT(1e5, 300.0).h

# --- define profile ---
profile = TestProfile(
    name="fill_sequence",
    phases=[
        # Phase 1: trim to steady state with no inflow
        PhaseDefinition(
            name="pre_fill_trim",
            mode=PhaseMode.STEADY_TRIM,
            duration=5.0,
            trim_targets={"inlet.mdot": 0.0},
        ),
        # Phase 2: open valve ramp — mdot rises linearly 0→0.1 kg/s over 2 s
        PhaseDefinition(
            name="fill",
            mode=PhaseMode.TRANSIENT,
            duration=2.0,
            control_commands=[
                ControlCommand("inlet.mdot", fn=lambda t: 0.05 * t),
                ControlCommand("inlet.h",   fn=lambda t: h_ref),
            ],
            recording_rate_hz=100.0,   # sample rate for stored time series
        ),
        # Phase 3: hold at end state
        PhaseDefinition(
            name="hold",
            mode=PhaseMode.DWELL,
            duration=1.0,
        ),
    ],
    global_limits=[
        # Hard limit: abort if chamber pressure exceeds 2 bar
        SafetyLimit("P_max", component_name="chamber", state_name="P",
                    upper_limit=2e5, is_hard=True),
    ],
)

result = profile.execute(layout, X0)
```

### Reading Results

```python
if result.success:
    print(f"Total duration: {result.total_duration:.2f} s")

    # Access a single phase
    fill = result.get_phase("fill")
    P = fill.get("chamber", "P")   # numpy array at recorded time points
    print(f"Peak pressure: {P.max()/1e5:.2f} bar")

    # Stitch all phases into a single continuous timeline
    t_all, X_all = result.get_combined()

    # Quick matplotlib overview of every state across all phases
    result.plot_timeline()

else:
    print(f"Abort: {result.abort_reason} at t={result.abort_time:.3f} s")
```

### Saving and Loading

```python
from atha.profiles.io import save_profile_result, load_profile_result

save_profile_result(result, "fill_sequence.hdf5")
loaded = load_profile_result("fill_sequence.hdf5")
```

The HDF5 file stores one group per phase (`phase_000`, `phase_001`, …), each containing `t`, `X`, and `state_names` datasets plus metadata attributes for `abort_triggered`, `name`, etc.

---

## Monte Carlo Analysis

The `atha.monte_carlo` package propagates parametric uncertainty through any callable model, records per-run results, and computes Sobol first-order and total-order sensitivity indices.

### Defining Uncertain Parameters

```python
from atha.monte_carlo import UncertainParameter, ParameterType

params = [
    # Normal distribution: mean=0.975, σ=0.005
    UncertainParameter("eta_cstar", ParameterType.NORMAL, mean=0.975, std=0.005),
    # Uniform distribution: chamber pressure ±5%
    UncertainParameter("P_chamber", ParameterType.UNIFORM, low=19.0e6, high=21.0e6),
    # Log-normal: positive-definite quantities (e.g., oxidiser flow rate)
    UncertainParameter("mdot_ox", ParameterType.LOGNORMAL, mean=400.0, std=8.0),
]
```

### Running a Monte Carlo Sweep

```python
from atha.monte_carlo import MonteCarloRunner

def evaluate(X: dict) -> float:
    """Return a scalar metric for one sample.  X maps param.name → value."""
    eff.eta_cstar = X["eta_cstar"]
    res = jannaf.compute(P_chamber=X["P_chamber"], T_chamber=3560.0,
                         MR=6.0, mdot_total=468.0)
    return res.Isp

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate,
    n_samples=500,
    sampler="lhs",      # "lhs" (Latin Hypercube) or "saltelli" (Sobol sequences)
    n_jobs=4,           # parallel workers via joblib; -1 = all cores
    seed=42,
)
result = runner.run()
result.print_summary()
```

`print_summary()` output:

```
MonteCarloResult — 500 samples
  mean     : 452.7 s
  std      :   3.1 s
  CV       :   0.7 %
  5th/95th : 447.5 / 457.9 s
  95 % CI  : [452.4, 453.0]
```

### Plotting

```python
result.plot_histogram(bins=40, title="Isp distribution")
```

### Sobol Sensitivity Indices

Sobol indices require samples generated with the Saltelli scheme (`sampler="saltelli"`). Use a power-of-2 base count for clean convergence.

```python
from atha.monte_carlo.sensitivity import compute_sobol_indices

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate,
    n_samples=128,         # N_base; total evaluations = N_base * (k + 2)
    sampler="saltelli",
    seed=42,
)
result = runner.run()

sobol = compute_sobol_indices(
    params=params,
    param_samples=result.param_samples,   # shape (N_total, k)
    Y=result.Y,                           # shape (N_total,)
    N_base=128,
)

print(sobol)
# {
#   "S1":      {"eta_cstar": 0.62, "P_chamber": 0.03, "mdot_ox": 0.01, ...},
#   "ST":      {"eta_cstar": 0.65, "P_chamber": 0.05, "mdot_ox": 0.02, ...},
#   "S1_conf": {...},
#   "ST_conf": {...},
# }

result.plot_sobol_indices(sobol)
```

### Monte Carlo over Full Test Profiles

`ProfileMonteCarloRunner` applies parameter perturbations to a complete `TestProfile` execution. Failed runs (aborts or solver errors) are recorded as `NaN` and excluded from statistics.

```python
from atha.monte_carlo import ProfileMonteCarloRunner

def apply_params(profile_template, param_values: dict) -> "TestProfile":
    """Return a new profile with perturbed parameters."""
    # Clone the profile, update component parameters, return modified copy
    ...

def extract_metric(profile_result) -> float:
    """Pull a scalar from the completed TestProfileResult."""
    fill = profile_result.get_phase("fill")
    return float(fill.get("chamber", "P").max())

runner = ProfileMonteCarloRunner(
    params=params,
    profile=profile,
    layout=layout,
    X0=X0,
    apply_params_fn=apply_params,
    extract_metric=extract_metric,
    n_samples=200,
    sampler="lhs",
    n_jobs=-1,
    seed=0,
)
result = runner.run()
result.print_summary()
```

### Saving and Loading MC Results

```python
result.save("isp_mc_500.hdf5")

from atha.monte_carlo.results import MonteCarloResult
loaded = MonteCarloResult.load("isp_mc_500.hdf5")
```

---

## Performance Maps

`PerformanceMap` is a general-purpose multi-axis interpolation object. Its axes can be **any named quantity** — component states, flow inputs, BCS values, computed outputs, or cross-component values. Maps are used to replace scalar parameters in any component with physics derived from test data, simulations, or analytical correlations.

### Creating Maps

#### From structured grid data

```python
import numpy as np
from atha.maps import PerformanceMap

# 1-D map: CdA as a function of upstream pressure
P_cal   = np.array([0.5e5, 1e5, 2e5, 3e5, 5e5])
CdA_cal = np.array([8e-5,  1e-4, 1.4e-4, 1.65e-4, 1.9e-4])

cda_map = PerformanceMap.from_arrays(
    axes={"inlet.P": P_cal},
    outputs={"CdA": CdA_cal},
    extrapolation="clamp",   # "clamp" | "warn" | "error"
)

# 2-D map: pump efficiency over speed × flow
omega_vals = np.linspace(15000, 28000, 8) * np.pi / 30  # rad/s
mdot_vals  = np.linspace(0.5, 2.5, 10)                  # kg/s
eta_grid   = ...                                          # shape (8, 10)

eta_map = PerformanceMap.from_arrays(
    axes={"shaft.omega": omega_vals, "inlet.mdot": mdot_vals},
    outputs={"efficiency": eta_grid},
)
```

#### From scattered test data (RBF interpolation)

```python
# Test-measured discharge coefficients at arbitrary (ΔP, T) points
dP_meas  = np.array([5e4, 1e5, 2e5, 3e5, ...])   # Pa
T_meas   = np.array([270, 295, 310, 290,  ...])   # K
Cd_meas  = np.array([0.70, 0.72, 0.74, 0.73, ...])

cd_map = PerformanceMap.from_scattered(
    points={"dP": dP_meas, "inlet.T": T_meas},
    outputs={"Cd": Cd_meas},
)
```

#### From a callable (analytical model or another simulator)

```python
def eta_cstar_model(**kw):
    Pc = kw["chamber.P"]
    MR = kw["MR"]
    return {"eta_cstar": 0.97 * (1.0 - 0.01 * abs(MR - 2.8))}

cstar_map = PerformanceMap.from_callable(
    fn=eta_cstar_model,
    axes=["chamber.P", "MR"],
    outputs=["eta_cstar"],
    bounds={"chamber.P": (0.5e6, 5e6), "MR": (1.5, 5.0)},
)
```

#### Constant (no axes)

```python
pm = PerformanceMap.constant(eta_cstar=0.975, Cd=0.72)
pm()   # → {"eta_cstar": 0.975, "Cd": 0.72}
```

### Evaluating Maps

All maps share a unified evaluation interface.  At integration time, each component builds a **context dict** from its states and inputs, then passes it to the map.  The map extracts only the axes it declared by name — all other keys are silently ignored.

```python
# Simple axis names — keyword convenience
result = cda_map(inlet_P=2e5)

# Axis names with dots (e.g. "inlet.P") — must use evaluate()
result = cda_map.evaluate({"inlet.P": 2e5, "shaft.omega": 2094.4})
# {"CdA": 1.4e-4}

# evaluate() accepts the full BCS context — extra keys ignored
bcs = {"inlet.P": 2e5, "inlet.T": 300.0, "inlet.rho": 1141.0, "outlet.P": 1e5}
result = cda_map.evaluate(bcs)
```

### Using Maps with Components

Pass a `PerformanceMap` as an optional keyword argument to any supported component.  The component falls back to its scalar value if no map is provided.


| Component             | Argument              | Map output key      | Replaces                              |
| --------------------- | --------------------- | ------------------- | ------------------------------------- |
| `Pump`                | `efficiency_map`      | `"efficiency"`      | scalar `efficiency`                   |
| `Turbine`             | `efficiency_map`      | `"efficiency"`      | scalar `efficiency`                   |
| `Nozzle`              | `cf_map`              | `"Cf"`              | JANNAF Cf calculation                 |
| `Nozzle`              | `discharge_coeff_map` | `"discharge_coeff"` | scalar `discharge_coeff`              |
| `OrificeCompressible` | `cd_map`              | `"Cd"`              | scalar `discharge_coeff`              |
| `OrificeCompressible` | `cda_map`             | `"CdA"`             | `discharge_coeff × area`              |
| `Valve`               | `cd_map`              | `"Cd"`              | scalar `discharge_coeff`              |
| `Valve`               | `cda_map`             | `"CdA"`             | `discharge_coeff × A_frac × max_area` |


```python
from atha.components.orifice import OrificeCompressible

# Bellows orifice — CdA grows with pressure
orifice = OrificeCompressible("bellows", area=1e-4, cda_map=cda_map)

# Pump with full hill chart
from atha.components.pump import Pump
pump = Pump("lox_pump", delta_P_design=2.7e6, mdot_design=1.93,
            omega_design=22000 * np.pi / 30, efficiency_map=eta_map)

# Cross-component map: Isp is a function of turbine exit temperature
#   axes span two components — this is fully supported
isp_map = PerformanceMap.from_arrays(
    axes={"turbine.T_exit": np.linspace(900, 1300, 5),
          "chamber.P":      np.linspace(1e6, 3e6, 5)},
    outputs={"Isp": isp_grid},
)
# At evaluation time, pass the combined BCS context:
isp_map.evaluate({"turbine.T_exit": 1050.0, "chamber.P": 2.1e6})
```

### Extrapolation Modes


| Mode                | Behaviour                                 |
| ------------------- | ----------------------------------------- |
| `"clamp"` (default) | Silently clips axis values to data bounds |
| `"warn"`            | Clips, but emits a `UserWarning`          |
| `"error"`           | Raises `ValueError`                       |


### Saving and Loading

```python
# Array-backed and scattered maps save directly
pm.save("my_map.h5")
pm2 = PerformanceMap.load("my_map.h5")

# Callable maps must be rasterised first
grid = {"chamber.P": np.linspace(0.5e6, 5e6, 30),
        "MR":        np.linspace(1.5, 5.0, 30)}
pm_arr = callable_map.rasterize(grid)
pm_arr.save("eta_cstar_rasterised.h5")
```

---

## Regenerative Cooling

`RegenChannel` models a lumped regenerative cooling jacket around the combustion chamber and nozzle.  Methane (or any coolant) flows through the channel, absorbs heat from the hot gas wall, then enters the injector at elevated enthalpy.

### Physics


| Side           | Correlation                       | Notes                                         |
| -------------- | --------------------------------- | --------------------------------------------- |
| Hot (gas)      | Simplified Bartz, pressure-scaled | `h_hot = h_hot_design × (Pc/Pc_design)^0.8`   |
| Cool (coolant) | Dittus-Boelter, turbulent heating | `Nu = 0.023 Re^0.8 Pr^0.4` (laminar: Nu=3.66) |
| Pressure drop  | Darcy-Weisbach + Churchill (1977) | Valid all Re, smooth to rough                 |


Wall temperature (single dynamic state):

```
dT_wall/dt = (Q_hot − Q_cool) / (m_wall × Cp_wall)
```

Coolant energy and momentum balance (algebraic):

```
h_out = h_in + Q_cool / ṁ
ΔP    = f (L/D_h) ρ v² / 2
```

### Ports and Inputs


| Port / Input         | Description                          | Units |
| -------------------- | ------------------------------------ | ----- |
| `coolant_inlet.mdot` | Coolant mass flow                    | kg/s  |
| `coolant_inlet.P`    | Coolant inlet pressure               | Pa    |
| `coolant_inlet.h`    | Coolant inlet enthalpy               | J/kg  |
| `gas.T` *(BCS)*      | Hot gas temperature (from chamber)   | K     |
| `gas.P` *(BCS)*      | Hot gas pressure (for Bartz scaling) | Pa    |


### Outputs


| Output             | Description                       | Units    |
| ------------------ | --------------------------------- | -------- |
| `coolant_outlet.P` | Outlet pressure after ΔP          | Pa       |
| `coolant_outlet.h` | Outlet enthalpy after heat pickup | J/kg     |
| `Q_cool`           | Heat absorbed by coolant          | W        |
| `Q_hot`            | Heat from hot gas to wall         | W        |
| `h_hot_coeff`      | Bartz hot-side HTC                | W/(m²·K) |
| `h_cool_coeff`     | Dittus-Boelter coolant-side HTC   | W/(m²·K) |
| `T_bulk_out`       | Coolant outlet temperature        | K        |
| `delta_P`          | Channel pressure drop             | Pa       |


### Example

```python
from atha.components.regen_channel import RegenChannel
from atha.thermo.coolprop_backend import CoolPropBackend

methane = CoolPropBackend("Methane")

regen = RegenChannel(
    "regen",
    fluid=methane,
    channel_area=5e-5,        # m²   total cross-section
    hydraulic_diam=3e-3,      # m
    channel_length=0.8,       # m    chamber + nozzle contour
    hot_area=0.15,            # m²   gas-side area
    cool_area=0.18,           # m²   coolant-side area
    wall_mass=2.5,            # kg   CuCrZr jacket
    wall_cp=390.0,            # J/(kg·K)
    h_hot_design=55000.0,     # W/(m²·K)  Bartz at design Pc
    Pc_design=10e6,           # Pa
    recovery_factor=0.90,
    initial_T_wall=300.0,     # K   cold start
)

# Wire into engine: fuel pump → regen → fuel injector
engine.connect(fuel_pump.port("outlet"),      regen.port("coolant_inlet"))
engine.connect(regen.port("coolant_outlet"),  fuel_inj.port("inlet"))

# BCS must supply hot-gas conditions
bcs["gas.T"] = 3500.0   # K  (or read from chamber._state_values in transient)
bcs["gas.P"] = 10e6     # Pa
```

The hot-gas inputs are not connected via a typed port — they are provided as flat BCS keys.  In a transient simulation, update them each step from the live chamber state:

```python
def make_bcs(t):
    return {
        ...,
        "gas.T": chamber._state_values.get("T", 3500.0),
        "gas.P": chamber._state_values.get("P", 10e6),
    }
```

See `examples/11_regen_channel.py` for a complete GG LOX/CH4 engine with cold-start heat soak transient and throttle sensitivity sweep.

---

## Architecture Overview

ATHA uses a **compile-then-solve** pattern:

1. **Build phase** — Python objects (`Engine`, `Volume`, `Rotor`, ...) connected via typed ports
2. **Compile** — `Engine.compile()` assigns integer offsets and returns `EngineLayout` with flat numpy arrays
3. **Solve** — `TransientSolver` or `SteadyStateSolver` operates only on `EngineLayout` and a state vector `X`

This keeps Python overhead out of the hot integration loop. See `development/09_simulation_technical_reference.md` for full technical details.

## References

- ROCKETS System: NASA/P&W Final Report, November 1991 (NASA Technical Report 19910011919)
- JANNAF Performance Manual: CPIA Publication 246, April 1975
- Cantera: [https://cantera.org](https://cantera.org)
- CoolProp: [http://www.coolprop.org](http://www.coolprop.org)

