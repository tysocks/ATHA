# ATHA Component Rig — Design Spec
## Date: 2026-05-02
## Status: APPROVED — ready for implementation

---

## Goal

Enable single-component simulation: evaluate any ATHA component in isolation at a fixed operating point, sweep it across parameter ranges to generate performance maps, or run a transient simulation driven by time-varying boundary conditions. Supports component-level model validation against real test data.

## Architecture

A new `atha/analysis/` package with two classes:

- **`ComponentRig`** — wraps a single `BaseComponent`, provides `evaluate()` and `transient()` methods. No `Engine`, `compile()`, or port wiring required.
- **`ComponentSweep`** — structured parameter sweep over a `ComponentRig`. Parallel grid evaluation, returns `SweepResult` with plot helpers.

Both classes reuse existing ATHA infrastructure (`SteadyStateSolver`, `TransientSolver`, joblib) rather than reinventing evaluation logic.

---

## File Structure

```
atha/
└── analysis/
    ├── __init__.py          # exports: ComponentRig, ComponentSweep, SweepAxis, SweepResult
    ├── rig.py               # ComponentRig, RigResult
    └── sweep.py             # ComponentSweep, SweepAxis, SweepResult

tests/
└── unit/
    ├── test_rig.py
    └── test_sweep.py
```

No modifications to existing files except adding `analysis/` exports to `atha/__init__.py` if desired.

---

## `ComponentRig` — `atha/analysis/rig.py`

### Construction

```python
rig = ComponentRig(component: BaseComponent)
```

Internally calls `Engine("rig").add_component(component).compile()` to produce a minimal single-component `EngineLayout`. This layout is reused across all calls to avoid repeated compilation.

### `rig.required_inputs() -> List[str]`

Returns the list of BCS keys the component reads from `inputs` in its `compute_outputs()` and `get_state_derivatives()` calls. Implemented by running a probe evaluation with a recording proxy dict.

Used so callers can discover what to provide without reading source:
```python
>>> rig.required_inputs()
['inlet.P', 'inlet.h', 'inlet.mdot', 'omega']
```

### `rig.evaluate(bcs: dict) -> dict`

Single operating-point evaluation. Behaviour depends on component type:

**Algebraic component** (`n_states == 0`):
- Calls `comp.compute_outputs(0.0, {}, bcs)` directly.
- Returns the outputs dict.
- No solver involved.

**Dynamic component** (`n_states > 0`):
- Runs `SteadyStateSolver` to find `X*` where all `dX/dt = 0`.
- Uses `comp._state_values` as the initial guess for `X0`.
- Updates `comp._state_values` in-place (same behaviour as `SteadyStateSolver.solve()`).
- Returns outputs dict at the solved state.

```python
point = rig.evaluate({
    "inlet.P": 3e5,
    "inlet.h": h_lox,
    "inlet.mdot": 1.2,
    "omega": 15000 * np.pi / 30,
})
# point keys are whatever the component's compute_outputs() returns
# e.g. {"outlet.P": ..., "eta": ..., "power": ..., "tau": ...}
```

### `rig.transient(t_span, bcs_fn, X0=None, recording_rate_hz=100.0) -> RigResult`

Time-domain simulation with time-varying boundary conditions.

**Algebraic component** (`n_states == 0`):
- Evaluates `rig.evaluate(bcs_fn(t))` at each point in a uniform time grid derived from `recording_rate_hz`.
- Returns `RigResult` with a uniform time array and output arrays.
- No ODE integration — instantaneous response at each sample.

**Dynamic component** (`n_states > 0`):
- Runs `TransientSolver(layout, method="Radau")`.
- `X0` defaults to `layout.assemble_state_vector()` (current component state values).
- Returns `RigResult` wrapping `TransientSolution`.

```python
result = rig.transient(
    t_span=(0.0, 2.0),
    bcs_fn=lambda t: {
        "inlet.P": 3e5,
        "inlet.h": h_lox,
        "inlet.mdot": 1.2,
        "omega": (10000 if t < 0.5 else 20000) * np.pi / 30,
    },
    recording_rate_hz=200.0,
)
```

### `RigResult`

```python
@dataclass
class RigResult:
    component_name: str
    t: np.ndarray              # shape (N,)
    outputs: dict              # key -> np.ndarray shape (N,)
    state_names: List[str]     # empty for algebraic components
    X: Optional[np.ndarray]    # shape (N, n_states), None for algebraic

    def get(self, key: str) -> np.ndarray
    def plot(self, *keys, title=None, xlabel="Time [s]")
    def save(self, path: str)          # HDF5
```

`get()` raises `KeyError` with available keys listed if the key is not found.
`plot()` uses matplotlib; accepts one or more output keys as separate subplots.

---

## `ComponentSweep` — `atha/analysis/sweep.py`

### `SweepAxis`

```python
@dataclass
class SweepAxis:
    key: str           # BCS key to vary (e.g. "inlet.mdot")
    values: np.ndarray # the values to sweep over
```

### `ComponentSweep`

```python
sweep = ComponentSweep(
    rig: ComponentRig,
    axes: List[SweepAxis],          # 1 or 2 axes supported
    fixed_bcs: dict,                # BCS held constant across all points
    outputs: List[str],             # output keys to record
    n_jobs: int = 1,                # parallel workers (joblib)
    seed: Optional[int] = None,     # unused for deterministic sweep, reserved
)
```

Generates the full Cartesian grid of all axis combinations. For a 2-axis sweep of M×N points, runs M×N `rig.evaluate()` calls. Failed evaluations (solver errors) are stored as `NaN` and do not abort the sweep.

```python
result = sweep.run()
```

### `SweepResult`

```python
@dataclass
class SweepResult:
    axes: List[SweepAxis]
    outputs: dict               # key -> np.ndarray, shape matches axes grid
    n_failed: int

    def get(self, key: str) -> np.ndarray
    def plot_map(self, output_key, x_axis, y_axis,
                 x_label=None, y_label=None,
                 x_scale=1.0, y_scale=1.0,
                 title=None, colorbar_label=None)
    def plot_curve(self, output_key, sweep_axis,
                   fixed: dict = None,
                   label=None, xlabel=None, ylabel=None)
    def save(self, path: str)    # HDF5
    @classmethod
    def load(cls, path: str) -> "SweepResult"
```

**`plot_map()`**: 2D contour/pcolor plot. Requires exactly 2 axes. `x_scale`/`y_scale` convert stored SI values to display units (e.g., `y_scale=30/np.pi` to display rad/s as rpm).

**`plot_curve()`**: 1D line plot. For a 2-axis sweep, `fixed` pins one axis to the nearest stored value; for a 1-axis sweep, `fixed` is ignored. Overlays multiple curves if called repeatedly on the same axes object.

---

## Data Flow

```
User provides component + BCS
        │
        ▼
ComponentRig
  ├── evaluate()  ──►  comp.compute_outputs()  [algebraic]
  │                    SteadyStateSolver       [dynamic]
  │                    returns: dict
  │
  └── transient() ──►  point-by-point eval     [algebraic]
                       TransientSolver         [dynamic]
                       returns: RigResult

ComponentSweep
  └── run()  ──►  Cartesian grid of evaluate() calls (parallel)
                  returns: SweepResult
```

---

## Component Compatibility

| Component | n_states | evaluate() | transient() | Notes |
|-----------|----------|------------|-------------|-------|
| Pump | 0 | ✓ direct | ✓ point-by-point | BCS: inlet.P, inlet.h, inlet.mdot, omega |
| Turbine | 0 | ✓ direct | ✓ point-by-point | BCS: inlet.P, inlet.h, inlet.mdot, omega |
| Nozzle | 0 | ✓ direct | ✓ point-by-point | BCS: chamber.P, chamber.T, mdot |
| OrificeCompressible | 0 | ✓ direct | ✓ point-by-point | BCS: inlet.P, inlet.h, outlet.P |
| Valve | 0 | ✓ direct | ✓ point-by-point | BCS: inlet.P, inlet.h, inlet.mdot, position |
| Volume | 2 | ✓ Newton SS | ✓ Radau ODE | BCS: all port flows |
| Rotor | 1 | ✓ Newton SS | ✓ Radau ODE | BCS: tau_drive, tau_load |
| ThrottleValve* | 1 | ✓ Newton SS | ✓ Radau ODE | BCS: cmd; state: position |
| MetalNode | 1 | ✓ Newton SS | ✓ Radau ODE | BCS: Q_hot, Q_cool |

*ThrottleValve is a planned component (per ecosystem integration proposal).

---

## Testing Plan

### `tests/unit/test_rig.py`

```python
def test_rig_evaluate_algebraic_pump()
    # Pump with known operating point → verify dP matches expected
    # Uses IdealGasBackend or mock pump map

def test_rig_evaluate_dynamic_volume_steady_state()
    # Volume with equal in/out flow → SS solve gives dP/dt ≈ 0

def test_rig_transient_algebraic_step_response()
    # Valve with step in inlet.P → output flow changes instantaneously
    # Verify len(result.t) == expected from recording_rate_hz

def test_rig_transient_dynamic_volume_pressure_rise()
    # Volume with constant inflow → P rises over time (same as existing test)

def test_rig_required_inputs_returns_list()
    # Verify required_inputs() returns a non-empty list

def test_rig_result_get_missing_raises_keyerror()
    # RigResult.get("nonexistent") raises KeyError with available keys listed

def test_rig_result_save_load_roundtrip()
    # Save to HDF5, reload, assert arrays match
```

### `tests/unit/test_sweep.py`

```python
def test_sweep_1d_pump_flow_vs_head()
    # 1D sweep: vary inlet.mdot, fixed omega
    # Verify result shape is (N,) matching SweepAxis.values

def test_sweep_2d_pump_map_shape()
    # 2D sweep: vary mdot × omega
    # Verify result["outlet.P"] shape is (M, N)

def test_sweep_failed_points_nan()
    # Provide a BCS that causes solver failure at one grid point
    # Verify that point is NaN, n_failed == 1, others are valid

def test_sweep_result_save_load_roundtrip()
    # Save to HDF5, reload, assert arrays match

def test_sweep_1axis_raises_if_plot_map_called()
    # plot_map() with 1-axis sweep raises ValueError
```

---

## Example Usage (End-to-End)

```python
from atha.analysis import ComponentRig, ComponentSweep, SweepAxis
from atha.components.pump import Pump
from atha.thermo.coolprop_backend import CoolPropBackend
import numpy as np

lox = CoolPropBackend("Oxygen")
h_inlet = lox.state_from_PT(5e5, 90.0).h

pump = Pump("lox_pump", diameter=0.12, pump_map=load_map("lox_pump_map.csv"))
rig = ComponentRig(pump)

# 1. Single point
op = rig.evaluate({
    "inlet.P": 5e5, "inlet.h": h_inlet,
    "inlet.mdot": 1.5,
    "omega": 18000 * np.pi / 30,
})
print(f"Head rise: {(op['outlet.P'] - 5e5)/1e5:.1f} bar   η={op['eta']:.3f}")

# 2. Transient: ramp shaft speed 10k→20k rpm over 0.5s
result = rig.transient(
    t_span=(0.0, 1.0),
    bcs_fn=lambda t: {
        "inlet.P": 5e5, "inlet.h": h_inlet, "inlet.mdot": 1.5,
        "omega": np.interp(t, [0, 0.5, 1.0],
                           [10000, 20000, 20000]) * np.pi / 30,
    },
)
result.plot("outlet.P", "eta")

# 3. Performance map
sweep = ComponentSweep(
    rig=rig,
    axes=[
        SweepAxis("omega",      np.linspace(8000, 22000, 15) * np.pi / 30),
        SweepAxis("inlet.mdot", np.linspace(0.5, 3.0, 12)),
    ],
    fixed_bcs={"inlet.P": 5e5, "inlet.h": h_inlet},
    outputs=["outlet.P", "eta", "power"],
    n_jobs=-1,
)
map_result = sweep.run()
map_result.plot_map("eta",
    x_axis="inlet.mdot", y_axis="omega",
    x_label="Flow [kg/s]", y_label="Speed [rpm]",
    y_scale=30/np.pi, colorbar_label="Efficiency")
map_result.save("lox_pump_map_predicted.hdf5")
```
